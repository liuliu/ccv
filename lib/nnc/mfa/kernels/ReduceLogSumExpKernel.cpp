#include "ReduceLogSumExpKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

ReduceLogSumExpKernel::ReduceLogSumExpKernel(ReduceLogSumExpKernelDescriptor descriptor, MTL::Device* const device)
{
  memoryPrecision = descriptor.memoryPrecision;
  source = createSource();
  auto source_string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(source_string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string ReduceLogSumExpKernel::createSource() const noexcept
{
  std::string source;
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    source = "typedef float real;\n";
  } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
    source = "typedef bfloat real;\n";
  } else {
    source = "typedef half real;\n";
  }
  source += R"(
#include <metal_stdlib>
using namespace metal;

constant uint column_count [[function_constant(0)]];
constant uint partition_size [[function_constant(1)]];
constant uint partition_count [[function_constant(2)]];
constant float input_scale [[function_constant(3)]];

inline float reduce_max_pair(float lhs, float rhs)
{
  return rhs > lhs ? rhs : lhs;
}

inline float reduce_max_simd(float value)
{
  for (ushort delta = 16; delta > 0; delta >>= 1)
    value = reduce_max_pair(value, simd_shuffle_down(value, delta));
  return value;
}

inline float reduce_sum_simd(float value)
{
  for (ushort delta = 16; delta > 0; delta >>= 1)
    value += simd_shuffle_down(value, delta);
  return value;
}

inline float reduce_max_threadgroup(
  float value,
  ushort lane_id,
  ushort simdgroup_id,
  ushort threads_per_threadgroup,
  threadgroup float* partials)
{
  value = reduce_max_simd(value);
  if (lane_id == 0)
    partials[simdgroup_id] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    const ushort simdgroup_count = (threads_per_threadgroup + 31) / 32;
    value = lane_id < simdgroup_count ? partials[lane_id] : -INFINITY;
    value = reduce_max_simd(value);
    if (lane_id == 0)
      partials[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return partials[0];
}

inline float reduce_sum_threadgroup(
  float value,
  ushort lane_id,
  ushort simdgroup_id,
  ushort threads_per_threadgroup,
  threadgroup float* partials)
{
  value = reduce_sum_simd(value);
  if (lane_id == 0)
    partials[simdgroup_id] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    const ushort simdgroup_count = (threads_per_threadgroup + 31) / 32;
    value = lane_id < simdgroup_count ? partials[lane_id] : 0;
    value = reduce_sum_simd(value);
  }
  return value;
}

inline float reduce_logsumexp_range(
  device const real* input,
  ulong row_offset,
  uint range_begin,
  uint range_end,
  ushort thread_id,
  ushort lane_id,
  ushort simdgroup_id,
  ushort threads_per_threadgroup,
  threadgroup float* partials,
  thread float* maximum)
{
  float local_maximum = -INFINITY;
  for (uint first_column = range_begin + uint(thread_id) * 4;
       first_column < range_end;
       first_column += uint(threads_per_threadgroup) * 4) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; i++) {
      const uint column = first_column + i;
      if (column < range_end) {
        const float value = input_scale * float(input[row_offset + column]);
        local_maximum = reduce_max_pair(local_maximum, value);
      }
    }
  }
  *maximum = reduce_max_threadgroup(local_maximum, lane_id, simdgroup_id, threads_per_threadgroup, partials);

  float local_sum = 0;
  if (isfinite(*maximum)) {
    for (uint first_column = range_begin + uint(thread_id) * 4;
         first_column < range_end;
         first_column += uint(threads_per_threadgroup) * 4) {
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 4; i++) {
        const uint column = first_column + i;
        if (column < range_end) {
          const float value = input_scale * float(input[row_offset + column]);
          local_sum += precise::exp(value - *maximum);
        }
      }
    }
  } else if (*maximum < 0) {
    // Preserve an all-NaN partition so it poisons a finite maximum found by
    // another partition. An all-negative-infinity partition contributes zero.
    for (uint column = range_begin + thread_id;
         column < range_end;
         column += threads_per_threadgroup) {
      if (isnan(input_scale * float(input[row_offset + column])))
        local_sum = NAN;
    }
  }
  return reduce_sum_threadgroup(local_sum, lane_id, simdgroup_id, threads_per_threadgroup, partials);
}

kernel void reduce_logsumexp_one_pass(
  device const real* input [[buffer(0)]],
  device real* output [[buffer(1)]],
  uint row [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup float partials[32];
  float maximum;
  const float sum = reduce_logsumexp_range(input, ulong(row) * column_count, 0, column_count, thread_id, lane_id, simdgroup_id, threads_per_threadgroup, partials, &maximum);
  if (simdgroup_id == 0 && lane_id == 0)
    output[row] = real(isfinite(maximum) ? precise::log(sum) + maximum : maximum);
}

kernel void reduce_logsumexp_partition(
  device const real* input [[buffer(0)]],
  device float2* partition_values [[buffer(1)]],
  uint group_id [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup float partials[32];
  const uint row = group_id / partition_count;
  const uint partition = group_id - row * partition_count;
  const uint partition_begin = partition * partition_size;
  const uint partition_end = partition_begin + min(partition_size, column_count - partition_begin);
  float maximum;
  const float sum = reduce_logsumexp_range(input, ulong(row) * column_count, partition_begin, partition_end, thread_id, lane_id, simdgroup_id, threads_per_threadgroup, partials, &maximum);
  if (simdgroup_id == 0 && lane_id == 0)
    partition_values[ulong(row) * partition_count + partition] = float2(maximum, sum);
}

kernel void reduce_logsumexp_merge_partitions(
  device const float2* partition_values [[buffer(0)]],
  device real* output [[buffer(1)]],
  uint row [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup float partials[32];
  const ulong row_offset = ulong(row) * partition_count;
  float local_maximum = -INFINITY;
  for (uint partition = thread_id;
       partition < partition_count;
       partition += threads_per_threadgroup)
    local_maximum = reduce_max_pair(local_maximum, partition_values[row_offset + partition].x);
  const float maximum = reduce_max_threadgroup(local_maximum, lane_id, simdgroup_id, threads_per_threadgroup, partials);

  float local_sum = 0;
  if (isfinite(maximum)) {
    for (uint partition = thread_id;
         partition < partition_count;
         partition += threads_per_threadgroup) {
      const float2 value = partition_values[row_offset + partition];
      local_sum += value.y * precise::exp(value.x - maximum);
    }
  }
  const float sum = reduce_sum_threadgroup(local_sum, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    output[row] = real(isfinite(maximum) ? precise::log(sum) + maximum : maximum);
}
  )";
  return source;
}
