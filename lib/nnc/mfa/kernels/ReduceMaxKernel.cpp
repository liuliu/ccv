#include "ReduceMaxKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

ReduceMaxKernel::ReduceMaxKernel(ReduceMaxKernelDescriptor descriptor, MTL::Device* const device)
{
  memoryPrecision = descriptor.memoryPrecision;
  source = createSource();
  auto sourceString = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(sourceString, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string ReduceMaxKernel::createSource() const noexcept
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
  }
  return value;
}

kernel void reduce_max_one_pass(
  device const real* input [[buffer(0)]],
  device real* output [[buffer(1)]],
  uint row [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup float partials[32];
  float best = -INFINITY;
  for (uint first_column = uint(thread_id) * 4;
       first_column < column_count;
       first_column += uint(threads_per_threadgroup) * 4) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; i++) {
      const uint column = first_column + i;
      if (column < column_count)
        best = reduce_max_pair(best, float(input[ulong(row) * column_count + column]));
    }
  }
  best = reduce_max_threadgroup(best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    output[row] = real(best);
}

kernel void reduce_max_partition(
  device const real* input [[buffer(0)]],
  device float* partition_values [[buffer(1)]],
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
  float best = -INFINITY;
  for (uint first_column = partition_begin + uint(thread_id) * 4;
       first_column < partition_end;
       first_column += uint(threads_per_threadgroup) * 4) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; i++) {
      const uint column = first_column + i;
      if (column < partition_end)
        best = reduce_max_pair(best, float(input[ulong(row) * column_count + column]));
    }
  }
  best = reduce_max_threadgroup(best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    partition_values[ulong(row) * partition_count + partition] = best;
}

kernel void reduce_max_merge_partitions(
  device const float* partition_values [[buffer(0)]],
  device real* output [[buffer(1)]],
  uint row [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup float partials[32];
  float best = -INFINITY;
  for (uint partition = thread_id;
       partition < partition_count;
       partition += threads_per_threadgroup)
    best = reduce_max_pair(best, partition_values[ulong(row) * partition_count + partition]);
  best = reduce_max_threadgroup(best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    output[row] = real(best);
}
  )";
  return source;
}
