#include "ArgmaxKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

ArgmaxKernel::ArgmaxKernel(ArgmaxKernelDescriptor descriptor, MTL::Device* const device)
{
  memoryPrecision = descriptor.memoryPrecision;
  source = createSource();
  auto sourceString = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(sourceString, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string ArgmaxKernel::createSource() const noexcept
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

constant uint invalid_index = 0xffffffffu;
constant uint column_count [[function_constant(0)]];
constant uint partition_size [[function_constant(1)]];
constant uint partition_count [[function_constant(2)]];
constant float gumbel_scale [[function_constant(3)]];

struct ArgmaxPair {
  float value;
  uint index;
};

struct ArgmaxParams {
  uint counter_0;
  uint counter_1;
  uint counter_2;
  uint counter_3;
  uint key_0;
  uint key_1;
};

inline ArgmaxPair invalid_pair()
{
  return ArgmaxPair{-INFINITY, invalid_index};
}

inline ArgmaxPair argmax_pair(ArgmaxPair lhs, ArgmaxPair rhs)
{
  if (rhs.index == invalid_index)
    return lhs;
  if (lhs.index == invalid_index)
    return rhs;
  const bool lhs_nan = isnan(lhs.value);
  const bool rhs_nan = isnan(rhs.value);
  // Match the existing serial CPU / CUDA behavior: a NaN in column 0
  // remains the winner, while NaNs in later columns are ignored.
  if (lhs_nan && lhs.index == 0)
    return lhs;
  if (rhs_nan && rhs.index == 0)
    return rhs;
  if (lhs_nan != rhs_nan)
    return lhs_nan ? rhs : lhs;
  if (rhs.value > lhs.value ||
      (rhs.value == lhs.value && rhs.index < lhs.index))
    return rhs;
  return lhs;
}

inline ArgmaxPair argmax_simd(ArgmaxPair value)
{
  for (ushort delta = 16; delta > 0; delta >>= 1) {
    const ArgmaxPair other = {
      simd_shuffle_down(value.value, delta),
      simd_shuffle_down(value.index, delta),
    };
    value = argmax_pair(value, other);
  }
  return value;
}

inline ArgmaxPair argmax_threadgroup(
  ArgmaxPair value,
  ushort lane_id,
  ushort simdgroup_id,
  ushort threads_per_threadgroup,
  threadgroup ArgmaxPair* partials)
{
  value = argmax_simd(value);
  if (lane_id == 0)
    partials[simdgroup_id] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    const ushort simdgroup_count = (threads_per_threadgroup + 31) / 32;
    value = lane_id < simdgroup_count ? partials[lane_id] : invalid_pair();
    value = argmax_simd(value);
  }
  return value;
}

inline ArgmaxPair plain_score(
  device const real* input,
  uint row,
  uint column,
  uint column_count)
{
  return ArgmaxPair{
    float(input[ulong(row) * column_count + column]),
    column,
  };
}

inline uint4 philox4x32_10(uint4 counter, uint2 key)
{
  constexpr uint multiplier_0 = 0xd2511f53u;
  constexpr uint multiplier_1 = 0xcd9e8d57u;
  constexpr uint key_increment_0 = 0x9e3779b9u;
  constexpr uint key_increment_1 = 0xbb67ae85u;
  for (ushort round = 0; round < 10; round++) {
    const uint high_0 = mulhi(multiplier_0, counter.x);
    const uint low_0 = multiplier_0 * counter.x;
    const uint high_1 = mulhi(multiplier_1, counter.z);
    const uint low_1 = multiplier_1 * counter.z;
    counter = uint4(
      high_1 ^ counter.y ^ key.x,
      low_1,
      high_0 ^ counter.w ^ key.y,
      low_0);
    key += uint2(key_increment_0, key_increment_1);
  }
  return counter;
}

inline float uniform_open_01(uint random_bits)
{
  return (float(random_bits >> 8) + 0.5f) * 0x1.0p-24f;
}

inline float gumbel_from_bits(uint random_bits)
{
  const float uniform_value = uniform_open_01(random_bits);
  return -fast::log(-fast::log(uniform_value));
}

inline uint4 philox_counter(
  ulong block_index,
  constant ArgmaxParams& params)
{
  const ulong counter_low =
    ulong(params.counter_0) | (ulong(params.counter_1) << 32);
  const ulong counter_high =
    ulong(params.counter_2) | (ulong(params.counter_3) << 32);
  const ulong updated_low = counter_low + block_index;
  const ulong updated_high = counter_high + ulong(updated_low < counter_low);
  return uint4(
    uint(updated_low),
    uint(updated_low >> 32),
    uint(updated_high),
    uint(updated_high >> 32));
}

inline uint4 gumbel_random_bits(
  uint row,
  uint first_column,
  constant ArgmaxParams& params)
{
  const ulong blocks_per_row = (ulong(column_count) + 3) / 4;
  const ulong block_index =
    ulong(row) * blocks_per_row + ulong(first_column / 4);
  return philox4x32_10(
    philox_counter(block_index, params),
    uint2(params.key_0, params.key_1));
}

inline ArgmaxPair gumbel_score(
  device const real* input,
  uint row,
  uint column,
  uint random_bits,
  uint column_count)
{
  return ArgmaxPair{
    float(input[ulong(row) * column_count + column]) +
      gumbel_scale * gumbel_from_bits(random_bits),
    column,
  };
}

kernel void argmax_one_pass(
  device const real* input [[buffer(0)]],
  device int* output [[buffer(1)]],
  constant ArgmaxParams& params [[buffer(2)]],
  uint row [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup ArgmaxPair partials[32];
  ArgmaxPair best = invalid_pair();
  for (uint first_column = uint(thread_id) * 4;
       first_column < column_count;
       first_column += uint(threads_per_threadgroup) * 4) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; i++) {
      const uint column = first_column + i;
      if (column < column_count)
        best = argmax_pair(
          best,
          plain_score(input, row, column, column_count));
    }
  }
  best = argmax_threadgroup(
    best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    output[row] = int(best.index);
}

kernel void argmax_partition(
  device const real* input [[buffer(0)]],
  device ArgmaxPair* partition_pairs [[buffer(1)]],
  constant ArgmaxParams& params [[buffer(2)]],
  uint group_id [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup ArgmaxPair partials[32];
  const uint row = group_id / partition_count;
  const uint partition = group_id - row * partition_count;
  const uint partition_begin = partition * partition_size;
  const uint partition_end = partition_begin + min(
    partition_size,
    column_count - partition_begin);
  ArgmaxPair best = invalid_pair();
  for (uint first_column = partition_begin + uint(thread_id) * 4;
       first_column < partition_end;
       first_column += uint(threads_per_threadgroup) * 4) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; i++) {
      const uint column = first_column + i;
      if (column < partition_end)
        best = argmax_pair(
          best,
          plain_score(input, row, column, column_count));
    }
  }
  best = argmax_threadgroup(
    best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    partition_pairs[ulong(row) * partition_count + partition] = best;
}

kernel void gumbel_argmax_one_pass(
  device const real* input [[buffer(0)]],
  device int* output [[buffer(1)]],
  constant ArgmaxParams& params [[buffer(2)]],
  uint row [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup ArgmaxPair partials[32];
  ArgmaxPair best = invalid_pair();
  for (uint first_column = uint(thread_id) * 4;
       first_column < column_count;
       first_column += uint(threads_per_threadgroup) * 4) {
    const uint4 random_bits =
      gumbel_random_bits(row, first_column, params);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; i++) {
      const uint column = first_column + i;
      if (column < column_count)
        best = argmax_pair(
          best,
          gumbel_score(
            input, row, column, random_bits[i], column_count));
    }
  }
  best = argmax_threadgroup(
    best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    output[row] = int(best.index);
}

kernel void gumbel_argmax_partition(
  device const real* input [[buffer(0)]],
  device ArgmaxPair* partition_pairs [[buffer(1)]],
  constant ArgmaxParams& params [[buffer(2)]],
  uint group_id [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup ArgmaxPair partials[32];
  const uint row = group_id / partition_count;
  const uint partition = group_id - row * partition_count;
  const uint partition_begin = partition * partition_size;
  const uint partition_end = partition_begin + min(
    partition_size,
    column_count - partition_begin);
  ArgmaxPair best = invalid_pair();
  for (uint first_column = partition_begin + uint(thread_id) * 4;
       first_column < partition_end;
       first_column += uint(threads_per_threadgroup) * 4) {
    const uint4 random_bits =
      gumbel_random_bits(row, first_column, params);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; i++) {
      const uint column = first_column + i;
      if (column < partition_end)
        best = argmax_pair(
          best,
          gumbel_score(
            input, row, column, random_bits[i], column_count));
    }
  }
  best = argmax_threadgroup(
    best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    partition_pairs[ulong(row) * partition_count + partition] = best;
}

kernel void argmax_merge_partitions(
  device const ArgmaxPair* partition_pairs [[buffer(0)]],
  device int* output [[buffer(1)]],
  constant ArgmaxParams& params [[buffer(2)]],
  uint row [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
  ushort threads_per_threadgroup [[threads_per_threadgroup]])
{
  threadgroup ArgmaxPair partials[32];
  ArgmaxPair best = invalid_pair();
  for (uint partition = thread_id;
       partition < partition_count;
       partition += threads_per_threadgroup) {
    best = argmax_pair(
      best,
      partition_pairs[ulong(row) * partition_count + partition]);
  }
  best = argmax_threadgroup(
    best, lane_id, simdgroup_id, threads_per_threadgroup, partials);
  if (simdgroup_id == 0 && lane_id == 0)
    output[row] = int(best.index);
}
  )";
  return source;
}
