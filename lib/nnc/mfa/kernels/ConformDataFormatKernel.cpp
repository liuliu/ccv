#include "ConformDataFormatKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

ConformDataFormatKernel::ConformDataFormatKernel(ConformDataFormatKernelDescriptor, MTL::Device* const device)
{
  const std::string source = createSource();
  threadgroupSize = MTL::Size(64, 1, 1);
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

MTL::Size ConformDataFormatKernel::gridSize(const uint32_t rowCount, const uint32_t headDim, const uint32_t preservedTail) const noexcept
{
  const uint32_t blocksPerRow = (headDim - preservedTail) / 64;
  return MTL::Size((NS::UInteger)rowCount * blocksPerRow, 1, 1);
}

std::string ConformDataFormatKernel::createSource() const noexcept
{
  return R"(
#include <metal_stdlib>
using namespace metal;

constant uint row_count [[function_constant(0)]];
constant uint head_dim [[function_constant(1)]];
constant uint preserved_tail [[function_constant(2)]];

inline float round_ties_to_even(const float x)
{
  const float lower = floor(x);
  const float fraction = x - lower;
  if (fraction < 0.5f)
    return lower;
  if (fraction > 0.5f)
    return lower + 1.0f;
  return fmod(lower, 2.0f) == 0.0f ? lower : lower + 1.0f;
}

inline float conform_e4m3(const float x)
{
  const float magnitude = min(abs(x), 448.0f);
  if (magnitude == 0.0f)
    return 0.0f;
  const float step = magnitude < 0.015625f ? 0.001953125f : exp2(floor(log2(magnitude)) - 3.0f);
  const float dequantized = min(round_ties_to_even(magnitude / step) * step, 448.0f);
  return x < 0.0f ? -dequantized : dequantized;
}

kernel void conform_data_format(
  device const float* source [[buffer(0)]],
  device float* destination [[buffer(1)]],
  uint block [[threadgroup_position_in_grid]],
  ushort tid [[thread_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort simdgroup_id [[simdgroup_index_in_threadgroup]])
{
  const uint prefix = head_dim - preserved_tail;
  const uint blocks_per_row = prefix / 64;
  const uint row = block / blocks_per_row;
  if (row >= row_count)
    return;
  const uint block_in_row = block - row * blocks_per_row;
  const uint row_base = row * head_dim;
  const uint index = row_base + block_in_row * 64 + tid;
  const float value = source[index];

  threadgroup float partial_max[2];
  const float simdgroup_max = simd_max(abs(value));
  if (lane_id == 0)
    partial_max[simdgroup_id] = simdgroup_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0)
    partial_max[0] = max(max(partial_max[0], partial_max[1]), 1.0e-4f);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float scale = exp2(ceil(log2(partial_max[0] / 448.0f)));
  destination[index] = conform_e4m3(clamp(value / scale, -448.0f, 448.0f)) * scale;

  if (block_in_row == 0)
    for (uint i = tid; i < preserved_tail; i += 64)
      destination[row_base + prefix + i] = source[row_base + prefix + i];
}
  )";
}
