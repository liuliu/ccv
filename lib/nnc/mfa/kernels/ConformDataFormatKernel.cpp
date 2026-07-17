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

constant float conform_e4m3_exp_scale[16] = {
  0.0f, 0.015625f, 0.03125f, 0.0625f,
  0.125f, 0.25f, 0.5f, 1.0f,
  2.0f, 4.0f, 8.0f, 16.0f,
  32.0f, 64.0f, 128.0f, 256.0f,
};

static inline float conform_e4m3_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * conform_e4m3_exp_scale[exp];
}

static inline float conform_e4m3(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (conform_e4m3_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - conform_e4m3_value(best));
        const float next_diff = abs(ax - conform_e4m3_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return sign * conform_e4m3_value(best);
}

// x is positive and normal: partial_max has a 1e-4 floor, and finite Float32
// values remain normal after division by 448. The exponent field therefore
// encodes floor(log2(x)); a nonzero mantissa advances it to ceil(log2(x)).
static inline float ceil_power_of_two(float x) {
    const uint bits = as_type<uint>(x);
    const uint exponent = bits & 0x7f800000u;
    const uint mantissa = bits & 0x007fffffu;
    return as_type<float>(exponent + (mantissa != 0 ? 0x00800000u : 0u));
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

  const float scale = ceil_power_of_two(partial_max[0] / 448.0f);
  destination[index] = conform_e4m3(clamp(value / scale, -448.0f, 448.0f)) * scale;

  if (block_in_row == 0)
    for (uint i = tid; i < preserved_tail; i += 64)
      destination[row_base + prefix + i] = source[row_base + prefix + i];
}
  )";
}
