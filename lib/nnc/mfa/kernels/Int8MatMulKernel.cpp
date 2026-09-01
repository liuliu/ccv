#include "Int8MatMulKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "GEMMHeaders.hpp"

Int8MatMulKernel::Int8MatMulKernel(Int8MatMulKernelDescriptor descriptor, MTL::Device* const device)
{
	(void)descriptor;
	threadgroupSize = MTL::Size(256, 1, 1);
	source = createMetalSimdgroupMatrixStorage(false) + createSource();
	auto sourceString = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(sourceString, nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string Int8MatMulKernel::createSource() const noexcept
{
	return R"(
#include <metal_stdlib>
using namespace metal;

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];
constant uint expert_count [[function_constant(3)]];
constant uint bincount [[function_constant(4)]];
constant uint weight_count [[function_constant(5)]];

inline float int8_matmul_reduce_max(float value,
                                  threadgroup float* scratch,
                                  ushort sgid,
                                  ushort lane_id)
{
  value = max(value, simd_shuffle_xor(value, 16));
  value = max(value, simd_shuffle_xor(value, 8));
  value = max(value, simd_shuffle_xor(value, 4));
  value = max(value, simd_shuffle_xor(value, 2));
  value = max(value, simd_shuffle_xor(value, 1));
  if (lane_id == 0)
    scratch[sgid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    value = lane_id < 8 ? scratch[lane_id] : 0.0f;
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane_id == 0)
      scratch[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

kernel void int8_matmul_quantize_activation(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x;
  if (row >= M)
    return;
  threadgroup float scratch[8];
  const ulong base = (ulong)row * K;
  float local_max = 0.0f;
  for (uint col = tid; col < K; col += 256)
    local_max = max(local_max, fabs(source[base + col]));
  const float max_abs = int8_matmul_reduce_max(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? (128.0f * max_abs / 127.0f) : (128.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[row] = scale;
  for (uint col = tid; col < K; col += 256) {
    const int rounded = (int)rint(source[base + col] * inv_scale);
    destination[base + col] = (half)clamp(rounded, -127, 127) * half(0.0078125f);
  }
}

kernel void int8_matmul_cast_weights(
    device const char* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    uint3 position [[thread_position_in_grid]])
{
  const uint index = position.x;
  if (index < weight_count)
    destination[index] = (half)source[index] * half(0.0078125f);
}

kernel void int8_matmul_dequantize_output(
    device float* output [[buffer(0)]],
    device const float* activation_scales [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    uint3 position [[thread_position_in_grid]])
{
  const uint index = position.x;
  const ulong count = (ulong)M * N;
  if ((ulong)index < count) {
    const uint row = index / N;
    const uint col = index - row * N;
    output[index] *= activation_scales[row] * (128.0f * weight_scales[col]);
  }
}

kernel void int8_matmul_segmented(
    device const half* A [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const int* counts [[buffer(2)]],
    device const char* B [[buffer(3)]],
    device float* C [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint bin = gid.y;
  if (bin >= bincount)
    return;
  const int expert = indices[bin];
  const int count_i = counts[bin];
  if (expert < 0 || expert >= (int)expert_count || count_i <= 0)
    return;
  uint row_offset = 0;
  for (uint i = 0; i < bin; ++i)
    row_offset += (uint)max(counts[i], 0);
  const uint n0 = gid.x * 8;
  if (n0 >= N)
    return;
  const ushort2 morton = morton_order(lane_id);
  const device char* B_base = B + (ulong)expert * N * K;
  const uint full_count = (uint)count_i & ~15u;
  for (uint m0 = 0; m0 < full_count; m0 += 16) {
    simdgroup_matrix_storage<float> accum[2];
    accum[0] = simdgroup_matrix_storage<float>(0);
    accum[1] = simdgroup_matrix_storage<float>(0);
    for (uint k0 = 0; k0 < K; k0 += 8) {
      const device char* B_lane = B_base + (ulong)(n0 + morton.x) * K + k0 + morton.y;
      simdgroup_matrix_storage<half> a_reg[2];
      simdgroup_matrix_storage<half> b_reg;
#pragma clang loop unroll(full)
      for (ushort m = 0; m < 2; ++m) {
        const uint row = m0 + m * 8 + morton.y;
        const device half* A_lane = A + (ulong)(row_offset + row) * K + k0 + morton.x;
        a_reg[m].load(A_lane, K, ushort2(0, 0), false);
      }
      b_reg.load(B_lane, K, ushort2(0, 0), true);
      *b_reg.thread_elements() *= half(0.0078125f);
      accum[0].multiply(a_reg[0], b_reg);
      accum[1].multiply(a_reg[1], b_reg);
    }
#pragma clang loop unroll(full)
    for (ushort m = 0; m < 2; ++m) {
      const uint row = m0 + m * 8 + morton.y;
      device float* C_lane = C + (ulong)(row_offset + row) * N + n0 + morton.x;
      accum[m].store(C_lane, N, ushort2(0, 0), false);
    }
  }
  if (full_count < (uint)count_i) {
    simdgroup_matrix_storage<float> accum[2];
    accum[0] = simdgroup_matrix_storage<float>(0);
    accum[1] = simdgroup_matrix_storage<float>(0);
    for (uint k0 = 0; k0 < K; k0 += 8) {
      const device char* B_lane = B_base + (ulong)(n0 + morton.x) * K + k0 + morton.y;
      simdgroup_matrix_storage<half> a_reg[2];
      simdgroup_matrix_storage<half> b_reg;
#pragma clang loop unroll(full)
      for (ushort m = 0; m < 2; ++m) {
        const uint row = full_count + m * 8 + morton.y;
        if (row < (uint)count_i) {
          const ulong address = (ulong)(row_offset + row) * K + k0 + morton.x;
          *a_reg[m].thread_elements() = half2(A[address], A[address + 1]);
        } else {
          a_reg[m] = simdgroup_matrix_storage<half>(0);
        }
      }
      b_reg.load(B_lane, K, ushort2(0, 0), true);
      *b_reg.thread_elements() *= half(0.0078125f);
      accum[0].multiply(a_reg[0], b_reg);
      accum[1].multiply(a_reg[1], b_reg);
    }
#pragma clang loop unroll(full)
    for (ushort m = 0; m < 2; ++m) {
      const uint row = full_count + m * 8 + morton.y;
      if (row < (uint)count_i) {
        const float2 values = *accum[m].thread_elements();
        const ulong address = (ulong)(row_offset + row) * N + n0 + morton.x;
        C[address] = values.x;
        C[address + 1] = values.y;
      }
    }
  }
}

kernel void int8_matmul_dequantize_segmented_output(
    device float* output [[buffer(0)]],
    device const float* activation_scales [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    device const int* indices [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  const uint bin = tgid.y;
  if (bin >= bincount)
    return;
  const int expert = indices[bin];
  const int count_i = counts[bin];
  if (expert < 0 || expert >= (int)expert_count || count_i <= 0)
    return;
  uint row_offset = 0;
  for (uint i = 0; i < bin; ++i)
    row_offset += (uint)max(counts[i], 0);
  const uint col = tgid.x * 256 + tid;
  if (col >= N)
    return;
  const float weight_scale = 128.0f * weight_scales[(ulong)expert * N + col];
  for (uint row = 0; row < (uint)count_i; ++row) {
    const ulong output_index = (ulong)(row_offset + row) * N + col;
    output[output_index] *= activation_scales[row_offset + row] * weight_scale;
  }
}
)";
}
