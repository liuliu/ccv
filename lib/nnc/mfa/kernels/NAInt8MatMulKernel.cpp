#include "NAInt8MatMulKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

namespace {

static uint32_t ceilLog2(uint64_t x) noexcept {
  if (x <= 1)
    return 0;
  --x;
  uint32_t bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}

}

NAInt8MatMulKernel::NAInt8MatMulKernel(
    NAInt8MatMulKernelDescriptor descriptor,
    MTL::Device *const device)
{
  blockDimensions = descriptor.blockDimensions;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  ioPrecision = descriptor.ioPrecision;
  useBias = descriptor.useBias;
  activationQuantizeThreads = descriptor.activationQuantizeThreads;
  groupM = descriptor.groupM;
  groupN = descriptor.groupN;

  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t NAInt8MatMulKernel::threadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept {
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

MTL::Size NAInt8MatMulKernel::threadgroupsPerGrid(uint32_t M, uint32_t N, uint32_t batchDimension) const noexcept {
  auto ceilDivide =
    [=](int64_t target, uint16_t granularity) -> int64_t {
      return (target + int64_t(granularity) - 1) / int64_t(granularity);
    };
  const int64_t M_tiles = ceilDivide(int64_t(M), blockDimensions[0]);
  const int64_t N_tiles = ceilDivide(int64_t(N), blockDimensions[1]);
  const uint32_t M_bits = ceilLog2(M_tiles);
  const uint32_t N_bits = ceilLog2(N_tiles);
  return MTL::Size(int64_t(1) << (M_bits + N_bits), 1, batchDimension);
}

std::string NAInt8MatMulKernel::createSource() const noexcept {
  CodeWriter source;
  source.SetValue("BLOCK_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_K", std::to_string(blockDimensions[2]));
  source.SetValue("SIMDGROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("QUANT_THREADS", std::to_string(activationQuantizeThreads));
  source.SetValue("QUANT_SIMDGROUPS", std::to_string(activationQuantizeThreads / 32));
  source.SetValue("GROUP_M", std::to_string(groupM));
  source.SetValue("GROUP_N", std::to_string(groupN));
  source.SetValue("IO_TYPE", ioPrecision.name());
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

inline uint compact_morton_even_bits(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

inline uint2 morton_decode_2d(uint code) {
  return uint2(compact_morton_even_bits(code),
               compact_morton_even_bits(code >> 1));
}

inline uint lower_bits_mask(uint bit_count) {
  if (bit_count == 0)
    return 0;
  return (1u << bit_count) - 1;
}

inline uint2 morton_decode_rectangular_2d(uint code,
                                          uint x_bits,
                                          uint y_bits) {
  const uint paired_bits = min(x_bits, y_bits);
  const uint paired_code = code & lower_bits_mask(paired_bits * 2);
  uint2 tile = morton_decode_2d(paired_code);
  uint tail = code >> (paired_bits * 2);
  if (x_bits > paired_bits) {
    const uint x_extra_bits = x_bits - paired_bits;
    tile.x |= (tail & lower_bits_mask(x_extra_bits)) << paired_bits;
    tail >>= x_extra_bits;
  }
  if (y_bits > paired_bits) {
    tile.y |= tail << paired_bits;
  }
  return tile;
}

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];
constant bool batched [[function_constant(11)]];
constant uint A_batch_stride [[function_constant(15)]];
constant uint B_batch_stride [[function_constant(16)]];
constant uint C_batch_stride [[function_constant(17)]];
constant uint bias_batch_stride [[function_constant(18)]];
constant uint A_scale_batch_stride [[function_constant(19)]];
constant uint B_scale_batch_stride [[function_constant(20)]];

inline float quantize_reduce_max(float value,
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
    value = lane_id < {{QUANT_SIMDGROUPS}} ? scratch[lane_id] : 0.0f;
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

kernel void quantize_activation(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device {{IO_TYPE}}* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[{{QUANT_SIMDGROUPS}}];
  const uint row = tgid.x;
  if (row >= M)
    return;
  if (batched) {
    src += A_batch_stride * tgid.z;
    dst += A_batch_stride * tgid.z;
    scales += A_scale_batch_stride * tgid.z;
  }
  float local_max = 0.0f;
  const uint base = row * K;
  if ((K % 4) == 0) {
    const uint vectors_per_row = K / 4;
    device const {{IO_TYPE}}4* src4 = reinterpret_cast<device const {{IO_TYPE}}4*>(src);
    device char4* dst4 = reinterpret_cast<device char4*>(dst);
    const uint vector_base = row * vectors_per_row;
    for (uint i = tid; i < vectors_per_row; i += {{QUANT_THREADS}}) {
      const float4 value = float4(src4[vector_base + i]);
      local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])), max(fabs(value[2]), fabs(value[3]))));
    }
    const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
    const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0)
      scales[row] = ({{IO_TYPE}})scale;
    for (uint i = tid; i < vectors_per_row; i += {{QUANT_THREADS}}) {
      const int4 rounded = int4(rint(float4(src4[vector_base + i]) * inv_scale));
      dst4[vector_base + i] = char4(clamp(rounded, int4(-127), int4(127)));
    }
  } else {
    for (uint i = tid; i < K; i += {{QUANT_THREADS}})
      local_max = max(local_max, fabs((float)src[base + i]));
    const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
    const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0)
      scales[row] = ({{IO_TYPE}})scale;
    for (uint i = tid; i < K; i += {{QUANT_THREADS}}) {
      const int rounded = (int)rint((float)src[base + i] * inv_scale);
      dst[base + i] = (int8_t)clamp(rounded, -127, 127);
    }
  }
}

kernel void int8_matmul(
    device int8_t *A_buf [[buffer(0)]],
    device int8_t *B_buf [[buffer(1)]],
    device {{IO_TYPE}} *C_buf [[buffer(2)]],
    device const {{IO_TYPE}} *A_scale_buf [[buffer(3)]],
    device const {{IO_TYPE}} *B_scale_buf [[buffer(4)]],
)";
  if (useBias) {
    source += R"(
    device const {{IO_TYPE}} *bias_buf [[buffer(5)]],
)";
  }
  source += R"(
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint M_tiles = (M + {{BLOCK_M}} - 1) / {{BLOCK_M}};
  const uint N_tiles = (N + {{BLOCK_N}} - 1) / {{BLOCK_N}};
  const uint M_tile_bits = M_tiles <= 1 ? 0 : 32 - clz(M_tiles - 1);
  const uint N_tile_bits = N_tiles <= 1 ? 0 : 32 - clz(N_tiles - 1);
  uint2 morton_tile = morton_decode_rectangular_2d(tgid.x, N_tile_bits, M_tile_bits);
  tgid.x = morton_tile.x;
  tgid.y = morton_tile.y;
  if (tgid.x >= N_tiles || tgid.y >= M_tiles) {
    return;
  }

  const uint M_block_start = tgid.y * {{BLOCK_M}};
  const uint M_block_size = min((uint){{BLOCK_M}}, M - M_block_start);
  const uint N_block_start = tgid.x * {{BLOCK_N}};
  const uint N_block_size = min((uint){{BLOCK_N}}, N - N_block_start);
  const uint M_group_start = {{GROUP_M}} ? (M_block_start / {{GROUP_M}}) * {{GROUP_M}} : M_block_start;
  const uint M_group_offset = M_block_start - M_group_start;
  const uint M_group_size = M - M_group_start;
  const uint N_group_start = {{GROUP_N}} ? (N_block_start / {{GROUP_N}}) * {{GROUP_N}} : N_block_start;
  const uint N_group_offset = N_block_start - N_group_start;
  const uint N_group_size = N - N_group_start;

  if (batched) {
    A_buf += A_batch_stride * tgid.z;
    B_buf += B_batch_stride * tgid.z;
    C_buf += C_batch_stride * tgid.z;
    A_scale_buf += A_scale_batch_stride * tgid.z;
    B_scale_buf += B_scale_batch_stride * tgid.z;
)";
  if (useBias) {
    source += R"(
    bias_buf += bias_batch_stride * tgid.z;
)";
  }
  source += R"(
  }

  A_buf += M_group_start * K;
  B_buf += N_group_start * K;
  C_buf += M_group_start * N;
  A_scale_buf += M_group_start;
  B_scale_buf += N_group_start;
)";
  if (useBias) {
    source += R"(
  bias_buf += N_group_start;
)";
  }
  source += R"(
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M_group_size));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N_group_size));
  if (N_block_start + {{BLOCK_N}} - 1 < N && M_block_start + {{BLOCK_M}} - 1 < M) {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_M}},
        {{BLOCK_N}},
        {{BLOCK_K}},
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> matmul_op;

    auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(0, M_group_offset);
    auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(0, N_group_offset);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    #pragma clang loop unroll(full)
    for (uint k = 0; k + {{BLOCK_K}} <= K; k += {{BLOCK_K}}) {
      auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(k, M_group_offset);
      auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(k, N_group_offset);
      matmul_op.run(mA, mB, cT);
    }
    if (K % {{BLOCK_K}} != 0) {
      constexpr auto residual_descriptor = matmul2d_descriptor(
          {{BLOCK_M}},
          {{BLOCK_N}},
          dynamic_length_v<int>,
          false,
          true,
          true,
          matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<residual_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> residual_op;
      auto mA = A.slice<dynamic_extent, {{BLOCK_M}}>(K / {{BLOCK_K}} * {{BLOCK_K}}, M_group_offset);
      auto mB = B.slice<dynamic_extent, {{BLOCK_N}}>(K / {{BLOCK_K}} * {{BLOCK_K}}, N_group_offset);
      residual_op.run(mA, mB, cT);
    }
    auto mC = C_buf + M_group_offset * N + N_block_start;
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = M_group_offset + (uint)idx[1];
        const uint col = N_group_offset + (uint)idx[0];
        float value = (float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col];
)";
  if (useBias) {
    source += R"(
        value += (float)bias_buf[col];
)";
  }
  source += R"(
        mC[idx[1] * N + idx[0]] = ({{IO_TYPE}})value;
      }
    }
  } else {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_M}},
        {{BLOCK_N}},
        dynamic_length_v<int>,
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> matmul_op;
    auto mA = A.slice(0, M_group_offset);
    auto mB = B.slice(0, N_group_offset);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    matmul_op.run(mA, mB, cT);
    auto mC = C_buf + M_group_offset * N + N_block_start;
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = (uint)idx[1];
        const uint col = (uint)idx[0];
        if (col < N_block_size && row < M_block_size) {
          float value = (float)cT[i] *
              (float)A_scale_buf[M_group_offset + row] *
              (float)B_scale_buf[N_group_offset + col];
)";
  if (useBias) {
    source += R"(
          value += (float)bias_buf[N_group_offset + col];
)";
  }
  source += R"(
          mC[row * N + col] = ({{IO_TYPE}})value;
        }
      }
    }
  }
}
)";
  return source.ToString();
}
