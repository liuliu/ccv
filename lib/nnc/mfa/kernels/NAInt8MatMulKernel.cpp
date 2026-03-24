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
  outputFloat = descriptor.outputFloat;
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

MTL::Size NAInt8MatMulKernel::threadgroupsPerGrid(uint32_t M, uint32_t N) const noexcept {
  auto ceilDivide =
    [=](int64_t target, uint16_t granularity) -> int64_t {
      return (target + int64_t(granularity) - 1) / int64_t(granularity);
    };
  const int64_t M_tiles = ceilDivide(int64_t(M), blockDimensions[0]);
  const int64_t N_tiles = ceilDivide(int64_t(N), blockDimensions[1]);
  const uint32_t M_bits = ceilLog2(M_tiles);
  const uint32_t N_bits = ceilLog2(N_tiles);
  return MTL::Size(int64_t(1) << (M_bits + N_bits), 1, 1);
}

std::string NAInt8MatMulKernel::createSource() const noexcept {
  CodeWriter source;
  source.SetValue("BLOCK_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_K", std::to_string(blockDimensions[2]));
  source.SetValue("BLOCK_K_2", std::to_string(blockDimensions[2] * 2));
  source.SetValue("SIMDGROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("GROUP_M", std::to_string(groupM));
  source.SetValue("GROUP_N", std::to_string(groupN));
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

kernel void int8_matmul(
    device int8_t *A_buf [[buffer(0)]],
    device int8_t *B_buf [[buffer(1)]],
)";
  if (outputFloat) {
    source += R"(
    device half *C_buf [[buffer(2)]],
    device const half *A_scale_buf [[buffer(3)]],
    device const half *B_scale_buf [[buffer(4)]],
)";
  } else {
    source += R"(
    device int32_t *C_buf [[buffer(2)]],
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

  A_buf += M_group_start * K;
  B_buf += N_group_start * K;
  C_buf += M_group_start * N;
)";
  if (outputFloat) {
    source += R"(
  A_scale_buf += M_group_start;
  B_scale_buf += N_group_start;
)";
  }
  source += R"(

  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M_group_size));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N_group_size));
)";
  if (!outputFloat) {
    source += R"(
  auto C = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(N, M_group_size));
)";
  }
  source += R"(
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
    for (uint k = 0; k + {{BLOCK_K_2}} <= K; k += {{BLOCK_K_2}}) {
      auto mA0 = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(k, M_group_offset);
      auto mB0 = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(k, N_group_offset);
      auto mA1 = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(k + {{BLOCK_K}}, M_group_offset);
      auto mB1 = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(k + {{BLOCK_K}}, N_group_offset);
      matmul_op.run(mA0, mB0, cT);
      matmul_op.run(mA1, mB1, cT);
    }
    if (K % {{BLOCK_K_2}} >= {{BLOCK_K}}) {
      auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(K / {{BLOCK_K_2}} * {{BLOCK_K_2}}, M_group_offset);
      auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(K / {{BLOCK_K_2}} * {{BLOCK_K_2}}, N_group_offset);
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
)";
  if (outputFloat) {
    source += R"(
    auto mC = C_buf + M_group_offset * N + N_block_start;
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = M_group_offset + idx[1];
        const uint col = N_group_offset + idx[0];
        mC[idx[1] * N + idx[0]] = (half)((float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col]);
      }
    }
)";
  } else {
    source += R"(
    auto mC = C.slice<{{BLOCK_N}}, {{BLOCK_M}}>(N_block_start, M_group_offset);
    cT.store(mC);
)";
  }
  source += R"(
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
)";
  if (outputFloat) {
    source += R"(
    auto mC = C_buf + M_group_offset * N + N_block_start;
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        if (idx[0] < N_block_size && idx[1] < M_block_size) {
          mC[idx[1] * N + idx[0]] = (half)((float)cT[i] *
              (float)A_scale_buf[M_group_offset + idx[1]] *
              (float)B_scale_buf[N_group_offset + idx[0]]);
        }
      }
    }
)";
  } else {
    source += R"(
    auto mC = C_buf + M_group_offset * N + N_block_start;
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        if (idx[0] < N_block_size && idx[1] < M_block_size) {
          mC[idx[1] * N + idx[0]] = cT[i];
        }
      }
    }
)";
  }
  source += R"(
  }
}
)";
  return source.ToString();
}
