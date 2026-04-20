#include "NAMatMulSmallMKernel.hpp"
#include "NAMatMulSmallMDescriptor.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

static uint32_t ceil_div(const uint32_t x, const uint32_t y) noexcept
{
  return (x + y - 1) / y;
}

std::string NAMatMulSmallMKernel::memoryName(char operand) const noexcept
{
  switch (operand) {
  case 'A':
    return memoryPrecisions.A.name();
  case 'B':
    return memoryPrecisions.B.name();
  case 'C':
    return memoryPrecisions.C.name();
  case 'S':
    return memoryPrecisions.bias.name();
  default:
    return "";
  }
}

NAMatMulSmallMKernel::NAMatMulSmallMKernel(NAMatMulSmallMKernelDescriptor descriptor, MTL::Device* const device)
{
  blockDimensions = descriptor.blockDimensions;
  memoryPrecisions = descriptor.memoryPrecisions;
  pack = descriptor.pack;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  useBias = descriptor.useBias;

  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t NAMatMulSmallMKernel::threadgroupSize(MTL::ComputePipelineState* const pipelineState) const noexcept
{
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

MTL::Size NAMatMulSmallMKernel::threadgroupsPerGrid(const NAMatMulSmallMDescriptor& descriptor) const noexcept
{
  const uint32_t M = descriptor.matrixDimensions[0];
  const uint32_t N = descriptor.matrixDimensions[1];
  return MTL::Size(
      ceil_div(M * pack, blockDimensions[1]),
      ceil_div(N * pack, blockDimensions[0]),
      descriptor.batchDimension * descriptor.splitK());
}

std::string NAMatMulSmallMKernel::createSource() const noexcept
{
  CodeWriter source;
  source.SetValue("MEMORY_NAME_A", memoryName('A'));
  source.SetValue("MEMORY_NAME_B", memoryName('B'));
  source.SetValue("MEMORY_NAME_C", memoryName('C'));
  source.SetValue("MEMORY_NAME_BIAS", memoryName('S'));
  source.SetValue("BLOCK_DIMENSIONS_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_K", std::to_string(blockDimensions[2]));
  source.SetValue("PACK", std::to_string(pack));
  source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));

  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];
constant uint KPACK [[function_constant(3)]];
constant uint SPLIT_K [[function_constant(4)]];
constant uint SPLIT_KPACK [[function_constant(5)]];
constant uint RHS_PACKED_ROWS = N * {{PACK}};
constant uint LHS_PACKED_COLS = M * {{PACK}};

kernel void matmul_small_m_block_view(device {{MEMORY_NAME_B}}* A_buf [[buffer(0)]],
                                      device {{MEMORY_NAME_A}}* B_buf [[buffer(1)]],
                                      device {{MEMORY_NAME_C}}* C_buf [[buffer(2)]],
                                      uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row_block = tgid.y * {{BLOCK_DIMENSIONS_M}};
  const uint col_block = tgid.x * {{BLOCK_DIMENSIONS_N}};
  const uint split_id = tgid.z % SPLIT_K;
  if (row_block >= RHS_PACKED_ROWS || col_block >= LHS_PACKED_COLS || split_id >= SPLIT_K) {
    return;
  }
  const uint row_size = min((uint){{BLOCK_DIMENSIONS_M}}, RHS_PACKED_ROWS - row_block);
  const uint col_size = min((uint){{BLOCK_DIMENSIONS_N}}, LHS_PACKED_COLS - col_block);
  const uint split_begin = split_id * SPLIT_KPACK;
  const uint split_end = split_begin + SPLIT_KPACK;

  auto A = tensor<device {{MEMORY_NAME_B}}, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(KPACK, RHS_PACKED_ROWS));
  auto B = tensor<device {{MEMORY_NAME_A}}, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(KPACK, LHS_PACKED_COLS));
  if (row_size == {{BLOCK_DIMENSIONS_M}} && col_size == {{BLOCK_DIMENSIONS_N}}) {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_DIMENSIONS_M}},
        {{BLOCK_DIMENSIONS_N}},
        {{BLOCK_DIMENSIONS_K}},
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{EXECUTION_SIMD_GROUPS}}>> matmul_op;
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(A), decltype(B), {{MEMORY_NAME_C}}>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        cT[i] = {{MEMORY_NAME_C}}(0);
      }
    }
    #pragma clang loop unroll(full)
    for (uint k0 = split_begin; k0 + {{BLOCK_DIMENSIONS_K}} <= split_end; k0 += {{BLOCK_DIMENSIONS_K}}) {
      auto mA = A.slice<{{BLOCK_DIMENSIONS_K}}, {{BLOCK_DIMENSIONS_M}}>(k0, row_block);
      auto mB = B.slice<{{BLOCK_DIMENSIONS_K}}, {{BLOCK_DIMENSIONS_N}}>(k0, col_block);
      matmul_op.run(mA, mB, cT);
    }
    if (SPLIT_K == 1 && (KPACK % {{BLOCK_DIMENSIONS_K}}) != 0) {
      constexpr auto residual_descriptor = matmul2d_descriptor(
          {{BLOCK_DIMENSIONS_M}},
          {{BLOCK_DIMENSIONS_N}},
          dynamic_length_v<int>,
          false,
          true,
          true,
          matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<residual_descriptor, execution_simdgroups<{{EXECUTION_SIMD_GROUPS}}>> residual_op;
      auto mAr = A.slice<dynamic_extent, {{BLOCK_DIMENSIONS_M}}>(KPACK / {{BLOCK_DIMENSIONS_K}} * {{BLOCK_DIMENSIONS_K}}, row_block);
      auto mBr = B.slice<dynamic_extent, {{BLOCK_DIMENSIONS_N}}>(KPACK / {{BLOCK_DIMENSIONS_K}} * {{BLOCK_DIMENSIONS_K}}, col_block);
      residual_op.run(mAr, mBr, cT);
    }
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint out_col = col_block + idx[0];
        const uint out_row = row_block + idx[1];
        if (out_col < LHS_PACKED_COLS && out_row < RHS_PACKED_ROWS &&
            (out_col % {{PACK}}) == (out_row % {{PACK}})) {
          const uint m = out_col / {{PACK}};
          const uint p = out_col - m * {{PACK}};
          const uint n = out_row / {{PACK}};
          if (m < M && n < N) {
            C_buf[(split_id * M * {{PACK}} + m * {{PACK}} + p) * N + n] = cT[i];
          }
        }
      }
    }
  } else {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_DIMENSIONS_M}},
        {{BLOCK_DIMENSIONS_N}},
        dynamic_length_v<int>,
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{EXECUTION_SIMD_GROUPS}}>> matmul_op;
    auto mA = A.slice(0, row_block);
    auto mB = B.slice(0, col_block);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), {{MEMORY_NAME_C}}>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        cT[i] = {{MEMORY_NAME_C}}(0);
      }
    }
    matmul_op.run(mA, mB, cT);
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint out_col = col_block + idx[0];
        const uint out_row = row_block + idx[1];
        if (out_col < LHS_PACKED_COLS && out_row < RHS_PACKED_ROWS &&
            (out_col % {{PACK}}) == (out_row % {{PACK}})) {
          const uint m = out_col / {{PACK}};
          const uint p = out_col - m * {{PACK}};
          const uint n = out_row / {{PACK}};
          if (m < M && n < N) {
            C_buf[(m * {{PACK}} + p) * N + n] = cT[i];
          }
        }
      }
    }
  }
}

kernel void reduce_diagonal(device const {{MEMORY_NAME_C}}* src [[buffer(0)]],
                            device {{MEMORY_NAME_C}}* dst [[buffer(1)]],
)";
  if (useBias) {
    source += R"(
                            device const {{MEMORY_NAME_BIAS}}* bias [[buffer(2)]],
)";
  }
  source += R"(
                            uint gid [[thread_position_in_grid]])
{
  const uint total = M * N;
  if (gid >= total) {
    return;
  }
  const uint m = gid / N;
  const uint n = gid - m * N;
)";
  if (useBias) {
    source += R"(
  float sum = float(bias[n]);
)";
  } else {
    source += R"(
  float sum = 0;
)";
  }
  source += R"(
  #pragma clang loop unroll(full)
  for (uint split_id = 0; split_id < SPLIT_K; ++split_id) {
    for (uint p = 0; p < {{PACK}}; ++p) {
      sum += float(src[(split_id * M * {{PACK}} + m * {{PACK}} + p) * N + n]);
    }
  }
  dst[gid] = {{MEMORY_NAME_C}}(sum);
}
)";

  return source.ToString();
}
