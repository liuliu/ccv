#include "SegmentedScaledGEMMKernel.hpp"
#include "SegmentedScaledGEMMKernelDescriptor.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

SegmentedScaledGEMMKernel::SegmentedScaledGEMMKernel(
    SegmentedScaledGEMMKernelDescriptor descriptor,
    MTL::Device* const device)
{
  blockDimensions = descriptor.blockDimensions;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  ioPrecision = descriptor.ioPrecision;
  useBias = descriptor.useBias;

  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t SegmentedScaledGEMMKernel::threadgroupSize(MTL::ComputePipelineState* const pipelineState) const noexcept
{
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

uint32_t SegmentedScaledGEMMKernel::maxTileRecords(uint32_t originalM, uint32_t segments) const noexcept
{
  const uint32_t tilesPerSegment = (originalM + blockDimensions[0] - 1) / blockDimensions[0];
  return segments * (tilesPerSegment > 0 ? tilesPerSegment : 1u);
}

std::string SegmentedScaledGEMMKernel::createSource() const noexcept
{
  CodeWriter source;
  source.SetValue("BLOCK_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_K", std::to_string(blockDimensions[2]));
  source.SetValue("SIMDGROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("IO_TYPE", ioPrecision.name());
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct TileRecord {
  uint row_offset;
  uint local_row;
  uint count;
  uint expert;
};

constant uint N [[function_constant(0)]];
constant uint K [[function_constant(1)]];
constant uint segments [[function_constant(2)]];
constant uint M_block [[function_constant(3)]];
constant uint N_block [[function_constant(4)]];
constant uint K_block [[function_constant(5)]];
constant uint max_tile_records [[function_constant(6)]];

kernel void segmented_scaled_gemm_plan(
    device const int* indices [[buffer(0)]],
    device const int* counts [[buffer(1)]],
    device TileRecord* records [[buffer(2)]],
    device uint* dispatch_args [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
  if (tid != 0)
    return;
  uint row_offset = 0;
  uint record_count = 0;
  for (uint segment = 0; segment < segments; ++segment) {
    const int count_i = counts[segment];
    const int expert_i = indices[segment];
    const uint count = count_i > 0 ? (uint)count_i : 0u;
    if (count > 0 && expert_i >= 0 && expert_i < (int)segments) {
      for (uint local = 0; local < count; local += M_block) {
        if (record_count < max_tile_records) {
          records[record_count] = TileRecord {
            row_offset,
            local,
            count,
            (uint)expert_i,
          };
        }
        ++record_count;
      }
    }
    row_offset += count;
  }
  dispatch_args[0] = min(record_count, max_tile_records);
  dispatch_args[1] = (N + N_block - 1) / N_block;
  dispatch_args[2] = 1;
}

kernel void segmented_scaled_gemm(
    device int8_t* A_buf [[buffer(0)]],
    device int8_t* B_buf [[buffer(1)]],
    device {{IO_TYPE}}* C_buf [[buffer(2)]],
    device const {{IO_TYPE}}* A_scale_buf [[buffer(3)]],
    device const {{IO_TYPE}}* B_scale_buf [[buffer(4)]],
    device const TileRecord* records [[buffer(5)]],
)";
  if (useBias) {
    source += R"(
    device const {{IO_TYPE}}* bias_buf [[buffer(6)]],
)";
  }
  source += R"(
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const TileRecord record = records[tgid.x];
  const uint N_block_start = tgid.y * N_block;
  if (N_block_start >= N)
    return;
  const uint local_row = record.local_row;
  if (local_row >= record.count)
    return;
  const uint M_dynamic = min(M_block, record.count - local_row);
  const uint N_dynamic = min(N_block, N - N_block_start);
  const uint global_row = record.row_offset + local_row;

  A_buf += global_row * K;
  B_buf += record.expert * (ulong)(N * K) + N_block_start * K;
  C_buf += global_row * N + N_block_start;
  A_scale_buf += global_row;
  B_scale_buf += record.expert * N + N_block_start;
)";
  if (useBias) {
    source += R"(
  bias_buf += record.expert * N + N_block_start;
)";
  }
  source += R"(

  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M_dynamic));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N_dynamic));

  if (M_dynamic == M_block && N_dynamic == N_block) {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_M}},
        {{BLOCK_N}},
        {{BLOCK_K}},
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> matmul_op;
    auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(0, 0);
    auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(0, 0);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    #pragma clang loop unroll(full)
    for (uint k = 0; k + K_block <= K; k += K_block) {
      auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(k, 0);
      auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(k, 0);
      matmul_op.run(mA, mB, cT);
    }
    if (K % K_block != 0) {
      constexpr auto residual_descriptor = matmul2d_descriptor(
          {{BLOCK_M}},
          {{BLOCK_N}},
          dynamic_length_v<int>,
          false,
          true,
          true,
          matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<residual_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> residual_op;
      auto mA = A.slice<dynamic_extent, {{BLOCK_M}}>(K / K_block * K_block, 0);
      auto mB = B.slice<dynamic_extent, {{BLOCK_N}}>(K / K_block * K_block, 0);
      residual_op.run(mA, mB, cT);
    }
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = (uint)idx[1];
        const uint col = (uint)idx[0];
        float value = (float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col];
)";
  if (useBias) {
    source += R"(
        value += (float)bias_buf[col];
)";
  }
  source += R"(
        C_buf[row * N + col] = ({{IO_TYPE}})value;
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
    auto mA = A.slice(0, 0);
    auto mB = B.slice(0, 0);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    matmul_op.run(mA, mB, cT);
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = (uint)idx[1];
        const uint col = (uint)idx[0];
        if (row < M_dynamic && col < N_dynamic) {
          float value = (float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col];
)";
  if (useBias) {
    source += R"(
          value += (float)bias_buf[col];
)";
  }
  source += R"(
          C_buf[row * N + col] = ({{IO_TYPE}})value;
        }
      }
    }
  }
}
)";
  return source.ToString();
}
