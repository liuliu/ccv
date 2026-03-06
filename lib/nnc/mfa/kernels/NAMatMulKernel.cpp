#include "NAMatMulKernel.hpp"
#include "NAMatMulDescriptor.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

std::string NAMatMulKernel::memoryName(char operand) const noexcept {
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

bool NAMatMulKernel::transposed(char operand) const noexcept {
  switch (operand) {
  case 'A':
    return transposeState[0];
  case 'B':
    return transposeState[1];
  case 'C':
    return false;
  default:
    return false;
  }
}

NAMatMulKernel::NAMatMulKernel(NAMatMulKernelDescriptor descriptor, MTL::Device *const device) {
  blockDimensions = descriptor.blockDimensions;
  memoryPrecisions = descriptor.memoryPrecisions;
  registerPrecisions = descriptor.registerPrecisions;
  splitK = descriptor.splitK;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  transposeState = descriptor.transposeState;
  useBias = descriptor.useBias;
  loadM = descriptor.loadM;

  /// The number of threads per group.
  source = createSource();

  // Compile the shader source.
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t NAMatMulKernel::threadgroupSize(MTL::ComputePipelineState *const pipelineState, const NAMatMulDescriptor &descriptor) const noexcept {
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

MTL::Size NAMatMulKernel::threadgroupsPerGrid(const NAMatMulDescriptor &descriptor) const noexcept {
  auto ceilDivide =
  [=](int64_t target, uint16_t granularity) -> int64_t {
    return (target + int64_t(granularity) - 1) / int64_t(granularity);
  };
  if (descriptor.dispatchMMajor) {
    return MTL::Size(ceilDivide(int64_t(descriptor.matrixDimensions[0]), blockDimensions[0]) * splitK, ceilDivide(int64_t(descriptor.matrixDimensions[1]), blockDimensions[1]), descriptor.batchDimension);
  } else {
    return MTL::Size(ceilDivide(int64_t(descriptor.matrixDimensions[1]), blockDimensions[1]) * splitK, ceilDivide(int64_t(descriptor.matrixDimensions[0]), blockDimensions[0]), descriptor.batchDimension);
  }
}

#pragma mark - Source

std::string NAMatMulKernel::createSource() const noexcept {
  CodeWriter source;
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

)";

  source.SetValue("TRANSPOSE_STATE_A", std::to_string(bool(transposeState[0])));
  source.SetValue("TRANSPOSE_STATE_B", std::to_string(bool(transposeState[1])));
  source.SetValue("TRANSPOSE_STATE_BIAS", std::to_string(bool(transposeState[2])));
  source.SetValue("BLOCK_DIMENSIONS_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_K", std::to_string(blockDimensions[2]));
  source.SetValue("BLOCK_DIMENSIONS_K_2", std::to_string(blockDimensions[2] * 2));
  source.SetValue("SPLIT_K", std::to_string(splitK));

  source += createConstants();

  source.SetValue("MEMORY_NAME_A", memoryName('A'));
  source.SetValue("MEMORY_NAME_B", memoryName('B'));
  source.SetValue("MEMORY_NAME_C", memoryName('C'));
  source.SetValue("MEMORY_NAME_BIAS", memoryName('S'));
  if (registerPrecisions.C.value == GEMMOperandPrecision::FP32 && memoryPrecisions.C.value != GEMMOperandPrecision::FP32) {
    source.SetValue("RELAXED_PRECISION", "false");
  } else {
    source.SetValue("RELAXED_PRECISION", "true");
  }
  source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("REGISTER_NAME_C", memoryName('C'));

  source += R"(

// Metal function arguments.
//
// A: the left-hand side matrix
// - dimensions: M x K
//               K x M (transposed)
// - memory precision: memA
//
// B: the right-hand side matrix
// - dimensions: K x N
//               N x K (transposed)
// - memory precision: memB
//
// C: the output matrix, alternatively the dot product accumulator
// - dimensions: M x N
// - memory precision: memC
//
// threadgroup_block: the chunk of threadgroup memory allocated at runtime
// - ideally 10 KB or less
// - precision: void/8-bit integer to make the pointer arithmetic more legible

kernel void matmul(device {{MEMORY_NAME_A}} *A_buf [[buffer(0)]],
                   device {{MEMORY_NAME_B}} *B_buf [[buffer(1)]],
                   device {{MEMORY_NAME_C}} *C_buf [[buffer(2)]],
)";
  if (useBias) {
    source += R"(
                   device {{MEMORY_NAME_BIAS}} *bias_buf [[buffer(3)]],
)";
    if (loadM) {
      source += R"(
                   const device uint *loadM [[buffer(4)]],
)";
    }
  } else {
    if (loadM) {
      source += R"(
                   const device uint *loadM [[buffer(3)]],
)";
    }
  }
  source += R"(
                 uint3 tgid [[threadgroup_position_in_grid]])
{
)";
  if (loadM) {
    source += R"(
  const uint M = loadM[0];
)";
  }
  source += R"(
  if (batched) {
    A_buf = A_buf + A_batch_stride * tgid.z;
    B_buf = B_buf + B_batch_stride * tgid.z;
)";
  if (splitK > 1) {
    source += R"(
    C_buf = C_buf + M * N * {{SPLIT_K}} * tgid.z;
)";
  } else {
    source += R"(
    C_buf = C_buf + C_batch_stride * tgid.z;
)";
  }
  if (useBias) {
    source += R"(
    bias_buf = bias_buf + bias_batch_stride * tgid.z;
)";
  }
  source += R"(
  }
)";
  if (transposed('A')) {
    source.SetValue("A_SLICE", std::to_string(blockDimensions[0]) + ", " + std::to_string(blockDimensions[2]));
    source.SetValue("A_MATRIX_SIZE", "M, K");
    source.SetValue("A_TILE_0_SIZE", "tgid.y * " + std::to_string(blockDimensions[0]) + ", 0");
    source.SetValue("A_TILE_K1_SIZE", "tgid.y * " + std::to_string(blockDimensions[0]) + ", k");
    source.SetValue("A_TILE_K2_SIZE", "tgid.y * " + std::to_string(blockDimensions[0]) + ", k + " + std::to_string(blockDimensions[2]));
    source.SetValue("A_TILE_LAST_K2_SIZE", "tgid.y * " + std::to_string(blockDimensions[0]) + ", K / " + std::to_string(blockDimensions[2] * 2) + " * " + std::to_string(blockDimensions[2] * 2));
    source.SetValue("A_TILE_LAST_K_SIZE", "tgid.y * " + std::to_string(blockDimensions[0]) + ", K / " + std::to_string(blockDimensions[2]) + " * " + std::to_string(blockDimensions[2]));
    source.SetValue("A_RESIDUAL_SLICE", std::to_string(blockDimensions[0]) + ", dynamic_extent");
  } else {
    source.SetValue("A_SLICE", std::to_string(blockDimensions[2]) + ", " + std::to_string(blockDimensions[0]));
    source.SetValue("A_MATRIX_SIZE", "K, M");
    source.SetValue("A_TILE_0_SIZE", "0, tgid.y * " + std::to_string(blockDimensions[0]));
    source.SetValue("A_TILE_K1_SIZE", "k, tgid.y * " + std::to_string(blockDimensions[0]));
    source.SetValue("A_TILE_K2_SIZE", "k + " + std::to_string(blockDimensions[2]) + ", tgid.y * " + std::to_string(blockDimensions[0]));
    source.SetValue("A_TILE_LAST_K2_SIZE", "K / " + std::to_string(blockDimensions[2] * 2) + " * " + std::to_string(blockDimensions[2] * 2) + ", tgid.y * " + std::to_string(blockDimensions[0]));
    source.SetValue("A_TILE_LAST_K_SIZE", "K / " + std::to_string(blockDimensions[2]) + " * " + std::to_string(blockDimensions[2]) + ", tgid.y * " + std::to_string(blockDimensions[0]));
    source.SetValue("A_RESIDUAL_SLICE", "dynamic_extent, " + std::to_string(blockDimensions[0]));
  }
  if (transposed('B')) {
    source.SetValue("B_SLICE", std::to_string(blockDimensions[2]) + ", " + std::to_string(blockDimensions[1]));
    source.SetValue("B_MATRIX_SIZE", "K, N");
    source.SetValue("B_TILE_0_SIZE", "0, tgid.x * " + std::to_string(blockDimensions[1]));
    source.SetValue("B_TILE_K1_SIZE", "k, tgid.x * " + std::to_string(blockDimensions[1]));
    source.SetValue("B_TILE_K2_SIZE", "k + " + std::to_string(blockDimensions[2]) + ", tgid.x * " + std::to_string(blockDimensions[1]));
    source.SetValue("B_TILE_LAST_K2_SIZE", "K / " + std::to_string(blockDimensions[2] * 2) + " * " + std::to_string(blockDimensions[2] * 2) + ", tgid.x * " + std::to_string(blockDimensions[1]));
    source.SetValue("B_TILE_LAST_K_SIZE", "K / " + std::to_string(blockDimensions[2]) + " * " + std::to_string(blockDimensions[2]) + ", tgid.x * " + std::to_string(blockDimensions[1]));
    source.SetValue("B_RESIDUAL_SLICE", "dynamic_extent, " + std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("B_SLICE", std::to_string(blockDimensions[1]) + ", " + std::to_string(blockDimensions[2]));
    source.SetValue("B_MATRIX_SIZE", "N, K");
    source.SetValue("B_TILE_0_SIZE", "tgid.x * " + std::to_string(blockDimensions[1]) + ", 0");
    source.SetValue("B_TILE_K1_SIZE", "tgid.x * " + std::to_string(blockDimensions[1]) + ", k");
    source.SetValue("B_TILE_K2_SIZE", "tgid.x * " + std::to_string(blockDimensions[1]) + ", k + " + std::to_string(blockDimensions[2]));
    source.SetValue("B_TILE_LAST_K2_SIZE", "tgid.x * " + std::to_string(blockDimensions[1]) + ", K / " + std::to_string(blockDimensions[2] * 2) + " * " + std::to_string(blockDimensions[2] * 2));
    source.SetValue("B_TILE_LAST_K_SIZE", "tgid.x * " + std::to_string(blockDimensions[1]) + ", K / " + std::to_string(blockDimensions[2]) + " * " + std::to_string(blockDimensions[2]));
    source.SetValue("B_RESIDUAL_SLICE", std::to_string(blockDimensions[1]) + ", dynamic_extent");
  }
  createInitializeC(&source);
  if (splitK > 1) {
    source.SetValue("SPLIT_K_STORE_OFFSET", "tgid.x * " + std::to_string(blockDimensions[1] * splitK) + " + k_split_idx * " + std::to_string(blockDimensions[1]));
    source.SetValue("BLOCK_DIMENSIONS_M_SPLIT_K", std::to_string(blockDimensions[0] * splitK));
  } else {
    source.SetValue("SPLIT_K_STORE_OFFSET", "tgid.x * " + std::to_string(blockDimensions[1]));
    source.SetValue("BLOCK_DIMENSIONS_M_SPLIT_K", std::to_string(blockDimensions[0]));
  }
  source += R"(

  // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
  auto A = tensor<device {{MEMORY_NAME_A}},  dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>({{A_MATRIX_SIZE}}));
  auto B = tensor<device {{MEMORY_NAME_B}},  dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>({{B_MATRIX_SIZE}}));
  auto C = tensor<device {{MEMORY_NAME_C}},  dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(N * {{SPLIT_K}}, M));
)";
  if (splitK > 1) {
    source += R"(
  uint k_split_idx;
  if (swap_mn_order) {
    k_split_idx = tgid.x / ((M + {{BLOCK_DIMENSIONS_M}} - 1) / {{BLOCK_DIMENSIONS_M}});
    tgid.x = tgid.x % ((M + {{BLOCK_DIMENSIONS_M}} - 1) / {{BLOCK_DIMENSIONS_M}});
    tgid.xy = tgid.yx;
  } else {
    k_split_idx = tgid.x / ((N + {{BLOCK_DIMENSIONS_N}} - 1) / {{BLOCK_DIMENSIONS_N}});
    tgid.x = tgid.x % ((N + {{BLOCK_DIMENSIONS_N}} - 1) / {{BLOCK_DIMENSIONS_N}});
  }
)";
  } else {
    source += R"(
  if (swap_mn_order) {
    tgid.xy = tgid.yx;
  }
)";
  }
  if (useBias) {
    source += R"(
  bias_buf = bias_buf + tgid.x * {{BLOCK_DIMENSIONS_N}};
)";
  }
  source += R"(

  if (tgid.x * {{BLOCK_DIMENSIONS_N}} + {{BLOCK_DIMENSIONS_N}} - 1 < N && tgid.y * {{BLOCK_DIMENSIONS_M}} + {{BLOCK_DIMENSIONS_M}} - 1 < M) {
    // Use static slice.
    // descriptor to create matmul operation that does {{BLOCK_DIMENSIONS_K}}x{{BLOCK_DIMENSIONS_M}} times {{BLOCK_DIMENSIONS_N}}x{{BLOCK_DIMENSIONS_K}} producing {{BLOCK_DIMENSIONS_N}}x{{BLOCK_DIMENSIONS_M}}
    constexpr auto matmul_descriptor = matmul2d_descriptor({{BLOCK_DIMENSIONS_M}}, {{BLOCK_DIMENSIONS_N}}, {{BLOCK_DIMENSIONS_K}}, {{TRANSPOSE_STATE_A}}, {{TRANSPOSE_STATE_B}}, {{RELAXED_PRECISION}}, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with {{EXECUTION_SIMD_GROUPS}} SIMD-Groups.
    matmul2d<matmul_descriptor, execution_simdgroups<{{EXECUTION_SIMD_GROUPS}}>> matmul_op;

    auto mA = A.slice<{{A_SLICE}}>({{A_TILE_0_SIZE}});
    auto mB = B.slice<{{B_SLICE}}>({{B_TILE_0_SIZE}});
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), {{REGISTER_NAME_C}}>();
{{INITIALIZE_C}}
)";

  if (splitK > 1) {
    source += R"(
    if (k_split_idx == 0) {
      #pragma clang loop unroll(full)
      for (uint k = 0; k < K_split; k += {{BLOCK_DIMENSIONS_K_2}}) {
        // Create appropriate slice for this thread group to work on.
        auto mA0 = A.slice<{{A_SLICE}}>({{A_TILE_K1_SIZE}});
        auto mB0 = B.slice<{{B_SLICE}}>({{B_TILE_K1_SIZE}});
        auto mA1 = A.slice<{{A_SLICE}}>({{A_TILE_K2_SIZE}});
        auto mB1 = B.slice<{{B_SLICE}}>({{B_TILE_K2_SIZE}});
        matmul_op.run(mA0, mB0, cT);
        matmul_op.run(mA1, mB1, cT);
      }
    }
)";
    source.SetValue("SPLIT_K_1", std::to_string(splitK - 1));
    if (splitK > 2) {
      source += R"(
    else if (k_split_idx > 0 && k_split_idx < {{SPLIT_K_1}}) {
      const uint k_start = k_split_idx * K_split;
      #pragma clang loop unroll(full)
      for (uint i_k = 0; i_k < K_split; i_k += {{BLOCK_DIMENSIONS_K_2}}) {
        const uint k = k_start + i_k;
        // Create appropriate slice for this thread group to work on.
        auto mA0 = A.slice<{{A_SLICE}}>({{A_TILE_K1_SIZE}});
        auto mB0 = B.slice<{{B_SLICE}}>({{B_TILE_K1_SIZE}});
        auto mA1 = A.slice<{{A_SLICE}}>({{A_TILE_K2_SIZE}});
        auto mB1 = B.slice<{{B_SLICE}}>({{B_TILE_K2_SIZE}});
        matmul_op.run(mA0, mB0, cT);
        matmul_op.run(mA1, mB1, cT);
      }
    }
)";
	}
    source += R"(
    else {
      #pragma clang loop unroll(full)
      for (uint k = {{SPLIT_K_1}} * K_split; k < K_edge; k += {{BLOCK_DIMENSIONS_K_2}}) {
        // Create appropriate slice for this thread group to work on.
        auto mA0 = A.slice<{{A_SLICE}}>({{A_TILE_K1_SIZE}});
        auto mB0 = B.slice<{{B_SLICE}}>({{B_TILE_K1_SIZE}});
        auto mA1 = A.slice<{{A_SLICE}}>({{A_TILE_K2_SIZE}});
        auto mB1 = B.slice<{{B_SLICE}}>({{B_TILE_K2_SIZE}});
        matmul_op.run(mA0, mB0, cT);
        matmul_op.run(mA1, mB1, cT);
      }
    }
)";
  } else {
    source += R"(
    #pragma clang loop unroll(full)
    for (uint k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_K_2}}) {
      // Create appropriate slice for this thread group to work on.
      auto mA0 = A.slice<{{A_SLICE}}>({{A_TILE_K1_SIZE}});
      auto mB0 = B.slice<{{B_SLICE}}>({{B_TILE_K1_SIZE}});
      auto mA1 = A.slice<{{A_SLICE}}>({{A_TILE_K2_SIZE}});
      auto mB1 = B.slice<{{B_SLICE}}>({{B_TILE_K2_SIZE}});
      matmul_op.run(mA0, mB0, cT);
      matmul_op.run(mA1, mB1, cT);
    }
)";
  }
  source += R"(
    if (K % ({{BLOCK_DIMENSIONS_K}} * 2) >= {{BLOCK_DIMENSIONS_K}}) {
      auto mA = A.slice<{{A_SLICE}}>({{A_TILE_LAST_K2_SIZE}});
      auto mB = B.slice<{{B_SLICE}}>({{B_TILE_LAST_K2_SIZE}});
      matmul_op.run(mA, mB, cT);
    }
    if (K % {{BLOCK_DIMENSIONS_K}} != 0) {
      constexpr auto matmul_descriptor = matmul2d_descriptor({{BLOCK_DIMENSIONS_M}}, {{BLOCK_DIMENSIONS_N}}, dynamic_length_v<int>, {{TRANSPOSE_STATE_A}}, {{TRANSPOSE_STATE_B}}, {{RELAXED_PRECISION}}, matmul2d_descriptor::mode::multiply_accumulate);
      // create matmul op from above descriptor with {{EXECUTION_SIMD_GROUPS}} SIMD-Groups.
      matmul2d<matmul_descriptor, execution_simdgroups<{{EXECUTION_SIMD_GROUPS}}>> matmul_op;
      auto mA = A.slice<{{A_RESIDUAL_SLICE}}>({{A_TILE_LAST_K_SIZE}});
      auto mB = B.slice<{{B_RESIDUAL_SLICE}}>({{B_TILE_LAST_K_SIZE}});
      matmul_op.run(mA, mB, cT);
    }
    auto mC = C.slice<{{BLOCK_DIMENSIONS_N}}, {{BLOCK_DIMENSIONS_M}}>({{SPLIT_K_STORE_OFFSET}}, tgid.y * {{BLOCK_DIMENSIONS_M}});
    cT.store(mC);
  } else {
    // Use dynamic slice for this edge case.
    // descriptor to create matmul operation that does {{BLOCK_DIMENSIONS_K}}x{{BLOCK_DIMENSIONS_M}} times {{BLOCK_DIMENSIONS_N}}x{{BLOCK_DIMENSIONS_K}} producing {{BLOCK_DIMENSIONS_N}}x{{BLOCK_DIMENSIONS_M}}
    constexpr auto matmul_descriptor = matmul2d_descriptor({{BLOCK_DIMENSIONS_M}}, {{BLOCK_DIMENSIONS_N}}, dynamic_length_v<int>, {{TRANSPOSE_STATE_A}}, {{TRANSPOSE_STATE_B}}, {{RELAXED_PRECISION}}, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with {{EXECUTION_SIMD_GROUPS}} SIMD-Groups.
    matmul2d<matmul_descriptor, execution_simdgroups<{{EXECUTION_SIMD_GROUPS}}>> matmul_op;

    auto mA = A.slice({{A_TILE_0_SIZE}});
    auto mB = B.slice({{B_TILE_0_SIZE}});
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), {{MEMORY_NAME_C}}>();
{{INITIALIZE_C}}
)";
  if (splitK > 1) {
    source += R"(
    if (k_split_idx == 0) {
      matmul_op.run(mA, mB, cT);
    }
)";
  } else {
    source += R"(
    matmul_op.run(mA, mB, cT);
)";
  }
  source += R"(
    // Since OS 26.2, cT.store(mC) is no longer safe store (not respecting C size). Doing this manually.
    auto mC = C_buf + tgid.y * {{BLOCK_DIMENSIONS_M_SPLIT_K}} * N + {{SPLIT_K_STORE_OFFSET}};
    const int N_edge = N - tgid.x * {{BLOCK_DIMENSIONS_N}};
    const int M_edge = M - tgid.y * {{BLOCK_DIMENSIONS_M}};
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
      if(cT.is_valid_element(k)) {
        auto idx = cT.get_multidimensional_index(k);
        if (idx[0] < N_edge && idx[1] < M_edge) {
          mC[idx[1] * {{SPLIT_K}} * N + idx[0]] = cT[k];
        }
      }
    }
  }
}
)";

  if (splitK > 1) {
    source.SetValue("BLOCK_DIMENSIONS_N_DIV_2", std::to_string(blockDimensions[1] / 2));
    source.SetValue("BLOCK_DIMENSIONS_N_DIV_2_SPLIT_K", std::to_string(blockDimensions[1] / 2 * splitK));
    source.SetValue("BLOCK_DIMENSIONS_N_SPLIT_K", std::to_string(blockDimensions[1] * splitK));
    source += R"(
kernel void reduce_sum_2(device {{MEMORY_NAME_C}}2 *A_buf [[buffer(0)]],
                         device {{MEMORY_NAME_C}}2 *B_buf [[buffer(1)]],
)";
    if (loadM) {
      source += R"(
                         const device uint *loadM [[buffer(2)]],
)";
    }
    source += R"(
                         uint2 tpig [[thread_position_in_grid]]) {
)";
    if (loadM) {
      source += R"(
  const uint M = loadM[0];
)";
    }
    source += R"(
  if (tpig.x >= M * N / 2) {
    return;
  }
  if (batched) {
    A_buf = A_buf + M * N / 2 * {{SPLIT_K}} * tpig.y;
    B_buf = B_buf + C_batch_stride / 2 * tpig.y;
  }
  A_buf += tpig.x / {{BLOCK_DIMENSIONS_N_DIV_2}} * {{BLOCK_DIMENSIONS_N_DIV_2_SPLIT_K}} + tpig.x % {{BLOCK_DIMENSIONS_N_DIV_2}};
  {{MEMORY_NAME_C}}2 val = A_buf[0];
  #pragma clang loop unroll(full)
  for (unsigned int k = 1; k < {{SPLIT_K}}; k++) {
    val += A_buf[k * {{BLOCK_DIMENSIONS_N_DIV_2}}];
  }
  B_buf[tpig.x] = val;
}

kernel void reduce_sum(device {{MEMORY_NAME_C}} *A_buf [[buffer(0)]],
                       device {{MEMORY_NAME_C}} *B_buf [[buffer(1)]],
)";
    if (loadM) {
      source += R"(
                       const device uint *loadM [[buffer(2)]],
)";
    }
    source += R"(
                       uint2 tpig [[thread_position_in_grid]]) {
)";
    if (loadM) {
      source += R"(
  const uint M = loadM[0];
)";
    }
    source += R"(
  if (tpig.x >= M * N) {
    return;
  }
  if (batched) {
    A_buf = A_buf + M * N * {{SPLIT_K}} * tpig.y;
    B_buf = B_buf + C_batch_stride * tpig.y;
  }
  A_buf += tpig.x / {{BLOCK_DIMENSIONS_N}} * {{BLOCK_DIMENSIONS_N_SPLIT_K}} + tpig.x % {{BLOCK_DIMENSIONS_N}};
  {{MEMORY_NAME_C}} val = A_buf[0];
  #pragma clang loop unroll(full)
  for (unsigned int k = 1; k < {{SPLIT_K}}; k++) {
    val += A_buf[k * {{BLOCK_DIMENSIONS_N}}];
  }
  B_buf[tpig.x] = val;
}
)";
  }

  return source.ToString();
}

std::string NAMatMulKernel::createConstants() const noexcept {
  std::string constants = R"(
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Whether we swap MN order (hence need to swap tgid.xy)
constant bool swap_mn_order [[function_constant(10)]];
// Specify the batch / batch strides at PSO creation time.
constant bool batched [[function_constant(11)]];

constant uint A_batch_stride [[function_constant(15)]];
constant uint B_batch_stride [[function_constant(16)]];
constant uint C_batch_stride [[function_constant(17)]];
constant uint bias_batch_stride [[function_constant(18)]];
)";
  if (!loadM) {
    constants += R"(
constant uint M [[function_constant(0)]];
)";
  }
  constants += R"(
constant uint K_edge = K > {{BLOCK_DIMENSIONS_K_2}} - 1 ? K + 1 - {{BLOCK_DIMENSIONS_K_2}} : 0;
)";
  if (splitK > 1) {
    constants += R"(
constant uint K_split = K / {{SPLIT_K}} / {{BLOCK_DIMENSIONS_K_2}} * {{BLOCK_DIMENSIONS_K_2}};
)";
  }
  return constants;
}

#pragma mark - Caching

void NAMatMulKernel::createInitializeC(CodeWriter *source) const noexcept {
  if (useBias) {
    if (splitK > 1) {
      source->SetValue("INITIALIZE_C", R"(
        if (k_split_idx == 0) {
          #pragma clang loop unroll(full)
          for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
            if(cT.is_valid_element(k)) {
              auto idx = cT.get_multidimensional_index(k);
              cT[k] = bias_buf[idx[0]];
            }
          }
        } else {
          #pragma clang loop unroll(full)
          for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
            if(cT.is_valid_element(k)) {
              cT[k] = 0;
            }
          }
        }
)");
    } else {
      source->SetValue("INITIALIZE_C", R"(
        #pragma clang loop unroll(full)
        for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
          if(cT.is_valid_element(k)) {
            auto idx = cT.get_multidimensional_index(k);
            cT[k] = bias_buf[idx[0]];
          }
        }
)");
    }
  } else {
    source->SetValue("INITIALIZE_C", R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
      if(cT.is_valid_element(k)) {
        cT[k] = 0;
      }
    }
)");
  }
}
