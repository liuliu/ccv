#include "NAAttentionKernel.hpp"
#include "NAAttentionDescriptor.hpp"
#include "GEMMHeaders.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>
#include <iomanip>

NAAttentionKernel::NAAttentionKernel(NAAttentionKernelDescriptor descriptor, MTL::Device *const device) {
  type = descriptor.type;
  memoryPrecisions = descriptor.memoryPrecisions;
  blockDimensions = descriptor.blockDimensions;
  headDimension = descriptor.headDimension;
  Hq = descriptor.Hq;
  Hk = descriptor.Hk;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  checkCEdge1 = descriptor.checkCEdge1;
  scale = descriptor.scale;
  bypassThreadgroupMemory = descriptor.bypassThreadgroupMemory;

  source = createSource();

  // Compile the shader source.
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  if (!library) {
    bypassThreadgroupMemory = false;
    source = createSource();
    string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  }
  CCV_NNC_MFA_CHECK_ERROR(error);
}

// MARK: - NAAttentionKernel

unsigned short NAAttentionKernel::threadgroupMemoryAllocation(MTL::ComputePipelineState *const pipelineState, const NAAttentionDescriptor &descriptor) const noexcept {
  if (type.value == AttentionKernelType::forward) {
    unsigned short threadgroupMemoryAllocation = blockDimensions[0] * blockDimensions[1] * executionSIMDGroups * memoryPrecisions[AttentionOperand::O].value().size();
    return threadgroupMemoryAllocation;
  }
  if (bypassThreadgroupMemory)
    return 0;
  if (memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32) {
    const uint16_t memorySize = memoryPrecisions[AttentionOperand::Q].value().size();
    const unsigned short kBlocks = (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
    const unsigned short sharedTileBytes =
        blockDimensions[2] *
        (blockDimensions[1] * 2 * kBlocks + blockDimensions[0] * 2 * executionSIMDGroups * kBlocks) *
        memorySize;
    return sharedTileBytes;
  }
  return 0;
}

/// The number of threads per group.
uint16_t NAAttentionKernel::threadgroupSize(MTL::ComputePipelineState *const pipelineState, const NAAttentionDescriptor &descriptor) const noexcept {
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

MTL::Size NAAttentionKernel::threadgroupsPerGrid(const NAAttentionDescriptor &descriptor) const noexcept {
  auto ceilDivide =
  [=](int64_t target, uint16_t granularity) -> int64_t {
    return (target + int64_t(granularity) - 1) / int64_t(granularity);
  };
  switch (type.value) {
  case AttentionKernelType::forward:
  case AttentionKernelType::backwardQuery:
    return MTL::Size(ceilDivide(descriptor.matrixDimensions[0], blockDimensions[0] * executionSIMDGroups) * Hq * descriptor.batchDimension, 1, 1);
  case AttentionKernelType::backwardKeyValue:
    return MTL::Size(ceilDivide(descriptor.matrixDimensions[1], blockDimensions[0] * executionSIMDGroups) * Hk * descriptor.batchDimension, 1, 1);
  }
  return MTL::Size(0, 0, 0);
}

std::string NAAttentionKernel::memoryName(AttentionOperand operand) const noexcept {
  auto value = memoryPrecisions[operand];
  return value.value().name();
}

std::string NAAttentionKernel::sequenceLength(AttentionOperand operand) const noexcept {
  switch (operand.value) {
  case AttentionOperand::Q:
  case AttentionOperand::dQ:
    return "R";
  case AttentionOperand::K:
  case AttentionOperand::dK:
    return "C";
  case AttentionOperand::V:
  case AttentionOperand::dV:
    return "C";
  case AttentionOperand::O:
  case AttentionOperand::dO:
    return "R";
  default:
    CCV_NNC_MFA_PRECONDITION(false);
  }
  return "";
}

unsigned short NAAttentionKernel::blockSequenceLength(AttentionOperand operand) const noexcept {
  switch (type.value) {
  case AttentionKernelType::forward:
  case AttentionKernelType::backwardQuery:
    switch (operand.value) {
    case AttentionOperand::Q:
    case AttentionOperand::dQ:
      return blockDimensions[0];
    case AttentionOperand::K:
    case AttentionOperand::dK:
      return blockDimensions[1];
    case AttentionOperand::V:
    case AttentionOperand::dV:
      return blockDimensions[1];
    case AttentionOperand::O:
    case AttentionOperand::dO:
      return blockDimensions[0];
    default:
      CCV_NNC_MFA_PRECONDITION(false);
    }

  case AttentionKernelType::backwardKeyValue:
    switch (operand.value) {
    case AttentionOperand::Q:
    case AttentionOperand::dQ:
      return blockDimensions[1];
    case AttentionOperand::K:
    case AttentionOperand::dK:
      return blockDimensions[0];
    case AttentionOperand::V:
    case AttentionOperand::dV:
      return blockDimensions[0];
    case AttentionOperand::O:
    case AttentionOperand::dO:
      return blockDimensions[1];
    default:
      CCV_NNC_MFA_PRECONDITION(false);
    }
  }
  CCV_NNC_MFA_PRECONDITION(false);
  return 0;
}

// MARK: - NAAttentionKernel+Source

std::string NAAttentionKernel::createSource() const noexcept {
  CodeWriter source;

  // Inject the contents of the headers.
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

)";

  createConstants(source);

  if (type.value == AttentionKernelType::backwardQuery) {
    source += createComputeD();
  }

  source += R"(
    
    // Declare the function.
    kernel void attention(
)";
  source += createBufferBindings() + "\n";
  switch (type.value) {
  case AttentionKernelType::forward:
    source.SetValue("DISPATCH_DIMENSION", "R");
    source.SetValue("DISPATCH_HEADS", "Hq");
    break;
  case AttentionKernelType::backwardQuery:
    source.SetValue("DISPATCH_DIMENSION", "R");
    source.SetValue("DISPATCH_HEADS", "Hq");
    break;
  case AttentionKernelType::backwardKeyValue:
    source.SetValue("DISPATCH_DIMENSION", "C");
    source.SetValue("DISPATCH_HEADS", "Hk");
    break;
  }
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
  if (type.value == AttentionKernelType::forward || !bypassThreadgroupMemory) {
    source += R"(
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
)";
  }
  if (type.value == AttentionKernelType::forward) {
    source += R"(
      ushort sgid [[simdgroup_index_in_threadgroup]],
      uint3 tgid [[threadgroup_position_in_grid]]
    ) {
)";
  } else {
    source += R"(
      ushort tid [[thread_index_in_threadgroup]],
      ushort sgid [[simdgroup_index_in_threadgroup]],
      uint3 tgid [[threadgroup_position_in_grid]]
    ) {
)";
  }
  source += R"(
  tgid = { (tgid.x / {{DISPATCH_HEADS}}) % (({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}})), tgid.x % {{DISPATCH_HEADS}}, tgid.x / {{DISPATCH_HEADS}} / (({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}})) };
  tgid.x = tgid.x * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{DISPATCH_DIMENSION}}) {
    return;
  }
)";
  source += createAdjustOffsets() + "\n";
  switch (type.value) {
  case AttentionKernelType::forward:
    loopForward(source);
    break;
  case AttentionKernelType::backwardQuery:
    loopBackwardQuery(source);
    break;
  case AttentionKernelType::backwardKeyValue:
    loopBackwardKeyValue(source);
    break;
  }
  source += "}\n";

  return source.ToString();
}

void NAAttentionKernel::createConstants(CodeWriter &source) const noexcept {
  source += R"(

// R = row dimension (output sequence)
// C = column dimension (input sequence)
// Hq = number of query heads.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];

)";
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  source.SetValue("HQ", std::to_string(Hq));
  source.SetValue("HK", std::to_string(Hk));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_2", std::to_string(blockDimensions[1] * 2));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source += R"(
constant uint Hq = {{HQ}};
constant uint Hk = {{HK}};
)";
  if (type.value == AttentionKernelType::forward) {
    source += R"(
// In this special case, leaving the rest to the trailing block to process.
constant uint C_remainder = (C % {{BLOCK_DIMENSIONS_TRAVERSAL_2}}) == {{BLOCK_DIMENSIONS_TRAVERSAL}} ? {{BLOCK_DIMENSIONS_TRAVERSAL}} : (C % {{BLOCK_DIMENSIONS_TRAVERSAL}});
)";
    if (checkCEdge1) {
      source += R"(
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint C_edge_1 = C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL_2}} : 0;
)";
    } else {
      // When we are not checking C_edge, C_edge makes sure we process entire blockDimensions.C * 2 block, rather than one of.
      // And leaving the rest to the C_remainder path.
      source += R"(
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL_2}} : 0;
)";
    }
    source += R"(
constant uint R_edge = R >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint R_remainder = R % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
constant uint K_edge = {{HEAD_DIMENSION}} + 1 - {{BLOCK_DIMENSIONS_HEAD}};
)";
  }
  source += R"(
constant uint K_Hq = {{HEAD_DIMENSION}} * Hq;
constant uint K_Hk = {{HEAD_DIMENSION}} * Hk;
)";
  for (const auto& operand : operands) {
    source.SetValue("OPERAND_NAME", operand.name());
    source.SetValue("OPERAND_BUFFER_INDEX", std::to_string(operand.bufferIndex() + 2));
    source += R"(
constant uint {{OPERAND_NAME}}_batch_stride [[function_constant({{OPERAND_BUFFER_INDEX}})]];
)";
  }
}

std::string NAAttentionKernel::createBufferBindings() const noexcept {
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  std::string output = "";
  for (const auto& operand : operands) {
    output += "  device ";
    output += memoryName(operand);
    output += "* " + operand.name() + "_buf [[buffer(";
    output += std::to_string(operand.bufferIndex()) + ")]],\n";
  }
  return output;
}

std::string NAAttentionKernel::operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept {
  CodeWriter source;
  source.SetValue("OPERAND", operand.name());
  if (operand.value == AttentionOperand::L || operand.value == AttentionOperand::D) {
    source += "{{OPERAND}}_buf + (tgid.z * Hq + tgid.y) * R\\";
  } else {
    source += "{{OPERAND}}_buf + tgid.z * {{OPERAND}}_batch_stride\\";
  }
  return source.ToString();
}

std::string NAAttentionKernel::createAdjustOffsets() const noexcept {
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  CodeWriter source;
  for (const auto& operand : operands) {
    source.SetValue("OPERAND", operand.name());
    source.SetValue("OPERAND_LOCATION", operandLocationWithHeadOffsetValue(operand));
      source += R"(
  {{OPERAND}}_buf = {{OPERAND_LOCATION}};
)";
  }
  return source.ToString();
}

// MARK: - Outer Loop

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
//   L = m + logBaseE(l)
//
// Backward Query
//   D = dO * O
//
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//
//     load V[c]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//
//     load K[c]
//     dQ += dS * K
//   }
//
// Backward Key-Value
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//
//     load dO[r]
//     dV += P^T * dO
//
//     load dO[r]
//     load D[r]
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//
//     load Q[r]
//     dK += dS^T * Q
//   }

static std::string high_precision_to_string(float value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

static std::string dotProductScale(float rsqrtD, bool derivative) {
  float logBase2E = 1.442695041;

  if (!derivative) {
    return high_precision_to_string(logBase2E * rsqrtD);
  } else {
    return high_precision_to_string(rsqrtD);
  }
}

std::string NAAttentionKernel::createComputeD() const noexcept {
  CodeWriter source;
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
  source.SetValue("MEMORY_NAME_D", memoryName(AttentionOperand::D));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("COMPUTE_D_THREADS", std::to_string(computeDThreads));
  source.SetValue("DOT_SCALE_DERIVATIVE", dotProductScale(scale, true));
  source += R"(

kernel void compute_d(
    device const {{MEMORY_NAME_O}}* O_buf [[buffer(3)]],
    device const {{MEMORY_NAME_DO}}* dO_buf [[buffer(6)]],
    device {{MEMORY_NAME_D}}* D_buf [[buffer(5)]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x % R;
  const uint head = tgid.x / R;
  O_buf += tgid.z * O_batch_stride;
  dO_buf += tgid.z * dO_batch_stride;
  D_buf += (tgid.z * Hq + head) * R;

  const uint offset = row * K_Hq + head * {{HEAD_DIMENSION}};
  float D_accumulator = 0;
  for (uint d = lane_id; d < {{HEAD_DIMENSION}}; d += {{COMPUTE_D_THREADS}}) {
    D_accumulator += (float)O_buf[offset + d] * (float)dO_buf[offset + d];
  }
  D_accumulator += simd_shuffle_xor(D_accumulator, 16);
  D_accumulator += simd_shuffle_xor(D_accumulator, 8);
  D_accumulator += simd_shuffle_xor(D_accumulator, 4);
  D_accumulator += simd_shuffle_xor(D_accumulator, 2);
  D_accumulator += simd_shuffle_xor(D_accumulator, 1);
  if (lane_id == 0) {
    D_buf[row] = ({{MEMORY_NAME_D}})(D_accumulator * {{DOT_SCALE_DERIVATIVE}});
  }
}

)";
  return source.ToString();
}

void NAAttentionKernel::loopForward(CodeWriter &source) const noexcept {
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
  // In OS 26.1, K no longer can be arbitrary number, it has to be multiple of 32. This might / might not be
  // a bug. A workaround is to use dynamic_length_v<int> which will result correct value.
  if (blockDimensions[1] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if ((headDimension % blockDimensions[2]) % 32 == 0) {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", std::to_string(headDimension % blockDimensions[2]));
  } else {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (Hq != Hk) {
  source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}},  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{MEMORY_NAME_K}},  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{MEMORY_NAME_V}},  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  threadgroup {{MEMORY_NAME_O}} *P_buf = (threadgroup {{MEMORY_NAME_O}}*)threadgroup_block + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sgid;
  auto P = tensor<threadgroup {{MEMORY_NAME_O}}, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
)";
  if (headDimension % blockDimensions[2] > 0) {
    source += R"(
  constexpr auto qk_desc_remainder = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc_remainder, execution_simdgroups<1>> matmul_qk_op_remainder;
)";
  }
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cS_1 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
  const unsigned short kBlocks = (std::max(headDimension, blockDimensions[2]) + blockDimensions[2] - 1) / blockDimensions[2];
  if (bypassThreadgroupMemory) {
    source += "  auto cP = matmul_pv_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_O}}, {{MEMORY_NAME_V}}, float>();\n";
    // Allocate O
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(cP), decltype(mV), float>();\n";
    }
  } else {
    // Allocate O
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();\n";
    }
  }
  source += R"(
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL_2}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
)";
  if (checkCEdge1) {
    source += R"(
        if (c < C_edge_1) {
          cS_1[k] = 0;
        } else {
          auto idx = cS_1.get_multidimensional_index(k);
          if (idx[0] >= (int)C_remainder) {
            cS_1[k] = -numeric_limits<float>::infinity();
          } else {
            cS_1[k] = 0;
          }
        }
)";
  } else {
    source += R"(
        cS_1[k] = 0;
)";
  }
  source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      auto mK_1 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_qk_op.run(mQ, mK_0, cS_0);
      matmul_qk_op.run(mQ, mK_1, cS_1);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      auto mK_1 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
      matmul_qk_op_remainder.run(mQ, mK_1, cS_1);
    }
)";
  }
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    auto cM_1_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_1, cM_1_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = max(cM_0_new[k], cM_1_new[k]) * {{DOT_SCALE}};
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
)";
  if (checkCEdge1) {
    source += R"(
        if (c < C_edge_1) {
          cS_1[k] = fast::exp2(cS_1[k] * {{DOT_SCALE}} - *dst_it);
        } else {
          auto idx = cS_1.get_multidimensional_index(k);
          if (idx[0] >= (int)C_remainder) {
            cS_1[k] = 0;
          } else {
            cS_1[k] = fast::exp2(cS_1[k] * {{DOT_SCALE}} - *dst_it);
          }
        }
)";
  } else {
    source += R"(
        cS_1[k] = fast::exp2(cS_1[k] * {{DOT_SCALE}} - *dst_it);
)";
  }
  source += R"(
      }
    }
    // Online reduce sum.
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    auto cL_1_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_1, cL_1_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_0_new[k] + cL_1_new[k];
      }
    }
    if (c == 0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] = 0;\n";
  }
  source += R"(
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = correction.map_iterator(it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
  }
  source += R"(
        }
      }
    }
)";
  if (bypassThreadgroupMemory) {
    source += R"(
    simdgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        cP[k] = ({{MEMORY_NAME_O}})cS_0[k];
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(cP, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
  } else {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(P, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
  }
  if (checkCEdge1) {
    if (bypassThreadgroupMemory) {
      source += R"(
    if (c < C_edge_1) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
        if(cS_1.is_valid_element(k)) {
          cP[k] = ({{MEMORY_NAME_O}})cS_1[k];
        }
      }
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
      auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_pv_op.run(cP, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    } else {
      source += R"(
    if (c < C_edge_1) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
        if(cS_1.is_valid_element(k)) {
          auto idx = cS_1.get_multidimensional_index(k);
          P_buf[idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_1[k];
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
      auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_pv_op.run(P, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    }
    source += R"(
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
        if(cS_1.is_valid_element(k)) {
          auto idx = cS_0.get_multidimensional_index(k);
          if (idx[0] >= (int)C_remainder) {
            P_buf[idx[0] - C_remainder + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
          } else {
            P_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_1[k];
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      // The reason to do this is because when K (in GEMM sense) is smaller (in this case, C_remainder is smaller than blockDimensions.C),
      // we need to start a new matmul descriptor with dynamic_extent for that, hence we copied the P_buf in this way and then sliced it.
      auto mP = P.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder, 0);
      constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
      auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, C - C_remainder);
      matmul_pv_op.run(mP, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
    source += R"(
    }
)";
  } else {
    if (bypassThreadgroupMemory) {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
      if(cS_1.is_valid_element(k)) {
        cP[k] = ({{MEMORY_NAME_O}})cS_1[k];
      }
    }
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
    auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
    matmul_pv_op.run(cP, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    } else {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
      if(cS_1.is_valid_element(k)) {
        auto idx = cS_1.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_1[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
    auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
    matmul_pv_op.run(P, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    }
  }
  source += R"(
  }
)";
  if (!checkCEdge1) { // Process the remainder path.
    source += R"(
  if (C_remainder > 0) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cS_0[k] = -numeric_limits<float>::infinity();
        } else {
          cS_0[k] = 0;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, C - C_remainder);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
    if (headDimension % blockDimensions[2] > 0) {
      source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
      source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, C - C_remainder);
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
    }
    source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_0_new[k] * {{DOT_SCALE}};
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
    // Online reduce sum.
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_0_new[k];
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = correction.map_iterator(it);
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "        cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
    }
    source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          P_buf[idx[0] - C_remainder + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
        } else {
          P_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // The reason to do this is because when K (in GEMM sense) is smaller (in this case, C_remainder is smaller than blockDimensions.C),
    // we need to start a new matmul descriptor with dynamic_extent for that, hence we copied the P_buf in this way and then sliced it.
    auto mP = P.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder, 0);
    constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, C - C_remainder);
    matmul_pv_op.run(mP, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
    source += R"(
  }
)";
  }
  source += R"(
  auto O = O_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = cL.map_iterator(it);
          auto L_reciprocal = fast::divide(1, *dst_it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
    } else {
      source += R"(
          if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
            O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
          }
)";
    }
  }
source += R"(
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        if (idx[0] < (int)R_remainder) {
          float L_sram = cM[k] + fast::log2(cL[k]);
          L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
        O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
    } else {
      source += R"(
        if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
        }
)";
    }
  }
source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        float L_sram = cM[k] + fast::log2(cL[k]);
        L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
      }
    }
  }
)";
}

void NAAttentionKernel::loopBackwardQuery(CodeWriter &source) const noexcept {
  const bool lowPrecisionInputs = memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32;
  const bool useThreadgroupSharing = lowPrecisionInputs && !bypassThreadgroupMemory;
  const unsigned short kBlocks = (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
  source.SetValue("MEMORY_NAME_DQ", memoryName(AttentionOperand::dQ));
  source.SetValue("MEMORY_NAME_DS", memoryName(AttentionOperand::Q));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("KBLOCKS", std::to_string(kBlocks));
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source.SetValue("DOT_SCALE_DERIVATIVE", dotProductScale(scale, true));
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  auto dO = tensor<device {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(dO_buf, dextents<int32_t, 2>(K_Hq, R));
)";
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  constexpr auto dsk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dsk_desc, execution_simdgroups<1>> matmul_dsk_op;
)";
  if (useThreadgroupSharing) {
    source += R"(
  threadgroup uchar *Q_shared_bytes = threadgroup_block;
  threadgroup uchar *dO_shared_bytes = Q_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_Q}}) * {{EXECUTION_SIMD_GROUPS}} * {{KBLOCKS}};
  threadgroup uchar *K_shared_bytes = dO_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_DO}}) * {{EXECUTION_SIMD_GROUPS}} * {{KBLOCKS}};
  threadgroup uchar *V_shared_bytes = K_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sizeof({{MEMORY_NAME_K}}) * {{KBLOCKS}};
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += R"(
  threadgroup uchar *Q{{LOOP_INDEX}}_shared_bytes = Q_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_Q}}) * {{EXECUTION_SIMD_GROUPS}} * {{LOOP_INDEX}} + sgid * {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_Q}});
  threadgroup uchar *dO{{LOOP_INDEX}}_shared_bytes = dO_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_DO}}) * {{EXECUTION_SIMD_GROUPS}} * {{LOOP_INDEX}} + sgid * {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_DO}});
  threadgroup uchar *K{{LOOP_INDEX}}_shared_bytes = K_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sizeof({{MEMORY_NAME_K}}) * {{LOOP_INDEX}};
  threadgroup uchar *V{{LOOP_INDEX}}_shared_bytes = V_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sizeof({{MEMORY_NAME_V}}) * {{LOOP_INDEX}};
  threadgroup {{MEMORY_NAME_Q}} *Q{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_Q}}*)Q{{LOOP_INDEX}}_shared_bytes;
  threadgroup {{MEMORY_NAME_DO}} *dO{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_DO}}*)dO{{LOOP_INDEX}}_shared_bytes;
  threadgroup {{MEMORY_NAME_K}} *K{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_K}}*)K{{LOOP_INDEX}}_shared_bytes;
  threadgroup {{MEMORY_NAME_V}} *V{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_V}}*)V{{LOOP_INDEX}}_shared_bytes;
  auto Q{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(Q{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto dO{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(dO{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto K{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(K{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>());
  auto V{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(V{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>());
)";
    }
    source += R"(
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<float, {{MEMORY_NAME_K}}, float>();
  auto cDP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mdO), decltype(mV), float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDQ_{{LOOP_INDEX}} = matmul_dsk_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mK), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
    if (cDQ_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDQ_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  const uint lane = tid % 32;
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
  for (uint load_index = lane; load_index < {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{BLOCK_DIMENSIONS_HEAD}};
    const uint row_idx = load_index / {{BLOCK_DIMENSIONS_HEAD}};
    const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
    Q{{LOOP_INDEX}}_shared_buf[load_index] = Q_buf[tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + row * K_Hq];
    dO{{LOOP_INDEX}}_shared_buf[load_index] = dO_buf[tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + row * K_Hq];
  }
)";
    }
    source += R"(
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto D = D_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  for (uint c = 0; c < C; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    for (uint load_index = tid; load_index < {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}}; load_index += 32 * {{EXECUTION_SIMD_GROUPS}}) {
      const uint head_idx = load_index % {{BLOCK_DIMENSIONS_HEAD}};
      const uint col_idx = load_index / {{BLOCK_DIMENSIONS_HEAD}};
      K{{LOOP_INDEX}}_shared_buf[load_index] = K_buf[tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + (c + col_idx) * K_Hk];
      V{{LOOP_INDEX}}_shared_buf[load_index] = V_buf[tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + (c + col_idx) * K_Hk];
    }
)";
    }
    source += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += R"(
    matmul_qk_op.run(Q{{LOOP_INDEX}}_shared, K{{LOOP_INDEX}}_shared, cS);
    matmul_qk_op.run(dO{{LOOP_INDEX}}_shared, V{{LOOP_INDEX}}_shared, cDP);
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
        cDS[k] = P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]);
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "    matmul_dsk_op.run(cDS, K{{LOOP_INDEX}}_shared, cDQ_{{LOOP_INDEX}});\n";
    }
    source += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
)";
  } else {
    source += R"(
  auto cDP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mdO), decltype(mV), float>();
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<float, {{MEMORY_NAME_K}}, float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDQ_{{LOOP_INDEX}} = matmul_dsk_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mK), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
    if (cDQ_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDQ_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto D = D_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  for (uint c = 0; c < C; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_qk_op.run(mQ_{{LOOP_INDEX}}, mK_{{LOOP_INDEX}}, cS);
      matmul_qk_op.run(mdO_{{LOOP_INDEX}}, mV_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
        cDS[k] = P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]);
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_dsk_op.run(cDS, mK_{{LOOP_INDEX}}, cDQ_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
)";
  }
  source += R"(
  auto dQ = dQ_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
    if (cDQ_0.is_valid_element(k)) {
      auto idx = cDQ_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += "      dQ[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_DQ}})cDQ_{{LOOP_INDEX}}[k];\n";
  }
  source += R"(
    }
  }
)";
}

void NAAttentionKernel::loopBackwardKeyValue(CodeWriter &source) const noexcept {
  const bool lowPrecisionInputs = memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32;
  const bool useThreadgroupSharing = lowPrecisionInputs && !bypassThreadgroupMemory;
  const unsigned short kBlocks = (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
  source.SetValue("MEMORY_NAME_DK", memoryName(AttentionOperand::dK));
  source.SetValue("MEMORY_NAME_DV", memoryName(AttentionOperand::dV));
  source.SetValue("MEMORY_NAME_P", memoryName(AttentionOperand::dO));
  source.SetValue("MEMORY_NAME_DS", memoryName(AttentionOperand::Q));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("KBLOCKS", std::to_string(kBlocks));
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source.SetValue("DOT_SCALE_DERIVATIVE", dotProductScale(scale, true));
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  auto dO = tensor<device {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(dO_buf, dextents<int32_t, 2>(K_Hq, R));
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
)";
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  constexpr auto kqt_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<kqt_desc, execution_simdgroups<1>> matmul_kqt_op;
  auto cST = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mK), decltype(mQ), float>();
  constexpr auto pdo_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pdo_desc, execution_simdgroups<1>> matmul_pdo_op;
)";
  if (useThreadgroupSharing) {
    source += R"(
  threadgroup uchar *K_shared_bytes = threadgroup_block;
  threadgroup uchar *V_shared_bytes = K_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_K}}) * {{EXECUTION_SIMD_GROUPS}} * {{KBLOCKS}};
  threadgroup uchar *Q_shared_bytes = V_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_V}}) * {{EXECUTION_SIMD_GROUPS}} * {{KBLOCKS}};
  threadgroup uchar *dO_shared_bytes = Q_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sizeof({{MEMORY_NAME_Q}}) * {{KBLOCKS}};
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += R"(
  threadgroup uchar *K{{LOOP_INDEX}}_shared_bytes = K_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_K}}) * {{EXECUTION_SIMD_GROUPS}} * {{LOOP_INDEX}} + sgid * {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_K}});
  threadgroup uchar *V{{LOOP_INDEX}}_shared_bytes = V_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_V}}) * {{EXECUTION_SIMD_GROUPS}} * {{LOOP_INDEX}} + sgid * {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * sizeof({{MEMORY_NAME_V}});
  threadgroup uchar *Q{{LOOP_INDEX}}_shared_bytes = Q_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sizeof({{MEMORY_NAME_Q}}) * {{LOOP_INDEX}};
  threadgroup uchar *dO{{LOOP_INDEX}}_shared_bytes = dO_shared_bytes + {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sizeof({{MEMORY_NAME_DO}}) * {{LOOP_INDEX}};
  threadgroup {{MEMORY_NAME_K}} *K{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_K}}*)K{{LOOP_INDEX}}_shared_bytes;
  threadgroup {{MEMORY_NAME_V}} *V{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_V}}*)V{{LOOP_INDEX}}_shared_bytes;
  threadgroup {{MEMORY_NAME_Q}} *Q{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_Q}}*)Q{{LOOP_INDEX}}_shared_bytes;
  threadgroup {{MEMORY_NAME_DO}} *dO{{LOOP_INDEX}}_shared_buf = (threadgroup {{MEMORY_NAME_DO}}*)dO{{LOOP_INDEX}}_shared_bytes;
  auto K{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(K{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto V{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(V{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto Q{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(Q{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>());
  auto dO{{LOOP_INDEX}}_shared = tensor<threadgroup {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(dO{{LOOP_INDEX}}_shared_buf, extents<int32_t, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>());
)";
    }
    source += R"(
  auto cP = matmul_pdo_op.get_left_input_cooperative_tensor<float, {{MEMORY_NAME_DO}}, float>();
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<float, {{MEMORY_NAME_Q}}, float>();
  auto cDP = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mV), decltype(mdO), float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDV_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cP), decltype(mdO), float>();\n";
      source += "  auto cDK_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mQ), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
    if (cDV_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDV_{{LOOP_INDEX}}[k] = 0;\n";
      source += "      cDK_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  const uint lane = tid % 32;
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
  for (uint load_index = lane; load_index < {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{BLOCK_DIMENSIONS_HEAD}};
    const uint col_idx = load_index / {{BLOCK_DIMENSIONS_HEAD}};
    const uint col = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + col_idx;
    K{{LOOP_INDEX}}_shared_buf[load_index] = K_buf[tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + col * K_Hk];
    V{{LOOP_INDEX}}_shared_buf[load_index] = V_buf[tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + col * K_Hk];
  }
)";
    }
    source += R"(
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0; r < R; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    for (uint load_index = tid; load_index < {{BLOCK_DIMENSIONS_HEAD}} * {{BLOCK_DIMENSIONS_TRAVERSAL}}; load_index += 32 * {{EXECUTION_SIMD_GROUPS}}) {
      const uint head_idx = load_index % {{BLOCK_DIMENSIONS_HEAD}};
      const uint row_idx = load_index / {{BLOCK_DIMENSIONS_HEAD}};
      Q{{LOOP_INDEX}}_shared_buf[load_index] = Q_buf[tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + (r + row_idx) * K_Hq];
      dO{{LOOP_INDEX}}_shared_buf[load_index] = dO_buf[tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + head_idx + (r + row_idx) * K_Hq];
    }
)";
    }
    source += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += R"(
    matmul_kqt_op.run(K{{LOOP_INDEX}}_shared, Q{{LOOP_INDEX}}_shared, cST);
    matmul_kqt_op.run(V{{LOOP_INDEX}}_shared, dO{{LOOP_INDEX}}_shared, cDP);
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        const float L_value = (float)L_buf[r + idx[0]];
        const float D_value = (float)D_buf[r + idx[0]];
        const float P_value = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
        cP[k] = P_value;
        cDS[k] = P_value * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value);
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += "    matmul_pdo_op.run(cP, dO{{LOOP_INDEX}}_shared, cDV_{{LOOP_INDEX}});\n";
      source += "    matmul_pdo_op.run(cDS, Q{{LOOP_INDEX}}_shared, cDK_{{LOOP_INDEX}});\n";
    }
    source += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
)";
  } else {
    source += R"(
  auto cDP = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mV), decltype(mdO), float>();
  auto cP = matmul_pdo_op.get_left_input_cooperative_tensor<float, {{MEMORY_NAME_DO}}, float>();
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<float, {{MEMORY_NAME_Q}}, float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDV_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cP), decltype(mdO), float>();\n";
      source += "  auto cDK_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mQ), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
    if (cDV_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDV_{{LOOP_INDEX}}[k] = 0;\n";
      source += "      cDK_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  for (uint r = 0; r < R; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_kqt_op.run(mK_{{LOOP_INDEX}}, mQ_{{LOOP_INDEX}}, cST);
      matmul_kqt_op.run(mV_{{LOOP_INDEX}}, mdO_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        const float L_value = (float)L_buf[r + idx[0]];
        const float D_value = (float)D_buf[r + idx[0]];
        const float P = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
        cP[k] = P;
        cDS[k] = P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value);
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
)";
  }
  source += R"(
  auto dK = dK_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  auto dV = dV_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
    if (cDV_0.is_valid_element(k)) {
      auto idx = cDV_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += "      dV[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{MEMORY_NAME_DV}})cDV_{{LOOP_INDEX}}[k];\n";
    source += "      dK[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{MEMORY_NAME_DK}})cDK_{{LOOP_INDEX}}[k];\n";
  }
  source += R"(
    }
  }
)";
}
