#include "NAAttentionKernel.hpp"
#include "NAAttentionDescriptor.hpp"
#include "GEMMHeaders.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>
#include <iomanip>

namespace {

uint32_t ceil_log2_u32_host(uint32_t x) {
  if (x <= 1)
    return 0;
  x -= 1;
  uint32_t bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}

}

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
  isCausal = descriptor.isCausal;
  masked = descriptor.masked;
  isVarlen = descriptor.isVarlen;
  loadC = descriptor.loadC;
  attentionSinks = descriptor.attentionSinks;
  slidingWindow = descriptor.slidingWindow;
  splitKV = descriptor.splitKV;

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
  if (type.value == AttentionKernelType::backwardQuery &&
      memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32 &&
      !bypassThreadgroupMemory) {
    return headDimension * blockDimensions[0] * executionSIMDGroups *
        memoryPrecisions[AttentionOperand::Q].value().size() * 2;
  }
  if (type.value == AttentionKernelType::backwardKeyValue &&
      memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32 &&
      !bypassThreadgroupMemory) {
    return headDimension * blockDimensions[0] * executionSIMDGroups *
        memoryPrecisions[AttentionOperand::K].value().size() * 2;
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
  case AttentionKernelType::forward: {
    if (splitKV > 1) {
      const uint32_t row_groups = (uint32_t)ceilDivide(descriptor.matrixDimensions[0], blockDimensions[0]);
      return MTL::Size(Hq * ceilDivide(splitKV, executionSIMDGroups), row_groups, descriptor.batchDimension);
    }
    const uint32_t row_groups = (uint32_t)ceilDivide(descriptor.matrixDimensions[0], blockDimensions[0] * executionSIMDGroups);
    const uint32_t morton_bits = ceil_log2_u32_host(row_groups) + ceil_log2_u32_host(Hq);
    return MTL::Size(uint64_t(1) << morton_bits, 1, descriptor.batchDimension);
  }
  case AttentionKernelType::backwardQuery: {
    return MTL::Size(ceilDivide(descriptor.matrixDimensions[0], blockDimensions[0] * executionSIMDGroups) * Hq * descriptor.batchDimension, 1, 1);
  }
  case AttentionKernelType::backwardKeyValue: {
    return MTL::Size(ceilDivide(descriptor.matrixDimensions[1], blockDimensions[0] * executionSIMDGroups) * Hk * descriptor.batchDimension, 1, 1);
  }
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

std::string NAAttentionKernel::createSplitKVSource() const noexcept {
  CodeWriter source;

  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

)";

  createConstants(source);
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
  source += R"(
kernel void attention(
  device {{MEMORY_NAME_Q}}* Q_buf [[buffer(0)]],
  device {{MEMORY_NAME_K}}* K_buf [[buffer(1)]],
  device {{MEMORY_NAME_V}}* V_buf [[buffer(2)]],
  device float* PartialO_buf [[buffer(5)]],
  device float* PartialL_buf [[buffer(6)]],
)";
  if (attentionSinks) {
    source += R"(
  device const {{MEMORY_NAME_Q}}* Sinks_buf [[buffer(19)]],
  constant uint& Sink_head_stride [[buffer(20)]],
)";
  }
  if (loadC) {
    source += R"(
  const device uint* C_buf [[buffer(21)]],
)";
  }
  source += R"(
  threadgroup uchar *threadgroup_block [[threadgroup(0)]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  uint3 tgid [[threadgroup_position_in_grid]]
) {
)";
  if (loadC) {
    createLoadCConstants(source);
  }
  source += R"(
  const uint split_group = tgid.x / Hq;
  const uint head = tgid.x - split_group * Hq;
  const uint row_group = tgid.y;
  const uint split_id = split_group * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (split_id >= SplitKV_splits) {
    return;
  }
  tgid = uint3(row_group, head, tgid.z);
  Q_buf += tgid.z * Q_batch_stride;
  K_buf += tgid.z * K_batch_stride;
  V_buf += tgid.z * V_batch_stride;
)";
  loopForwardSplitKV(source);
  source += R"(
}

)";
  source += createSplitKVCombine();
  return source.ToString();
}

std::string NAAttentionKernel::createSource() const noexcept {
  if (type.value == AttentionKernelType::forward && splitKV > 1) {
    return createSplitKVSource();
  }
  CodeWriter source;
  const bool lowPrecisionInputs =
      memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32;
  const bool usesThreadgroupBlock =
      type.value == AttentionKernelType::forward ||
      (type.value == AttentionKernelType::backwardQuery &&
       lowPrecisionInputs && !bypassThreadgroupMemory) ||
      (type.value == AttentionKernelType::backwardKeyValue &&
       lowPrecisionInputs && !bypassThreadgroupMemory);

  // Inject the contents of the headers.
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

)";

  createConstants(source);

  if (type.value == AttentionKernelType::forward && masked) {
    source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
    source += R"(
kernel void generate_attention_block_mask(
    device const {{MEMORY_NAME_Q}} *Mask_buf [[buffer(15)]],
    device uchar *Block_mask_buf [[buffer(16)]],
    threadgroup uint *block_mask_scratch [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
  const uint q_start = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  const uint c_start = tgid.y * {{BLOCK_DIMENSIONS_TRAVERSAL}};
  const uint q_extent = min((uint){{BLOCK_DIMENSIONS_PARALLELIZATION}}, R - q_start);
  const uint c_extent = min((uint){{BLOCK_DIMENSIONS_TRAVERSAL}}, C - c_start);
  const uint element_count = q_extent * c_extent;
  uint all_zero = 1;
  uint all_masked = 1;
  for (uint i = tid; i < element_count; i += {{BLOCK_MASK_THREADS}}) {
    const uint row = q_start + i / c_extent;
    const uint column = c_start + i - (i / c_extent) * c_extent;
)";
    if (isCausal) {
      source += R"(
    if (int(column) > int(row) + int(C) - int(R)) {
      continue;
    }
)";
    }
    source += R"(
    const float value = (float)Mask_buf[tgid.z * Mask_batch_stride + row * C + column];
    if (value != 0.0f) {
      all_zero = 0;
    }
    if (!(value <= {{MASKED_THRESHOLD}})) {
      all_masked = 0;
    }
  }
  threadgroup uint *zero_scratch = block_mask_scratch;
  threadgroup uint *masked_scratch = block_mask_scratch + {{BLOCK_MASK_THREADS}};
  zero_scratch[tid] = all_zero;
  masked_scratch[tid] = all_masked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint tile_all_zero = 1;
    uint tile_all_masked = 1;
    for (uint i = 0; i < {{BLOCK_MASK_THREADS}}; ++i) {
      tile_all_zero &= zero_scratch[i];
      tile_all_masked &= masked_scratch[i];
    }
    const uchar flag = tile_all_masked ? 0 : (tile_all_zero ? 1 : 2);
    Block_mask_buf[tgid.z * Block_mask_batch_stride + tgid.x * K_block_tiles + tgid.y] = flag;
  }
}

)";
  }

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
  if (usesThreadgroupBlock) {
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
    if (loadC) {
      createLoadCConstants(source);
    }
  } else {
    source += R"(
      ushort tid [[thread_index_in_threadgroup]],
      ushort sgid [[simdgroup_index_in_threadgroup]],
      uint3 tgid [[threadgroup_position_in_grid]]
    ) {
)";
  }
  if (type.value == AttentionKernelType::forward) {
    source += R"(
  const uint row_group_count = ({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group_bits = ceil_log2_u32(row_group_count);
  const uint head_bits = ceil_log2_u32({{DISPATCH_HEADS}});
  const uint2 morton_tile = morton_decode_rectangular_2d(tgid.x, row_group_bits, head_bits);
  if (morton_tile.y >= {{DISPATCH_HEADS}} || morton_tile.x >= row_group_count) {
    return;
  }
  tgid = uint3(morton_tile.x, morton_tile.y, tgid.z);
)";
    source += R"(
  tgid.x = tgid.x * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{DISPATCH_DIMENSION}}) {
    return;
  }
)";
  } else {
    source += R"(
  const uint linear_group = tgid.x;
  const uint row_group_count = ({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group = )";
  if (type.value == AttentionKernelType::backwardKeyValue) {
    source += "linear_group % row_group_count";
  } else {
    source += "(linear_group / {{DISPATCH_HEADS}}) % row_group_count";
  }
  source += R"(;
  const uint head = )";
  if (type.value == AttentionKernelType::backwardKeyValue) {
    source += "(linear_group / row_group_count) % {{DISPATCH_HEADS}}";
  } else {
    source += "linear_group % {{DISPATCH_HEADS}}";
  }
  source += R"(;
  const uint batch = linear_group / ({{DISPATCH_HEADS}} * row_group_count);
  tgid = uint3(row_group, head, batch);
  tgid.x = row_group * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{DISPATCH_DIMENSION}}) {
    return;
  }
)";
  }
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
)";
  if (!loadC) {
    source += R"(
constant uint C [[function_constant(1)]];
)";
  }
  source += R"(

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
  source.SetValue("BLOCK_MASK_THREADS", std::to_string(blockMaskThreads));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  std::ostringstream maskedThreshold;
  maskedThreshold << std::setprecision(std::numeric_limits<float>::max_digits10)
      << ((memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::FP16) ?
          (-65504.0f * 0.5f) :
          (-std::numeric_limits<float>::max() * 0.5f));
  source.SetValue("MASKED_THRESHOLD", maskedThreshold.str());
  source += R"(
constant uint Hq = {{HQ}};
constant uint Hk = {{HK}};
)";
  if (type.value == AttentionKernelType::forward) {
    if (!isVarlen) {
      if (!loadC && (isCausal || masked)) {
        source += R"(
constant uint C_single_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint C_single_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
)";
      } else if (!loadC) {
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
      }
      source += R"(
constant uint R_edge = R >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint R_remainder = R % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
constant uint K_edge = {{HEAD_DIMENSION}} + 1 - {{BLOCK_DIMENSIONS_HEAD}};
)";
    }
    if (isVarlen) {
      source += R"(
constant uint K_edge = {{HEAD_DIMENSION}} + 1 - {{BLOCK_DIMENSIONS_HEAD}};
)";
    }
  } else if (type.value == AttentionKernelType::backwardQuery) {
    source += R"(
constant uint C_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint R_edge = R >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint R_remainder = R % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
)";
  } else if (type.value == AttentionKernelType::backwardKeyValue) {
    source += R"(
constant uint KV_R_edge = R >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? R + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint KV_R_remainder = R % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint KV_C_edge = C >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? C + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint KV_C_remainder = C % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
)";
  }
  source += R"(
constant uint K_Hq = {{HEAD_DIMENSION}} * Hq;
constant uint K_Hk = {{HEAD_DIMENSION}} * Hk;
)";
  if (type.value == AttentionKernelType::forward && splitKV > 1) {
    source += R"(
constant uint SplitKV_splits [[function_constant(19)]];
constant uint Batch_dimension [[function_constant(20)]];
)";
    if (!loadC) {
      source += R"(
constant uint SplitKV_c_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant bool SplitKV_has_c_remainder = SplitKV_c_remainder > 0;
constant uint SplitKV_c_blocks = (C + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) / {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint SplitKV_blocks_per_split = (SplitKV_c_blocks + SplitKV_splits - 1) / SplitKV_splits;
)";
    }
  }
  for (const auto& operand : operands) {
    source.SetValue("OPERAND_NAME", operand.name());
    source.SetValue("OPERAND_BUFFER_INDEX", std::to_string(operand.bufferIndex() + 2));
    source += R"(
constant uint {{OPERAND_NAME}}_batch_stride [[function_constant({{OPERAND_BUFFER_INDEX}})]];
)";
  }
  if (type.value == AttentionKernelType::forward && masked) {
    source += R"(
constant uint Mask_batch_stride [[function_constant(15)]];
constant uint Block_mask_batch_stride [[function_constant(16)]];
constant uint K_block_tiles = (C + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) / {{BLOCK_DIMENSIONS_TRAVERSAL}};
)";
  }
  if (type.value == AttentionKernelType::forward) {
    source += R"(

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

inline uint ceil_log2_u32(uint x) {
  if (x <= 1)
    return 0;
  x -= 1;
  uint bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}
)";
  }
}

void NAAttentionKernel::createLoadCConstants(CodeWriter &source) const noexcept {
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_2", std::to_string(blockDimensions[1] * 2));
  source += R"(
  const uniform<uint> C = make_uniform(C_buf[0]);
)";
  if (type.value != AttentionKernelType::forward || isVarlen) {
    return;
  }
  if (splitKV > 1) {
    source += R"(
  const uniform<uint> SplitKV_c_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}}u;
  const uniform<bool> SplitKV_has_c_remainder = SplitKV_c_remainder > 0u;
  const uniform<uint> SplitKV_c_blocks = (C + {{BLOCK_DIMENSIONS_TRAVERSAL}}u - 1u) / {{BLOCK_DIMENSIONS_TRAVERSAL}}u;
  const uniform<uint> SplitKV_splits_uniform = make_uniform(SplitKV_splits);
  const uniform<uint> SplitKV_blocks_per_split = (SplitKV_c_blocks + SplitKV_splits_uniform - 1u) / SplitKV_splits_uniform;
)";
  } else if (isCausal || masked) {
    source += R"(
  const uniform<uint> C_single_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}}u;
  uniform<uint> C_single_edge = 0u;
  if (C >= {{BLOCK_DIMENSIONS_TRAVERSAL}}u) {
    C_single_edge = C + 1u - {{BLOCK_DIMENSIONS_TRAVERSAL}}u;
  }
)";
  } else {
    source += R"(
  uniform<uint> C_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}}u;
  if ((C % {{BLOCK_DIMENSIONS_TRAVERSAL_2}}u) == {{BLOCK_DIMENSIONS_TRAVERSAL}}u) {
    C_remainder = {{BLOCK_DIMENSIONS_TRAVERSAL}}u;
  }
)";
    if (checkCEdge1) {
      source += R"(
  uniform<uint> C_edge = 0u;
  if (C >= {{BLOCK_DIMENSIONS_TRAVERSAL}}u) {
    C_edge = C + 1u - {{BLOCK_DIMENSIONS_TRAVERSAL}}u;
  }
  uniform<uint> C_edge_1 = 0u;
  if (C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}}u) {
    C_edge_1 = C + 1u - {{BLOCK_DIMENSIONS_TRAVERSAL_2}}u;
  }
)";
    } else {
      source += R"(
  uniform<uint> C_edge = 0u;
  if (C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}}u) {
    C_edge = C + 1u - {{BLOCK_DIMENSIONS_TRAVERSAL_2}}u;
  }
)";
    }
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
  if (type.value == AttentionKernelType::forward && masked) {
    output += "  device const ";
    output += memoryName(AttentionOperand::Q);
    output += "* Mask_buf [[buffer(15)]],\n";
    output += "  device const uchar* Block_mask_buf [[buffer(16)]],\n";
  }
  if (type.value == AttentionKernelType::forward && isVarlen) {
    output += "  device const int* QSeqOffsets_buf [[buffer(17)]],\n";
    output += "  device const int* KVSeqOffsets_buf [[buffer(18)]],\n";
  }
  if (type.value == AttentionKernelType::forward && attentionSinks) {
    output += "  device const ";
    output += memoryName(AttentionOperand::Q);
    output += "* Sinks_buf [[buffer(19)]],\n";
    output += "  constant uint& Sink_head_stride [[buffer(20)]],\n";
  }
  if (type.value == AttentionKernelType::forward && loadC) {
    output += "  const device uint* C_buf [[buffer(21)]],\n";
  }
  return output;
}

std::string NAAttentionKernel::operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept {
  CodeWriter source;
  source.SetValue("OPERAND", operand.name());
  if (operand.value == AttentionOperand::L || operand.value == AttentionOperand::D) {
    if (type.value == AttentionKernelType::backwardKeyValue) {
      source.SetValue("QUERY_HEAD_RATIO", std::to_string(Hq / Hk));
      source += "{{OPERAND}}_buf + (tgid.z * Hq + tgid.y * {{QUERY_HEAD_RATIO}}) * R\\";
    } else {
      source += "{{OPERAND}}_buf + (tgid.z * Hq + tgid.y) * R\\";
    }
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
  if (type.value == AttentionKernelType::forward && isVarlen) {
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
    source += R"(
  const uint q_start = uint(QSeqOffsets_buf[tgid.z]);
  const uint q_end = uint(QSeqOffsets_buf[tgid.z + 1]);
  const uint kv_start = uint(KVSeqOffsets_buf[tgid.z]);
  const uint kv_end = uint(KVSeqOffsets_buf[tgid.z + 1]);
  const uint R_seq = q_end - q_start;
  const uint C_seq = kv_end - kv_start;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_seq) {
    return;
  }
  const uint C_single_remainder_seq = C_seq % {{BLOCK_DIMENSIONS_TRAVERSAL}};
  const uint C_single_edge_seq = C_seq >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C_seq + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
  const uint R_edge_seq = R_seq >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R_seq + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
  const uint R_remainder_seq = R_seq % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  Q_buf = Q_buf + q_start * K_Hq;
  K_buf = K_buf + kv_start * K_Hk;
  V_buf = V_buf + kv_start * K_Hk;
  O_buf = O_buf + q_start * K_Hq;
  L_buf = L_buf + (tgid.z * Hq + tgid.y) * R;
)";
    return source.ToString();
  }
  for (const auto& operand : operands) {
    source.SetValue("OPERAND", operand.name());
    source.SetValue("OPERAND_LOCATION", operandLocationWithHeadOffsetValue(operand));
      source += R"(
  {{OPERAND}}_buf = {{OPERAND_LOCATION}};
)";
  }
  if (type.value == AttentionKernelType::forward && masked) {
    source += R"(
  Mask_buf += tgid.z * Mask_batch_stride;
  Block_mask_buf += tgid.z * Block_mask_batch_stride;
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

void NAAttentionKernel::loopForwardSplitKV(CodeWriter &source) const noexcept {
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
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
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  const unsigned short kBlocks = (std::max(headDimension, blockDimensions[2]) + blockDimensions[2] - 1) / blockDimensions[2];
  source += R"(
  const uint c_block_begin = split_id * SplitKV_blocks_per_split;
  const uint c_block_end = min(SplitKV_c_blocks, c_block_begin + SplitKV_blocks_per_split);
  const uint c_begin = c_block_begin * {{BLOCK_DIMENSIONS_TRAVERSAL}};
  const uint c_end = min(uint(C), c_block_end * {{BLOCK_DIMENSIONS_TRAVERSAL}});
  const uint row_start = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  device float* PartialO = PartialO_buf + (((tgid.z * Hq + tgid.y) * SplitKV_splits + split_id) * R) * {{HEAD_DIMENSION}};
  device float* PartialL = PartialL_buf + ((tgid.z * Hq + tgid.y) * SplitKV_splits + split_id) * R;
  if (c_block_begin >= c_block_end) {
    for (uint row_idx = lane_id; row_idx < {{BLOCK_DIMENSIONS_PARALLELIZATION}}; row_idx += 32) {
      const uint row = row_start + row_idx;
      if (row >= R) {
        continue;
      }
      PartialL[row] = -numeric_limits<float>::infinity();
    }
    for (uint idx = lane_id; idx < {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{HEAD_DIMENSION}}; idx += 32) {
      const uint row = row_start + idx / {{HEAD_DIMENSION}};
      if (row >= R) {
        continue;
      }
      PartialO[row * {{HEAD_DIMENSION}} + idx % {{HEAD_DIMENSION}}] = 0;
    }
    return;
  }
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
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, row_start);
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
)";
  if (attentionSinks) {
    source += R"(
  const bool include_sink = split_id == 0;
  const float sink_m = include_sink ? (float)Sinks_buf[tgid.y * Sink_head_stride] * 1.442695041 : -numeric_limits<float>::infinity();
  const float sink_l = include_sink ? 1 : numeric_limits<float>::denorm_min();
)";
  }
  source += attentionSinks ? R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = sink_m;
      cL[k] = sink_l;
    }
  }
)" : R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
)";
  source += R"(
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();\n";
  }
  source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "      cO_{{LOOP_INDEX}}[k] = 0;\n";
  }
  source += R"(
    }
  }
  for (uint c = c_begin; c < c_end; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    const bool c_tail = SplitKV_has_c_remainder && c + {{BLOCK_DIMENSIONS_TRAVERSAL}} > C;
    uint c_remainder = {{BLOCK_DIMENSIONS_TRAVERSAL}};
    if (c_tail) {
      c_remainder = uint(SplitKV_c_remainder);
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        cS_0[k] = (c_tail && idx[0] >= (int)c_remainder) ? -numeric_limits<float>::infinity() : 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, row_start);
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, row_start);
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
  }
  if (isCausal) {
    source += R"(
    const bool causal_mask_0 = int(c + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) > int(C) - int(R);
    if (causal_mask_0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto idx = cS_0.get_multidimensional_index(k);
          const int causal_column_limit = int(row_start + idx[1]) + int(C) - int(R);
          if (int(c) + idx[0] > causal_column_limit) {
            cS_0[k] = -numeric_limits<float>::infinity();
          }
        }
      }
    }
)";
  }
  source += R"(
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
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
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (c_tail && idx[0] >= (int)c_remainder) {
          cS_0[k] = 0;
        } else {
          auto it = cS_0.get_iterator(k);
          auto dst_it = cM.map_iterator(it);
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if (cL.is_valid_element(k)) {
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
    if (c_tail) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto idx = cS_0.get_multidimensional_index(k);
          if (idx[0] >= (int)c_remainder) {
            P_buf[idx[0] - c_remainder + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
          } else {
            P_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - c_remainder + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      auto mP = P.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - c_remainder, 0);
      constexpr auto pv_desc_tail = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<pv_desc_tail, execution_simdgroups<1>> matmul_pv_tail_op;
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += R"(
      auto mV_0_tail_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_pv_tail_op.run(mP, mV_0_tail_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
  }
  source += R"(
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
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
  source += R"(
    }
  }
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {
      auto idx = cO_0.get_multidimensional_index(k);
      const uint row = row_start + idx[1];
      if (row < R) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        const uint partial_row_offset = row * {{HEAD_DIMENSION}};
        const float L_reciprocal = fast::divide(1, *dst_it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
        PartialO[partial_row_offset + idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}] = cO_{{LOOP_INDEX}}[k] * L_reciprocal;
)";
    } else {
      source += R"(
        if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
          PartialO[partial_row_offset + idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}] = cO_{{LOOP_INDEX}}[k] * L_reciprocal;
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
      const uint row = row_start + idx[0];
      if (row < R) {
        PartialL[row] = cM[k] + fast::log2(cL[k]);
      }
    }
  }
)";
}

std::string NAAttentionKernel::createSplitKVCombine() const noexcept {
  CodeWriter source;
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source += R"(
kernel void attention_splitkv_combine(
  device {{MEMORY_NAME_O}}* O_buf [[buffer(3)]],
  device {{MEMORY_NAME_L}}* L_buf [[buffer(4)]],
  device const float* PartialO_buf [[buffer(5)]],
  device const float* PartialL_buf [[buffer(6)]],
  uint gid [[thread_position_in_grid]]
) {
  const uint total = Batch_dimension * Hq * R * {{HEAD_DIMENSION}};
  if (gid >= total) {
    return;
  }
  const uint d = gid % {{HEAD_DIMENSION}};
  const uint row_head_linear = gid / {{HEAD_DIMENSION}};
  const uint row = row_head_linear % R;
  const uint head_linear = row_head_linear / R;
  const uint head = head_linear % Hq;
  const uint batch = head_linear / Hq;
  const uint split_base = (batch * Hq + head) * SplitKV_splits;
  float lse_max = -numeric_limits<float>::infinity();
  for (uint split = 0; split < SplitKV_splits; ++split) {
    lse_max = max(lse_max, PartialL_buf[(split_base + split) * R + row]);
  }
  float lse_sum = 0;
  for (uint split = 0; split < SplitKV_splits; ++split) {
    lse_sum += fast::exp2(PartialL_buf[(split_base + split) * R + row] - lse_max);
  }
  const float lse = lse_max + fast::log2(lse_sum);
  float output = 0;
  for (uint split = 0; split < SplitKV_splits; ++split) {
    const uint partial_idx = (split_base + split) * R + row;
    const float weight = fast::exp2(PartialL_buf[partial_idx] - lse);
    output += weight * PartialO_buf[partial_idx * {{HEAD_DIMENSION}} + d];
  }
  O_buf += batch * O_batch_stride;
  O_buf[(row * Hq + head) * {{HEAD_DIMENSION}} + d] = ({{MEMORY_NAME_O}})output;
  if (d == 0) {
    L_buf[(batch * Hq + head) * R + row] = ({{MEMORY_NAME_L}})lse;
  }
}
)";
  return source.ToString();
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

void NAAttentionKernel::loopForwardSingleCausal(CodeWriter &source) const noexcept {
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
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
  source.SetValue("R_LENGTH", isVarlen ? "R_seq" : "R");
  source.SetValue("C_LENGTH", isVarlen ? "C_seq" : "C");
  source.SetValue("R_EDGE", isVarlen ? "R_edge_seq" : "R_edge");
  source.SetValue("R_REMAINDER", isVarlen ? "R_remainder_seq" : "R_remainder");
  source.SetValue("C_SINGLE_EDGE", isVarlen ? "C_single_edge_seq" : "C_single_edge");
  source.SetValue("C_SINGLE_REMAINDER", isVarlen ? "C_single_remainder_seq" : "C_single_remainder");
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source.SetValue("MASK_SCALE", dotProductScale(1.0f, false));
  source.SetValue("SLIDING_WINDOW", std::to_string(slidingWindow));
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}},  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, {{R_LENGTH}}));
  auto K = tensor<device {{MEMORY_NAME_K}},  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, {{C_LENGTH}}));
  auto V = tensor<device {{MEMORY_NAME_V}},  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, {{C_LENGTH}}));
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
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
)";
  if (attentionSinks) {
    source += R"(
  const float sink_m = (float)Sinks_buf[tgid.y * Sink_head_stride] * 1.442695041;
)";
  }
  source += attentionSinks ? R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = sink_m;
      cL[k] = 1;
    }
  }
)" : R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
)";
  source += R"(
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
  const unsigned short kBlocks = (std::max(headDimension, blockDimensions[2]) + blockDimensions[2] - 1) / blockDimensions[2];
  if (bypassThreadgroupMemory) {
    source += "  auto cP = matmul_pv_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_O}}, {{MEMORY_NAME_V}}, float>();\n";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(cP), decltype(mV), float>();\n";
    }
  } else {
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();\n";
    }
  }
  if (isCausal || !isVarlen) {
    source += R"(
  const int causal_row_start = int(tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
)";
  }
  if (isCausal) {
    source += R"(
  const int causal_column_offset = int({{C_LENGTH}}) - int({{R_LENGTH}});
  const int causal_last_column = causal_row_start + int({{BLOCK_DIMENSIONS_PARALLELIZATION}}) - 1 + causal_column_offset;
  const int causal_first_column_limit = causal_row_start + causal_column_offset;
  const uint single_c_edge = causal_last_column < 0 ? 0 : min({{C_SINGLE_EDGE}}, uint(causal_last_column) + 1);
)";
    if (slidingWindow > 0) {
      source += R"(
  const int causal_first_visible_column = max(0, causal_first_column_limit + 1 - int({{SLIDING_WINDOW}}));
  const uint single_c_begin = uint(causal_first_visible_column) / {{BLOCK_DIMENSIONS_TRAVERSAL}} * {{BLOCK_DIMENSIONS_TRAVERSAL}};
)";
    }
  } else {
    source += R"(
  const uint single_c_edge = {{C_SINGLE_EDGE}};
)";
  }
  source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "      cO_{{LOOP_INDEX}}[k] = 0;\n";
  }
  if (slidingWindow > 0) {
    source += R"(
    }
  }

  for (uint c = single_c_begin; c < single_c_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
)";
  } else {
    source += R"(
    }
  }

  for (uint c = 0; c < single_c_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
)";
  }
  if (masked) {
    source += R"(
    const uchar mask_flags = Block_mask_buf[tgid.x * K_block_tiles + c / {{BLOCK_DIMENSIONS_TRAVERSAL}}];
    if (mask_flags == 0) {
      continue;
    }
)";
  }
  source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
  }
  if (masked) {
    if (isCausal) {
      source += R"(
    const bool causal_mask_0 = int(c + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) > causal_first_column_limit;
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int row = causal_row_start + idx[1];
        const int column = int(c) + idx[0];
        float score = cS_0[k] * {{DOT_SCALE}};
        if (mask_flags == 2 && row < int({{R_LENGTH}})) {
          score += (float)Mask_buf[row * C + column] * {{MASK_SCALE}};
        }
)";
    if (isCausal) {
      if (slidingWindow > 0) {
        source += R"(
        const int causal_column_limit = row + causal_column_offset;
        const int causal_column_begin = causal_column_limit + 1 - int({{SLIDING_WINDOW}});
        if ((causal_mask_0 && column > causal_column_limit) || column < causal_column_begin) {
          score = -numeric_limits<float>::infinity();
        }
)";
      } else {
        source += R"(
        const int causal_column_limit = row + causal_column_offset;
        if (causal_mask_0 && column > causal_column_limit) {
          score = -numeric_limits<float>::infinity();
        }
)";
      }
    }
    source += R"(
        cS_0[k] = score;
      }
    }
)";
  } else if (isCausal) {
    if (slidingWindow > 0) {
      source += R"(
    const bool causal_mask_0 = int(c + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) > causal_first_column_limit;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        const int causal_column_begin = causal_column_limit + 1 - int({{SLIDING_WINDOW}});
        const int column = int(c) + idx[0];
        if ((causal_mask_0 && column > causal_column_limit) || column < causal_column_begin) {
          cS_0[k] = -numeric_limits<float>::infinity();
        }
      }
    }
)";
    } else {
      source += R"(
    const bool causal_mask_0 = int(c + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) > causal_first_column_limit;
    if (causal_mask_0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto idx = cS_0.get_multidimensional_index(k);
          const int causal_row = causal_row_start + idx[1];
          const int causal_column_limit = causal_row + causal_column_offset;
          if (int(c) + idx[0] > causal_column_limit) {
            cS_0[k] = -numeric_limits<float>::infinity();
          }
        }
      }
    }
)";
    }
  }
  source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
)";
  if (masked) {
    source += R"(
        const float M_new = cM_0_new[k];
)";
  } else {
    source += R"(
        const float M_new = cM_0_new[k] * {{DOT_SCALE}};
)";
  }
  source += R"(
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
)";
  if (masked) {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS_0[k] = fast::exp2(cS_0[k] - *dst_it);
      }
    }
)";
  } else if (isCausal) {
    if (slidingWindow > 0) {
      source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        const int causal_column_begin = causal_column_limit + 1 - int({{SLIDING_WINDOW}});
        const int column = int(c) + idx[0];
        cS_0[k] = (column <= causal_column_limit && column >= causal_column_begin) ?
            fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it) :
            0;
      }
    }
)";
    } else {
      source += R"(
    // Softmax. cS becomes cP.
    if (causal_mask_0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto it = cS_0.get_iterator(k);
          auto dst_it = cM.map_iterator(it);
          auto idx = cS_0.get_multidimensional_index(k);
          const int causal_row = causal_row_start + idx[1];
          const int causal_column_limit = causal_row + causal_column_offset;
          cS_0[k] = int(c) + idx[0] <= causal_column_limit ?
              fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it) :
              0;
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto it = cS_0.get_iterator(k);
          auto dst_it = cM.map_iterator(it);
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
)";
    }
  } else {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
      }
    }
)";
  }
  source += R"(
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
  source += R"(
  }
)";
  if (isCausal) {
    if (slidingWindow > 0) {
      source += R"(
  if ({{C_SINGLE_REMAINDER}} > 0 && int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) <= causal_last_column && int({{C_LENGTH}}) > causal_first_visible_column) {
)";
    } else {
      source += R"(
  if ({{C_SINGLE_REMAINDER}} > 0 && int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) <= causal_last_column) {
)";
    }
  } else {
    source += R"(
  if ({{C_SINGLE_REMAINDER}} > 0) {
)";
  }
  if (masked) {
    source += R"(
    const uint c = {{C_LENGTH}} - {{C_SINGLE_REMAINDER}};
    const uchar mask_flags = Block_mask_buf[tgid.x * K_block_tiles + c / {{BLOCK_DIMENSIONS_TRAVERSAL}}];
    if (mask_flags != 0) {
)";
  }
  source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = -numeric_limits<float>::infinity();
        } else {
          cS_0[k] = 0;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, {{C_LENGTH}} - {{C_SINGLE_REMAINDER}});
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, {{C_LENGTH}} - {{C_SINGLE_REMAINDER}});
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
  }
  if (masked) {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int row = causal_row_start + idx[1];
        const int column = int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0];
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = -numeric_limits<float>::infinity();
        } else {
          float score = cS_0[k] * {{DOT_SCALE}};
          if (mask_flags == 2 && row < int({{R_LENGTH}})) {
            score += (float)Mask_buf[row * C + column] * {{MASK_SCALE}};
          }
)";
    if (isCausal) {
      if (slidingWindow > 0) {
        source += R"(
          const int causal_column_limit = row + causal_column_offset;
          const int causal_column_begin = causal_column_limit + 1 - int({{SLIDING_WINDOW}});
          if (column > causal_column_limit || column < causal_column_begin) {
            score = -numeric_limits<float>::infinity();
          }
)";
      } else {
        source += R"(
          const int causal_column_limit = row + causal_column_offset;
          if (column > causal_column_limit) {
            score = -numeric_limits<float>::infinity();
          }
)";
      }
    }
    source += R"(
          cS_0[k] = score;
        }
      }
    }
)";
  } else if (isCausal) {
    if (slidingWindow > 0) {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        const int causal_column_begin = causal_column_limit + 1 - int({{SLIDING_WINDOW}});
        const int column = int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0];
        if (column > causal_column_limit || column < causal_column_begin) {
          cS_0[k] = -numeric_limits<float>::infinity();
        }
      }
    }
)";
    } else {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        if (int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0] > causal_column_limit) {
          cS_0[k] = -numeric_limits<float>::infinity();
        }
      }
    }
)";
    }
  } else {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = -numeric_limits<float>::infinity();
        }
      }
    }
)";
  }
  source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
)";
  if (masked) {
    source += R"(
        const float M_new = cM_0_new[k];
)";
  } else {
    source += R"(
        const float M_new = cM_0_new[k] * {{DOT_SCALE}};
)";
  }
  source += R"(
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
)";
  if (masked) {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] - *dst_it);
        }
      }
    }
)";
  } else if (isCausal) {
    if (slidingWindow > 0) {
      source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        const int causal_column_begin = causal_column_limit + 1 - int({{SLIDING_WINDOW}});
        const int column = int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0];
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}} || column > causal_column_limit || column < causal_column_begin) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
)";
    } else {
      source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}} || int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0] > causal_column_limit) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
)";
    }
  } else {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
)";
  }
  source += R"(
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
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          P_buf[idx[0] - {{C_SINGLE_REMAINDER}} + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
        } else {
          P_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - {{C_SINGLE_REMAINDER}} + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto mP = P.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - {{C_SINGLE_REMAINDER}}, 0);
    constexpr auto pv_remainder_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_remainder_desc, execution_simdgroups<1>> matmul_pv_remainder_op;
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, {{C_LENGTH}} - {{C_SINGLE_REMAINDER}});
    matmul_pv_remainder_op.run(mP, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
  }
  source += R"(
  }
)";
  if (masked) {
    source += R"(
  }
)";
  }
  source += R"(
  auto O = O_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if ({{R_REMAINDER}} > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{R_EDGE}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int){{R_REMAINDER}}) {
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
        if (idx[0] < (int){{R_REMAINDER}}) {
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

void NAAttentionKernel::loopForward(CodeWriter &source) const noexcept {
  if (isCausal || masked || isVarlen) {
    loopForwardSingleCausal(source);
    return;
  }
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
)";
  if (attentionSinks) {
    source += R"(
  const float sink_m = (float)Sinks_buf[tgid.y * Sink_head_stride] * 1.442695041;
)";
  }
  source += attentionSinks ? R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = sink_m;
      cL[k] = 1;
    }
  }
)" : R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
)";
  source += R"(
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
)";
  source += R"(
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
)";
  source += R"(
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
)";
    source += R"(
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
  source.SetValue("MEMORY_NAME_DS", memoryName(AttentionOperand::D));
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
  threadgroup {{MEMORY_NAME_Q}} *Q_shared_buf = (threadgroup {{MEMORY_NAME_Q}}*)threadgroup_block +
      sgid * ({{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * 2);
  threadgroup {{MEMORY_NAME_DO}} *dO_shared_buf = Q_shared_buf + {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto Q_shared = tensor<threadgroup {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(
      Q_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  auto dO_shared = tensor<threadgroup {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(
      dO_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  const uint lane = tid % 32;
  for (uint load_index = lane; load_index < {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{HEAD_DIMENSION}};
    const uint row_idx = load_index / {{HEAD_DIMENSION}};
    const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
    if (row < R) {
      Q_shared_buf[load_index] = Q_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hq];
      dO_shared_buf[load_index] = dO_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hq];
    } else {
      Q_shared_buf[load_index] = 0;
      dO_shared_buf[load_index] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_K}}, float>();
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
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto D = D_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  const bool query_row_tail = (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge);
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
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
      auto mQ_{{LOOP_INDEX}} = Q_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mdO_{{LOOP_INDEX}} = dO_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
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
        if (query_row_tail && idx[1] >= (int)R_remainder) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
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
  if (C_remainder > 0) {
    const uint c = C - C_remainder;
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
      auto mQ_{{LOOP_INDEX}} = Q_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mdO_{{LOOP_INDEX}} = dO_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
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
        if (idx[0] >= (int)C_remainder || (query_row_tail && idx[1] >= (int)R_remainder)) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
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
  } else {
    source += R"(
  auto cDP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mdO), decltype(mV), float>();
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_K}}, float>();
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
  const bool query_row_tail = (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge);
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
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
        if (query_row_tail && idx[1] >= (int)R_remainder) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
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
  if (C_remainder > 0) {
    const uint c = C - C_remainder;
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
        if (idx[0] >= (int)C_remainder || (query_row_tail && idx[1] >= (int)R_remainder)) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
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
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
      if (cDQ_0.is_valid_element(k)) {
        auto idx = cDQ_0.get_multidimensional_index(k);
        if (idx[1] >= (int)R_remainder) {
          continue;
        }
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += "      dQ[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_DQ}})cDQ_{{LOOP_INDEX}}[k];\n";
  }
  source += R"(
      }
    }
  } else {
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
  source.SetValue("MEMORY_NAME_P", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("MEMORY_NAME_DS", memoryName(AttentionOperand::D));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("KBLOCKS", std::to_string(kBlocks));
  source.SetValue("QUERY_HEAD_RATIO", std::to_string(Hq / Hk));
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
  const uint query_head_start = tgid.y * {{QUERY_HEAD_RATIO}};
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head_start * {{HEAD_DIMENSION}}, 0);
  auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head_start * {{HEAD_DIMENSION}}, 0);
  constexpr auto kqt_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<kqt_desc, execution_simdgroups<1>> matmul_kqt_op;
  auto cST = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mK), decltype(mQ), float>();
  constexpr auto pdo_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pdo_desc, execution_simdgroups<1>> matmul_pdo_op;
)";
  if (useThreadgroupSharing) {
    source += R"(
  threadgroup {{MEMORY_NAME_K}} *K_shared_buf = (threadgroup {{MEMORY_NAME_K}}*)threadgroup_block +
      sgid * ({{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * 2);
  threadgroup {{MEMORY_NAME_V}} *V_shared_buf = K_shared_buf + {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto K_shared = tensor<threadgroup {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(
      K_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  auto V_shared = tensor<threadgroup {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(
      V_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  const uint lane = tid % 32;
  for (uint load_index = lane; load_index < {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{HEAD_DIMENSION}};
    const uint row_idx = load_index / {{HEAD_DIMENSION}};
    const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
    if (row < C) {
      K_shared_buf[load_index] = K_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
      V_shared_buf[load_index] = V_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
    } else {
      K_shared_buf[load_index] = 0;
      V_shared_buf[load_index] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto cP = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_P}}, {{MEMORY_NAME_DO}}, float>();
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_Q}}, float>();
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
  #pragma clang loop unroll(full)
  for (uint query_head_offset = 0; query_head_offset < {{QUERY_HEAD_RATIO}}; ++query_head_offset) {
    const uint query_head = query_head_start + query_head_offset;
    const device {{MEMORY_NAME_L}}* L = L_buf + query_head_offset * R;
    const device {{MEMORY_NAME_DS}}* D = D_buf + query_head_offset * R;
  for (uint r = 0; r < KV_R_edge; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
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
      auto mK_{{LOOP_INDEX}} = K_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mV_{{LOOP_INDEX}} = V_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
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
        const float L_value = (float)L[r + idx[0]];
        const float D_value = (float)D[r + idx[0]];
        const float P_value = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
        cP[k] = ({{MEMORY_NAME_P}})P_value;
        cDS[k] = ({{MEMORY_NAME_DS}})(P_value * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  if (KV_R_remainder > 0) {
    const uint r = R - KV_R_remainder;
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
      auto mK_{{LOOP_INDEX}} = K_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mV_{{LOOP_INDEX}} = V_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
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
        if (idx[0] >= (int)KV_R_remainder) {
          cP[k] = 0;
          cDS[k] = 0;
        } else {
          const float L_value = (float)L[r + idx[0]];
          const float D_value = (float)D[r + idx[0]];
          const float P_value = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
          cP[k] = ({{MEMORY_NAME_P}})P_value;
          cDS[k] = ({{MEMORY_NAME_DS}})(P_value * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  }
)";
  } else {
    source += R"(
  auto cDP = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mV), decltype(mdO), float>();
  auto cP = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_P}}, {{MEMORY_NAME_DO}}, float>();
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_Q}}, float>();
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
  #pragma clang loop unroll(full)
  for (uint query_head_offset = 0; query_head_offset < {{QUERY_HEAD_RATIO}}; ++query_head_offset) {
    const uint query_head = query_head_start + query_head_offset;
    const device {{MEMORY_NAME_L}}* L = L_buf + query_head_offset * R;
    const device {{MEMORY_NAME_DS}}* D = D_buf + query_head_offset * R;
  for (uint r = 0; r < KV_R_edge; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
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
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
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
        const float L_value = (float)L[r + idx[0]];
        const float D_value = (float)D[r + idx[0]];
        const float P = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
        cP[k] = ({{MEMORY_NAME_P}})P;
        cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  if (KV_R_remainder > 0) {
    const uint r = R - KV_R_remainder;
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
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
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
        if (idx[0] >= (int)KV_R_remainder) {
          cP[k] = 0;
          cDS[k] = 0;
        } else {
          const float L_value = (float)L[r + idx[0]];
          const float D_value = (float)D[r + idx[0]];
          const float P = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
          cP[k] = ({{MEMORY_NAME_P}})P;
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(query_head * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  }
)";
  }
  source += R"(
  auto dK = dK_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  auto dV = dV_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  if (KV_C_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= KV_C_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
      if (cDV_0.is_valid_element(k)) {
        auto idx = cDV_0.get_multidimensional_index(k);
        if (idx[1] >= (int)KV_C_remainder) {
          continue;
        }
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
  } else {
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
  }
)";
}
