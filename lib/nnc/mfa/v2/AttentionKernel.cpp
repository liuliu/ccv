#include "AttentionKernel.hpp"
#include "GEMMHeaders.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>
#include <iomanip>

AttentionKernel::AttentionKernel(AttentionKernelDescriptor descriptor, MTL::Device *const device) {
  type = descriptor.type;
  cacheState = descriptor.cacheState;
  memoryPrecisions = descriptor.memoryPrecisions;
  preferAsyncCache = descriptor.preferAsyncCache;
  preferAsyncLoad = descriptor.preferAsyncLoad;
  registerPrecisions = descriptor.registerPrecisions;
  transposeState = descriptor.transposeState;

  blockDimensions = descriptor.blockDimensions;
  headDimension = descriptor.headDimension;
  leadingDimensions = descriptor.leadingDimensions;
  disableAsyncCopy = false;

  threadgroupSize = 32 * (blockDimensions[0] / 8);

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();

  source = createSource();

  // Compile the shader source.
  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    if (error) {
      error = nil;
      library = NS::TransferPtr(findPrecompiledLibrary(descriptor, device, &error));
      if (!library) {
        preferAsyncCache = false;
        preferAsyncLoad = false;
        disableAsyncCopy = true;
        source = createSource();
        string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
        error = nil;
        library = NS::TransferPtr(device->newLibrary(string, nil, &error));
      }
    }
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

// MARK: - AttentionKernel

std::string AttentionKernel::memoryName(AttentionOperand operand) const noexcept {
  auto value = memoryPrecisions[operand];
  return value.value().name();
}

std::string AttentionKernel::registerName(AttentionOperand operand) const noexcept {
  auto value = registerPrecisions[operand];
  return value.value().name();
}

std::string AttentionKernel::loadFunction(AttentionOperand operand) const noexcept {
  auto memoryPrecision = memoryPrecisions[operand].value();
  auto registerPrecision = registerPrecisions[operand].value();
  switch (memoryPrecision.value) {
  case GEMMOperandPrecision::FP16:
    switch (registerPrecision.value) {
    case GEMMOperandPrecision::FP16:
      return "load";
    case GEMMOperandPrecision::BF16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::FP32:
      return "load";
    }
    break;
  case GEMMOperandPrecision::BF16:
    switch (registerPrecision.value) {
    case GEMMOperandPrecision::FP16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::BF16:
      return "load";
    case GEMMOperandPrecision::FP32:
      return "load_bfloat";
    }
    break;
  case GEMMOperandPrecision::FP32:
    switch (registerPrecision.value) {
    case GEMMOperandPrecision::FP16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::BF16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::FP32:
      return "load";
    }
    break;
  }
  return "load";
}

std::string AttentionKernel::storeFunction(AttentionOperand operand) const noexcept {
  auto memoryPrecision = memoryPrecisions[operand].value();
  auto registerPrecision = registerPrecisions[operand].value();
  switch (memoryPrecision.value) {
  case GEMMOperandPrecision::FP16:
    switch (registerPrecision.value) {
    case GEMMOperandPrecision::FP16:
      return "store";
    case GEMMOperandPrecision::BF16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::FP32:
      return "store";
    }
    break;
  case GEMMOperandPrecision::BF16:
    switch (registerPrecision.value) {
    case GEMMOperandPrecision::FP16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::BF16:
      return "store";
    case GEMMOperandPrecision::FP32:
      return "store_bfloat";
    }
    break;
  case GEMMOperandPrecision::FP32:
    switch (registerPrecision.value) {
    case GEMMOperandPrecision::FP16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::BF16:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
    case GEMMOperandPrecision::FP32:
      return "store";
    }
    break;
  }
  return "store";
}

bool AttentionKernel::cached(AttentionOperand operand) const noexcept {
  auto value = cacheState[operand];
  return value.value();
}

bool AttentionKernel::transposed(AttentionOperand operand) const noexcept {
  auto value = transposeState[operand];
  return value.value();
}

std::string AttentionKernel::sequenceLength(AttentionOperand operand) const noexcept {
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

unsigned short AttentionKernel::blockSequenceLength(AttentionOperand operand) const noexcept {
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

std::string AttentionKernel::leadingDimension(AttentionOperand operand) const noexcept {
  auto leadingDimension = leadingDimensions[operand];
  if (leadingDimension.value_or(false)) { // Prefer this value.
    return operand.name() + "_leading_dimension";
  }
  if (transposed(operand)) {
    return sequenceLength(operand);
  } else {
    return std::to_string(headDimension);
  }
}

unsigned short AttentionKernel::leadingBlockDimension(AttentionOperand operand) const noexcept {
  if (transposed(operand)) {
    return blockSequenceLength(operand);
  } else {
    return blockDimensions[2];
  }
}

std::string AttentionKernel::parallelizationDimensionValue() const noexcept {
  switch (type.value) {
  case AttentionKernelType::forward:
  case AttentionKernelType::backwardQuery:
    return "R";
  case AttentionKernelType::backwardKeyValue:
    return "C";
  }
  return "";
}

std::string AttentionKernel::parallelizationGroupOffsetValue() const noexcept {
  return "parallelization_group_offset";
}

std::string AttentionKernel::unsafeParallelizationThreadOffsetValue() const noexcept {
  return parallelizationGroupOffsetValue() + " + sidx * 8 + morton_offset.y";
}

std::string AttentionKernel::clampedParallelizationThreadOffsetValue() const noexcept {
  return "min(" + unsafeParallelizationThreadOffsetValue() + ", " + parallelizationDimensionValue() + " - 1)";
}

std::string AttentionKernel::traversalDimensionValue() const noexcept {
  switch (type.value) {
  case AttentionKernelType::forward:
  case AttentionKernelType::backwardQuery:
    return "C";
  case AttentionKernelType::backwardKeyValue:
    return "R";
  }
  return "";
}

std::string AttentionKernel::traversalOffsetValue() const noexcept {
  switch (type.value) {
  case AttentionKernelType::forward:
  case AttentionKernelType::backwardQuery:
    return "c";
  case AttentionKernelType::backwardKeyValue:
    return "r";
  }
}

std::string AttentionKernel::paddedTraversalEdgeValue() const noexcept {
  auto blockDim = blockDimensions[1];
  auto remainder = traversalDimensionValue() + " % " + std::to_string(blockDim);

  std::string output = "(" + remainder + " == 0) ? " + std::to_string(blockDim) + " : " + remainder;
  output = "((" + output + ") + 7) / 8 * 8";
  return output;
}

unsigned short AttentionKernel::paddedHeadDimensionValue() const noexcept {
  return (headDimension + 8 - 1) / 8 * 8;
}

unsigned short AttentionKernel::paddedHeadEdgeValue() const noexcept {
  auto blockDim = blockDimensions[2];
  auto remainder = (headDimension) % (blockDim);

  auto output = (remainder) == 0 ? (blockDim) : (remainder);
  output = (((output)) + 7) / 8 * 8;
  return output;
}

unsigned short AttentionKernel::threadgroupSizeValue() const noexcept {
  return 32 * (blockDimensions[0] / 8);
}

unsigned short AttentionKernel::createThreadgroupMemoryAllocation() const noexcept {
  unsigned short output = 0;
  unsigned short* outputRef = &output;

    // Sets the allocation to the maximum of this and the previous allocated
    // size.
  auto allocateParallelization =
  [=](AttentionOperand operand) -> void {
    auto memoryPrecision = memoryPrecisions[operand].value();

    unsigned short blockBytes = 1;
    blockBytes *= blockDimensions[0];
    blockBytes *= blockDimensions[2];
    blockBytes *= (unsigned short)memoryPrecision.size();

    *outputRef = std::max(*outputRef, blockBytes);
  };
  auto allocateTraversal =
  [=](AttentionOperand operand) -> void {
    auto memoryPrecision = memoryPrecisions[operand].value();

    unsigned short blockBytes = 1;
    blockBytes *= blockDimensions[1];
    blockBytes *= blockDimensions[2];
    blockBytes *= (unsigned short)memoryPrecision.size();

    *outputRef = std::max(*outputRef, blockBytes);
  };

  // Allocate memory for the GEMM operands.
  switch (type.value) {
  case AttentionKernelType::forward:
    // S = Q * K^T
    allocateParallelization(AttentionOperand::Q);
    allocateTraversal(AttentionOperand::K);

    // O += P * V
    allocateParallelization(AttentionOperand::O);
    allocateTraversal(AttentionOperand::V);
    break;
  case AttentionKernelType::backwardQuery:
    // S = Q * K^T
    allocateParallelization(AttentionOperand::Q);
    allocateTraversal(AttentionOperand::K);

    // dP = dO * V^T
    allocateParallelization(AttentionOperand::dO);
    allocateTraversal(AttentionOperand::V);

    // dQ += dS * K
    allocateParallelization(AttentionOperand::dQ);
    allocateTraversal(AttentionOperand::K);
    break;
  case AttentionKernelType::backwardKeyValue:
    // S^T = K * Q^T
    allocateParallelization(AttentionOperand::K);
    allocateTraversal(AttentionOperand::Q);

    // dV += P^T * dO
    allocateParallelization(AttentionOperand::dV);
    allocateTraversal(AttentionOperand::dO);

    // dP^T = V * dO^T
    allocateParallelization(AttentionOperand::V);
    allocateTraversal(AttentionOperand::dO);

    // dK += dS^T * Q
    allocateParallelization(AttentionOperand::dK);
    allocateTraversal(AttentionOperand::Q);
    break;
  }

  // dO * O
  //
  // Will never exceed 4 KB (128 threads/group), 8 KB (256 threads/group).
  if (AttentionKernelType::backwardQuery == type.value) {
    output = std::max(
      output,
      (unsigned short)(2 * blockDimensions[0] * 8 * 4));
  }

  // L or D
  //
  // Will never exceed ~512 bytes.
  if (AttentionKernelType::backwardKeyValue == type.value) {
    output = std::max(
      output,
      (unsigned short)(blockDimensions[1] * 4));
  }

  return output;
}

// MARK: - AttentionKernel+Source

std::string AttentionKernel::createSource() const noexcept {
  CodeWriter source;

  bool injectBF16Methods = false;
  switch (type.value) {
    case AttentionKernelType::forward:
      if ((memoryPrecisions[AttentionOperand::Q] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::K] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::S] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::P] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::V] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::O] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::L] == GEMMOperandPrecision::BF16)) {
        injectBF16Methods = true;
      }
      break;
    case AttentionKernelType::backwardQuery:
      if ((memoryPrecisions[AttentionOperand::Q] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::K] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::S] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::P] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::V] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::O] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::L] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::D] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dO] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dP] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dS] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dQ] == GEMMOperandPrecision::BF16)) {
        injectBF16Methods = true;
      }
      break;
    case AttentionKernelType::backwardKeyValue:
      if ((memoryPrecisions[AttentionOperand::Q] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::K] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::S] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::P] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::V] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::O] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::L] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::D] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dO] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dV] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dP] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dS] == GEMMOperandPrecision::BF16) || (memoryPrecisions[AttentionOperand::dK] == GEMMOperandPrecision::BF16)) {
        injectBF16Methods = true;
      }
      break;
  }

  // Inject the contents of the headers.
  source += createMetalSimdgroupEvent(disableAsyncCopy) + "\n";
  source += createMetalSimdgroupMatrixStorage(injectBF16Methods) + "\n";
  source += "using namespace metal;\n\n";

  source += createConstants() + "\n";

  source += R"(
    
    // Declare the function.
    kernel void attention(
)";
  source += createBufferBindings() + "\n";
  switch (type.value) {
  case AttentionKernelType::forward:
    source.SetValue("DISPATCH_DIMENSION", "R");
    break;
  case AttentionKernelType::backwardQuery:
    source.SetValue("DISPATCH_DIMENSION", "R");
    break;
  case AttentionKernelType::backwardKeyValue:
    source.SetValue("DISPATCH_DIMENSION", "C");
    break;
  }
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
  source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
  source += R"(
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
      
      uint3 gid [[threadgroup_position_in_grid]],
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]]
    ) {
      ushort2 morton_offset = morton_order(lane_id);
      gid = { gid.x % (({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} - 1) / {{BLOCK_DIMENSIONS_PARALLELIZATION}}), (gid.x / (({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} - 1) / {{BLOCK_DIMENSIONS_PARALLELIZATION}})) % Hq, gid.x / (Hq * (({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} - 1) / {{BLOCK_DIMENSIONS_PARALLELIZATION}}))};
      uint parallelization_group_offset = gid.x;
      parallelization_group_offset *= {{BLOCK_DIMENSIONS_PARALLELIZATION}};
      
      // Return early if the entire SIMD is out of bounds.
      if ({{PARALLELIZATION_GROUP_OFFSET}} >= {{PARALLELIZATION_DIMENSION}}) {
        return;
      }
)";
  source += createAdjustOffsets() + "\n";
  source += createSetup() + "\n";
  switch (type.value) {
  case AttentionKernelType::forward:
    source += loopForward() + "\n";
    break;
  case AttentionKernelType::backwardQuery:
    source += loopBackwardQuery() + "\n";
    break;
  case AttentionKernelType::backwardKeyValue:
    source += loopBackwardKeyValue() + "\n";
    break;
  }
  source += createCleanup(type) + "\n}\n";

  return source.ToString();
}

std::string AttentionKernel::createConstants() const noexcept {
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  std::string output = "";
  for (const auto& operand : operands) {
    output += "  constant uint " + operand.name() + "_batch_stride [[function_constant(";
    output += std::to_string(operand.bufferIndex() + 5) + ")]];\n";
    if (leadingDimensions[operand].value_or(false)) {
      output += "  constant uint " + operand.name() + "_leading_dimension [[function_constant(";
      output += std::to_string(operand.bufferIndex() + 15) + ")]];\n";
    }
  }
  return R"(

    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    // Hq = number of query heads.
    constant uint R [[function_constant(0)]];
    constant uint C [[function_constant(1)]];

    constant uint Hq [[function_constant(2)]];
    constant uint H_Hk_ratio [[function_constant(3)]];

	constant float dot_product_scale_derivative [[function_constant(4)]];
	constant float dot_product_scale = dot_product_scale_derivative * 1.442695041;

)" + output;
}

std::string AttentionKernel::createBufferBindings() const noexcept {
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  std::string output = "";
  for (const auto& operand : operands) {
    output += "  device ";
    output += memoryName(operand);
    output += "* " + operand.name() + " [[buffer(";
    output += std::to_string(operand.bufferIndex()) + ")]],\n";
  }
  return output;
}

std::string AttentionKernel::operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept {
  CodeWriter source;
  source.SetValue("OPERAND", operand.name());
  if (operand.value == AttentionOperand::L || operand.value == AttentionOperand::D) {
    source += "{{OPERAND}} + (gid.z * Hq + gid.y) * R\\";
  } else {
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    if (operand.value == AttentionOperand::K || operand.value == AttentionOperand::V || operand.value == AttentionOperand::dK || operand.value == AttentionOperand::dV) {
      if (!transposed(operand)) {
        source += "{{OPERAND}} + gid.z * {{OPERAND}}_batch_stride + gid.y / H_Hk_ratio * {{HEAD_DIMENSION}}\\";
      } else {
        source.SetValue("SEQUENCE_LENGTH", sequenceLength(operand));
        source += "{{OPERAND}} + gid.z * {{OPERAND}}_batch_stride + gid.y / H_Hk_ratio * {{HEAD_DIMENSION}} * {{SEQUENCE_LENGTH}}\\";
      }
    } else {
      if (!transposed(operand)) {
        source += "{{OPERAND}} + gid.z * {{OPERAND}}_batch_stride + gid.y * {{HEAD_DIMENSION}}\\";
      } else {
        source.SetValue("SEQUENCE_LENGTH", sequenceLength(operand));
        source += "{{OPERAND}} + gid.z * {{OPERAND}}_batch_stride + gid.y * {{HEAD_DIMENSION}} * {{SEQUENCE_LENGTH}}\\";
      }
    }
  }
  return source.ToString();
}

std::string AttentionKernel::operandLocationValue(AttentionOperand operand) const noexcept {
  return operand.name();
}

std::string AttentionKernel::createAdjustOffsets() const noexcept {
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  CodeWriter source;
  for (const auto& operand : operands) {
    source.SetValue("OPERAND", operand.name());
    source.SetValue("OPERAND_LOCATION", operandLocationWithHeadOffsetValue(operand));
      source += R"(
    {{OPERAND}} = {{OPERAND_LOCATION}};
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

struct AttentionOuterProductDescriptor {
  AttentionOperand A;
  AttentionOperand B;
  AttentionOperand C;

  constexpr AttentionOuterProductDescriptor(AttentionOperand aA, AttentionOperand aB, AttentionOperand aC) : A(aA), B(aB), C(aC) { }
};

struct AttentionAccumulateDescriptor {
  AttentionOperand A;
  AttentionOperand B;
  AttentionOperand C;

  std::string everyIterationScale;
  std::string lastIterationScale;

  AttentionAccumulateDescriptor(AttentionOperand aA, AttentionOperand aB, AttentionOperand aC, const std::string& aEveryIterationScale, const std::string& aLastIterationScale) : A(aA), B(aB), C(aC), everyIterationScale(aEveryIterationScale), lastIterationScale(aLastIterationScale) { }
};

std::string AttentionKernel::loopForward() const noexcept {
  AttentionOuterProductDescriptor outerProductDesc(AttentionOperand::Q, AttentionOperand::K, AttentionOperand::S);
  auto QKT = outerProduct(outerProductDesc);
  
  AttentionAccumulateDescriptor accumulateDesc(AttentionOperand::P, AttentionOperand::V, AttentionOperand::O, "correction", "fast::divide(1, l)");
  auto PV = accumulate(accumulateDesc);
  
  CodeWriter source;
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("QKT", QKT);
  source.SetValue("MASK_ATTENTION_MATRIX_EDGE", maskAttentionMatrixEdge());
  source.SetValue("ONLINE_REDUCE_MAXIMUM", onlineReduceMaximum());
  source.SetValue("ONLINE_CORRECT_O", onlineCorrectO());
  source.SetValue("SOFTMAX", softmax(false));
  source.SetValue("ONLINE_REDUCE_SUM", onlineReduceSum());
  source.SetValue("PV", PV);
  source += R"(

  // Outer loop over the traversal dimension.
  for (uint c = 0; c < C; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    // S = Q * K^T
    {{QKT}}
    {{MASK_ATTENTION_MATRIX_EDGE}}

    // m = reduce(m)
    {{ONLINE_REDUCE_MAXIMUM}}

    // correction = exp(m_old) / exp(m_new)
    {{ONLINE_CORRECT_O}}

    // P = softmax(S * scaleFactor)
    {{SOFTMAX}}

    // l = reduce(l)
    {{ONLINE_REDUCE_SUM}}

    // O *= correction
    // O += P * V
    // O /= l
    {{PV}}
  }

)";
  return source.ToString();
}

std::string AttentionKernel::loopBackwardQuery() const noexcept {
  AttentionOuterProductDescriptor outerProductDesc( AttentionOperand::Q, AttentionOperand::K, AttentionOperand::S);
  auto QKT = outerProduct(outerProductDesc);

  outerProductDesc.A = AttentionOperand::dO;
  outerProductDesc.B = AttentionOperand::V;
  outerProductDesc.C = AttentionOperand::dP;
  auto dOVT = outerProduct(outerProductDesc);

  AttentionAccumulateDescriptor accumulateDesc(AttentionOperand::dS, AttentionOperand::K, AttentionOperand::dQ, "", "");
  auto dSK = accumulate(accumulateDesc);

  CodeWriter source;
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("QKT", QKT);
  source.SetValue("SOFTMAX", softmax(false));
  source.SetValue("DOVT", dOVT);
  source.SetValue("D_SOFTMAX", softmax(true));
  source.SetValue("DSK", dSK);
  source += R"(

  // Outer loop over the traversal dimension.
  for (uint c = 0; c < C; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    // S = Q * K^T
    {{QKT}}

    // P = softmax(S * scaleFactor)
    {{SOFTMAX}}

    // dP = dO * V^T
    {{DOVT}}

    // dS = P * (dP - D) * scaleFactor
    {{D_SOFTMAX}}

    // dQ += dS * K
    {{DSK}}
  }

)";
  return source.ToString();

}

std::string AttentionKernel::loopBackwardKeyValue() const noexcept {
  AttentionOuterProductDescriptor outerProductDesc(AttentionOperand::K, AttentionOperand::Q, AttentionOperand::S /* S^T */);
  auto KQT = outerProduct(outerProductDesc);
  
  AttentionAccumulateDescriptor accumulateDesc(AttentionOperand::P, /* P^T */ AttentionOperand::dO, AttentionOperand::dV, "", "");
  auto PTdO = accumulate(accumulateDesc);
  
  outerProductDesc.A = AttentionOperand::V;
  outerProductDesc.B = AttentionOperand::dO;
  outerProductDesc.C = AttentionOperand::dP; // dP^T
  auto VdOT = outerProduct(outerProductDesc);
  
  accumulateDesc.A = AttentionOperand::dS; // dS^T
  accumulateDesc.B = AttentionOperand::Q;
  accumulateDesc.C = AttentionOperand::dK;
  auto dSTQ = accumulate(accumulateDesc);
  
  CodeWriter source;
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("KQT", KQT);
  source.SetValue("SOFTMAX", softmax(false));
  source.SetValue("PTDO", PTdO);
  source.SetValue("VDOT", VdOT);
  source.SetValue("D_SOFTMAX", softmax(true));
  source.SetValue("DSTQ", dSTQ);
  source += R"(

  // Outer loop over the traversal dimension.
  for (uint r = 0; r < R; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    // S^T = K * Q^T
    {{KQT}}

    // P^T = exp(S^T - L)
    {{SOFTMAX}}

    // dV += P^T * dO
    {{PTDO}}

    // dP^T = V * dO^T
    {{VDOT}}

    // dS^T = P^T * (dP^T - D) * scaleFactor
    {{D_SOFTMAX}}

    // dK += dS^T * Q
    {{DSTQ}}
  }

)";

  return source.ToString();
}

// MARK: - AttentionKernel+Accumulate

class MTLAddressSpace {
  // Hijack some C++ syntax, making it look like Swift's enumerations with
  // member functions.
  //
  // Source: https://stackoverflow.com/a/53284026
public:
  enum Value: uint16_t {
    device = 0,
    threadgroup = 1,
  };

  MTLAddressSpace() = default;
  constexpr MTLAddressSpace(Value aKernelType) : value(aKernelType) { }

  explicit operator bool() const = delete;

  constexpr bool operator==(const MTLAddressSpace &rhs) const { return value == rhs.value; }
  constexpr bool operator!=(const MTLAddressSpace &rhs) const { return value != rhs.value; }

  std::string keyword() const noexcept {
    switch (value) {
      case device:
        return "device";
      case threadgroup:
        return "threadgroup";
    }
  }

  std::string offsetType() const noexcept {
    switch (value) {
      case device:
        return "uint";
      case threadgroup:
        return "ushort";
    }
  }

  Value value;
};

std::string AttentionKernel::accumulate(const AttentionAccumulateDescriptor& accumulateDesc) const noexcept {

  struct LoopIterationDescriptor {
    MTLAddressSpace addressSpaceLHS;
    MTLAddressSpace addressSpaceRHS;
    std::string registerOffset;
    unsigned short registerSize;
    LoopIterationDescriptor(MTLAddressSpace aAddressSpaceLHS, MTLAddressSpace aAddressSpaceRHS) : addressSpaceLHS(aAddressSpaceLHS), addressSpaceRHS(aAddressSpaceRHS), registerOffset(""), registerSize(0) { }
  };

  // MARK: - Initialize
  auto A = accumulateDesc.A;
  auto B = accumulateDesc.B;
  auto C = accumulateDesc.C;

  auto allocateAccumulator =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    if (cached(C)) {
      return "";
    }
    CodeWriter source;
    source.SetValue("REGISTER_NAME_C", registerName(C));
    source.SetValue("C", C.name());
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source += R"(

    simdgroup_matrix_storage<{{REGISTER_NAME_C}}> {{C}}_sram[{{DESCRIPTOR_REGISTER_SIZE}} / 8];

)";
    return source.ToString();
  };
    
  auto initializeAccumulator =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("REGISTER_NAME_C", registerName(C));
    source.SetValue("C", C.name());
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("DESCRIPTOR_REGISTER_OFFSET", descriptor.registerOffset);
    source += R"(
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
      auto {{C}} = {{C}}_sram + ({{DESCRIPTOR_REGISTER_OFFSET}} + d) / 8;
      *{{C}} = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(0);
    }

)";
    return source.ToString();
  };

  auto scaleAccumulator =
  [=](std::string scale, LoopIterationDescriptor descriptor) -> std::string {
    if (scale.empty()) {
      return "";
    }
    CodeWriter source;
    source.SetValue("SCALE", scale);
    source.SetValue("C", C.name());
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("DESCRIPTOR_REGISTER_OFFSET", descriptor.registerOffset);
    source += R"(
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
      auto {{C}} = {{C}}_sram + ({{DESCRIPTOR_REGISTER_OFFSET}} + d) / 8;
      *({{C}}->thread_elements()) *= {{SCALE}};
    }

)";
    return source.ToString();
  };

  // MARK: - Load/Store Accumulator

  auto declareAccumulatorLocation =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("C", C.name());
    source.SetValue("C_LOCATION", operandLocationValue(C));
    source.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
    source.SetValue("MEMORY_NAME_C", memoryName(C));
    source.SetValue("LEADING_DIMENSION_C", leadingDimension(C));
    source.SetValue("LEADING_BLOCK_DIMENSION_C", std::to_string(leadingBlockDimension(C)));
    source.SetValue("TRANSPOSED_C", transposed(C) ? "true" : "false");
    switch (descriptor.addressSpaceLHS.value) {
    case MTLAddressSpace::device:
      source += R"(

       uint2 {{C}}_src_offset(
         morton_offset.x + d_outer,
         {{CLAMPED_PARALLELIZATION_THREAD_OFFSET}});
       auto {{C}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>
       ::apply_offset(
         {{C_LOCATION}}, {{LEADING_DIMENSION_C}},
         {{C}}_src_offset, {{TRANSPOSED_C}});

)";
       break;
     case MTLAddressSpace::threadgroup:
       source += R"(

       ushort2 {{C}}_block_offset(
         morton_offset.x,
         morton_offset.y + sidx * 8);
       auto {{C}}_src = (threadgroup {{MEMORY_NAME_C}}*)(threadgroup_block);
       {{C}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>
       ::apply_offset(
         {{C}}_src, {{LEADING_BLOCK_DIMENSION_C}},
         {{C}}_block_offset, {{TRANSPOSED_C}});
       threadgroup_barrier(mem_flags::mem_threadgroup);

)";
      break;
    }
    return source.ToString();
  };

  auto asyncLoadAccumulator = [=]() -> std::string {
    CodeWriter source;
    source.SetValue("C", C.name());
    source.SetValue("C_LOCATION", operandLocationValue(C));
    source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
    source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("MEMORY_NAME_C", memoryName(C));
    source.SetValue("LEADING_DIMENSION_C", leadingDimension(C));
    source.SetValue("LEADING_BLOCK_DIMENSION_C", std::to_string(leadingBlockDimension(C)));
    source.SetValue("TRANSPOSED_C", transposed(C) ? "true" : "false");
    if (disableAsyncCopy) {
      source.SetValue("ASYNC_LANE_ID", ", lane_id");
    } else {
      source.SetValue("ASYNC_LANE_ID", "");
    }
    source += R"(

     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (sidx == 0) {
       uint2 {{C}}_offset(d_outer, {{PARALLELIZATION_GROUP_OFFSET}});
       auto src = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>
       ::apply_offset(
         {{C_LOCATION}}, {{LEADING_DIMENSION_C}},
         {{C}}_offset, {{TRANSPOSED_C}});
       auto dst = (threadgroup {{MEMORY_NAME_C}}*)(threadgroup_block);
       
       ushort D_dimension = min(
         ushort({{BLOCK_DIMENSIONS_HEAD}}),
         ushort({{HEAD_DIMENSION}} - d_outer));
       ushort R_dimension = min(
         uint({{BLOCK_DIMENSIONS_PARALLELIZATION}}),
         uint({{PARALLELIZATION_DIMENSION}} - {{PARALLELIZATION_GROUP_OFFSET}}));
       ushort2 tile(D_dimension, R_dimension);
       
       simdgroup_event event;
       event.async_copy<{{LEADING_BLOCK_DIMENSION_C}}, 32>(
         dst, tile,
         src, {{LEADING_DIMENSION_C}}, tile{{ASYNC_LANE_ID}}, {{TRANSPOSED_C}});
       simdgroup_event::wait(1, &event);
     }

)";
    return source.ToString();
  };

  auto asyncStoreAccumulator =
  [=]() -> std::string {
    CodeWriter source;
    source.SetValue("C", C.name());
    source.SetValue("C_LOCATION", operandLocationValue(C));
    source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
    source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("MEMORY_NAME_C", memoryName(C));
    source.SetValue("LEADING_DIMENSION_C", leadingDimension(C));
    source.SetValue("LEADING_BLOCK_DIMENSION_C", std::to_string(leadingBlockDimension(C)));
    source.SetValue("TRANSPOSED_C", transposed(C) ? "true" : "false");
    if (disableAsyncCopy) {
      source.SetValue("ASYNC_LANE_ID", ", lane_id");
    } else {
      source.SetValue("ASYNC_LANE_ID", "");
    }
    source += R"(

     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (sidx == 0) {
       uint2 {{C}}_offset(d_outer, {{PARALLELIZATION_GROUP_OFFSET}});
       auto src = (threadgroup {{MEMORY_NAME_C}}*)(threadgroup_block);
       auto dst = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>
       ::apply_offset(
         {{C_LOCATION}}, {{LEADING_DIMENSION_C}},
         {{C}}_offset, {{TRANSPOSED_C}});
       
       ushort D_dimension = min(
         ushort({{BLOCK_DIMENSIONS_HEAD}}),
         ushort({{HEAD_DIMENSION}} - d_outer));
       ushort R_dimension = min(
         uint({{BLOCK_DIMENSIONS_PARALLELIZATION}}),
         uint({{PARALLELIZATION_DIMENSION}} - {{PARALLELIZATION_GROUP_OFFSET}}));
       ushort2 tile(D_dimension, R_dimension);
       
       simdgroup_event event;
       event.async_copy<{{LEADING_BLOCK_DIMENSION_C}}, 32>(
         dst, {{LEADING_DIMENSION_C}}, tile,
         src, tile{{ASYNC_LANE_ID}}, {{TRANSPOSED_C}});
       simdgroup_event::wait(1, &event);
     }
     
)";
    return source.ToString();
  };
   

  auto loadAccumulator =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("C", C.name());
    source.SetValue("DECLARE_ACCUMULATOR_LOCATION", declareAccumulatorLocation(descriptor));
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("LOAD_FUNCTION_C", loadFunction(C));
    source.SetValue("LEADING_DIMENSION_C", leadingDimension(C));
    source.SetValue("LEADING_BLOCK_DIMENSION_C", std::to_string(leadingBlockDimension(C)));
    source.SetValue("TRANSPOSED_C", transposed(C) ? "true" : "false");
    switch (descriptor.addressSpaceLHS.value) {
    case MTLAddressSpace::device:
      source += R"(

      {{DECLARE_ACCUMULATOR_LOCATION}}

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
        ushort2 {{C}}_origin(d, 0);
        {{C}}_sram[d / 8].{{LOAD_FUNCTION_C}}(
          {{C}}_src, {{LEADING_DIMENSION_C}},
          {{C}}_origin, {{TRANSPOSED_C}});
      }

)";
      break;
    case MTLAddressSpace::threadgroup:
      source.SetValue("ASYNC_LOAD_ACCUMULATOR", asyncLoadAccumulator());
      source += R"(

      {{ASYNC_LOAD_ACCUMULATOR}}
      {{DECLARE_ACCUMULATOR_LOCATION}}
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
        ushort2 {{C}}_origin(d, 0);
        {{C}}_sram[d / 8].{{LOAD_FUNCTION_C}}(
          {{C}}_src, {{LEADING_BLOCK_DIMENSION_C}}, 
          {{C}}_origin, {{TRANSPOSED_C}});
      }
      
)";
      break;
    }
    return source.ToString();
  };

  auto storeAccumulator =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("C", C.name());
    source.SetValue("DECLARE_ACCUMULATOR_LOCATION", declareAccumulatorLocation(descriptor));
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("STORE_FUNCTION_C", storeFunction(C));
    source.SetValue("LEADING_DIMENSION_C", leadingDimension(C));
    source.SetValue("LEADING_BLOCK_DIMENSION_C", std::to_string(leadingBlockDimension(C)));
    source.SetValue("TRANSPOSED_C", transposed(C) ? "true" : "false");
    source.SetValue("UNSAFE_PARALLELIZATION_THREAD_OFFSET", unsafeParallelizationThreadOffsetValue());
    source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    switch (descriptor.addressSpaceLHS.value) {
    case MTLAddressSpace::device:
      source += R"(

      {{DECLARE_ACCUMULATOR_LOCATION}}

      if ({{UNSAFE_PARALLELIZATION_THREAD_OFFSET}} < {{PARALLELIZATION_DIMENSION}}) {
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
          ushort2 {{C}}_origin(d, 0);
          {{C}}_sram[d / 8].{{STORE_FUNCTION_C}}(
            {{C}}_src, {{LEADING_DIMENSION_C}},
            {{C}}_origin, {{TRANSPOSED_C}});
        }
      }

)";
      break;
    case MTLAddressSpace::threadgroup:
      source.SetValue("ASYNC_STORE_ACCUMULATOR", asyncStoreAccumulator());
      source += R"(

      {{DECLARE_ACCUMULATOR_LOCATION}}

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
        ushort2 {{C}}_origin(d, 0);
        {{C}}_sram[d / 8].{{STORE_FUNCTION_C}}(
          {{C}}_src, {{LEADING_BLOCK_DIMENSION_C}},
          {{C}}_origin, {{TRANSPOSED_C}});
      }

      {{ASYNC_STORE_ACCUMULATOR}}

)";
      break;
    }
    return source.ToString();
  };

  // MARK: - Load RHS
  
  auto leadingDimensionRHS =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    switch (descriptor.addressSpaceRHS.value) {
    case MTLAddressSpace::device:
      return leadingDimension(B);
    case MTLAddressSpace::threadgroup:
      return std::to_string(leadingBlockDimension(B));
    }
  };
  
  auto declareRHSLocation =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("B", B.name());
    source.SetValue("B_LOCATION", operandLocationValue(B));
    source.SetValue("MEMORY_NAME_B", memoryName(B));
    source.SetValue("LEADING_DIMENSION_B", leadingDimension(B));
    source.SetValue("LEADING_BLOCK_DIMENSION_B", std::to_string(leadingBlockDimension(B)));
    source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    switch (descriptor.addressSpaceRHS.value) {
    case MTLAddressSpace::device:
      source += R"(

      uint2 {{B}}_src_offset(
        morton_offset.x + d_outer,
        morton_offset.y + {{TRAVERSAL_OFFSET}});
      auto {{B}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
      ::apply_offset(
        {{B_LOCATION}}, {{LEADING_DIMENSION_B}},
        {{B}}_src_offset, {{TRANSPOSED_B}});

)";
      break;
    case MTLAddressSpace::threadgroup:
      source += R"(

      ushort2 {{B}}_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto {{B}}_src = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block);
      {{B}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
      ::apply_offset(
        {{B}}_src, {{LEADING_BLOCK_DIMENSION_B}},
        {{B}}_block_offset, {{TRANSPOSED_B}});
      threadgroup_barrier(mem_flags::mem_threadgroup);

)";
      break;
    }
    return source.ToString();
  };
  
  auto loadRHS =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    switch (descriptor.addressSpaceRHS.value) {
    case MTLAddressSpace::device:
      return declareRHSLocation(descriptor);
    case MTLAddressSpace::threadgroup:
      CodeWriter source;
      source.SetValue("B", B.name());
      source.SetValue("B_LOCATION", operandLocationValue(B));
      source.SetValue("MEMORY_NAME_B", memoryName(B));
      source.SetValue("LEADING_DIMENSION_B", leadingDimension(B));
      source.SetValue("LEADING_BLOCK_DIMENSION_B", std::to_string(leadingBlockDimension(B)));
      source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
      source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
      source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
      source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
      source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
      source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
      source.SetValue("PADDED_TRAVERSAL_EDGE", paddedTraversalEdgeValue());
      source.SetValue("DECLARE_RHS_LOCATION", declareRHSLocation(descriptor));
      if (disableAsyncCopy) {
        source.SetValue("ASYNC_LANE_ID", ", lane_id");
      } else {
        source.SetValue("ASYNC_LANE_ID", "");
      }
      source += R"(
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 {{B}}_offset(d_outer, {{TRAVERSAL_OFFSET}});
        auto src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
        ::apply_offset(
          {{B_LOCATION}}, {{LEADING_DIMENSION_B}},
          {{B}}_offset, {{TRANSPOSED_B}});
        auto dst = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort({{BLOCK_DIMENSIONS_HEAD}}),
          ushort({{HEAD_DIMENSION}} - d_outer));
        ushort C_src_dimension = min(
          uint({{BLOCK_DIMENSIONS_TRAVERSAL}}),
          uint({{TRAVERSAL_DIMENSION}} - {{TRAVERSAL_OFFSET}}));
        ushort C_dst_dimension = max(
          ushort({{PADDED_TRAVERSAL_EDGE}}),
          ushort(C_src_dimension));
        ushort2 tile_src(D_dimension, C_src_dimension);
        ushort2 tile_dst(D_dimension, C_dst_dimension);
        
        simdgroup_event event;
        event.async_copy<{{LEADING_BLOCK_DIMENSION_B}}, 32>(
          dst, tile_dst,
          src, {{LEADING_DIMENSION_B}}, tile_src{{ASYNC_LANE_ID}}, {{TRANSPOSED_B}});
        simdgroup_event::wait(1, &event);
      }

      {{DECLARE_RHS_LOCATION}}
      
 )";
      return source.ToString();
    }
    return "";
  };
  
  // MARK: - Inner Loop
  
  auto innerLoopHead =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("DESCRIPTOR_REGISTER_OFFSET", descriptor.registerOffset);
    source.SetValue("REGISTER_NAME_B", registerName(B));
    source.SetValue("A", A.name());
    source.SetValue("B", B.name());
    source.SetValue("C", C.name());
    source.SetValue("LOAD_FUNCTION_B", loadFunction(B));
    source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
    source.SetValue("LEADING_DIMENSION_RHS", leadingDimensionRHS(descriptor));
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
      // Load the RHS from memory.
      ushort2 {{B}}_origin(d, c);
      simdgroup_matrix_storage<{{REGISTER_NAME_B}}> {{B}};
      {{B}}.{{LOAD_FUNCTION_B}}(
        {{B}}_src, {{LEADING_DIMENSION_RHS}},
        {{B}}_origin, {{TRANSPOSED_B}});

      // Issue one SIMD matmul instruction.
      {{C}}_sram[({{DESCRIPTOR_REGISTER_OFFSET}} + d) / 8].multiply(
        {{A}}_sram[c / 8], {{B}}, /*accumulate=*/true);
    }

)";
    return source.ToString();
  };
  
  auto innerLoopTraversal =
  [=](std::string traversalStart, std::string traversalEnd, LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("TRAVERSAL_START", traversalStart);
    source.SetValue("TRAVERSAL_END", traversalEnd);
    source.SetValue("INNER_LOOP_HEAD", innerLoopHead(descriptor));
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort c = {{TRAVERSAL_START}}; c < {{TRAVERSAL_END}}; c += 8) {
      {{INNER_LOOP_HEAD}}
    }

)";
    return source.ToString();
  };

  // MARK: - Outer Loop
  
  auto loopIteration =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    auto multiplyAB =
    [=]() -> std::string {
      CodeWriter source;
      if (descriptor.addressSpaceLHS == MTLAddressSpace::device ||
          descriptor.addressSpaceRHS == MTLAddressSpace::device) {
        auto blockDim = blockDimensions[1];
        source.SetValue("INNER_LOOP_TRAVERSAL", innerLoopTraversal("0", std::to_string(blockDim), descriptor));
        source.SetValue("BLOCK_DIM", std::to_string(blockDim));
        source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
        source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
        source.SetValue("SCALE_ACCUMULATOR", scaleAccumulator(accumulateDesc.lastIterationScale, descriptor));
        source += R"(

        {{INNER_LOOP_TRAVERSAL}}
        if (
          ({{TRAVERSAL_DIMENSION}} % {{BLOCK_DIM}} == 0) &&
          ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} == {{TRAVERSAL_DIMENSION}})
        ) {
           {{SCALE_ACCUMULATOR}}
        }

)";
      } else {
        source.SetValue("INNER_LOOP_TRAVERSAL_0", innerLoopTraversal("0", paddedTraversalEdgeValue(), descriptor));
        source.SetValue("INNER_LOOP_TRAVERSAL_1", innerLoopTraversal(paddedTraversalEdgeValue(), std::to_string(blockDimensions[1]), descriptor));
        source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
        source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
        source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
        source.SetValue("SCALE_ACCUMULATOR", scaleAccumulator(accumulateDesc.lastIterationScale, descriptor));
        source += R"(

        {{INNER_LOOP_TRAVERSAL_0}}
        if ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIMENSIONS_TRAVERSAL}}
            < {{TRAVERSAL_DIMENSION}}) {
          {{INNER_LOOP_TRAVERSAL_1}}
        } else {
          {{SCALE_ACCUMULATOR}}
        }

)";
      }
      return source.ToString();
    };
    
    CodeWriter source;
    source.SetValue("ALLOCATE_ACCUMULATOR", allocateAccumulator(descriptor));
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("INITIALIZE_ACCUMULATOR", initializeAccumulator(descriptor));
    if (cached(C)) {
      source.SetValue("LOAD_ACCUMULATOR", "");
      source.SetValue("STORE_ACCUMULATOR", "");
    } else {
      source.SetValue("LOAD_ACCUMULATOR", loadAccumulator(descriptor));
      source.SetValue("STORE_ACCUMULATOR", storeAccumulator(descriptor));
    }
    source.SetValue("LOAD_RHS", loadRHS(descriptor));
    source.SetValue("MULTIPLY_AB", multiplyAB());
    source.SetValue("SCALE_ACCUMULATOR", scaleAccumulator(accumulateDesc.everyIterationScale, descriptor));
    source += R"(

    {{ALLOCATE_ACCUMULATOR}}
    if ({{TRAVERSAL_OFFSET}} == 0) {
      {{INITIALIZE_ACCUMULATOR}}
    } else {
      {{LOAD_ACCUMULATOR}}
      {{SCALE_ACCUMULATOR}}
    }
    {{LOAD_RHS}}
    {{MULTIPLY_AB}}
    {{STORE_ACCUMULATOR}}

)";
    return source.ToString();
  };
  
  auto gatedLoopIteration =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    auto descriptorThreadgroup = descriptor;
    descriptorThreadgroup.addressSpaceLHS = MTLAddressSpace::threadgroup;
    descriptorThreadgroup.addressSpaceRHS = MTLAddressSpace::threadgroup;
    if (preferAsyncCache && preferAsyncLoad) {
      return loopIteration(descriptorThreadgroup);
    }

    auto descriptorDevice = descriptor;
    if (preferAsyncCache) {
      descriptorDevice.addressSpaceLHS = MTLAddressSpace::threadgroup;
    } else {
      descriptorDevice.addressSpaceLHS = MTLAddressSpace::device;
    }
    if (preferAsyncLoad) {
      descriptorDevice.addressSpaceRHS = MTLAddressSpace::threadgroup;
    } else {
      descriptorDevice.addressSpaceRHS = MTLAddressSpace::device;
    }
    
    auto blockDim = blockDimensions[1];
    CodeWriter source;
    source.SetValue("BLOCK_DIM", std::to_string(blockDim));
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("LOOP_ITERATION_DEVICE", loopIteration(descriptorDevice));
    source.SetValue("LOOP_ITERATION_THREADGROUP", loopIteration(descriptorThreadgroup));

    source += R"(

    if ((
          ({{TRAVERSAL_DIMENSION}} % {{BLOCK_DIM}} == 0) ||
          ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} <= {{TRAVERSAL_DIMENSION}})
        ) && (
          ({{HEAD_DIMENSION}} % 8 == 0) ||
          (d_outer + {{DESCRIPTOR_REGISTER_SIZE}} <= {{HEAD_DIMENSION}})
        )) {
      {{LOOP_ITERATION_DEVICE}}
    } else {
      {{LOOP_ITERATION_THREADGROUP}}
    }

)";
    return source.ToString();
  };

  // MARK: - Top Level Specification
  
  auto loopEnd =
  [=]() -> unsigned short {
    return paddedHeadDimensionValue();
  };
  
  auto loopEndFloor =
  [=]() -> unsigned short {
    return loopEnd() - loopEnd() % blockDimensions[2];
  };
  
  auto unrollStatement =
  [=]() -> std::string {
    if (cached(C)) {
      return "#pragma clang loop unroll(full)";
    } else {
      return "#pragma clang loop unroll(disable)";
    }
  };
  
  auto registerOffset =
  [=]() -> std::string {
    if (cached(C)) {
      return "d_outer";
    } else {
      return "0";
    }
  };
  
  auto firstIterations =
  [=]() -> std::string {
    LoopIterationDescriptor descriptor(MTLAddressSpace::device, MTLAddressSpace::device);
    descriptor.registerOffset = registerOffset();
    descriptor.registerSize = blockDimensions[2];
    CodeWriter source;
    source.SetValue("UNROLL_STATEMENT", unrollStatement());
    source.SetValue("LOOP_END_FLOOR", std::to_string(loopEndFloor()));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration(descriptor));
    
    source += R"(
    
    {{UNROLL_STATEMENT}}
    for (
      ushort d_outer = 0;
      d_outer < {{LOOP_END_FLOOR}};
      d_outer += {{BLOCK_DIMENSIONS_HEAD}}
    ) {
      {{GATED_LOOP_ITERATION}}
    }
    
)";
    return source.ToString();
  };
  
  auto lastIteration =
  [=]() -> std::string {
    LoopIterationDescriptor descriptor(MTLAddressSpace::device, MTLAddressSpace::device);
    descriptor.registerOffset = registerOffset();
    descriptor.registerSize = paddedHeadEdgeValue();
    
    CodeWriter source;
    source.SetValue("LOOP_END_FLOOR", std::to_string(loopEndFloor()));
    source.SetValue("LOOP_END_FLOOR_LESS_LOOP_END", (loopEndFloor() < loopEnd()) ? "true" : "false");
    source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration(descriptor));
    
    source += R"(

    if ({{LOOP_END_FLOOR_LESS_LOOP_END}}) {
      ushort d_outer = {{LOOP_END_FLOOR}};
      {{GATED_LOOP_ITERATION}}
    }

)";
    return source.ToString();
  };
  
  // Collect all of the statements into one string.
  return "\n" + firstIterations() + "\n" + lastIteration() + "\n";
}

// MARK: - AttentionKernel+Caching

std::string AttentionKernel::cache(AttentionOperand operand, CachingOperationType type) const noexcept {
  // MARK: - Operand

  auto allocateOperand =
  [=]() -> std::string {
    if (type == CachingOperationType::load) {
      CodeWriter source;
      source.SetValue("REGISTER_NAME_OPERAND", registerName(operand));
      source.SetValue("OPERAND", operand.name());
      source.SetValue("PADDED_HEAD_DIMENSION_8", std::to_string(paddedHeadDimensionValue() / 8));
      source += R"(

      simdgroup_matrix_storage<{{REGISTER_NAME_OPERAND}}> {{OPERAND}}_sram[{{PADDED_HEAD_DIMENSION_8}}];

)";
      return source.ToString();
    } else {
      return "";
    }
  };
  
  auto asyncAccessOperand =
  [=]() -> std::string {
    if (type == CachingOperationType::load) {
      CodeWriter source;
      source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
      source.SetValue("OPERAND", operand.name());
      source.SetValue("OPERAND_LOCATION", operandLocationValue(operand));
      source.SetValue("LEADING_BLOCK_DIMENSION_OPERAND", std::to_string(leadingBlockDimension(operand)));
      source.SetValue("LEADING_DIMENSION_OPERAND", leadingDimension(operand));
      source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
      source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
      source.SetValue("PADDED_HEAD_DIMENSION", std::to_string(paddedHeadDimensionValue()));
      source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
      source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
      source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
      source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
      if (disableAsyncCopy) {
        source.SetValue("ASYNC_LANE_ID", ", lane_id");
      } else {
        source.SetValue("ASYNC_LANE_ID", "");
      }
      source += R"(

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 {{OPERAND}}_offset(d_outer, {{PARALLELIZATION_GROUP_OFFSET}});
        auto src = simdgroup_matrix_storage<{{MEMORY_NAME_OPERAND}}>
        ::apply_offset(
          {{OPERAND_LOCATION}}, {{LEADING_DIMENSION_OPERAND}},
          {{OPERAND}}_offset, {{TRANSPOSED_OPERAND}});
        auto dst = (threadgroup {{MEMORY_NAME_OPERAND}}*)(threadgroup_block);

        ushort D_src_dimension = min(
          ushort({{BLOCK_DIMENSIONS_HEAD}}),
          ushort({{HEAD_DIMENSION}} - d_outer));
        ushort D_dst_dimension = min(
          ushort({{BLOCK_DIMENSIONS_HEAD}}),
          ushort({{PADDED_HEAD_DIMENSION}} - d_outer));
        ushort R_dimension = min(
          uint({{BLOCK_DIMENSIONS_PARALLELIZATION}}),
          uint({{PARALLELIZATION_DIMENSION}} - {{PARALLELIZATION_GROUP_OFFSET}}));
        ushort2 tile_src(D_src_dimension, R_dimension);
        ushort2 tile_dst(D_dst_dimension, R_dimension);

        simdgroup_event event;
        event.async_copy<{{LEADING_BLOCK_DIMENSION_OPERAND}}, 32>(
          dst, tile_dst,
          src, {{LEADING_DIMENSION_OPERAND}}, tile_src{{ASYNC_LANE_ID}},
          {{TRANSPOSED_OPERAND}});
        simdgroup_event::wait(1, &event);
      }

)";
      return source.ToString();
    } else {
      CodeWriter source;
      source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
      source.SetValue("OPERAND", operand.name());
      source.SetValue("OPERAND_LOCATION", operandLocationValue(operand));
      source.SetValue("LEADING_BLOCK_DIMENSION_OPERAND", std::to_string(leadingBlockDimension(operand)));
      source.SetValue("LEADING_DIMENSION_OPERAND", leadingDimension(operand));
      source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
      source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
      source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
      source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
      source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
      source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
      if (disableAsyncCopy) {
        source.SetValue("ASYNC_LANE_ID", ", lane_id");
      } else {
        source.SetValue("ASYNC_LANE_ID", "");
      }
      source += R"(

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 {{OPERAND}}_offset(d_outer, {{PARALLELIZATION_GROUP_OFFSET}});
        auto src = (threadgroup {{MEMORY_NAME_OPERAND}}*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<{{MEMORY_NAME_OPERAND}}>
        ::apply_offset(
          {{OPERAND_LOCATION}}, {{LEADING_DIMENSION_OPERAND}},
          {{OPERAND}}_offset, {{TRANSPOSED_OPERAND}});

        ushort D_dimension = min(
          ushort({{BLOCK_DIMENSIONS_HEAD}}),
          ushort({{HEAD_DIMENSION}} - d_outer));
        ushort R_dimension = min(
          uint({{BLOCK_DIMENSIONS_PARALLELIZATION}}),
          uint({{PARALLELIZATION_DIMENSION}} - {{PARALLELIZATION_GROUP_OFFSET}}));
        ushort2 tile(D_dimension, R_dimension);

        simdgroup_event event;
        event.async_copy<{{LEADING_BLOCK_DIMENSION_OPERAND}}, 32>(
          dst, {{LEADING_DIMENSION_OPERAND}}, tile,
          src, tile{{ASYNC_LANE_ID}},
          {{TRANSPOSED_OPERAND}});
        simdgroup_event::wait(1, &event);
      }

)";
      return source.ToString();
    }
  };

  struct LoopIterationDescriptor {
    MTLAddressSpace addressSpace;
  };
  
  auto leadingDimensionOperand =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    if (descriptor.addressSpace == MTLAddressSpace::device) {
      return leadingDimension(operand);
    } else {
      return std::to_string(leadingBlockDimension(operand));
    }
  };
  
  auto declareOperandLocation =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    if (descriptor.addressSpace == MTLAddressSpace::device) {
      CodeWriter source;
      source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
      source.SetValue("OPERAND", operand.name());
      source.SetValue("OPERAND_LOCATION", operandLocationValue(operand));
      source.SetValue("LEADING_DIMENSION_OPERAND", leadingDimension(operand));
      source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
      source.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
      source += R"(

      uint2 {{OPERAND}}_src_offset(
        morton_offset.x + d_outer,
        {{CLAMPED_PARALLELIZATION_THREAD_OFFSET}});
      auto {{OPERAND}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_OPERAND}}>
      ::apply_offset(
        {{OPERAND_LOCATION}}, {{LEADING_DIMENSION_OPERAND}},
        {{OPERAND}}_src_offset, {{TRANSPOSED_OPERAND}});

)";
      return source.ToString();
    } else {
      CodeWriter source;
      source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
      source.SetValue("OPERAND", operand.name());
      source.SetValue("LEADING_BLOCK_DIMENSION_OPERAND", std::to_string(leadingBlockDimension(operand)));
      source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
      source += R"(

      ushort2 {{OPERAND}}_block_offset(
        morton_offset.x, 
        morton_offset.y + sidx * 8);
      auto {{OPERAND}}_src =
      (threadgroup {{MEMORY_NAME_OPERAND}}*)(threadgroup_block);

      {{OPERAND}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_OPERAND}}>
      ::apply_offset(
        {{OPERAND}}_src, {{LEADING_BLOCK_DIMENSION_OPERAND}},
        {{OPERAND}}_block_offset, {{TRANSPOSED_OPERAND}});
      threadgroup_barrier(mem_flags::mem_threadgroup);

)";
      return source.ToString();
    }
  };

  // MARK: - Inner Loop
  
  auto innerLoopHead =
  [=](unsigned short headStart, unsigned short headEnd, LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("HEAD_START", std::to_string(headStart));
    source.SetValue("HEAD_END", std::to_string(headEnd));
    source.SetValue("OPERAND", operand.name());
    source.SetValue("LEADING_DIMENSION_OPERAND", leadingDimensionOperand(descriptor));
    source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
    if (type == CachingOperationType::load) {
      source.SetValue("LOAD_FUNCTION_OPERAND", loadFunction(operand));
      source += R"(

      #pragma clang loop unroll(full)
      for (ushort d = {{HEAD_START}}; d < {{HEAD_END}}; d += 8) {
        ushort2 {{OPERAND}}_origin(d, 0);
        {{OPERAND}}_sram[(d_outer + d) / 8].{{LOAD_FUNCTION_OPERAND}}(
          {{OPERAND}}_src, {{LEADING_DIMENSION_OPERAND}},
          {{OPERAND}}_origin, {{TRANSPOSED_OPERAND}});
      }

)";
    } else {
      source.SetValue("STORE_FUNCTION_OPERAND", storeFunction(operand));
      source += R"(

      #pragma clang loop unroll(full)
      for (ushort d = {{HEAD_START}}; d < {{HEAD_END}}; d += 8) {
        ushort2 {{OPERAND}}_origin(d, 0);
        {{OPERAND}}_sram[(d_outer + d) / 8].{{STORE_FUNCTION_OPERAND}}(
          {{OPERAND}}_src, {{LEADING_DIMENSION_OPERAND}},
          {{OPERAND}}_origin, {{TRANSPOSED_OPERAND}});
      }

)";
    }
    return source.ToString();
  };
  
  // MARK: - Outer Loop

  auto loopIteration =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    auto loadOperand =
    [=]() -> std::string {
      if (type == CachingOperationType::load) {
        return asyncAccessOperand();
      } else {
        return "";
      }
    };

    auto storeOperand =
    [=]() -> std::string {
      if (type == CachingOperationType::load) {
        return "";
      } else {
        return asyncAccessOperand();
      }
    };

    if (descriptor.addressSpace == MTLAddressSpace::device) {
      CodeWriter source;
      source.SetValue("DECLARE_OPERAND_LOCATION", declareOperandLocation(descriptor));
      source.SetValue("TYPE_IS_LOAD", (type == CachingOperationType::load) ? "true" : "false");
      source.SetValue("UNSAFE_PARALLELIZATION_THREAD_OFFSET", unsafeParallelizationThreadOffsetValue());
      source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
      source.SetValue("INNER_LOOP_HEAD", innerLoopHead(0, blockDimensions[2], descriptor));
      source += R"(

      {{DECLARE_OPERAND_LOCATION}}
      if (
        {{TYPE_IS_LOAD}} ||
        ({{UNSAFE_PARALLELIZATION_THREAD_OFFSET}} < {{PARALLELIZATION_DIMENSION}})
      ) {
      {{INNER_LOOP_HEAD}}
      }

)";
      return source.ToString();
    } else {
      CodeWriter source;
      source.SetValue("LOAD_OPERAND", loadOperand());
      source.SetValue("DECLARE_OPERAND_LOCATION", declareOperandLocation(descriptor));
      source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
      source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
      source.SetValue("INNER_LOOP_HEAD_0", innerLoopHead(0, blockDimensions[2], descriptor));
      source.SetValue("INNER_LOOP_HEAD_1", innerLoopHead(0, headDimension % blockDimensions[2], descriptor));
      source.SetValue("STORE_OPERAND", storeOperand());
      source += R"(

      {{LOAD_OPERAND}}
      {{DECLARE_OPERAND_LOCATION}}
      if (d_outer + {{BLOCK_DIMENSIONS_HEAD}} <= {{HEAD_DIMENSION}}) {
        {{INNER_LOOP_HEAD_0}}
      } else {
        {{INNER_LOOP_HEAD_1}}
      }
      {{STORE_OPERAND}}

)";
      return source.ToString();
    }
  };
  
  auto gatedLoopIteration =
  [=]() -> std::string {
    LoopIterationDescriptor descriptorDevice;
    LoopIterationDescriptor descriptorThreadgroup;
    descriptorDevice.addressSpace = MTLAddressSpace::device;
    descriptorThreadgroup.addressSpace = MTLAddressSpace::threadgroup;
    CodeWriter source;
    source.SetValue("NOT_PREFER_ASYNC_CACHE", !preferAsyncCache ? "true" : "false");
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("LOOP_ITERATION_DEVICE", loopIteration(descriptorDevice));
    source.SetValue("LOOP_ITERATION_THREADGROUP", loopIteration(descriptorThreadgroup));

    source += R"(

    if ({{NOT_PREFER_ASYNC_CACHE}} && (
      ({{HEAD_DIMENSION}} % {{BLOCK_DIMENSIONS_HEAD}} == 0) ||
      (d_outer + {{BLOCK_DIMENSIONS_HEAD}} <= {{HEAD_DIMENSION}})
    )) {
      {{LOOP_ITERATION_DEVICE}}
    } else {
      {{LOOP_ITERATION_THREADGROUP}}
    }
)";
    return source.ToString();
  };

  CodeWriter source;
  source.SetValue("ALLOCATE_OPERAND", allocateOperand());
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration());
  source += R"(

  {{ALLOCATE_OPERAND}}

  #pragma clang loop unroll(full)
  for (
    ushort d_outer = 0;
    d_outer < {{HEAD_DIMENSION}};
    d_outer += {{BLOCK_DIMENSIONS_HEAD}}
  ) {
    {{GATED_LOOP_ITERATION}}
  }

)";
  return source.ToString();
}

std::string AttentionKernel::createSetup() const noexcept {
  // Allocate registers for the specified operand.
  auto allocate =
  [=](AttentionOperand operand) -> std::string {
    CodeWriter source;
    source.SetValue("REGISTER_NAME_OPERAND", registerName(operand));
    source.SetValue("OPERAND", operand.name());
    source.SetValue("PADDED_HEAD_DIMENSION_8", std::to_string(paddedHeadDimensionValue() / 8));
    source += R"(

    simdgroup_matrix_storage<{{REGISTER_NAME_OPERAND}}> {{OPERAND}}_sram[{{PADDED_HEAD_DIMENSION_8}}];

)";
    return source.ToString();
  };
  
  // Initialize the output string.
  CodeWriter output;

  switch (type.value) {
  case AttentionKernelType::forward:
    if (cached(AttentionOperand::Q)) {
      output += cache(AttentionOperand::Q, CachingOperationType::load);
    }
    if (cached(AttentionOperand::O)) {
      output += allocate(AttentionOperand::O);
    }
    output += R"(

    float m = -numeric_limits<float>::max();
    float l = numeric_limits<float>::denorm_min();

)";
    break;
  case AttentionKernelType::backwardQuery: {
    if (cached(AttentionOperand::Q)) {
      output += cache(AttentionOperand::Q, CachingOperationType::load);
    }
    if (cached(AttentionOperand::dO)) {
      output += cache(AttentionOperand::dO, CachingOperationType::load);
    }
    if (cached(AttentionOperand::dQ)) {
      output += allocate(AttentionOperand::dQ);
    }
    
    auto memoryPrecisionL = memoryPrecisions[AttentionOperand::L].value();
    if (memoryPrecisionL == GEMMOperandPrecision::BF16) {
      CCV_NNC_MFA_PRECONDITION(false);
    }
    
    // L is always either FP16 or FP32, so we don't need custom type
    // conversion code here.
    output.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
    output.SetValue("COMPUTE_D", computeD());
    output += R"(

    float L_sram = L[{{CLAMPED_PARALLELIZATION_THREAD_OFFSET}}];
    {{COMPUTE_D}}
    
)";
    break;
  }
  case AttentionKernelType::backwardKeyValue:
    if (cached(AttentionOperand::K)) {
      output += cache(AttentionOperand::K, CachingOperationType::load);
    }
    if (cached(AttentionOperand::V)) {
      output += cache(AttentionOperand::V, CachingOperationType::load);
    }
    if (cached(AttentionOperand::dK)) {
      output += allocate(AttentionOperand::dK);
    }
    if (cached(AttentionOperand::dV)) {
      output += allocate(AttentionOperand::dV);
    }
    break;
  }
    
  return output.ToString();
}

std::string AttentionKernel::createCleanup(const AttentionKernelType type) const noexcept {
  // Initialize the output string.
  CodeWriter output;
  
  switch (type.value) {
  case AttentionKernelType::forward:
    if (cached(AttentionOperand::O)) {
      output += cache(AttentionOperand::O, CachingOperationType::store);
    }
    
    // L is always either FP16 or FP32, so we don't need custom type
    // conversion code here.
    output.SetValue("L_LOCATION", operandLocationValue(AttentionOperand::L));
    output.SetValue("UNSAFE_PARALLELIZATION_THREAD_OFFSET", unsafeParallelizationThreadOffsetValue());
    output.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    output.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
    output += R"(

    if ({{UNSAFE_PARALLELIZATION_THREAD_OFFSET}} < {{PARALLELIZATION_DIMENSION}}) {
      // Premultiplied by log_base_2(e).
      float L_sram = m + fast::log2(l);
      ({{L_LOCATION}})[{{CLAMPED_PARALLELIZATION_THREAD_OFFSET}}] = L_sram;
    }

)";
    break;
  case AttentionKernelType::backwardQuery: {
    if (cached(AttentionOperand::dQ)) {
      output += cache(AttentionOperand::dQ, CachingOperationType::store);
    }
    
    // Cast D from FP32 to potentially BF16.
    auto storeD =
    [=]() -> std::string {
      CodeWriter source;
      source.SetValue("D_LOCATION", operandLocationValue(AttentionOperand::D));
      source.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
      switch (memoryPrecisions[AttentionOperand::D].value().value) {
      case GEMMOperandPrecision::FP32:
        source += R"(

        ({{D_LOCATION}})[{{CLAMPED_PARALLELIZATION_THREAD_OFFSET}}] = D_sram;

)";
        break;
      case GEMMOperandPrecision::BF16:
        source += R"(

        bfloat2 registerForm = *(thread bfloat2*)(&D_sram);
        bfloat memoryForm = registerForm[1];
        ({{D_LOCATION}})[{{CLAMPED_PARALLELIZATION_THREAD_OFFSET}}] = memoryForm;

)";
        break;
      default:
        CCV_NNC_MFA_PRECONDITION(false);
        break;
      }
      return source.ToString();
    };
    output.SetValue("UNSAFE_PARALLELIZATION_THREAD_OFFSET", unsafeParallelizationThreadOffsetValue());
    output.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    output.SetValue("STORE_D", storeD());
    output += R"(
    
    if ({{UNSAFE_PARALLELIZATION_THREAD_OFFSET}} < {{PARALLELIZATION_DIMENSION}}) {
      {{STORE_D}}
    }
    
)";
    break;
  }
  case AttentionKernelType::backwardKeyValue:
    if (cached(AttentionOperand::dK)) {
      output += cache(AttentionOperand::dK, CachingOperationType::store);
    }
    if (cached(AttentionOperand::dV)) {
      output += cache(AttentionOperand::dV, CachingOperationType::store);
    }
    break;
  }
  
  return output.ToString();
}

// MARK: - AttentionKernel+OuterProduct

std::string AttentionKernel::outerProduct(const AttentionOuterProductDescriptor& descriptor) const noexcept {
  auto A = descriptor.A;
  auto B = descriptor.B;
  auto C = descriptor.C;

  // MARK: - Initialize

  auto allocateAccumulator =
  [=]() -> std::string {
    CodeWriter source;
    source.SetValue("C", C.name());
    source.SetValue("REGISTER_NAME_C", registerName(C));
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
    source += R"(

    simdgroup_matrix_storage<{{REGISTER_NAME_C}}> {{C}}_sram[{{BLOCK_DIMENSIONS_TRAVERSAL}} / 8];

)";
    return source.ToString();
  };

  auto initializeAccumulator =
  [=]() -> std::string {
    CodeWriter source;
    source.SetValue("C", C.name());
    source.SetValue("REGISTER_NAME_C", registerName(C));
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
      auto {{C}} = {{C}}_sram + c / 8;
      *{{C}} = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(0);
    }

)";
    return source.ToString();
  };

  struct LoopIterationDescriptor {
    // Whether to accumulate in the SIMD matmul.
    std::string accumulateConditional;
    MTLAddressSpace addressSpaceLHS;
    MTLAddressSpace addressSpaceRHS;
    std::string registerOffset;
    unsigned short registerSize;
  };
  
  auto allocateLHS =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    if (cached(A)) {
      return "";
    }
    CodeWriter source;
    source.SetValue("A", A.name());
    source.SetValue("REGISTER_NAME_A", registerName(A));
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source += R"(

    simdgroup_matrix_storage<{{REGISTER_NAME_A}}> {{A}}_sram[{{DESCRIPTOR_REGISTER_SIZE}} / 8];

)";
    return source.ToString();
  };

  // MARK: - Load LHS

  auto declareLHSLocation =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("A", A.name());
    source.SetValue("MEMORY_NAME_A", memoryName(A));
    source.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
    source.SetValue("TRANSPOSED_A", transposed(A) ? "true" : "false");
    switch (descriptor.addressSpaceLHS.value) {
    case MTLAddressSpace::device:
      source.SetValue("LEADING_DIMENSION_A", leadingDimension(A));
      source.SetValue("A_LOCATION", operandLocationValue(A));
      source += R"(

      uint2 {{A}}_src_offset(
        morton_offset.x + d_outer,
        {{CLAMPED_PARALLELIZATION_THREAD_OFFSET}});
      auto {{A}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>
      ::apply_offset(
        {{A_LOCATION}}, {{LEADING_DIMENSION_A}},
        {{A}}_src_offset, {{TRANSPOSED_A}});

)";
      return source.ToString();
    case MTLAddressSpace::threadgroup:
      source.SetValue("LEADING_BLOCK_DIMENSION_A", std::to_string(leadingBlockDimension(A)));
      source += R"(

      ushort2 {{A}}_block_offset(
        morton_offset.x, 
        morton_offset.y + sidx * 8);
      auto {{A}}_src = (threadgroup {{MEMORY_NAME_A}}*)(threadgroup_block);
      {{A}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>
      ::apply_offset(
        {{A}}_src, {{LEADING_BLOCK_DIMENSION_A}},
        {{A}}_block_offset, {{TRANSPOSED_A}});
      threadgroup_barrier(mem_flags::mem_threadgroup);

)";
      return source.ToString();
    }
  };
  
  auto asyncLoadLHS =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("A", A.name());
    source.SetValue("A_LOCATION", operandLocationValue(A));
    source.SetValue("MEMORY_NAME_A", memoryName(A));
    source.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
    source.SetValue("TRANSPOSED_A", transposed(A) ? "true" : "false");
    source.SetValue("LEADING_DIMENSION_A", leadingDimension(A));
    source.SetValue("LEADING_BLOCK_DIMENSION_A", std::to_string(leadingBlockDimension(A)));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
    if (disableAsyncCopy) {
      source.SetValue("ASYNC_LANE_ID", ", lane_id");
    } else {
      source.SetValue("ASYNC_LANE_ID", "");
    }
    source += R"(

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      uint2 {{A}}_offset(d_outer, {{PARALLELIZATION_GROUP_OFFSET}});
      auto src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>
      ::apply_offset(
        {{A_LOCATION}}, {{LEADING_DIMENSION_A}},
        {{A}}_offset, {{TRANSPOSED_A}});
      auto dst = (threadgroup {{MEMORY_NAME_A}}*)(threadgroup_block);

      ushort D_src_dimension = min(
        ushort({{BLOCK_DIMENSIONS_HEAD}}),
        ushort({{HEAD_DIMENSION}} - d_outer));
      ushort D_dst_dimension = {{DESCRIPTOR_REGISTER_SIZE}};
      ushort R_dimension = min(
        uint({{BLOCK_DIMENSIONS_PARALLELIZATION}}),
        uint({{PARALLELIZATION_DIMENSION}} - {{PARALLELIZATION_GROUP_OFFSET}}));
      ushort2 tile_src(D_src_dimension, R_dimension);
      ushort2 tile_dst(D_dst_dimension, R_dimension);

      simdgroup_event event;
      event.async_copy<{{LEADING_BLOCK_DIMENSION_A}}, 32>(
        dst, tile_dst,
        src, {{LEADING_DIMENSION_A}}, tile_src{{ASYNC_LANE_ID}}, {{TRANSPOSED_A}});
      simdgroup_event::wait(1, &event);
    }

)";
    return source.ToString();
  };
  
  auto loadLHS =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    if (cached(A)) {
      return "";
    }
    CodeWriter source;
    source.SetValue("A", A.name());
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("LOAD_FUNCTION_A", loadFunction(A));
    source.SetValue("TRANSPOSED_A", transposed(A) ? "true" : "false");
    source.SetValue("DECLARE_LHS_LOCATION", declareLHSLocation(descriptor));
    switch (descriptor.addressSpaceLHS.value) {
    case MTLAddressSpace::device:
      source.SetValue("LEADING_DIMENSION_A", leadingDimension(A));
      source += R"(

      {{DECLARE_LHS_LOCATION}}

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
        ushort2 {{A}}_origin(d, 0);
        {{A}}_sram[d / 8].{{LOAD_FUNCTION_A}}(
          {{A}}_src, {{LEADING_DIMENSION_A}},
          {{A}}_origin, {{TRANSPOSED_A}});
      }

)";
      return source.ToString();
    case MTLAddressSpace::threadgroup:
      source.SetValue("ASYNC_LOAD_LHS", asyncLoadLHS(descriptor));
      source.SetValue("LEADING_BLOCK_DIMENSION_A", std::to_string(leadingBlockDimension(A)));
      source += R"(

      {{ASYNC_LOAD_LHS}}
      {{DECLARE_LHS_LOCATION}}

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
        ushort2 {{A}}_origin(d, 0);
        {{A}}_sram[d / 8].{{LOAD_FUNCTION_A}}(
          {{A}}_src, {{LEADING_BLOCK_DIMENSION_A}},
          {{A}}_origin, {{TRANSPOSED_A}});
      }

)";
      return source.ToString();
    }
  };

  // MARK: - Load RHS
  
  auto leadingDimensionRHS =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    switch (descriptor.addressSpaceRHS.value) {
    case MTLAddressSpace::device:
      return leadingDimension(B);
    case MTLAddressSpace::threadgroup:
      return std::to_string(leadingBlockDimension(B));
    }
  };

  auto declareRHSLocation =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("B", B.name());
    source.SetValue("MEMORY_NAME_B", memoryName(B));
    switch (descriptor.addressSpaceRHS.value) {
    case MTLAddressSpace::device:
      source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
      source.SetValue("LEADING_DIMENSION_B", leadingDimension(B));
      source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
      source.SetValue("B_LOCATION", operandLocationValue(B));
      source += R"(

      uint2 {{B}}_src_offset(
        morton_offset.y + d_outer,
        morton_offset.x + {{TRAVERSAL_OFFSET}});
      auto {{B}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
      ::apply_offset(
        {{B_LOCATION}}, {{LEADING_DIMENSION_B}},
        {{B}}_src_offset, {{TRANSPOSED_B}});

)";
      break;
    case MTLAddressSpace::threadgroup:
      source.SetValue("LEADING_BLOCK_DIMENSION_B", std::to_string(leadingBlockDimension(B)));
      source.SetValue("NOT_TRANSPOSED_B", !transposed(B) ? "true" : "false");
      source += R"(

      ushort2 {{B}}_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto {{B}}_src = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block);
      {{B}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
      ::apply_offset(
        {{B}}_src, {{LEADING_BLOCK_DIMENSION_B}},
        {{B}}_block_offset, {{NOT_TRANSPOSED_B}});
      threadgroup_barrier(mem_flags::mem_threadgroup);

)";
      break;
    }
    return source.ToString();
  };
  
  auto loadRHS =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    switch (descriptor.addressSpaceRHS.value) {
    case MTLAddressSpace::device:
      return declareRHSLocation(descriptor);
    case MTLAddressSpace::threadgroup:
      source.SetValue("B", B.name());
      source.SetValue("B_LOCATION", operandLocationValue(B));
      source.SetValue("MEMORY_NAME_B", memoryName(B));
      source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
      source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
      source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
      source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
      source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
      source.SetValue("PADDED_TRAVERSAL_EDGE", paddedTraversalEdgeValue());
      source.SetValue("LEADING_DIMENSION_B", leadingDimension(B));
      source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
      source.SetValue("LEADING_BLOCK_DIMENSION_B", std::to_string(leadingBlockDimension(B)));
      source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
      source.SetValue("DECLARE_RHS_LOCATION", declareRHSLocation(descriptor));
      if (disableAsyncCopy) {
        source.SetValue("ASYNC_LANE_ID", ", lane_id");
      } else {
        source.SetValue("ASYNC_LANE_ID", "");
      }
      source += R"(

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 {{B}}_offset(d_outer, {{TRAVERSAL_OFFSET}});
        auto src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
        ::apply_offset(
          {{B_LOCATION}}, {{LEADING_DIMENSION_B}},
          {{B}}_offset, {{TRANSPOSED_B}});
        auto dst = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block);
 
        ushort D_src_dimension = min(
          ushort({{BLOCK_DIMENSIONS_HEAD}}),
          ushort({{HEAD_DIMENSION}} - d_outer));
        ushort D_dst_dimension = {{DESCRIPTOR_REGISTER_SIZE}};
        ushort C_src_dimension = min(
          uint({{BLOCK_DIMENSIONS_TRAVERSAL}}),
          uint({{TRAVERSAL_DIMENSION}} - {{TRAVERSAL_OFFSET}}));
        ushort C_dst_dimension = max(
          ushort({{PADDED_TRAVERSAL_EDGE}}),
          ushort(C_src_dimension));
        ushort2 tile_src(D_src_dimension, C_src_dimension);
        ushort2 tile_dst(D_dst_dimension, C_dst_dimension);

        simdgroup_event event;
        event.async_copy<{{LEADING_BLOCK_DIMENSION_B}}, 32>(
          dst, tile_dst,
          src, {{LEADING_DIMENSION_B}}, tile_src{{ASYNC_LANE_ID}}, {{TRANSPOSED_B}});
        simdgroup_event::wait(1, &event);
      }

      {{DECLARE_RHS_LOCATION}}

)";
      break;
    }
    return source.ToString();
  };


  // MARK: - Inner Loop
  
  auto innerLoopTraversal =
  [=](std::string traversalStart, std::string traversalEnd, LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("TRAVERSAL_START", traversalStart);
    source.SetValue("TRAVERSAL_END", traversalEnd);
    source.SetValue("A", A.name());
    source.SetValue("B", B.name());
    source.SetValue("C", C.name());
    source.SetValue("REGISTER_NAME_B", registerName(B));
    source.SetValue("LOAD_FUNCTION_B", loadFunction(B));
    source.SetValue("LEADING_DIMENSION_RHS", leadingDimensionRHS(descriptor));
    source.SetValue("NOT_TRANSPOSED_B", !transposed(B) ? "true" : "false");
    source.SetValue("DESCRIPTOR_REGISTER_OFFSET", descriptor.registerOffset);
    source.SetValue("DESCRIPTOR_ACCUMULATE_CONDITIONAL", descriptor.accumulateConditional);
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort c = {{TRAVERSAL_START}}; c < {{TRAVERSAL_END}}; c += 8) {
      // Load the RHS from memory.
      ushort2 {{B}}_origin(c, d);
      simdgroup_matrix_storage<{{REGISTER_NAME_B}}> {{B}};
      {{B}}.{{LOAD_FUNCTION_B}}(
        {{B}}_src, {{LEADING_DIMENSION_RHS}},
        {{B}}_origin, {{NOT_TRANSPOSED_B}});

      // Issue one SIMD matmul instruction.
      {{C}}_sram[c / 8].multiply(
        {{A}}_sram[({{DESCRIPTOR_REGISTER_OFFSET}} + d) / 8],
        {{B}}, {{DESCRIPTOR_ACCUMULATE_CONDITIONAL}});
    }

)";
    return source.ToString();
  };
  
  auto innerLoopHead =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    if (descriptor.addressSpaceLHS == MTLAddressSpace::device ||
        descriptor.addressSpaceRHS == MTLAddressSpace::device) {
      source.SetValue("INNER_LOOP_TRAVERSAL", innerLoopTraversal("0", std::to_string(blockDimensions[1]), descriptor));
      source += R"(

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
        {{INNER_LOOP_TRAVERSAL}}
      }

)";
    } else {
      source.SetValue("INNER_LOOP_TRAVERSAL_0", innerLoopTraversal("0", paddedTraversalEdgeValue(), descriptor));
      source.SetValue("INNER_LOOP_TRAVERSAL_1", innerLoopTraversal(paddedTraversalEdgeValue(), std::to_string(blockDimensions[1]), descriptor));
      source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
      source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
      source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
      source += R"(

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
        {{INNER_LOOP_TRAVERSAL_0}}
        if ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIMENSIONS_TRAVERSAL}}
            < {{TRAVERSAL_DIMENSION}}) {
          {{INNER_LOOP_TRAVERSAL_1}}
        }
      }

)";
    }
    return source.ToString();
  };

  // MARK: - Outer Loop

  auto loopIteration =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    return "\n" + allocateLHS(descriptor) + "\n" + loadLHS(descriptor) + "\n" + loadRHS(descriptor) + "\n" + innerLoopHead(descriptor) + "\n";
  };
  
  auto gatedLoopIteration =
  [=](LoopIterationDescriptor descriptor) -> std::string {
    auto descriptorThreadgroup = descriptor;
    descriptorThreadgroup.addressSpaceLHS = MTLAddressSpace::threadgroup;
    descriptorThreadgroup.addressSpaceRHS = MTLAddressSpace::threadgroup;
    if (preferAsyncCache && preferAsyncLoad) {
      return loopIteration(descriptorThreadgroup);
    }

    auto descriptorDevice = descriptor;
    if (preferAsyncCache) {
      descriptorDevice.addressSpaceLHS = MTLAddressSpace::threadgroup;
    } else {
      descriptorDevice.addressSpaceLHS = MTLAddressSpace::device;
    }
    if (preferAsyncLoad) {
      descriptorDevice.addressSpaceRHS = MTLAddressSpace::threadgroup;
    } else {
      descriptorDevice.addressSpaceRHS = MTLAddressSpace::device;
    }
    
    auto blockDim = blockDimensions[1];
    CodeWriter source;
    source.SetValue("BLOCK_DIM", std::to_string(blockDim));
    source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("LOOP_ITERATION_DEVICE", loopIteration(descriptorDevice));
    source.SetValue("LOOP_ITERATION_THREADGROUP", loopIteration(descriptorThreadgroup));

    source += R"(

    if ((
        ({{TRAVERSAL_DIMENSION}} % {{BLOCK_DIM}} == 0) ||
        ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} <= {{TRAVERSAL_DIMENSION}})
      ) && (
        ({{HEAD_DIMENSION}} % 8 == 0) ||
        (d_outer + {{DESCRIPTOR_REGISTER_SIZE}} <= {{HEAD_DIMENSION}})
      )) {
      {{LOOP_ITERATION_DEVICE}}
    } else {
      {{LOOP_ITERATION_THREADGROUP}}
    }

)";
    return source.ToString();
  };

  // MARK: - Top Level Specification
  

  auto loopEnd =
  [=]() -> unsigned short {
    return paddedHeadDimensionValue();
  };

  auto loopEndFloor =
  [=]() -> unsigned short {
    return loopEnd() - loopEnd() % blockDimensions[2];
  };
 
  auto unrollStatement =
  [=]() -> std::string {
    if (cached(A)) {
      return "#pragma clang loop unroll(full)";
    } else {
      return "#pragma clang loop unroll(disable)";
    }
  };
 
  auto initializeStatement =
  [=]() -> std::string {
    if (cached(A)) {
      // Zero-initialize during the multiply-accumulate loop.
      return "";
    } else {
      // Zero-initialize beforehand.
      return initializeAccumulator();
    }
  };
  
  auto accumulateConditional =
  [=]() -> std::string {
    if (cached(A)) {
      return "((d_outer > 0) || (d > 0))";
    } else {
      // The accumulator is already initialized.
      return "true";
    }
  };
  
  auto registerOffset =
  [=]() -> std::string {
    if (cached(A)) {
      return "d_outer";
    } else {
      return "0";
    }
  };
  
  auto firstIterations =
  [=]() -> std::string {
    LoopIterationDescriptor descriptor;
    descriptor.accumulateConditional = accumulateConditional();
    descriptor.registerOffset = registerOffset();
    descriptor.registerSize = blockDimensions[2];

    CodeWriter source;
    source.SetValue("UNROLL_STATEMENT", unrollStatement());
    source.SetValue("LOOP_END_FLOOR", std::to_string(loopEndFloor()));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration(descriptor));
    source += R"(

    {{UNROLL_STATEMENT}}
    for (
      ushort d_outer = 0;
      d_outer < {{LOOP_END_FLOOR}};
      d_outer += {{BLOCK_DIMENSIONS_HEAD}}
    ) {
      {{GATED_LOOP_ITERATION}}
    }

)";
    return source.ToString();
  };
  
  auto lastIteration =
  [=]() -> std::string {
    LoopIterationDescriptor descriptor;
    descriptor.accumulateConditional = accumulateConditional();
    descriptor.registerOffset = registerOffset();
    descriptor.registerSize = paddedHeadEdgeValue();
    
    CodeWriter source;
    source.SetValue("LOOP_END_FLOOR", std::to_string(loopEndFloor()));
    source.SetValue("LOOP_END_FLOOR_LESS_LOOP_END", (loopEndFloor() < loopEnd()) ? "true" : "false");
    source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration(descriptor));
    source += R"(

    if ({{LOOP_END_FLOOR_LESS_LOOP_END}}) {
      ushort d_outer = {{LOOP_END_FLOOR}};
      {{GATED_LOOP_ITERATION}}
    }

)";
    return source.ToString();
  };
  
  // Collect all of the statements into one string.
  return "\n" + allocateAccumulator() + "\n" + initializeStatement() + "\n" + firstIterations() + "\n" + lastIteration() + "\n";
}

// MARK: - AttentionKernel+Softmax

/*
static std::string high_precision_to_string(float value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}
*/

static std::string dotProductScale(bool derivative) {
  if (!derivative) {
    return "dot_product_scale";
  } else {
    return "dot_product_scale_derivative";
  }
}
std::string AttentionKernel::computeD() const noexcept {

  // Parts of the dO * O reduction that fall within block bounds.
  auto bulkContributions =
  [=](unsigned short truncatedHeadDimension) -> std::string {
    // Recycle most of the cached values for dO.
    auto declareDerivativeOLocation =
    [=]() -> std::string {
      if (cached(AttentionOperand::dO)) {
        return "";
      } else {
        CodeWriter source;
        source.SetValue("DO_LOCATION", operandLocationValue(AttentionOperand::dO));
        source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
        source.SetValue("LEADING_DIMENSION_DO", leadingDimension(AttentionOperand::dO));
        source.SetValue("TRANSPOSED_DO", transposed(AttentionOperand::dO) ? "true" : "false");
        source += R"(

        // Where the dO data will be read from.
        auto dO_src = simdgroup_matrix_storage<{{MEMORY_NAME_DO}}>
        ::apply_offset(
          {{DO_LOCATION}}, {{LEADING_DIMENSION_DO}},
          offset_src, {{TRANSPOSED_DO}});

)";
        return source.ToString();
      }
    };

    auto loadDerivativeO =
    [=]() -> std::string {
      if (cached(AttentionOperand::dO)) {
        return R"(

        auto dO = dO_sram[d / 8];

)";
      } else {
        CodeWriter source;
        source.SetValue("REGISTER_NAME_DO", registerName(AttentionOperand::dO));
        source.SetValue("LOAD_FUNCTION_DO", loadFunction(AttentionOperand::dO));
        source.SetValue("LEADING_DIMENSION_DO", leadingDimension(AttentionOperand::dO));
        source.SetValue("TRANSPOSED_DO", transposed(AttentionOperand::dO) ? "true" : "false");
        source += R"(

        simdgroup_matrix_storage<{{REGISTER_NAME_DO}}> dO;
        dO.{{LOAD_FUNCTION_DO}}(
          dO_src, {{LEADING_DIMENSION_DO}},
          ushort2(d, 0), {{TRANSPOSED_DO}});

)";
        return source.ToString();
      }
    };

    CodeWriter source;
    source.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
    source.SetValue("DECLARE_DERIVATIVE_O_LOCATION", declareDerivativeOLocation());
    source.SetValue("LOAD_DERIVATIVE_O", loadDerivativeO());
    source.SetValue("O_LOCATION", operandLocationValue(AttentionOperand::O));
    source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
    source.SetValue("REGISTER_NAME_O", registerName(AttentionOperand::O));
    source.SetValue("LOAD_FUNCTION_O", loadFunction(AttentionOperand::O));
    source.SetValue("LEADING_DIMENSION_O", leadingDimension(AttentionOperand::O));
    source.SetValue("TRANSPOSED_O", transposed(AttentionOperand::O) ? "true" : "false");
    source.SetValue("TRUNCATED_HEAD_DIMENSION", std::to_string(truncatedHeadDimension));
    source += R"(

    // Threads outside of the matrix along the row dimension,
    // have their origin shifted in-bounds.
    uint D_offset = morton_offset.x;
    uint R_offset = {{CLAMPED_PARALLELIZATION_THREAD_OFFSET}};
    uint2 offset_src(D_offset, R_offset);

    {{DECLARE_DERIVATIVE_O_LOCATION}}

    // Where the O data will be read from.
    auto O_src = simdgroup_matrix_storage<{{MEMORY_NAME_O}}>
    ::apply_offset(
      {{O_LOCATION}}, {{LEADING_DIMENSION_O}},
      offset_src, {{TRANSPOSED_O}});

    // Going to use async copy to handle the matrix edge.
    #pragma clang loop unroll(disable)
    for (ushort d = 0; d < {{TRUNCATED_HEAD_DIMENSION}}; d += 8) {
      {{LOAD_DERIVATIVE_O}}

      simdgroup_matrix_storage<{{REGISTER_NAME_O}}> O;
      O.{{LOAD_FUNCTION_O}}(
        O_src, {{LEADING_DIMENSION_O}},
        ushort2(d, 0), {{TRANSPOSED_O}});

      // Perform the pointwise multiplication.
      auto dO_value = *(dO.thread_elements());
      auto O_value = *(O.thread_elements());
      D_accumulator += float2(dO_value) * float2(O_value);
    }
)";
    return source.ToString();
  };
  
  // Parts of the dO * O reduction that fall on an indivisible edge.
  auto edgeContributions =
  [=](unsigned short truncatedHeadDimension) -> std::string {
    if (headDimension % 8 == 0) {
      return "";
    }
    
    // Abbreviated block, only covers the last 8 elements.
    auto leadingBlockDimension =
    [=](AttentionOperand operand) -> unsigned short {
      if (transposed(operand)) {
        return blockSequenceLength(operand);
      } else {
        return 8;
      }
    };
    
    // Distinct from the block bytes that would be used to calculate
    // the threadgroup memory allocation.
    auto blockBytesDerivativeO =
    [=]() -> unsigned short {
      auto memoryPrecision = memoryPrecisions[AttentionOperand::dO].value();
      auto size = (unsigned short)memoryPrecision.size();
      return blockDimensions[0] * 8 * size;
    };

    CodeWriter source;
    source.SetValue("TRUNCATED_HEAD_DIMENSION", std::to_string(truncatedHeadDimension));
    source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
    source.SetValue("DO_LOCATION", operandLocationValue(AttentionOperand::dO));
    source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
    source.SetValue("REGISTER_NAME_DO", registerName(AttentionOperand::dO));
    source.SetValue("LOAD_FUNCTION_DO", loadFunction(AttentionOperand::dO));
    source.SetValue("LEADING_DIMENSION_DO", leadingDimension(AttentionOperand::dO));
    source.SetValue("LEADING_BLOCK_DIMENSION_DO", std::to_string(leadingBlockDimension(AttentionOperand::dO)));
    source.SetValue("TRANSPOSED_DO", transposed(AttentionOperand::dO) ? "true" : "false");
    source.SetValue("O_LOCATION", operandLocationValue(AttentionOperand::O));
    source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
    source.SetValue("REGISTER_NAME_O", registerName(AttentionOperand::O));
    source.SetValue("LOAD_FUNCTION_O", loadFunction(AttentionOperand::O));
    source.SetValue("LEADING_DIMENSION_O", leadingDimension(AttentionOperand::O));
    source.SetValue("LEADING_BLOCK_DIMENSION_O", std::to_string(leadingBlockDimension(AttentionOperand::O)));
    source.SetValue("TRANSPOSED_O", transposed(AttentionOperand::O) ? "true" : "false");
    source.SetValue("BLOCK_BYTES_DERIVATIVE_O", std::to_string(blockBytesDerivativeO()));
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
    if (disableAsyncCopy) {
      source.SetValue("ASYNC_LANE_ID", ", lane_id");
    } else {
      source.SetValue("ASYNC_LANE_ID", "");
    }
    source += R"(

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      uint D_offset = {{TRUNCATED_HEAD_DIMENSION}};
      uint R_offset = {{PARALLELIZATION_GROUP_OFFSET}};
      uint2 offset_src(D_offset, R_offset);

      auto dO_src = simdgroup_matrix_storage<{{MEMORY_NAME_DO}}>
      ::apply_offset(
        {{DO_LOCATION}}, {{LEADING_DIMENSION_DO}}, 
        offset_src, {{TRANSPOSED_DO}});
      auto O_src = simdgroup_matrix_storage<{{MEMORY_NAME_O}}>
      ::apply_offset(
        {{O_LOCATION}}, {{LEADING_DIMENSION_O}}, 
        offset_src, {{TRANSPOSED_O}});

      auto dO_dst = (threadgroup {{MEMORY_NAME_DO}}*)(threadgroup_block);
      auto O_dst = (threadgroup {{MEMORY_NAME_O}}*)(
        threadgroup_block + {{BLOCK_BYTES_DERIVATIVE_O}});

      ushort D_src_dimension = {{HEAD_DIMENSION}} % 8;
      ushort D_dst_dimension = 8;
      ushort R_dimension = min(
        uint({{BLOCK_DIMENSIONS_PARALLELIZATION}}),
        uint({{PARALLELIZATION_DIMENSION}} - {{PARALLELIZATION_GROUP_OFFSET}}));
      ushort2 tile_src(D_src_dimension, R_dimension);
      ushort2 tile_dst(D_dst_dimension, R_dimension);

      // Issue two async copies.
      simdgroup_event events[2];
      events[0].async_copy<{{LEADING_BLOCK_DIMENSION_DO}}, 32>(
        dO_dst, tile_dst,
        dO_src, {{LEADING_DIMENSION_DO}}, tile_src{{ASYNC_LANE_ID}}, {{TRANSPOSED_DO}});
      events[1].async_copy<{{LEADING_BLOCK_DIMENSION_O}}, 32>(
        O_dst, tile_dst,
        O_src, {{LEADING_DIMENSION_O}}, tile_src{{ASYNC_LANE_ID}}, {{TRANSPOSED_O}});
      simdgroup_event::wait(2, events);
    }

    // Where the dO and O data will be read from.
    ushort2 offset_src(morton_offset.x, morton_offset.y + sidx * 8);
    auto dO_block = (threadgroup {{MEMORY_NAME_DO}}*)(threadgroup_block);
    auto O_block = (threadgroup {{MEMORY_NAME_O}}*)(
      threadgroup_block + {{BLOCK_BYTES_DERIVATIVE_O}});

    dO_block = simdgroup_matrix_storage<{{MEMORY_NAME_DO}}>
    ::apply_offset(
      dO_block, {{LEADING_BLOCK_DIMENSION_DO}},
      offset_src, {{TRANSPOSED_DO}});
    O_block = simdgroup_matrix_storage<{{MEMORY_NAME_O}}>
    ::apply_offset(
      O_block, {{LEADING_BLOCK_DIMENSION_O}},
      offset_src, {{TRANSPOSED_O}});
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load the zero-padded edge data.
    ushort2 origin(0, 0);
    simdgroup_matrix_storage<{{REGISTER_NAME_DO}}> dO;
    simdgroup_matrix_storage<{{REGISTER_NAME_O}}> O;
    dO.{{LOAD_FUNCTION_DO}}(
      dO_block, {{LEADING_BLOCK_DIMENSION_DO}},
      origin, {{TRANSPOSED_DO}});
    O.{{LOAD_FUNCTION_O}}(
      O_block, {{LEADING_BLOCK_DIMENSION_O}},
      origin, {{TRANSPOSED_O}});

    // Perform the pointwise multiplication.
    auto dO_value = *(dO.thread_elements());
    auto O_value = *(O.thread_elements());
    D_accumulator += float2(dO_value) * float2(O_value);

    )";
    return source.ToString();
  };
  
  // Outer loop over the head dimension.
  auto loopEndFloor = headDimension - headDimension % 8;
  CodeWriter source;
  source.SetValue("BULK_CONTRIBUTIONS", bulkContributions(loopEndFloor));
  source.SetValue("EDGE_CONTRIBUTIONS", edgeContributions(loopEndFloor));
  source.SetValue("DOT_PRODUCT_SCALE", dotProductScale(true));
  source += R"(

  float2 D_accumulator(0);
  {
    {{BULK_CONTRIBUTIONS}}
  }
  {
    {{EDGE_CONTRIBUTIONS}}
  }

  float D_sram = D_accumulator[0] + D_accumulator[1];
  D_sram += simd_shuffle_xor(D_sram, 1);
  D_sram += simd_shuffle_xor(D_sram, 8);
  D_sram *= {{DOT_PRODUCT_SCALE}};

)";
  return source.ToString();
}

std::string AttentionKernel::maskAttentionMatrixEdge() const noexcept {
  auto blockDim = blockDimensions[1];
  std::string remainder = "(" + traversalDimensionValue() + " % "+ std::to_string(blockDim) + ")";
  std::string remainderFloor = "(" + remainder + " - (" + remainder + " % 8))";
  float logBase2E = 1.442695041;

  CodeWriter source;
  source.SetValue("REMAINDER", remainder);
  source.SetValue("REMAINDER_FLOOR", remainderFloor);
  source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
  source.SetValue("BLOCK_DIM", std::to_string(blockDim));
  source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
  source.SetValue("LOG_BASE_2E", std::to_string(logBase2E));
  source.SetValue("REGISTER_NAME_S", registerName(AttentionOperand::S));
  source += R"(

  if (({{REMAINDER}} != 0) &&
      ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} > {{TRAVERSAL_DIMENSION}})) {
    // Prevent the value from becoming -INF during the FMA before the
    // exponentiation. If the multiplication during FMA returns -INF,
    // subtracting a positive 'm' value will turn it into zero. We don't want
    // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
    const {{REGISTER_NAME_S}} mask_value =
    (0.875 / {{LOG_BASE_2E}}) * -numeric_limits<{{REGISTER_NAME_S}}>::max();
    
    #pragma clang loop unroll(full)
    for (ushort index = 0; index < 2; ++index) {
      if (morton_offset.x + index >= {{REMAINDER}} - {{REMAINDER_FLOOR}}) {
        auto S_elements = S_sram[{{REMAINDER_FLOOR}} / 8].thread_elements();
        (*S_elements)[index] = mask_value;
      }
    }
    #pragma clang loop unroll(full)
    for (ushort c = {{REMAINDER_FLOOR}} + 8; c < {{BLOCK_DIM}}; c += 8) {
      auto S_elements = S_sram[c / 8].thread_elements();
      *S_elements = mask_value;
    }
  }

)";
  return source.ToString();
}

std::string AttentionKernel::onlineReduceMaximum() const noexcept {
  CodeWriter source;
  source.SetValue("REGISTER_NAME_S", registerName(AttentionOperand::S));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("DOT_PRODUCT_SCALE", dotProductScale(false));
  source += R"(

  // update 'm'
  vec<{{REGISTER_NAME_S}}, 2> m_new_accumulator;
  #pragma clang loop unroll(full)
  for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
    auto S_elements = S_sram[c / 8].thread_elements();
    if (c == 0) {
      m_new_accumulator = *S_elements;
    } else {
      m_new_accumulator = max(m_new_accumulator, *S_elements);
    }
  }
  float m_new = max(m_new_accumulator[0], m_new_accumulator[1]);
  m_new = max(m_new, simd_shuffle_xor(m_new, 1));
  m_new = max(m_new, simd_shuffle_xor(m_new, 8));
  m_new *= {{DOT_PRODUCT_SCALE}};
  
)";
  return source.ToString();
}

std::string AttentionKernel::onlineCorrectO() const noexcept {
  return R"(

  // update 'O'
  float correction = 1;
  if (m_new > m) {
    correction = fast::exp2(m - m_new);
    m = m_new;
  }

)";
}

std::string AttentionKernel::onlineReduceSum() const noexcept {
  CodeWriter source;
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source += R"(
  
  // update 'l'
  float2 l_new_accumulator;
  #pragma clang loop unroll(full)
  for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
    auto P_elements = P_sram[c / 8].thread_elements();
    if (c == 0) {
      l_new_accumulator = float2(*P_elements);
    } else {
      l_new_accumulator += float2(*P_elements);
    }
  }
  float l_new = l_new_accumulator[0] + l_new_accumulator[1];
  l_new += simd_shuffle_xor(l_new, 1);
  l_new += simd_shuffle_xor(l_new, 8);
  l = l * correction + l_new;
  
)";
  return source.ToString();
}

std::string AttentionKernel::softmax(bool derivative) const noexcept {
  AttentionOperand operand = derivative ? AttentionOperand::D : AttentionOperand::L;

  auto allocateOutput =
  [=]() -> std::string {
    auto blockDim = blockDimensions[1];
    CodeWriter source;
    source.SetValue("BLOCK_DIM", std::to_string(blockDim));
    if (!derivative) {
      source.SetValue("REGISTER_NAME_P", registerName(AttentionOperand::P));
      source += R"(

      simdgroup_matrix_storage<{{REGISTER_NAME_P}}> P_sram[{{BLOCK_DIM}} / 8];

)";
    } else {
      source.SetValue("REGISTER_NAME_DS", registerName(AttentionOperand::dS));
      source += R"(

      simdgroup_matrix_storage<{{REGISTER_NAME_DS}}> dS_sram[{{BLOCK_DIM}} / 8];

)";
    }
    return source.ToString();
  };
  
  auto loadOperand =
  [=]() -> std::string {
    CodeWriter source;
    source.SetValue("OPERAND", operand.name());
    source.SetValue("OPERAND_LOCATION", operandLocationValue(operand));
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
    source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
    source.SetValue("PADDED_TRAVERSAL_EDGE", paddedTraversalEdgeValue());
    if (disableAsyncCopy) {
      source.SetValue("ASYNC_LANE_ID", ", lane_id");
    } else {
      source.SetValue("ASYNC_LANE_ID", "");
    }
    source += R"(

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      auto {{OPERAND}}_src = {{OPERAND_LOCATION}} + {{TRAVERSAL_OFFSET}};
      auto {{OPERAND}}_dst =
      (threadgroup {{MEMORY_NAME_OPERAND}}*)(threadgroup_block);

      ushort R_src_dimension = min(
        uint({{BLOCK_DIMENSIONS_TRAVERSAL}}),
        uint({{TRAVERSAL_DIMENSION}} - {{TRAVERSAL_OFFSET}}));
      ushort R_dst_dimension = max(
        ushort({{PADDED_TRAVERSAL_EDGE}}),
        ushort(R_src_dimension));

      // Issue an async copy.
      simdgroup_event event;
      event.async_copy<32>(
        {{OPERAND}}_dst, R_dst_dimension,
        {{OPERAND}}_src, R_src_dimension{{ASYNC_LANE_ID}});
      simdgroup_event::wait(1, &event);
    }

)";
    return source.ToString();
  };
  
  // Declares the source of L or D.
  //
  // Also guards against unsafe accesses to the declared pointer (barrier).

  auto declareOperandLocation =
  [=](MTLAddressSpace addressSpace) -> std::string {
    CodeWriter source;
    source.SetValue("OPERAND", operand.name());
    source.SetValue("OPERAND_LOCATION", operandLocationValue(operand));
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
    if (addressSpace == MTLAddressSpace::device) {
      source += R"(

      auto {{OPERAND}}_src = {{OPERAND_LOCATION}};
      {{OPERAND}}_src += {{TRAVERSAL_OFFSET}} + morton_offset.x;

)";
    } else {
      source += R"(

      auto {{OPERAND}}_src =
      (threadgroup {{MEMORY_NAME_OPERAND}}*)(threadgroup_block);
      {{OPERAND}}_src += morton_offset.x;
      threadgroup_barrier(mem_flags::mem_threadgroup);

)";
    }
    return source.ToString();
  };
  
  auto overwriteAttentionMatrixElements =
  [=]() -> std::string {
    CodeWriter source;
    source.SetValue("SCALE", dotProductScale(derivative));
 
    if (!derivative) {
      source.SetValue("REGISTER_NAME_P", registerName(AttentionOperand::P));
      source += R"(

      auto S = *(S_sram[c / 8].thread_elements());
      auto P = vec<{{REGISTER_NAME_P}}, 2>(
        fast::exp2(float2(S) * {{SCALE}} - float2(L_elements)));
      *(P_sram[c / 8].thread_elements()) = P;

)";
    } else {
      source.SetValue("REGISTER_NAME_DS", registerName(AttentionOperand::dS));
      source += R"(

      auto P = *(P_sram[c / 8].thread_elements());
      auto dP = *(dP_sram[c / 8].thread_elements());
      auto dS = vec<{{REGISTER_NAME_DS}}, 2>(
        float2(P) * (float2(dP) * {{SCALE}} - float2(D_elements)));
      *(dS_sram[c / 8].thread_elements()) = dS;

)";
    }
    return source.ToString();
  };
  
  auto innerLoop =
  [=]() -> std::string {
    CodeWriter source;
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
    source.SetValue("OVERWRITE_ATTENTION_MATRIX_ELEMENTS", overwriteAttentionMatrixElements());
    source.SetValue("OPERAND", operand.name());
    source.SetValue("LOAD_FUNCTION_OPERAND", loadFunction(operand));
    source.SetValue("REGISTER_NAME_OPERAND", registerName(operand));
    switch (type.value) {
    case AttentionKernelType::forward:
      source += R"(

      #pragma clang loop unroll(full)
      for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
        auto L_elements = m;
        {{OVERWRITE_ATTENTION_MATRIX_ELEMENTS}}
      }

)";
      break;
    case AttentionKernelType::backwardQuery:
      source += R"(

      #pragma clang loop unroll(full)
      for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
        auto {{OPERAND}}_elements = {{OPERAND}}_sram;
        {{OVERWRITE_ATTENTION_MATRIX_ELEMENTS}}
      }

)";
      break;
    case AttentionKernelType::backwardKeyValue:
      source += R"(

      #pragma clang loop unroll(full)
      for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
        ushort2 {{OPERAND}}_origin(c, 0);
        simdgroup_matrix_storage<{{REGISTER_NAME_OPERAND}}> {{OPERAND}};
        {{OPERAND}}.{{LOAD_FUNCTION_OPERAND}}(
          {{OPERAND}}_src, 1,
          {{OPERAND}}_origin, false);
        auto {{OPERAND}}_elements = *({{OPERAND}}.thread_elements());

        {{OVERWRITE_ATTENTION_MATRIX_ELEMENTS}}
      }

)";
      break;
    }
    return source.ToString();
  };
  
  CodeWriter source;
  source.SetValue("ALLOCATE_OUTPUT", allocateOutput());
  source.SetValue("INNER_LOOP", innerLoop());
  switch (type.value) {
  case AttentionKernelType::forward:
  case AttentionKernelType::backwardQuery:
    source += R"(

    {{ALLOCATE_OUTPUT}}
    {
      {{INNER_LOOP}}
    }

)";
    break;
  case AttentionKernelType::backwardKeyValue:
    auto blockDim = blockDimensions[1];
    source.SetValue("BLOCK_DIM", std::to_string(blockDim));
    source.SetValue("NOT_PREFER_ASYNC_LOAD", !preferAsyncLoad ? "true" : "false");
    source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("LOAD_OPERAND", loadOperand());
    source.SetValue("DECLARE_OPERAND_LOCATION_DEVICE", declareOperandLocation(MTLAddressSpace::device));
    source.SetValue("DECLARE_OPERAND_LOCATION_THREADGROUP", declareOperandLocation(MTLAddressSpace::threadgroup));

    source += R"(

    {{ALLOCATE_OUTPUT}}
    if ({{NOT_PREFER_ASYNC_LOAD}} && (
        ({{TRAVERSAL_DIMENSION}} % {{BLOCK_DIM}} == 0) ||
        ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} <= {{TRAVERSAL_DIMENSION}})
      )) {
      {{DECLARE_OPERAND_LOCATION_DEVICE}}
      {{INNER_LOOP}}
    } else {
      {{LOAD_OPERAND}}
      {{DECLARE_OPERAND_LOCATION_THREADGROUP}}
      {{INNER_LOOP}}
    }

)";
    break;
  }
  return source.ToString();
}
