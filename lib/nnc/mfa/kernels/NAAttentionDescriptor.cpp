#include "NAAttentionDescriptor.hpp"
#include "NAAttentionKernelDescriptor.hpp"
#include "NAAttentionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool NAAttentionDescriptor::operator==(const NAAttentionDescriptor& rhs) const {
  auto lhsMatrixDimensions = matrixDimensions;
  auto rhsMatrixDimensions = rhs.matrixDimensions;
  if (loadC && rhs.loadC) {
    lhsMatrixDimensions[1] = 0;
    rhsMatrixDimensions[1] = 0;
  }
  bool loadCSourceMatch = true;
  if (loadC && rhs.loadC) {
    const auto lhsBlockDimensions = blockDimensions();
    const auto rhsBlockDimensions = rhs.blockDimensions();
    const auto lhsExecutionSIMDGroups = executionSIMDGroups();
    const auto rhsExecutionSIMDGroups = rhs.executionSIMDGroups();
    loadCSourceMatch =
      simd_all(lhsBlockDimensions == rhsBlockDimensions) &&
      lhsExecutionSIMDGroups == rhsExecutionSIMDGroups &&
      checkCEdge1(lhsBlockDimensions) == rhs.checkCEdge1(rhsBlockDimensions) &&
      splitKV(lhsBlockDimensions, lhsExecutionSIMDGroups) == rhs.splitKV(rhsBlockDimensions, rhsExecutionSIMDGroups);
  }
  return
  batchDimension == rhs.batchDimension &&
  Hq == rhs.Hq &&
  Hk == rhs.Hk &&
  scale == rhs.scale &&
  isCausal == rhs.isCausal &&
  masked == rhs.masked &&
  isVarlen == rhs.isVarlen &&
  attentionSinks == rhs.attentionSinks &&
  slidingWindow == rhs.slidingWindow &&
  maskBatchStride == rhs.maskBatchStride &&
  loadC == rhs.loadC &&
  loadCSourceMatch &&
  type == rhs.type &&
  (lowPrecisionInputs == rhs.lowPrecisionInputs) &&
  (isBF16 == rhs.isBF16) &&
  (lowPrecisionIntermediates == rhs.lowPrecisionIntermediates) &&
  batchStrides == rhs.batchStrides &&
  simd_all(lhsMatrixDimensions == rhsMatrixDimensions);
}

std::size_t std::hash<NAAttentionDescriptor>::operator()(const NAAttentionDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.batchDimension);
  combine_32(seed, hash.Hq);
  combine_32(seed, hash.Hk);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.loadC ? 0 : hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  if (hash.loadC) {
    const auto blockDimensions = hash.blockDimensions();
    const uint16_t executionSIMDGroups = hash.executionSIMDGroups();
    combine_64(seed, pack_64(simd_make_ushort4(blockDimensions, 0)));
    combine_32(seed, pack_32(simd::ushort2 {
        executionSIMDGroups,
        hash.splitKV(blockDimensions, executionSIMDGroups) }));
    combine_32(seed, hash.checkCEdge1(blockDimensions) ? 1 : 0);
  }
  combine_32(seed, pack_32(simd::uchar4 { hash.lowPrecisionInputs, hash.isBF16, hash.lowPrecisionIntermediates, hash.isCausal }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.masked ? 1 : 0),
      (uint16_t)(hash.isVarlen ? 1 : 0) }));
  combine_32(seed, hash.attentionSinks ? 1 : 0);
  combine_32(seed, hash.slidingWindow);
  combine_32(seed, hash.maskBatchStride);
  combine_32(seed, pack_32(simd::ushort2 { hash.type.value, 0 } ));
  combine_32(seed, hash.loadC ? 1 : 0);
  return seed;
}

simd::ushort3 NAAttentionDescriptor::blockDimensions() const noexcept {
  unsigned short headDimension = matrixDimensions[2];
  unsigned short revisedHead = (headDimension + 15) / 16 * 16;
  if (type.value != AttentionKernelType::forward && lowPrecisionInputs && headDimension == 128) {
    revisedHead = 64;
  } else if (headDimension <= 128) {
    revisedHead = std::min(headDimension, revisedHead);
  } else {
    revisedHead = revisedHead / std::max(revisedHead / 128, 2); // At least it is 2, could be more.
  }
  if (type.value != AttentionKernelType::forward) {
    return simd::ushort3 { 16, 64, revisedHead };
  }
  if (isCausal) {
    return simd::ushort3 { 16, 32, revisedHead };
  }
  // Prefer ones without partial matrix multiplication (due to tiling).
  if (matrixDimensions[1] % 64 == 0) {
    return simd::ushort3 { 16, 64, revisedHead };
  } else if (matrixDimensions[1] % 48 == 0) {
    return simd::ushort3 { 16, 48, revisedHead };
  }
  // Prefer no trailing involved, so the compute is more evenly distributed.
  if (matrixDimensions[1] % 128 > 64 && matrixDimensions[1] % 96 < 48) {
    return simd::ushort3 { 16, 64, revisedHead };
  } else if (matrixDimensions[1] % 128 < 64 && matrixDimensions[1] % 96 > 48) {
    return simd::ushort3 { 16, 48, revisedHead };
  }
  // If we have to use matrix multiplication, calculate how much wasted compute we are going to be with.
  const unsigned short remainder64 = matrixDimensions[1] % 64;
  const unsigned short remainder48 = matrixDimensions[1] % 48;
  if (remainder64 * 48 < remainder48 * 64) {
    return simd::ushort3 { 16, 48, revisedHead };
  } else {
    return simd::ushort3 { 16, 64, revisedHead };
  }
}

uint16_t NAAttentionDescriptor::executionSIMDGroups() const noexcept {
  const unsigned short headDimension = matrixDimensions[2];
  if (type.value == AttentionKernelType::backwardQuery &&
      lowPrecisionInputs && headDimension >= 128) {
    return 6;
  }
  if (type.value == AttentionKernelType::backwardKeyValue &&
      lowPrecisionInputs && headDimension >= 128) {
    return 6;
  }
  if (type.value == AttentionKernelType::forward && isCausal) {
    return 8;
  }
  return (type.value == AttentionKernelType::forward) ? (lowPrecisionInputs ? 16 : 8) : 8;
}

bool NAAttentionDescriptor::checkCEdge1(simd::ushort3 blockDimensions) const noexcept {
  return (matrixDimensions[1] % (blockDimensions[1] * 2)) > blockDimensions[1];
}

NAAttentionKernelDescriptor NAAttentionDescriptor::kernelDescriptor(MTL::Device *const device, const DeviceProperties &dprops) const noexcept {
  auto createBypassThreadgroupMemory =
  [=]() -> bool {
    return false;
  };
  auto blockDimensions = this->blockDimensions();
  const uint16_t executionSIMDGroups = this->executionSIMDGroups();
  const bool checkCEdge1 = this->checkCEdge1(blockDimensions);
  return NAAttentionKernelDescriptor(blockDimensions, matrixDimensions[2], Hq, Hk, executionSIMDGroups, checkCEdge1, createMemoryPrecisions(), type, scale, createBypassThreadgroupMemory(), isCausal, masked, isVarlen, splitKV(blockDimensions, executionSIMDGroups), loadC, attentionSinks, slidingWindow);
}

uint16_t NAAttentionDescriptor::splitKV(simd::ushort3 blockDimensions, uint16_t executionSIMDGroups) const noexcept {
  if (type.value != AttentionKernelType::forward ||
      matrixDimensions[0] == 0 ||
      matrixDimensions[0] > blockDimensions[0] * 4 ||
      masked ||
      isVarlen ||
      slidingWindow > 0) {
    return 1;
  }
  const uint32_t minSequenceLength = matrixDimensions[0] == 1 ? 2048 : 4096;
  if (matrixDimensions[1] < minSequenceLength) {
    return 1;
  }
  const uint32_t cBlocks = (matrixDimensions[1] + blockDimensions[1] - 1) / blockDimensions[1];
  const uint32_t rowGroups = (matrixDimensions[0] + blockDimensions[0] - 1) / blockDimensions[0];
  const uint32_t activeTiles = batchDimension * Hq * rowGroups;
  if (activeTiles > 128) {
    return 1;
  }
  return (uint16_t)std::min<uint32_t>(executionSIMDGroups, cBlocks);
}

std::pair<NAAttentionKernelDescriptor, PipelineValue<NAAttentionKernel> *> NAAttentionDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAAttentionKernelDescriptor, std::unique_ptr<NAAttentionKernel>> *const libraryCache) const noexcept {
  auto createPipeline =
  [=](MTL::Library* library, const NAAttentionKernelDescriptor& kernelDesc, const char* name) -> MTL::ComputePipelineState* {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t rowDimension = matrixDimensions[0];
    uint32_t columnDimension = matrixDimensions[1];
    constants->setConstantValue(&rowDimension, MTL::DataTypeUInt, NS::Integer(0));
    if (!loadC) {
      constants->setConstantValue(&columnDimension, MTL::DataTypeUInt, 1);
    }
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
    for (const auto& operand : operands) {
      uint32_t batchStride = batchStrides[operand].value_or(0);
      constants->setConstantValue(&batchStride, MTL::DataTypeUInt, 2 + operand.bufferIndex());
    }
    if (type.value == AttentionKernelType::forward && masked) {
      const uint32_t qTiles = (matrixDimensions[0] + kernelDesc.blockDimensions[0] - 1) / kernelDesc.blockDimensions[0];
      const uint32_t kTiles = (matrixDimensions[1] + kernelDesc.blockDimensions[1] - 1) / kernelDesc.blockDimensions[1];
      const uint32_t maskBatchStride = masked ? this->maskBatchStride : 0;
      const uint32_t blockMaskBatchStride = masked && maskBatchStride > 0 ? qTiles * kTiles : 0;
      constants->setConstantValue(&maskBatchStride, MTL::DataTypeUInt, NS::UInteger(15));
      constants->setConstantValue(&blockMaskBatchStride, MTL::DataTypeUInt, NS::UInteger(16));
    }
    if (type.value == AttentionKernelType::forward && kernelDesc.splitKV > 1) {
      const uint32_t splitKV = kernelDesc.splitKV;
      const uint32_t batchDimension = this->batchDimension;
      constants->setConstantValue(&splitKV, MTL::DataTypeUInt, NS::UInteger(19));
      constants->setConstantValue(&batchDimension, MTL::DataTypeUInt, NS::UInteger(20));
    }

    NS::String* swiftName = NS::String::string(name, NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto pipelineDesc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDesc->setComputeFunction(NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error)).get());
    CCV_NNC_MFA_CHECK_ERROR(error);
    
    auto pipeline = device->newComputePipelineState(pipelineDesc.get(), MTL::PipelineOptionNone, NULL, &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto createKernel =
  [=](NAAttentionKernelDescriptor descriptor) -> NAAttentionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      NAAttentionKernel* kernel = new NAAttentionKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<NAAttentionKernel>(kernel);
      return kernel;
    }
  };

  auto kernelDesc = kernelDescriptor(device, dprops);
  NAAttentionKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), kernelDesc, "attention"));
  NS::SharedPtr<MTL::ComputePipelineState> second;
  if (type.value == AttentionKernelType::forward && kernelDesc.splitKV > 1) {
    second = NS::TransferPtr(createPipeline(kernel->library.get(), kernelDesc, "attention_splitkv_combine"));
  } else if (type.value == AttentionKernelType::forward && masked) {
    second = NS::TransferPtr(createPipeline(kernel->library.get(), kernelDesc, "generate_attention_block_mask"));
  }
  if (type.value == AttentionKernelType::backwardQuery) {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    uint32_t rowDimension = matrixDimensions[0];
    uint32_t columnDimension = matrixDimensions[1];
    const uint32_t oBatchStride = batchStrides[AttentionOperand::O].value_or(0);
    const uint32_t dOBatchStride = batchStrides[AttentionOperand::dO].value_or(0);
    constants->setConstantValue(&rowDimension, MTL::DataTypeUInt, NS::Integer(0));
    constants->setConstantValue(&columnDimension, MTL::DataTypeUInt, 1);
    constants->setConstantValue(&oBatchStride, MTL::DataTypeUInt, 2 + AttentionOperand(AttentionOperand::O).bufferIndex());
    constants->setConstantValue(&dOBatchStride, MTL::DataTypeUInt, 2 + AttentionOperand(AttentionOperand::dO).bufferIndex());
    NS::String* swiftName = NS::String::string("compute_d", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto pipelineDesc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDesc->setComputeFunction(NS::TransferPtr
    (kernel->library->newFunction(swiftName, constants.get(), &error)).get());
    CCV_NNC_MFA_CHECK_ERROR(error);
    second = NS::TransferPtr(device->newComputePipelineState(pipelineDesc.get(), MTL::PipelineOptionNone, NULL, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }

  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<NAAttentionKernel>* output = new PipelineValue<NAAttentionKernel> { kernel, pipeline };
  output->second = second;
  return std::make_pair(kernelDesc, output);
}

// MARK: - AttentionDescriptor+Precisions

AttentionOperands<GEMMOperandPrecision> NAAttentionDescriptor::createMemoryPrecisions() const noexcept {
  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;
  
  if (lowPrecisionInputs) {
    if (isBF16) {
      memoryPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::BF16;
      memoryPrecisions[AttentionOperand::K] = GEMMOperandPrecision::BF16;
      memoryPrecisions[AttentionOperand::V] = GEMMOperandPrecision::BF16;
      memoryPrecisions[AttentionOperand::O] = GEMMOperandPrecision::BF16;
      memoryPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::BF16;
    } else {
      memoryPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP16;
      memoryPrecisions[AttentionOperand::K] = GEMMOperandPrecision::FP16;
      memoryPrecisions[AttentionOperand::V] = GEMMOperandPrecision::FP16;
      memoryPrecisions[AttentionOperand::O] = GEMMOperandPrecision::FP16;
      memoryPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::FP16;
    }
  } else {
    memoryPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::K] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::V] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::O] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::FP32;
  }

  if (lowPrecisionIntermediates) {
    memoryPrecisions[AttentionOperand::L] = isBF16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
    memoryPrecisions[AttentionOperand::D] = GEMMOperandPrecision::BF16;
  } else {
    memoryPrecisions[AttentionOperand::L] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::D] = GEMMOperandPrecision::FP32;
  }

  if (lowPrecisionInputs) {
    const auto lowPrecision = isBF16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
    memoryPrecisions[AttentionOperand::dV] = lowPrecision;
    memoryPrecisions[AttentionOperand::dK] = lowPrecision;
    memoryPrecisions[AttentionOperand::dQ] = lowPrecision;
  } else {
    memoryPrecisions[AttentionOperand::dV] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::dK] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::dQ] = GEMMOperandPrecision::FP32;
  }
  
  return memoryPrecisions;
}
