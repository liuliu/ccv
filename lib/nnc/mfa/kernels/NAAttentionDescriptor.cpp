#include "NAAttentionDescriptor.hpp"
#include "NAAttentionKernelDescriptor.hpp"
#include "NAAttentionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool NAAttentionDescriptor::operator==(const NAAttentionDescriptor& rhs) const {
  return
  batchDimension == rhs.batchDimension &&
  Hq == rhs.Hq &&
  Hk == rhs.Hk &&
  scale == rhs.scale &&
  type == rhs.type &&
  (lowPrecisionInputs == rhs.lowPrecisionInputs) &&
  (isBF16 == rhs.isBF16) &&
  (lowPrecisionIntermediates == rhs.lowPrecisionIntermediates) &&
  batchStrides == rhs.batchStrides &&
  simd_all(matrixDimensions == rhs.matrixDimensions);
}

std::size_t std::hash<NAAttentionDescriptor>::operator()(const NAAttentionDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.batchDimension);
  combine_32(seed, hash.Hq);
  combine_32(seed, hash.Hk);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, pack_32(simd::uchar4 { hash.lowPrecisionInputs, hash.isBF16, hash.lowPrecisionIntermediates, 0 }));
  combine_32(seed, pack_32(simd::ushort2 { hash.type.value, 0 } ));
  return seed;
}

NAAttentionKernelDescriptor NAAttentionDescriptor::kernelDescriptor(MTL::Device *const device, const DeviceProperties &dprops) const noexcept {
  auto createHeadDimension = 
  [=]() -> unsigned short {
    return matrixDimensions[2];
  };
  auto createBlockDimensions =
  [=]() -> simd::ushort3 {
    unsigned short headDimension = createHeadDimension();
    unsigned short revisedHead = (headDimension + 15) / 16 * 16;
    if (headDimension <= 128) {
      revisedHead = std::min(headDimension, revisedHead);
    } else {
      revisedHead = revisedHead / std::max(revisedHead / 128, 2); // At least it is 2, could be more.
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
  };
  auto createExecutionSIMDGroups = 
  [=]() -> uint16_t {
    return lowPrecisionInputs ? 16 : 8;
  };
  auto blockDimensions = createBlockDimensions();
  bool checkCEdge1 = (matrixDimensions[1] % (blockDimensions[1] * 2)) > blockDimensions[1];
  return NAAttentionKernelDescriptor(blockDimensions, createHeadDimension(), Hq, Hk, createExecutionSIMDGroups(), checkCEdge1, createMemoryPrecisions(), type, scale);
}

std::pair<NAAttentionKernelDescriptor, PipelineValue<NAAttentionKernel> *> NAAttentionDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAAttentionKernelDescriptor, std::unique_ptr<NAAttentionKernel>> *const libraryCache) const noexcept {
  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t rowDimension = matrixDimensions[0];
    uint32_t columnDimension = matrixDimensions[1];
    constants->setConstantValue(&rowDimension, MTL::DataTypeUInt, NS::Integer(0));
    constants->setConstantValue(&columnDimension, MTL::DataTypeUInt, 1);
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

    NS::String* swiftName = NS::String::string("attention", NS::UTF8StringEncoding);
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
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<NAAttentionKernel>* output = new PipelineValue<NAAttentionKernel> { kernel, pipeline };
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

  memoryPrecisions[AttentionOperand::dV] = GEMMOperandPrecision::FP32;
  memoryPrecisions[AttentionOperand::dK] = GEMMOperandPrecision::FP32;
  memoryPrecisions[AttentionOperand::dQ] = GEMMOperandPrecision::FP32;
  
  return memoryPrecisions;
}
