#include "GatedDeltaDescriptor.hpp"
#include "GatedDeltaKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool GatedDeltaDescriptor::operator==(const GatedDeltaDescriptor& rhs) const {
  return
  batchSize == rhs.batchSize &&
  sequenceLength == rhs.sequenceLength &&
  keyHeadCount == rhs.keyHeadCount &&
  valueHeadCount == rhs.valueHeadCount &&
  keyDim == rhs.keyDim &&
  valueDim == rhs.valueDim &&
  inputMemoryPrecision == rhs.inputMemoryPrecision &&
  betaMemoryPrecision == rhs.betaMemoryPrecision &&
  logDecay == rhs.logDecay;
}

std::size_t std::hash<GatedDeltaDescriptor>::operator()(const GatedDeltaDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { hash.batchSize, hash.sequenceLength }));
  combine_64(seed, pack_64(simd::uint2 { hash.keyHeadCount, hash.valueHeadCount }));
  combine_64(seed, pack_64(simd::uint2 { hash.keyDim, hash.valueDim }));
  combine_64(seed, hash.inputMemoryPrecision.value);
  combine_64(seed, hash.betaMemoryPrecision.value);
  combine_64(seed, hash.logDecay ? 1 : 0);
  return seed;
}

std::pair<GatedDeltaKernelDescriptor, PipelineValue<GatedDeltaKernel> *> GatedDeltaDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GatedDeltaKernelDescriptor, std::unique_ptr<GatedDeltaKernel>>* const libraryCache) const noexcept {
  auto createKernel =
  [=](GatedDeltaKernelDescriptor descriptor) -> GatedDeltaKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      GatedDeltaKernel* kernel = new GatedDeltaKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<GatedDeltaKernel>(kernel);
      return kernel;
    }
  };

  GatedDeltaKernelDescriptor kernelDesc;
  kernelDesc.stateElementsPerLane = (uint8_t)((keyDim + 31) / 32);
  kernelDesc.inputMemoryPrecision = inputMemoryPrecision;
  kernelDesc.betaMemoryPrecision = betaMemoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&batchSize, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&sequenceLength, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&keyHeadCount, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&valueHeadCount, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&keyDim, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&valueDim, MTL::DataTypeUInt, NS::UInteger(5));
    constants->setConstantValue(&logDecay, MTL::DataTypeBool, NS::UInteger(6));
    const bool keyDimMultipleOf32 = (keyDim % 32) == 0;
    const bool valueDimMultipleOf4 = (valueDim % 4) == 0;
    constants->setConstantValue(&keyDimMultipleOf32, MTL::DataTypeBool, NS::UInteger(7));
    constants->setConstantValue(&valueDimMultipleOf4, MTL::DataTypeBool, NS::UInteger(8));

    NS::String* swiftName = NS::String::string("gated_delta", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  GatedDeltaKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<GatedDeltaKernel>* output = new PipelineValue<GatedDeltaKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
