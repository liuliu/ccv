#include "CausalConvolutionDescriptor.hpp"
#include "CausalConvolutionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool CausalConvolutionDescriptor::operator==(const CausalConvolutionDescriptor& rhs) const {
  return silu == rhs.silu &&
    dynamicWeights == rhs.dynamicWeights &&
    memoryPrecision == rhs.memoryPrecision &&
    channels == rhs.channels &&
    kernelSize == rhs.kernelSize &&
    nextRows == rhs.nextRows &&
    dynamicWeightGroups == rhs.dynamicWeightGroups;
}

std::size_t std::hash<CausalConvolutionDescriptor>::operator()(const CausalConvolutionDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (hash.silu ? 1u : 0u) | (hash.dynamicWeights ? 2u : 0u) }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.channels, hash.kernelSize }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.nextRows, hash.dynamicWeightGroups }));
  return seed;
}

std::pair<CausalConvolutionKernelDescriptor, PipelineValue<CausalConvolutionKernel>*> CausalConvolutionDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CausalConvolutionKernelDescriptor, std::unique_ptr<CausalConvolutionKernel>>* const libraryCache) const noexcept {
  auto createKernel =
  [=](CausalConvolutionKernelDescriptor descriptor) -> CausalConvolutionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      CausalConvolutionKernel* kernel = new CausalConvolutionKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<CausalConvolutionKernel>(kernel);
      return kernel;
    }
  };

  const CausalConvolutionKernelDescriptor kernelDesc = { silu, dynamicWeights, memoryPrecision };

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&channels, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&kernelSize, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&nextRows, MTL::DataTypeUInt, NS::UInteger(2));
    if (dynamicWeights)
      constants->setConstantValue(&dynamicWeightGroups, MTL::DataTypeUInt, NS::UInteger(3));

    NS::String* swiftName = NS::String::string("causal_convolution", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  CausalConvolutionKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<CausalConvolutionKernel>* output = new PipelineValue<CausalConvolutionKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
