#include "AdamDescriptor.hpp"
#include "AdamKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool AdamDescriptor::operator==(const AdamDescriptor& rhs) const {
  return
  adamw == rhs.adamw &&
  amsgrad == rhs.amsgrad &&
  memoryPrecision == rhs.memoryPrecision &&
  rate == rhs.rate &&
  scale == rhs.scale &&
  beta1 == rhs.beta1 &&
  beta2 == rhs.beta2 &&
  decay == rhs.decay &&
  epsilon == rhs.epsilon &&
  length == rhs.length;
}

std::size_t std::hash<AdamDescriptor>::operator()(const AdamDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.adamw | ((unsigned int)hash.amsgrad << 8) }));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.rate), *reinterpret_cast<const uint32_t*>(&hash.scale) }));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.beta1), *reinterpret_cast<const uint32_t*>(&hash.beta2) }));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.decay), *reinterpret_cast<const uint32_t*>(&hash.epsilon) }));
  combine_64(seed, hash.length);
  return seed;
}

std::pair<AdamKernelDescriptor, PipelineValue<AdamKernel>*> AdamDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<AdamKernelDescriptor, std::unique_ptr<AdamKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](AdamKernelDescriptor descriptor) -> AdamKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      AdamKernel* kernel = new AdamKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<AdamKernel>(kernel);
      return kernel;
    }
  };

  AdamKernelDescriptor kernelDesc;
  kernelDesc.adamw = adamw;
  kernelDesc.amsgrad = amsgrad;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&length, MTL::DataTypeUInt, NS::UInteger(0));
    const float decayValue = adamw ? rate * decay : decay;
    constants->setConstantValue(&decayValue, MTL::DataTypeFloat, NS::UInteger(1));
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(2));
    constants->setConstantValue(&beta1, MTL::DataTypeFloat, NS::UInteger(3));
    constants->setConstantValue(&beta2, MTL::DataTypeFloat, NS::UInteger(4));
    constants->setConstantValue(&epsilon, MTL::DataTypeFloat, NS::UInteger(5));

    NS::String* swiftName = NS::String::string("adam", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  AdamKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<AdamKernel>* output = new PipelineValue<AdamKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
