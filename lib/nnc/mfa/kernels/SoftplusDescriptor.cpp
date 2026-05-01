#include "SoftplusDescriptor.hpp"
#include "SoftplusKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool SoftplusDescriptor::operator==(const SoftplusDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  value == rhs.value &&
  length == rhs.length;
}

std::size_t std::hash<SoftplusDescriptor>::operator()(const SoftplusDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, hash.length);
  return seed;
}

std::pair<SoftplusKernelDescriptor, PipelineValue<SoftplusKernel>*> SoftplusDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SoftplusKernelDescriptor, std::unique_ptr<SoftplusKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](SoftplusKernelDescriptor descriptor) -> SoftplusKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      SoftplusKernel* kernel = new SoftplusKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<SoftplusKernel>(kernel);
      return kernel;
    }
  };

  SoftplusKernelDescriptor kernelDesc;
  kernelDesc.value = value;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t count;
    if (value == 0) {
    } else if (value == 1) {
      count = length / 4;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    } else {
      count = length;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    }

    NS::String* swiftName = NS::String::string("softplus_forward", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  SoftplusKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<SoftplusKernel>* output = new PipelineValue<SoftplusKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
