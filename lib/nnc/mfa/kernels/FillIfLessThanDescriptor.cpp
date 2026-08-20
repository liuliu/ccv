#include "FillIfLessThanDescriptor.hpp"
#include "FillIfLessThanKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool FillIfLessThanDescriptor::operator==(const FillIfLessThanDescriptor& rhs) const {
  return
    memoryPrecision == rhs.memoryPrecision &&
    value == rhs.value &&
    loadM == rhs.loadM &&
    (loadM || length == rhs.length);
}

std::size_t std::hash<FillIfLessThanDescriptor>::operator()(const FillIfLessThanDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, hash.loadM ? 0 : hash.length);
  combine_32(seed, hash.loadM ? 1 : 0);
  return seed;
}

std::pair<FillIfLessThanKernelDescriptor, PipelineValue<FillIfLessThanKernel>*> FillIfLessThanDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<FillIfLessThanKernelDescriptor, std::unique_ptr<FillIfLessThanKernel>>* const libraryCache) const noexcept {
  auto createKernel =
  [=](FillIfLessThanKernelDescriptor descriptor) -> FillIfLessThanKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    }
    FillIfLessThanKernel* kernel = new FillIfLessThanKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<FillIfLessThanKernel>(kernel);
    return kernel;
  };

  FillIfLessThanKernelDescriptor kernelDesc;
  kernelDesc.value = value;
  kernelDesc.loadM = loadM;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    if (!loadM && value != 0) {
      const uint32_t count = value == 1 ? length / 4 : length;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    }

    NS::String* swiftName = NS::String::string("fill_if_less_than", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  FillIfLessThanKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<FillIfLessThanKernel>* output = new PipelineValue<FillIfLessThanKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
