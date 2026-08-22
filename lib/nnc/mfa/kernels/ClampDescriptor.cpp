#include "ClampDescriptor.hpp"
#include "ClampKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool ClampDescriptor::operator==(const ClampDescriptor& rhs) const {
  return memoryPrecision == rhs.memoryPrecision && value == rhs.value && bounds == rhs.bounds && loadM == rhs.loadM && (loadM || length == rhs.length);
}

std::size_t std::hash<ClampDescriptor>::operator()(const ClampDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, pack_64(simd::uint2 { hash.loadM ? 0 : hash.length, (unsigned int)hash.bounds }));
  combine_32(seed, hash.loadM ? 1 : 0);
  return seed;
}

std::pair<ClampKernelDescriptor, PipelineValue<ClampKernel>*> ClampDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ClampKernelDescriptor, std::unique_ptr<ClampKernel>>* const libraryCache) const noexcept {
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;
  const ClampKernelDescriptor kernelDesc = {
    .value = value,
    .bounds = bounds,
    .loadM = loadM,
    .memoryPrecision = memoryPrecision,
  };
  ClampKernel* kernel;
  const auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end())
    kernel = iterator->second.get();
  else {
    auto newKernel = std::make_unique<ClampKernel>(kernelDesc, device);
    kernel = newKernel.get();
    (*libraryCache)[kernelDesc] = std::move(newKernel);
  }
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  if (!loadM && value != 0) {
    const uint32_t count = value == 1 ? length / 4 : length;
    constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
  }
  NS::Error* error = nil;
  auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string("clamp_forward", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  PipelineValue<ClampKernel>* output = new PipelineValue<ClampKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
