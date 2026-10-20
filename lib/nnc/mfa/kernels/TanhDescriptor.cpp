#include "TanhDescriptor.hpp"
#include "TanhKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool TanhDescriptor::operator==(const TanhDescriptor& rhs) const {
  return gradient == rhs.gradient && value == rhs.value && loadM == rhs.loadM &&
    memoryPrecision == rhs.memoryPrecision && soft_cap == rhs.soft_cap &&
    (loadM || length == rhs.length);
}

std::size_t std::hash<TanhDescriptor>::operator()(const TanhDescriptor& descriptor) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  const TanhKernelDescriptor kernelDesc = { descriptor.gradient, descriptor.value, descriptor.loadM, descriptor.memoryPrecision };
  std::size_t seed = std::hash<TanhKernelDescriptor>()(kernelDesc);
  seed = combine_32(seed, descriptor.loadM ? 0 : descriptor.length);
  seed = combine_64(seed, std::hash<float>()(descriptor.soft_cap));
  return seed;
}

std::pair<TanhKernelDescriptor, PipelineValue<TanhKernel>*> TanhDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<TanhKernelDescriptor, std::unique_ptr<TanhKernel>>* const libraryCache) const noexcept {
  const TanhKernelDescriptor kernelDesc = { gradient, value, loadM, memoryPrecision };
  TanhKernel* kernel;
  const auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end())
    kernel = iterator->second.get();
  else {
    auto newKernel = std::make_unique<TanhKernel>(kernelDesc, device);
    kernel = newKernel.get();
    (*libraryCache)[kernelDesc] = std::move(newKernel);
  }
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  if (value != 0 && !loadM) {
    const uint32_t count = value == 1 ? length / 4 : length;
    constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
  }
  constants->setConstantValue(&soft_cap, MTL::DataTypeFloat, NS::UInteger(1));
  NS::Error* error = nil;
  auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string("tanh_soft_cap", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return std::make_pair(kernelDesc, new PipelineValue<TanhKernel> { kernel, pipeline });
}
