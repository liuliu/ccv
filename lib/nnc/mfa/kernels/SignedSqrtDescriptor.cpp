#include "SignedSqrtDescriptor.hpp"
#include "SignedSqrtKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool SignedSqrtDescriptor::operator==(const SignedSqrtDescriptor& rhs) const {
  return gradient == rhs.gradient && value == rhs.value && loadM == rhs.loadM &&
    memoryPrecision == rhs.memoryPrecision && minimum_magnitude == rhs.minimum_magnitude &&
    (loadM || length == rhs.length);
}

std::size_t std::hash<SignedSqrtDescriptor>::operator()(const SignedSqrtDescriptor& descriptor) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  const SignedSqrtKernelDescriptor kernelDesc = { descriptor.gradient, descriptor.value, descriptor.loadM, descriptor.memoryPrecision };
  std::size_t seed = std::hash<SignedSqrtKernelDescriptor>()(kernelDesc);
  combine_32(seed, descriptor.loadM ? 0 : descriptor.length);
  combine_64(seed, std::hash<float>()(descriptor.minimum_magnitude));
  return seed;
}

std::pair<SignedSqrtKernelDescriptor, PipelineValue<SignedSqrtKernel>*> SignedSqrtDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SignedSqrtKernelDescriptor, std::unique_ptr<SignedSqrtKernel>>* const libraryCache) const noexcept {
  const SignedSqrtKernelDescriptor kernelDesc = { gradient, value, loadM, memoryPrecision };
  SignedSqrtKernel* kernel;
  const auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end())
    kernel = iterator->second.get();
  else {
    auto newKernel = std::make_unique<SignedSqrtKernel>(kernelDesc, device);
    kernel = newKernel.get();
    (*libraryCache)[kernelDesc] = std::move(newKernel);
  }
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  if (value != 0 && !loadM) {
    const uint32_t count = value == 1 ? length / 4 : length;
    constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
  }
  constants->setConstantValue(&minimum_magnitude, MTL::DataTypeFloat, NS::UInteger(1));
  NS::Error* error = nil;
  auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string("signed_sqrt", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return std::make_pair(kernelDesc, new PipelineValue<SignedSqrtKernel> { kernel, pipeline });
}
