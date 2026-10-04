#include "NARegisterAttentionDescriptor.hpp"
#include "NARegisterAttentionKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

std::pair<NARegisterAttentionKernelDescriptor, PipelineValue<NARegisterAttentionKernel>*>
NARegisterAttentionDescriptor::findKernel(
    MTL::Device* device, const DeviceProperties& dprops,
    NS::Array* binaryArchivesToRead, MTL::BinaryArchive* binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<NARegisterAttentionKernelDescriptor, std::unique_ptr<NARegisterAttentionKernel>>* libraryCache) const noexcept {
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;
  const NARegisterAttentionKernelDescriptor kernelDesc { D };
  auto iterator = libraryCache->find(kernelDesc);
  if (iterator == libraryCache->end()) {
    iterator = libraryCache->emplace(kernelDesc,
        std::make_unique<NARegisterAttentionKernel>(kernelDesc, device)).first;
  }
  auto* kernel = iterator->second.get();
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&alignedQ, MTL::DataTypeBool, NS::UInteger(200));
  constants->setConstantValue(&alignedK, MTL::DataTypeBool, NS::UInteger(201));
  const bool disabled = false;
  for (const auto index : {300, 302})
    constants->setConstantValue(&disabled, MTL::DataTypeBool, NS::UInteger(index));
  constants->setConstantValue(&causal, MTL::DataTypeBool, NS::UInteger(301));
  constants->setConstantValue(&saveL, MTL::DataTypeBool, NS::UInteger(303));
  NS::Error* error = nullptr;
  auto function = NS::TransferPtr(kernel->library->newFunction(
      NS::String::string("attention_register", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto* output = new PipelineValue<NARegisterAttentionKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
