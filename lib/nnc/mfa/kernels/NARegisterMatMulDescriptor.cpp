#include "NARegisterMatMulDescriptor.hpp"
#include "NARegisterMatMulKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

std::pair<NARegisterMatMulKernelDescriptor, PipelineValue<NARegisterMatMulKernel>*>
NARegisterMatMulDescriptor::findKernel(
    MTL::Device* device, const DeviceProperties& dprops,
    NS::Array* binaryArchivesToRead, MTL::BinaryArchive* binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<NARegisterMatMulKernelDescriptor, std::unique_ptr<NARegisterMatMulKernel>>* libraryCache) const noexcept {
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;
  const NARegisterMatMulKernelDescriptor kernelDesc { quantized, wideM, splitK, castOutputToFloat };
  auto iterator = libraryCache->find(kernelDesc);
  if (iterator == libraryCache->end()) {
    iterator = libraryCache->emplace(kernelDesc,
        std::make_unique<NARegisterMatMulKernel>(kernelDesc, device)).first;
  }
  auto* kernel = iterator->second.get();
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const bool disabled = false;
  constants->setConstantValue(&disabled, MTL::DataTypeBool, NS::UInteger(10));
  constants->setConstantValue(&useBias, MTL::DataTypeBool, NS::UInteger(100));
  constants->setConstantValue(&disabled, MTL::DataTypeBool, NS::UInteger(110));
  constants->setConstantValue(&alignedM, MTL::DataTypeBool, NS::UInteger(200));
  constants->setConstantValue(&alignedN, MTL::DataTypeBool, NS::UInteger(201));
  constants->setConstantValue(&alignedK, MTL::DataTypeBool, NS::UInteger(202));
  NS::Error* error = nullptr;
  auto function = NS::TransferPtr(kernel->library->newFunction(
      NS::String::string("matmul_register", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto* output = new PipelineValue<NARegisterMatMulKernel> { kernel, pipeline };
  if (splitK) {
    auto reduction = NS::TransferPtr(kernel->library->newFunction(
        NS::String::string("matmul_register_reduce", NS::UTF8StringEncoding), constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    output->second = NS::TransferPtr(device->newComputePipelineState(reduction.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
  return std::make_pair(kernelDesc, output);
}
