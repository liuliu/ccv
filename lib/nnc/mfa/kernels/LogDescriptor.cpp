#include "LogDescriptor.hpp"
#include "LogKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool LogDescriptor::operator==(const LogDescriptor& rhs) const {
  return memoryPrecision == rhs.memoryPrecision && value == rhs.value && loadM == rhs.loadM && (loadM || length == rhs.length);
}

std::size_t std::hash<LogDescriptor>::operator()(const LogDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, hash.loadM ? 0 : hash.length);
  combine_32(seed, hash.loadM ? 1 : 0);
  return seed;
}

std::pair<LogKernelDescriptor, PipelineValue<LogKernel>*> LogDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<LogKernelDescriptor, std::unique_ptr<LogKernel>>* const libraryCache) const noexcept {
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;
  const LogKernelDescriptor kernelDesc = {
    .value = value,
    .loadM = loadM,
    .memoryPrecision = memoryPrecision,
  };
  LogKernel* kernel;
  const auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end())
    kernel = iterator->second.get();
  else {
    auto newKernel = std::make_unique<LogKernel>(kernelDesc, device);
    kernel = newKernel.get();
    (*libraryCache)[kernelDesc] = std::move(newKernel);
  }

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  if (!loadM && value != 0) {
    const uint32_t count = value == 1 ? length / 4 : length;
    constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
  }
  NS::Error* error = nil;
  auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string("log_forward", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  PipelineValue<LogKernel>* output = new PipelineValue<LogKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
