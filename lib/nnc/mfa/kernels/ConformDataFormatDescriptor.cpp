#include "ConformDataFormatDescriptor.hpp"
#include "ConformDataFormatKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool ConformDataFormatDescriptor::operator==(const ConformDataFormatDescriptor& rhs) const
{
  return
  memoryPrecision == rhs.memoryPrecision &&
  loadM == rhs.loadM &&
  (loadM || rowCount == rhs.rowCount) &&
  headDim == rhs.headDim &&
  preservedTail == rhs.preservedTail;
}

std::size_t std::hash<ConformDataFormatDescriptor>::operator()(const ConformDataFormatDescriptor& hash) const noexcept
{
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { hash.loadM ? 0 : hash.rowCount, hash.headDim }));
  combine_32(seed, hash.preservedTail);
  combine_32(seed, hash.loadM ? 1 : 0);
  combine_32(seed, hash.memoryPrecision.value);
  return seed;
}

std::pair<ConformDataFormatKernelDescriptor, PipelineValue<ConformDataFormatKernel>*> ConformDataFormatDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<ConformDataFormatKernelDescriptor, std::unique_ptr<ConformDataFormatKernel>>* const libraryCache) const noexcept
{
  ConformDataFormatKernelDescriptor kernelDescriptor;
  kernelDescriptor.loadM = loadM;
  kernelDescriptor.memoryPrecision = memoryPrecision;
  ConformDataFormatKernel* kernel;
  auto iterator = libraryCache->find(kernelDescriptor);
  if (iterator != libraryCache->end()) {
    kernel = iterator->second.get();
  } else {
    kernel = new ConformDataFormatKernel(kernelDescriptor, device);
    (*libraryCache)[kernelDescriptor] = std::unique_ptr<ConformDataFormatKernel>(kernel);
  }

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  if (!loadM)
    constants->setConstantValue(&rowCount, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&headDim, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&preservedTail, MTL::DataTypeUInt, NS::UInteger(2));
  NS::String* functionName = NS::String::string("conform_data_format", NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  PipelineValue<ConformDataFormatKernel>* output = new PipelineValue<ConformDataFormatKernel> { kernel, pipeline };
  return std::make_pair(kernelDescriptor, output);
}
