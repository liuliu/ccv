#include "Int8SwiGLUDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool Int8SwiGLUDescriptor::operator==(const Int8SwiGLUDescriptor& rhs) const
{
  return N == rhs.N && K == rhs.K && clamp == rhs.clamp &&
    memoryPrecision == rhs.memoryPrecision;
}

std::size_t std::hash<Int8SwiGLUDescriptor>::operator()(
  const Int8SwiGLUDescriptor& value) const noexcept
{
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_32(seed, value.N);
  seed = combine_32(seed, value.K);
  seed = combine_32(seed, reinterpret_cast<const uint32_t&>(value.clamp));
  seed = combine_32(seed, (uint32_t)value.memoryPrecision.value);
  return seed;
}

std::pair<
  Int8SwiGLUKernelDescriptor,
  PipelineValue<Int8SwiGLUKernel>*> Int8SwiGLUDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<
      Int8SwiGLUKernelDescriptor,
      std::unique_ptr<Int8SwiGLUKernel>>* const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  Int8SwiGLUKernelDescriptor kernelDesc;
  kernelDesc.clamp = clamp > 0;
  kernelDesc.memoryPrecision = memoryPrecision;
  Int8SwiGLUKernel* kernel;
  auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end()) {
    kernel = iterator->second.get();
  } else {
    kernel = new Int8SwiGLUKernel(kernelDesc, device);
    (*libraryCache)[kernelDesc] = std::unique_ptr<Int8SwiGLUKernel>(kernel);
  }

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  if (clamp > 0)
    constants->setConstantValue(&clamp, MTL::DataTypeFloat, NS::UInteger(2));
  NS::String* functionName = NS::String::string(
    "int8_swiglu", NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto function = NS::TransferPtr(
    kernel->library->newFunction(functionName, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(
    device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto* output = new PipelineValue<Int8SwiGLUKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
