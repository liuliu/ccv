#include "SegmentedInt8SwiGLUDescriptor.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool SegmentedInt8SwiGLUDescriptor::operator==(const SegmentedInt8SwiGLUDescriptor& rhs) const
{
  return simd_all(matrixDimensions == rhs.matrixDimensions) &&
    expertCount == rhs.expertCount && routeCount == rhs.routeCount &&
    format == rhs.format && broadcastInput == rhs.broadcastInput &&
    clamp == rhs.clamp && memoryPrecision == rhs.memoryPrecision;
}

std::size_t std::hash<SegmentedInt8SwiGLUDescriptor>::operator()(
  const SegmentedInt8SwiGLUDescriptor& value) const noexcept
{
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(value.matrixDimensions));
  seed = combine_32(seed, value.expertCount);
  seed = combine_32(seed, value.routeCount);
  seed = combine_32(seed, value.format);
  seed = combine_32(seed, value.broadcastInput);
  seed = combine_32(seed, reinterpret_cast<const uint32_t&>(value.clamp));
  seed = combine_32(seed, (uint32_t)value.memoryPrecision.value);
  return seed;
}

uint32_t SegmentedInt8SwiGLUDescriptor::groupSize() const noexcept
{
  CCV_NNC_MFA_PRECONDITION(format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
  return 32;
}

uint32_t SegmentedInt8SwiGLUDescriptor::groupsPerRow() const noexcept
{
  return (matrixDimensions[1] + groupSize() - 1) / groupSize();
}

uint64_t SegmentedInt8SwiGLUDescriptor::inputScaleOffset() const noexcept
{
  const uint64_t payloadBytes =
    (uint64_t)expertCount * matrixDimensions[0] * groupsPerRow() * 8;
  return (payloadBytes + 127) & ~UINT64_C(127);
}

uint64_t SegmentedInt8SwiGLUDescriptor::weightExpertStride() const noexcept
{
  return (uint64_t)matrixDimensions[0] * groupsPerRow() * 8;
}

std::pair<
  SegmentedInt8SwiGLUKernelDescriptor,
  PipelineValue<SegmentedInt8SwiGLUKernel>*> SegmentedInt8SwiGLUDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<
      SegmentedInt8SwiGLUKernelDescriptor,
      std::unique_ptr<SegmentedInt8SwiGLUKernel>>* const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  SegmentedInt8SwiGLUKernelDescriptor kernelDesc;
  kernelDesc.format = format;
  kernelDesc.memoryPrecision = memoryPrecision;
  SegmentedInt8SwiGLUKernel* kernel;
  auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end()) {
    kernel = iterator->second.get();
  } else {
    kernel = new SegmentedInt8SwiGLUKernel(kernelDesc, device);
    (*libraryCache)[kernelDesc] =
      std::unique_ptr<SegmentedInt8SwiGLUKernel>(kernel);
  }

  const uint32_t K = matrixDimensions[1];
  const uint32_t N = matrixDimensions[0];
  const uint32_t groupSizeValue = groupSize();
  const uint32_t groupsPerRowValue = groupsPerRow();
  const uint64_t weightStride = weightExpertStride();
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&routeCount, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&groupSizeValue, MTL::DataTypeUInt, NS::UInteger(3));
  constants->setConstantValue(&groupsPerRowValue, MTL::DataTypeUInt, NS::UInteger(4));
  constants->setConstantValue(&expertCount, MTL::DataTypeUInt, NS::UInteger(5));
  constants->setConstantValue(&broadcastInput, MTL::DataTypeUInt, NS::UInteger(6));
  constants->setConstantValue(&weightStride, MTL::DataTypeULong, NS::UInteger(7));
  constants->setConstantValue(&clamp, MTL::DataTypeFloat, NS::UInteger(8));

  NS::String* functionName = NS::String::string(
    "segmented_int8_swiglu", NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto function = NS::TransferPtr(
    kernel->library->newFunction(functionName, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(
    device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto* output = new PipelineValue<SegmentedInt8SwiGLUKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
