#include "SegmentedInt8GemvDescriptor.hpp"
#include "SegmentedInt8GemvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool SegmentedInt8GemvDescriptor::operator==(const SegmentedInt8GemvDescriptor& rhs) const
{
  return
    simd_all(matrixDimensions == rhs.matrixDimensions) &&
    expertCount == rhs.expertCount &&
    binCount == rhs.binCount &&
    format == rhs.format &&
    memoryPrecision == rhs.memoryPrecision &&
    useBias == rhs.useBias;
}

std::size_t std::hash<SegmentedInt8GemvDescriptor>::operator()(
  const SegmentedInt8GemvDescriptor& hash) const noexcept
{
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(hash.matrixDimensions));
  seed = combine_32(seed, hash.expertCount);
  seed = combine_32(seed, hash.binCount);
  seed = combine_32(seed, hash.format);
  seed = combine_32(seed, pack_32(simd::uchar4 {
    (uint8_t)hash.memoryPrecision.value,
    (uint8_t)hash.useBias,
    0,
    0,
  }));
  return seed;
}

uint32_t SegmentedInt8GemvDescriptor::groupSize() const noexcept
{
  switch (format) {
    case 0:
      return 0;
    case CCV_NNC_QX_8I_ROWWISE_Q5_K:
    case CCV_NNC_QX_8I_ROWWISE_Q4_K:
    case CCV_NNC_QX_8I_ROWWISE_Q3_K:
    case CCV_NNC_QX_8I_ROWWISE_Q2_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
      return 16;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
      return 32;
    case CCV_NNC_QX_8I_ROWWISE_Q6_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
      return 8;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      return 0;
  }
}

uint32_t SegmentedInt8GemvDescriptor::groupsPerRow() const noexcept
{
  const uint32_t size = groupSize();
  return size > 0 ? (matrixDimensions[1] + size - 1) / size : 0;
}

uint32_t SegmentedInt8GemvDescriptor::groupBits() const noexcept
{
  switch (format) {
    case 0:
      return 0;
    case CCV_NNC_QX_8I_ROWWISE_Q5_K:
      return 88;
    case CCV_NNC_QX_8I_ROWWISE_Q4_K:
      return 72;
    case CCV_NNC_QX_8I_ROWWISE_Q3_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
      return 56;
    case CCV_NNC_QX_8I_ROWWISE_Q2_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
      return 42;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
      return 21;
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
      return 28;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
      return 64;
    case CCV_NNC_QX_8I_ROWWISE_Q6_K:
      return 52;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      return 0;
  }
}

uint64_t SegmentedInt8GemvDescriptor::inputScaleOffset() const noexcept
{
  const uint32_t N = matrixDimensions[0];
  const uint32_t K = matrixDimensions[1];
  if (format == 0)
    return ((uint64_t)expertCount * N * K + 127) & ~UINT64_C(127);
  const uint64_t payloadBits =
    (uint64_t)expertCount * N * groupsPerRow() * groupBits();
  const uint64_t payloadBytes = (payloadBits + 7) / 8;
  return (payloadBytes + 127) & ~UINT64_C(127);
}

uint64_t SegmentedInt8GemvDescriptor::weightExpertStride() const noexcept
{
  const uint32_t N = matrixDimensions[0];
  const uint32_t K = matrixDimensions[1];
  if (format == 0)
    return (uint64_t)N * K;
  const uint64_t expertBits = (uint64_t)N * groupsPerRow() * groupBits();
  CCV_NNC_MFA_PRECONDITION((expertBits % 8) == 0);
  return expertBits / 8;
}

std::pair<
  SegmentedInt8GemvKernelDescriptor,
  PipelineValue<SegmentedInt8GemvKernel>*> SegmentedInt8GemvDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<
      SegmentedInt8GemvKernelDescriptor,
      std::unique_ptr<SegmentedInt8GemvKernel>>* const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  const uint32_t N = matrixDimensions[0];
  const uint32_t K = matrixDimensions[1];
  CCV_NNC_MFA_PRECONDITION(N > 0 && K > 0);
  CCV_NNC_MFA_PRECONDITION(expertCount > 0 && binCount > 0);

  SegmentedInt8GemvKernelDescriptor kernelDesc;
  kernelDesc.fusedBias = useBias;
  kernelDesc.mrows = 1;
  kernelDesc.format = format;
  kernelDesc.memoryPrecision = memoryPrecision;
  SegmentedInt8GemvKernel* kernel;
  auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end()) {
    kernel = iterator->second.get();
  } else {
    kernel = new SegmentedInt8GemvKernel(kernelDesc, device);
    (*libraryCache)[kernelDesc] =
      std::unique_ptr<SegmentedInt8GemvKernel>(kernel);
  }

  const uint32_t groupSize = this->groupSize();
  const uint32_t groupsPerRow = this->groupsPerRow();
  const uint64_t weightStride = weightExpertStride();
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  if (format != 0) {
    constants->setConstantValue(
      &groupSize, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(
      &groupsPerRow, MTL::DataTypeUInt, NS::UInteger(4));
  }
  constants->setConstantValue(
    &expertCount, MTL::DataTypeUInt, NS::UInteger(5));
  constants->setConstantValue(
    &binCount, MTL::DataTypeUInt, NS::UInteger(6));
  constants->setConstantValue(
    &weightStride, MTL::DataTypeULong, NS::UInteger(7));

  NS::String* functionName = NS::String::string(
    "segmented_int8_gemv", NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto function = NS::TransferPtr(
    kernel->library->newFunction(functionName, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(
    device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  PipelineValue<SegmentedInt8GemvKernel>* output =
    new PipelineValue<SegmentedInt8GemvKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
