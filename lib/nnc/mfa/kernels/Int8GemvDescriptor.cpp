#include "Int8GemvDescriptor.hpp"
#include "Int8GemvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool Int8GemvDescriptor::operator==(const Int8GemvDescriptor& rhs) const {
  return
  fusedBias == rhs.fusedBias &&
  mrows == rhs.mrows &&
  format == rhs.format &&
  memoryPrecision == rhs.memoryPrecision &&
  nrows == rhs.nrows &&
  ncols == rhs.ncols;
}

std::size_t std::hash<Int8GemvDescriptor>::operator()(const Int8GemvDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.fusedBias | ((unsigned int)hash.mrows << 8) }));
  seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.format, (unsigned int)hash.nrows }));
  seed = combine_64(seed, hash.ncols);
  return seed;
}

uint32_t Int8GemvDescriptor::groupSize() const noexcept {
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

uint32_t Int8GemvDescriptor::groupsPerRow() const noexcept {
  const uint32_t size = groupSize();
  return size > 0 ? (ncols + size - 1) / size : 0;
}

uint32_t Int8GemvDescriptor::groupBits() const noexcept {
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

uint32_t Int8GemvDescriptor::inputScaleOffset() const noexcept {
  const uint64_t payloadBits = (uint64_t)nrows * groupsPerRow() * groupBits();
  const uint32_t payloadBytes = (uint32_t)((payloadBits + 7) / 8);
  return (payloadBytes + 127) & ~127u;
}

std::pair<Int8GemvKernelDescriptor, PipelineValue<Int8GemvKernel>*> Int8GemvDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Int8GemvKernelDescriptor, std::unique_ptr<Int8GemvKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](Int8GemvKernelDescriptor descriptor) -> Int8GemvKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      Int8GemvKernel* kernel = new Int8GemvKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<Int8GemvKernel>(kernel);
      return kernel;
    }
  };

  Int8GemvKernelDescriptor kernelDesc;
  kernelDesc.fusedBias = fusedBias;
  kernelDesc.mrows = mrows;
  kernelDesc.format = format;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    const uint32_t scaleOffset = format == 0 ? (uint32_t)(((uint64_t)nrows * ncols + 127) & ~UINT64_C(127)) : inputScaleOffset();
    const uint32_t groupSize = this->groupSize();
    const uint32_t groupsPerRow = this->groupsPerRow();
    constants->setConstantValue(&ncols, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&nrows, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&scaleOffset, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&groupSize, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&groupsPerRow, MTL::DataTypeUInt, NS::UInteger(4));

    NS::String* swiftName = NS::String::string("int8_gemv", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  Int8GemvKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<Int8GemvKernel>* output = new PipelineValue<Int8GemvKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
