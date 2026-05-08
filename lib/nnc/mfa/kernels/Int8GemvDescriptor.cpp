#include "Int8GemvDescriptor.hpp"
#include "Int8GemvKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool Int8GemvDescriptor::operator==(const Int8GemvDescriptor& rhs) const {
  return
  fusedBias == rhs.fusedBias &&
  memoryPrecision == rhs.memoryPrecision &&
  nrows == rhs.nrows &&
  ncols == rhs.ncols;
}

std::size_t std::hash<Int8GemvDescriptor>::operator()(const Int8GemvDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.fusedBias }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.nrows, (unsigned int)hash.ncols }));
  return seed;
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
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    const uint32_t scaleOffset = (uint32_t)(((uint64_t)nrows * ncols + 127) & ~UINT64_C(127));
    constants->setConstantValue(&ncols, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&nrows, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&scaleOffset, MTL::DataTypeUInt, NS::UInteger(2));

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
