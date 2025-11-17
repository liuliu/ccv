#include "CastDescriptor.hpp"
#include "CastKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool CastDescriptor::operator==(const CastDescriptor& rhs) const {
  return
  fromMemoryPrecision == rhs.fromMemoryPrecision &&
  memoryPrecision == rhs.memoryPrecision &&
  value == rhs.value &&
  length == rhs.length;
}

std::size_t std::hash<CastDescriptor>::operator()(const CastDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.fromMemoryPrecision.value }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.length, (unsigned int)hash.value }));
  return seed;
}

std::pair<CastKernelDescriptor, PipelineValue<CastKernel> *> CastDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CastKernelDescriptor, std::unique_ptr<CastKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](CastKernelDescriptor descriptor) -> CastKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      CastKernel* kernel = new CastKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<CastKernel>(kernel);
      return kernel;
    }
  };

  CastKernelDescriptor kernelDesc;
  kernelDesc.value = value;
  kernelDesc.fromMemoryPrecision = fromMemoryPrecision;
  kernelDesc.memoryPrecision = memoryPrecision;

  // WARNING: The owner must explicitly retain the compute pipeline.
  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t count;
    if (value == 0) {
    } else if (value == 1) {
      count = length / 4;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    } else {
      count = length;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    }

    NS::String* swiftName = NS::String::string("cast", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  CastKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
    
  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<CastKernel>* output = new PipelineValue<CastKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
