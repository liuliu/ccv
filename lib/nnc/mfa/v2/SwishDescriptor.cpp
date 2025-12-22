#include "SwishDescriptor.hpp"
#include "SwishKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool SwishDescriptor::operator==(const SwishDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  gradient == rhs.gradient &&
  value == rhs.value &&
  length == rhs.length;
}

std::size_t std::hash<SwishDescriptor>::operator()(const SwishDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.length, 0 }));
  return seed;
}

std::pair<SwishKernelDescriptor, PipelineValue<SwishKernel> *> SwishDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SwishKernelDescriptor, std::unique_ptr<SwishKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](SwishKernelDescriptor descriptor) -> SwishKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      SwishKernel* kernel = new SwishKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<SwishKernel>(kernel);
      return kernel;
    }
  };

  SwishKernelDescriptor kernelDesc;
  kernelDesc.gradient = gradient;
  kernelDesc.value = value;
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

    NS::String* swiftName = NS::String::string("swish", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  SwishKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
    
  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<SwishKernel>* output = new PipelineValue<SwishKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
