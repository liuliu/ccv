#include "GeluDescriptor.hpp"
#include "GeluKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool GeluDescriptor::operator==(const GeluDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  tanh == rhs.tanh &&
  gradient == rhs.gradient &&
  value == rhs.value &&
  length == rhs.length;
}

std::size_t std::hash<GeluDescriptor>::operator()(const GeluDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.length, (unsigned int)hash.tanh }));
  return seed;
}

std::pair<GeluKernelDescriptor, PipelineValue<GeluKernel> *> GeluDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GeluKernelDescriptor, std::unique_ptr<GeluKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](GeluKernelDescriptor descriptor) -> GeluKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      GeluKernel* kernel = new GeluKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<GeluKernel>(kernel);
      return kernel;
    }
  };

  GeluKernelDescriptor kernelDesc;
  kernelDesc.gradient = gradient;
  kernelDesc.tanh = tanh;
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
    } else if (tanh && value == 1) {
      count = length / 4;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    } else {
      count = length;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    }

    NS::String* swiftName = NS::String::string("gelu", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  GeluKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
    
  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<GeluKernel>* output = new PipelineValue<GeluKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
