#include "MulDescriptor.hpp"
#include "MulKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool MulDescriptor::operator==(const MulDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  value == rhs.value &&
  loadM == rhs.loadM &&
  (loadM || value == 0 || length == rhs.length);
}

std::size_t std::hash<MulDescriptor>::operator()(const MulDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_32(seed, hash.loadM || hash.value == 0 ? 0 : hash.length);
  combine_32(seed, hash.loadM ? 1 : 0);
  return seed;
}

std::pair<MulKernelDescriptor, PipelineValue<MulKernel> *> MulDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<MulKernelDescriptor, std::unique_ptr<MulKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](MulKernelDescriptor descriptor) -> MulKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      MulKernel* kernel = new MulKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<MulKernel>(kernel);
      return kernel;
    }
  };

  MulKernelDescriptor kernelDesc;
  kernelDesc.value = value;
  kernelDesc.loadM = loadM;
  kernelDesc.memoryPrecision = memoryPrecision;

  // WARNING: The owner must explicitly retain the compute pipeline.
  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    if (!loadM && value != 0) {
      const uint32_t count = value == 1 ? length / 4 : length;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    }

    NS::String* swiftName = NS::String::string("mul", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  MulKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<MulKernel>* output = new PipelineValue<MulKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
