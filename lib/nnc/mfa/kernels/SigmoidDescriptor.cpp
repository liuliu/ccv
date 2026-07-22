#include "SigmoidDescriptor.hpp"
#include "SigmoidKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool SigmoidDescriptor::operator==(const SigmoidDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  gradient == rhs.gradient &&
  value == rhs.value &&
  loadM == rhs.loadM &&
  (loadM || length == rhs.length);
}

std::size_t std::hash<SigmoidDescriptor>::operator()(const SigmoidDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, pack_64(simd::uint2 { hash.loadM ? 0 : (unsigned int)hash.length, (unsigned int)hash.gradient }));
  combine_32(seed, hash.loadM ? 1 : 0);
  return seed;
}

std::pair<SigmoidKernelDescriptor, PipelineValue<SigmoidKernel> *> SigmoidDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SigmoidKernelDescriptor, std::unique_ptr<SigmoidKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](SigmoidKernelDescriptor descriptor) -> SigmoidKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      SigmoidKernel* kernel = new SigmoidKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<SigmoidKernel>(kernel);
      return kernel;
    }
  };

  SigmoidKernelDescriptor kernelDesc;
  kernelDesc.gradient = gradient;
  kernelDesc.value = value;
  kernelDesc.loadM = loadM;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    if (!loadM) {
      uint32_t count;
      if (value == 0) {
      } else if (value == 1) {
        count = length / 4;
        constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
      } else {
        count = length;
        constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
      }
    }

    NS::String* swiftName = NS::String::string("sigmoid", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  SigmoidKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<SigmoidKernel>* output = new PipelineValue<SigmoidKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
