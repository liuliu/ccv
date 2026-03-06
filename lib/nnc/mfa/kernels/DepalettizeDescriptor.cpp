#include "DepalettizeDescriptor.hpp"
#include "DepalettizeKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool DepalettizeDescriptor::operator==(const DepalettizeDescriptor& rhs) const {
  return
  qbits == rhs.qbits &&
  memoryPrecision == rhs.memoryPrecision &&
  numberInBlocks == rhs.numberInBlocks &&
  length == rhs.length;
}

bool DepalettizeDescriptor::partial() const noexcept {
  return qbits == 8 && numberInBlocks > 0 && (length % numberInBlocks) != 0;
}

std::size_t std::hash<DepalettizeDescriptor>::operator()(const DepalettizeDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.qbits }));
  combine_64(seed, pack_64(simd::uint2 { hash.numberInBlocks, hash.length }));
  return seed;
}

std::pair<DepalettizeKernelDescriptor, PipelineValue<DepalettizeKernel>*> DepalettizeDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<DepalettizeKernelDescriptor, std::unique_ptr<DepalettizeKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](DepalettizeKernelDescriptor descriptor) -> DepalettizeKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      DepalettizeKernel* kernel = new DepalettizeKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<DepalettizeKernel>(kernel);
      return kernel;
    }
  };

  DepalettizeKernelDescriptor kernelDesc;
  kernelDesc.qbits = qbits;
  kernelDesc.partial = partial() ? 1 : 0;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    CCV_NNC_MFA_PRECONDITION(qbits == 5 || qbits == 6 || qbits == 8);
    if (qbits == 5) {
      CCV_NNC_MFA_PRECONDITION((length % numberInBlocks) == 0);
      CCV_NNC_MFA_PRECONDITION((numberInBlocks % (128 * 8)) == 0);
      const uint32_t blocks = numberInBlocks / 8;
      constants->setConstantValue(&blocks, MTL::DataTypeUInt, NS::UInteger(0));
    } else if (qbits == 6) {
      CCV_NNC_MFA_PRECONDITION((length % numberInBlocks) == 0);
      CCV_NNC_MFA_PRECONDITION((numberInBlocks % (256 * 4)) == 0);
      const uint32_t blocks = numberInBlocks / 4;
      constants->setConstantValue(&blocks, MTL::DataTypeUInt, NS::UInteger(0));
    } else {
      CCV_NNC_MFA_PRECONDITION((numberInBlocks % (256 * 4)) == 0);
      const uint32_t blocks = numberInBlocks / 4;
      constants->setConstantValue(&blocks, MTL::DataTypeUInt, NS::UInteger(0));
      if (partial()) {
        CCV_NNC_MFA_PRECONDITION((length % (256 * 4)) == 0);
        const uint32_t numberOfElementsPerSegment = numberInBlocks / (256 * 4);
        constants->setConstantValue(&numberOfElementsPerSegment, MTL::DataTypeUInt, NS::UInteger(1));
      }
    }

    NS::String* swiftName = NS::String::string("depalettize", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  DepalettizeKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<DepalettizeKernel>* output = new PipelineValue<DepalettizeKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
