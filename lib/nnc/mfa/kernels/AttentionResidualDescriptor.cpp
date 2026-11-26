#include "AttentionResidualDescriptor.hpp"
#include "AttentionResidualKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool AttentionResidualDescriptor::operator==(const AttentionResidualDescriptor& rhs) const {
  return queryCount == rhs.queryCount &&
    memoryPrecision == rhs.memoryPrecision &&
    channels == rhs.channels &&
    blockCount == rhs.blockCount &&
    epsilon == rhs.epsilon &&
    scale == rhs.scale;
}

std::size_t std::hash<AttentionResidualDescriptor>::operator()(const AttentionResidualDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(simd::uint2 { hash.queryCount, (unsigned int)hash.memoryPrecision.value }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.channels, hash.blockCount }));
  seed = combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.epsilon), *reinterpret_cast<const uint32_t*>(&hash.scale) }));
  return seed;
}

std::pair<AttentionResidualKernelDescriptor, PipelineValue<AttentionResidualKernel>*> AttentionResidualDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<AttentionResidualKernelDescriptor, std::unique_ptr<AttentionResidualKernel>>* const libraryCache) const noexcept {
  auto createKernel =
  [=](AttentionResidualKernelDescriptor descriptor) -> AttentionResidualKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      AttentionResidualKernel* kernel = new AttentionResidualKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<AttentionResidualKernel>(kernel);
      return kernel;
    }
  };

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&channels, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&blockCount, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&epsilon, MTL::DataTypeFloat, NS::UInteger(2));
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(3));

    NS::String* swiftName = NS::String::string("attention_residual", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  // The specialized pipeline can support fewer threads than the device.
  AttentionResidualKernelDescriptor kernelDesc = { queryCount, memoryPrecision, 1024 };
  AttentionResidualKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
  while (kernelDesc.threads > 32 && pipeline->maxTotalThreadsPerThreadgroup() < kernelDesc.threads) {
    kernelDesc.threads /= 2;
    kernel = createKernel(kernelDesc);
    pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
  }

  PipelineValue<AttentionResidualKernel>* output = new PipelineValue<AttentionResidualKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
