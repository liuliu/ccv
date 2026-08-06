#include "ScaledDotProductArgPartitionEnumerateDescriptor.hpp"
#include "ScaledDotProductArgPartitionEnumerateKernel.hpp"
#include "ScaledDotProductArgPartitionEnumerateKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool ScaledDotProductArgPartitionEnumerateDescriptor::operator==(const ScaledDotProductArgPartitionEnumerateDescriptor& rhs) const {
  return
  T == rhs.T &&
  C == rhs.C &&
  kth == rhs.kth &&
  compressionRatio == rhs.compressionRatio &&
  queryOffset == rhs.queryOffset &&
  isCausal == rhs.isCausal;
}

std::size_t std::hash<ScaledDotProductArgPartitionEnumerateDescriptor>::operator()(const ScaledDotProductArgPartitionEnumerateDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { hash.T, hash.C }));
  combine_64(seed, pack_64(simd::uint2 { hash.kth, hash.compressionRatio }));
  combine_32(seed, static_cast<uint32_t>(hash.queryOffset));
  combine_32(seed, hash.isCausal ? 1 : 0);
  return seed;
}

std::pair<ScaledDotProductArgPartitionEnumerateKernelDescriptor, PipelineValue<ScaledDotProductArgPartitionEnumerateKernel> *> ScaledDotProductArgPartitionEnumerateDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ScaledDotProductArgPartitionEnumerateKernelDescriptor, std::unique_ptr<ScaledDotProductArgPartitionEnumerateKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](ScaledDotProductArgPartitionEnumerateKernelDescriptor descriptor) -> ScaledDotProductArgPartitionEnumerateKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      ScaledDotProductArgPartitionEnumerateKernel* kernel = new ScaledDotProductArgPartitionEnumerateKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<ScaledDotProductArgPartitionEnumerateKernel>(kernel);
      return kernel;
    }
  };

  ScaledDotProductArgPartitionEnumerateKernelDescriptor kernelDesc;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&T, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&C, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&kth, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&compressionRatio, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&isCausal, MTL::DataTypeBool, NS::UInteger(4));
    constants->setConstantValue(&queryOffset, MTL::DataTypeInt, NS::UInteger(5));
    NS::String* swiftName = NS::String::string("enumerate", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  ScaledDotProductArgPartitionEnumerateKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
  PipelineValue<ScaledDotProductArgPartitionEnumerateKernel>* output = new PipelineValue<ScaledDotProductArgPartitionEnumerateKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
