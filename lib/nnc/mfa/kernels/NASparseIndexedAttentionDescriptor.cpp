#include "NASparseIndexedAttentionDescriptor.hpp"
#include "NASparseIndexedAttentionKernel.hpp"
#include "NASparseIndexedAttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool NASparseIndexedAttentionDescriptor::operator==(const NASparseIndexedAttentionDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  attentionSinks == rhs.attentionSinks &&
  T == rhs.T &&
  denseRows == rhs.denseRows &&
  sparseRows == rhs.sparseRows &&
  H == rhs.H &&
  K == rhs.K &&
  isCausal == rhs.isCausal &&
  slidingWindow == rhs.slidingWindow &&
  sinkHeadStride == rhs.sinkHeadStride &&
  scale == rhs.scale &&
  variant == rhs.variant;
}

std::size_t std::hash<NASparseIndexedAttentionDescriptor>::operator()(const NASparseIndexedAttentionDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, hash.T }));
  combine_64(seed, pack_64(simd::uint2 { hash.denseRows, hash.sparseRows }));
  combine_64(seed, pack_64(simd::uint2 { hash.H, hash.K }));
  combine_64(seed, pack_64(simd::uint2 { hash.sinkHeadStride, hash.isCausal ? 1u : 0u }));
  combine_32(seed, hash.slidingWindow);
  combine_32(seed, pack_32(simd::ushort2 { (unsigned short)(hash.attentionSinks ? 1 : 0), (unsigned short)hash.variant }));
  combine_32(seed, reinterpret_cast<const uint32_t&>(hash.scale));
  return seed;
}

std::pair<NASparseIndexedAttentionKernelDescriptor, PipelineValue<NASparseIndexedAttentionKernel> *> NASparseIndexedAttentionDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NASparseIndexedAttentionKernelDescriptor, std::unique_ptr<NASparseIndexedAttentionKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](NASparseIndexedAttentionKernelDescriptor descriptor) -> NASparseIndexedAttentionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      NASparseIndexedAttentionKernel* kernel = new NASparseIndexedAttentionKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<NASparseIndexedAttentionKernel>(kernel);
      return kernel;
    }
  };

  NASparseIndexedAttentionKernelDescriptor kernelDesc;
  kernelDesc.memoryPrecision = memoryPrecision;
  kernelDesc.attentionSinks = attentionSinks;
  kernelDesc.denseOnly = K == 0;
  kernelDesc.variant = variant;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&T, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&denseRows, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&sparseRows, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&H, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&isCausal, MTL::DataTypeBool, NS::UInteger(5));
    constants->setConstantValue(&sinkHeadStride, MTL::DataTypeUInt, NS::UInteger(6));
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(7));
    constants->setConstantValue(&slidingWindow, MTL::DataTypeUInt, NS::UInteger(8));
    NS::String* swiftName = NS::String::string("sparse_indexed_attention", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    descriptor->setComputeFunction(function.get());
    auto pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  NASparseIndexedAttentionKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<NASparseIndexedAttentionKernel>* output = new PipelineValue<NASparseIndexedAttentionKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
