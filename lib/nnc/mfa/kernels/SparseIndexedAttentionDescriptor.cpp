#include "SparseIndexedAttentionDescriptor.hpp"
#include "SparseIndexedAttentionKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool SparseIndexedAttentionDescriptor::operator==(const SparseIndexedAttentionDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  attentionSinks == rhs.attentionSinks &&
  T == rhs.T &&
  denseRows == rhs.denseRows &&
  sparseRows == rhs.sparseRows &&
  H == rhs.H &&
  D == rhs.D &&
  K == rhs.K &&
  isCausal == rhs.isCausal &&
  sinkHeadStride == rhs.sinkHeadStride &&
  scale == rhs.scale;
}

std::size_t std::hash<SparseIndexedAttentionDescriptor>::operator()(const SparseIndexedAttentionDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, hash.T }));
  combine_64(seed, pack_64(simd::uint2 { hash.denseRows, hash.sparseRows }));
  combine_64(seed, pack_64(simd::uint2 { hash.H, hash.D }));
  combine_64(seed, pack_64(simd::uint2 { hash.K, hash.sinkHeadStride }));
  combine_32(seed, pack_32(simd::ushort2 { (unsigned short)(hash.attentionSinks ? 1 : 0), (unsigned short)(hash.isCausal ? 1 : 0) }));
  combine_32(seed, reinterpret_cast<const uint32_t&>(hash.scale));
  return seed;
}

std::pair<SparseIndexedAttentionKernelDescriptor, PipelineValue<SparseIndexedAttentionKernel> *> SparseIndexedAttentionDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SparseIndexedAttentionKernelDescriptor, std::unique_ptr<SparseIndexedAttentionKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](SparseIndexedAttentionKernelDescriptor descriptor) -> SparseIndexedAttentionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      SparseIndexedAttentionKernel* kernel = new SparseIndexedAttentionKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<SparseIndexedAttentionKernel>(kernel);
      return kernel;
    }
  };

  SparseIndexedAttentionKernelDescriptor kernelDesc;
  kernelDesc.memoryPrecision = memoryPrecision;
  kernelDesc.attentionSinks = attentionSinks;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&T, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&denseRows, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&sparseRows, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&H, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&D, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(5));
    constants->setConstantValue(&isCausal, MTL::DataTypeBool, NS::UInteger(6));
    constants->setConstantValue(&sinkHeadStride, MTL::DataTypeUInt, NS::UInteger(7));
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(8));
    NS::String* swiftName = NS::String::string("sparse_indexed_attention", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  SparseIndexedAttentionKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<SparseIndexedAttentionKernel>* output = new PipelineValue<SparseIndexedAttentionKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
