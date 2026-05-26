#include "NAScaledDotProductArgPartitionDescriptor.hpp"
#include "NAScaledDotProductArgPartitionKernel.hpp"
#include "NAScaledDotProductArgPartitionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool NAScaledDotProductArgPartitionDescriptor::operator==(const NAScaledDotProductArgPartitionDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  T == rhs.T &&
  C == rhs.C &&
  H == rhs.H &&
  D == rhs.D &&
  kth == rhs.kth &&
  compressionRatio == rhs.compressionRatio &&
  scale == rhs.scale &&
  isCausal == rhs.isCausal &&
  scoreBlockM == rhs.scoreBlockM &&
  scoreBlockN == rhs.scoreBlockN &&
  scoreSIMDGroups == rhs.scoreSIMDGroups;
}

std::size_t std::hash<NAScaledDotProductArgPartitionDescriptor>::operator()(const NAScaledDotProductArgPartitionDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, hash.T }));
  combine_64(seed, pack_64(simd::uint2 { hash.C, hash.H }));
  combine_64(seed, pack_64(simd::uint2 { hash.D, hash.kth }));
  combine_64(seed, pack_64(simd::uint2 { hash.compressionRatio, hash.isCausal ? 1u : 0u }));
  combine_32(seed, pack_32(simd::ushort2 { hash.scoreBlockM, hash.scoreBlockN }));
  combine_32(seed, pack_32(simd::ushort2 { hash.scoreSIMDGroups, 0 }));
  combine_32(seed, reinterpret_cast<const uint32_t&>(hash.scale));
  return seed;
}

std::pair<NAScaledDotProductArgPartitionKernelDescriptor, PipelineValue<NAScaledDotProductArgPartitionKernel> *> NAScaledDotProductArgPartitionDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAScaledDotProductArgPartitionKernelDescriptor, std::unique_ptr<NAScaledDotProductArgPartitionKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](NAScaledDotProductArgPartitionKernelDescriptor descriptor) -> NAScaledDotProductArgPartitionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      NAScaledDotProductArgPartitionKernel* kernel = new NAScaledDotProductArgPartitionKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<NAScaledDotProductArgPartitionKernel>(kernel);
      return kernel;
    }
  };

  NAScaledDotProductArgPartitionKernelDescriptor kernelDesc;
  kernelDesc.memoryPrecision = memoryPrecision;
  kernelDesc.kth = kth;
  kernelDesc.scoreBlockM = scoreBlockM;
  kernelDesc.scoreBlockN = scoreBlockN;
  kernelDesc.scoreSIMDGroups = scoreSIMDGroups;

  auto createPipeline =
  [=](MTL::Library* library, const char* name) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&T, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&C, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&H, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&D, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&compressionRatio, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&isCausal, MTL::DataTypeBool, NS::UInteger(5));
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(6));
    NS::String* swiftName = NS::String::string(name, NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  NAScaledDotProductArgPartitionKernel* kernel = createKernel(kernelDesc);
  auto scorePipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "index_score"));
  auto topKSerialPipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "topk_serial"));
  auto topKTilePipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "topk_tile"));
  auto topKMergePipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "topk_merge"));

  PipelineValue<NAScaledDotProductArgPartitionKernel>* output = new PipelineValue<NAScaledDotProductArgPartitionKernel> { kernel, scorePipeline };
  output->second = topKSerialPipeline;
  output->third = topKTilePipeline;
  output->fourth = topKMergePipeline;
  return std::make_pair(kernelDesc, output);
}
