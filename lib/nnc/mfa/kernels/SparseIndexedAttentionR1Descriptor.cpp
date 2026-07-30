#include "SparseIndexedAttentionR1Descriptor.hpp"
#include "SparseIndexedAttentionR1Kernel.hpp"

#include <algorithm>
#include <simd/simd.h>

#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

static uint32_t alignUp(uint32_t value, uint32_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

static uint32_t directSIMDGroups(uint32_t D, GEMMOperandPrecision memoryPrecision) noexcept {
  const uint32_t maxThreadgroupMemoryBytes = 32768;
  const uint32_t queryBytes = alignUp(D * (uint32_t)memoryPrecision.size(), sizeof(float));
  const uint32_t fixedBytes = queryBytes + sizeof(uint32_t);
  if (fixedBytes >= maxThreadgroupMemoryBytes) {
    return 1;
  }
  const uint32_t partialBytes = (D + 2) * sizeof(float);
  uint32_t maxSIMDGroups = (maxThreadgroupMemoryBytes - fixedBytes) / partialBytes;
  if (maxSIMDGroups > 32) {
    maxSIMDGroups = 32;
  }
  if (maxSIMDGroups > 1 && (maxSIMDGroups % 2) != 0) {
    --maxSIMDGroups;
  }
  while (maxSIMDGroups > 2 &&
      alignUp(fixedBytes + maxSIMDGroups * partialBytes, 16) > maxThreadgroupMemoryBytes) {
    maxSIMDGroups -= 2;
  }
  return maxSIMDGroups < 1 ? 1 : maxSIMDGroups;
}

bool SparseIndexedAttentionR1Descriptor::operator==(const SparseIndexedAttentionR1Descriptor& rhs) const {
  return
      memoryPrecision == rhs.memoryPrecision &&
      (loadK ||
          (denseRows == rhs.denseRows &&
           sparseRows == rhs.sparseRows &&
           K == rhs.K)) &&
      loadK == rhs.loadK &&
      H == rhs.H &&
      D == rhs.D &&
      scale == rhs.scale &&
      attentionSinks == rhs.attentionSinks &&
      slidingWindow == rhs.slidingWindow &&
      simdgroups == rhs.simdgroups &&
      workgroups == rhs.workgroups &&
      mode == rhs.mode;
}

SparseIndexedAttentionR1Descriptor SparseIndexedAttentionR1Descriptor::select(
    GEMMOperandPrecision memoryPrecision,
    uint32_t denseRows,
    uint32_t sparseRows,
    uint32_t K,
    uint32_t H,
    uint32_t D,
    float scale,
    bool loadK,
    bool attentionSinks,
    uint32_t slidingWindow) noexcept
{
  SparseIndexedAttentionR1Descriptor descriptor;
  descriptor.memoryPrecision = memoryPrecision;
  descriptor.denseRows = denseRows;
  descriptor.sparseRows = sparseRows;
  descriptor.K = K;
  descriptor.H = H;
  descriptor.D = D;
  descriptor.scale = scale;
  descriptor.loadK = loadK;
  descriptor.attentionSinks = attentionSinks;
  descriptor.slidingWindow = slidingWindow;

  const uint32_t visibleDenseRows =
      slidingWindow > 0 ? std::min(denseRows, slidingWindow) : denseRows;
  const uint64_t maximumRows = (uint64_t)visibleDenseRows + K;
  if (maximumRows < 2048) {
    descriptor.mode = Mode::direct;
    descriptor.simdgroups = directSIMDGroups(D, memoryPrecision);
    descriptor.workgroups = 1;
  } else {
    descriptor.mode = Mode::splitReduce;
    descriptor.simdgroups = 8;
    descriptor.workgroups = 32;
  }
  return descriptor;
}

std::size_t std::hash<SparseIndexedAttentionR1Descriptor>::operator()(const SparseIndexedAttentionR1Descriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 {
      (unsigned int)hash.memoryPrecision.value,
      (unsigned int)hash.mode }));
  combine_64(seed, pack_64(simd::uint2 {
      hash.loadK ? 0 : hash.denseRows,
      hash.loadK ? 0 : hash.sparseRows }));
  combine_64(seed, pack_64(simd::uint2 { hash.loadK ? 0 : hash.K, hash.D }));
  combine_32(seed, hash.H);
  combine_64(seed, pack_64(simd::uint2 { hash.simdgroups, hash.workgroups }));
  combine_32(seed, hash.loadK ? 1 : 0);
  combine_32(seed, hash.slidingWindow);
  combine_32(seed, hash.attentionSinks ? 1 : 0);
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
  return seed;
}

std::pair<SparseIndexedAttentionR1KernelDescriptor, PipelineValue<SparseIndexedAttentionR1Kernel>*> SparseIndexedAttentionR1Descriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<SparseIndexedAttentionR1KernelDescriptor, std::unique_ptr<SparseIndexedAttentionR1Kernel>> *const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  auto createKernel =
  [=](SparseIndexedAttentionR1KernelDescriptor descriptor) -> SparseIndexedAttentionR1Kernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    }
    SparseIndexedAttentionR1Kernel* kernel =
        new SparseIndexedAttentionR1Kernel(descriptor, device);
    (*libraryCache)[descriptor] =
        std::unique_ptr<SparseIndexedAttentionR1Kernel>(kernel);
    return kernel;
  };

  SparseIndexedAttentionR1KernelDescriptor kernelDesc;
  kernelDesc.memoryPrecision = memoryPrecision;
  kernelDesc.loadK = loadK;
  kernelDesc.attentionSinks = attentionSinks;

  auto createPipeline =
  [=](MTL::Library* library, const char* functionNameString) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&H, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&D, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&simdgroups, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&workgroups, MTL::DataTypeUInt, NS::UInteger(3));
    float scale = this->scale;
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(4));
    constants->setConstantValue(
        &slidingWindow, MTL::DataTypeUInt, NS::UInteger(5));
    if (!loadK) {
      constants->setConstantValue(
          &denseRows, MTL::DataTypeUInt, NS::UInteger(6));
      constants->setConstantValue(
          &sparseRows, MTL::DataTypeUInt, NS::UInteger(7));
      constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(8));
    }

    auto functionName = NS::String::string(
        functionNameString, NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(
        library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  SparseIndexedAttentionR1Kernel* kernel = createKernel(kernelDesc);
  PipelineValue<SparseIndexedAttentionR1Kernel>* output =
      new PipelineValue<SparseIndexedAttentionR1Kernel>;
  output->kernel = kernel;
  if (mode == Mode::direct) {
    output->pipeline = NS::TransferPtr(
        createPipeline(kernel->library.get(), "sparse_indexed_attention_r1_direct"));
  } else {
    output->pipeline = NS::TransferPtr(
        createPipeline(kernel->library.get(), "sparse_indexed_attention_r1_split_partials"));
    output->second = NS::TransferPtr(
        createPipeline(kernel->library.get(), "sparse_indexed_attention_r1_split_reduce"));
  }
  return std::make_pair(kernelDesc, output);
}
