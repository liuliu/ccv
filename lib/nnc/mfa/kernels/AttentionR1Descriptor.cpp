#include "AttentionR1Descriptor.hpp"
#include "AttentionR1Kernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

static uint32_t alignUp(uint32_t value, uint32_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

static uint32_t directSIMDGroups(uint32_t D, GEMMOperandPrecision memoryPrecision) noexcept {
  const uint32_t maxThreadgroupMemoryBytes = 32768;
  const uint32_t queryBytes = alignUp(D * (uint32_t)memoryPrecision.size(), sizeof(float));
  if (queryBytes >= maxThreadgroupMemoryBytes) {
    return 1;
  }
  const uint32_t partialBytes = (D + 2) * sizeof(float);
  uint32_t maxSIMDGroups = (maxThreadgroupMemoryBytes - queryBytes) / partialBytes;
  if (maxSIMDGroups > 32) {
    maxSIMDGroups = 32;
  }
  if (maxSIMDGroups > 1 && (maxSIMDGroups % 2) != 0) {
    --maxSIMDGroups;
  }
  while (maxSIMDGroups > 2 &&
      alignUp(queryBytes + maxSIMDGroups * partialBytes, 16) > maxThreadgroupMemoryBytes) {
    maxSIMDGroups -= 2;
  }
  return maxSIMDGroups < 1 ? 1 : maxSIMDGroups;
}

bool AttentionR1Descriptor::operator==(const AttentionR1Descriptor& rhs) const {
  return
      memoryPrecision == rhs.memoryPrecision &&
      (loadC || C == rhs.C) &&
      loadC == rhs.loadC &&
      Hq == rhs.Hq &&
      Hk == rhs.Hk &&
      D == rhs.D &&
      scale == rhs.scale &&
      attentionSinks == rhs.attentionSinks &&
      simdgroups == rhs.simdgroups &&
      workgroups == rhs.workgroups &&
      mode == rhs.mode;
}

AttentionR1Descriptor AttentionR1Descriptor::select(
    GEMMOperandPrecision memoryPrecision,
    uint32_t C,
    uint32_t Hq,
    uint32_t Hk,
    uint32_t D,
    float scale,
    bool loadC,
    bool attentionSinks) noexcept
{
  AttentionR1Descriptor descriptor;
  descriptor.memoryPrecision = memoryPrecision;
  descriptor.C = C;
  descriptor.Hq = Hq;
  descriptor.Hk = Hk;
  descriptor.D = D;
  descriptor.scale = scale;
  descriptor.loadC = loadC;
  descriptor.attentionSinks = attentionSinks;

  if (C < 2048) {
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

std::size_t std::hash<AttentionR1Descriptor>::operator()(const AttentionR1Descriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.mode }));
  combine_64(seed, pack_64(simd::uint2 { hash.loadC ? 0 : hash.C, hash.D }));
  combine_64(seed, pack_64(simd::uint2 { hash.Hq, hash.Hk }));
  combine_64(seed, pack_64(simd::uint2 { hash.simdgroups, hash.workgroups }));
  combine_32(seed, hash.loadC ? 1 : 0);
  combine_32(seed, hash.attentionSinks ? 1 : 0);
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
  return seed;
}

std::pair<AttentionR1KernelDescriptor, PipelineValue<AttentionR1Kernel>*> AttentionR1Descriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<AttentionR1KernelDescriptor, std::unique_ptr<AttentionR1Kernel>> *const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  auto createKernel =
  [=](AttentionR1KernelDescriptor descriptor) -> AttentionR1Kernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    }
    AttentionR1Kernel* kernel = new AttentionR1Kernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<AttentionR1Kernel>(kernel);
    return kernel;
  };

  AttentionR1KernelDescriptor kernelDesc;
  kernelDesc.memoryPrecision = memoryPrecision;
  kernelDesc.loadC = loadC;
  kernelDesc.attentionSinks = attentionSinks;

  auto createPipeline =
  [=](MTL::Library* library, const char* functionNameString) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    if (!loadC) {
      constants->setConstantValue(&C, MTL::DataTypeUInt, NS::UInteger(0));
    }
    constants->setConstantValue(&Hq, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&Hk, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&D, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&simdgroups, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&workgroups, MTL::DataTypeUInt, NS::UInteger(5));
    float scale = this->scale;
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(6));

    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  AttentionR1Kernel* kernel = createKernel(kernelDesc);
  PipelineValue<AttentionR1Kernel>* output = new PipelineValue<AttentionR1Kernel>;
  output->kernel = kernel;
  if (mode == Mode::direct) {
    output->pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "attention_r1_direct"));
  } else {
    output->pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "attention_r1_split_partials"));
    output->second = NS::TransferPtr(createPipeline(kernel->library.get(), "attention_r1_split_reduce"));
  }
  return std::make_pair(kernelDesc, output);
}
