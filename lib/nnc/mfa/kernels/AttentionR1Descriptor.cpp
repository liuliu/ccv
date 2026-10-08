#include "AttentionR1Descriptor.hpp"
#include <algorithm>
#include "AttentionR1Kernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool AttentionR1Descriptor::operator==(const AttentionR1Descriptor& rhs) const {
  return
      memoryPrecision == rhs.memoryPrecision &&
      (loadC || C == rhs.C) &&
      loadC == rhs.loadC &&
      Hq == rhs.Hq &&
      Hk == rhs.Hk &&
      D == rhs.D && R == rhs.R && isCausal == rhs.isCausal &&
      scale == rhs.scale &&
      attentionSinks == rhs.attentionSinks &&
      simdgroups == rhs.simdgroups && headsPerSIMD == rhs.headsPerSIMD &&
      reductionSIMDGroups == rhs.reductionSIMDGroups &&
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
    bool attentionSinks,
    uint32_t R,
    bool isCausal,
    uint32_t batchDimension,
    uint32_t coreCount) noexcept
{
  AttentionR1Descriptor descriptor;
  descriptor.memoryPrecision = memoryPrecision;
  descriptor.C = C;
  descriptor.Hq = Hq;
  descriptor.Hk = Hk;
  descriptor.D = D;
  descriptor.R = R;
  descriptor.isCausal = isCausal;
  descriptor.scale = scale;
  descriptor.loadC = loadC;
  descriptor.attentionSinks = attentionSinks;

  // Put related GQA heads together so their KV reads share the same cache
  // lines, without a cross-SIMD merge.
  // Keep the group at most 128 threads for R1 and 256 for R2 so a large GQA ratio
  // does not force all of its queries into one large scheduling unit.
  const uint32_t gqa = Hq / Hk;
  uint32_t headsPerGroup = 1;
  for (uint32_t h = 2; h <= 4 && h <= gqa; ++h)
    if (gqa % h == 0)
      headsPerGroup = h;
  // Pair D=128 heads that share K/V. Each lane then holds the same number
  // of query and accumulator elements as one D=256 head. This bounds the
  // register footprint while halving K/V loads for each pair.
  descriptor.headsPerSIMD = D == 128 && headsPerGroup % 2 == 0 ? 2 : 1;
  descriptor.simdgroups = headsPerGroup * R / descriptor.headsPerSIMD;
  descriptor.workgroups = 1;
  const uint64_t groupsPerPartition = uint64_t(batchDimension) * (Hq / headsPerGroup);
  const uint32_t cores = coreCount ? coreCount : 32;
  // Pairing halves the SIMD count in a workgroup; supply more workgroups
  // to retain the same number of independent SIMD streams per core.
  const uint32_t targetGroups = cores * 4 * descriptor.headsPerSIMD;
  // Each SIMD owns complete query heads, so eight KV rows already amortize
  // their query loads. More batches/heads supply parallelism without extra splits.
  // Once the GPU has work, long traversals can still be limited by their
  // serial softmax loop. Balance that loop against the partial-buffer merge
  // with a square-root split budget. More query rows occupy more SIMD groups
  // within a workgroup; more independent batches/heads require fewer splits.
  // Bound a paired SIMD to 256 head/row steps. With fewer independent
  // queries than cores, also shorten a single head's recurrence to 128 rows:
  // extra KV splits reduce its serial latency. Larger batches already supply
  // independent queries; oversplitting them adds partial-buffer and merge work.
  const uint64_t independentQueries = uint64_t(batchDimension) * R * Hq;
  const uint32_t rowsPerPartition = descriptor.headsPerSIMD == 2 || independentQueries < cores ? 128 : 256;
  const uint64_t traversalWork = uint64_t(C) * cores * R * descriptor.headsPerSIMD;
  while (descriptor.workgroups < 1024 && descriptor.workgroups <= C / 16 &&
      (groupsPerPartition * descriptor.workgroups < targetGroups ||
       uint64_t(C) > uint64_t(rowsPerPartition) * descriptor.workgroups ||
       4ull * descriptor.workgroups * descriptor.workgroups * batchDimension * Hq < traversalWork))
    descriptor.workgroups *= 2;
  // Spread the vector merge across SIMD groups, retaining at least four
  // partials per SIMD when possible. Large splits amortize a wider reduction;
  // otherwise bound its shared memory at 16 KiB for D=256.
  const uint32_t reductionLimit = descriptor.workgroups >= 512 ? 32 : 16;
  descriptor.reductionSIMDGroups = std::min(reductionLimit, std::max(4u, descriptor.workgroups / 4));
  descriptor.mode = descriptor.workgroups == 1 ? Mode::direct : Mode::splitReduce;
  // Share short KV traversals among 16 SIMD streams in one workgroup per
  // query/head. Up to four rows per stream amortizes the merge even with
  // many queries. Allow up to sixteen rows when at most two workgroups per
  // core must run; larger batches benefit from grouped-query cache reuse.
  if (C >= 8 && C <= 16 * 16 &&
      (C <= 16 * 4 || uint64_t(batchDimension) * R * Hq <= uint64_t(cores) * 2)) {
    descriptor.mode = Mode::cooperative;
    descriptor.simdgroups = 16;
    descriptor.headsPerSIMD = 1;
    descriptor.workgroups = 1;
    descriptor.reductionSIMDGroups = 1;
  }
  return descriptor;
}

std::size_t std::hash<AttentionR1Descriptor>::operator()(const AttentionR1Descriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.mode }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.loadC ? 0 : hash.C, hash.D }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.Hq, hash.Hk }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.simdgroups, hash.workgroups }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.R, hash.reductionSIMDGroups }));
  seed = combine_32(seed, hash.headsPerSIMD);
  seed = combine_32(seed, hash.isCausal ? 1 : 0);
  seed = combine_32(seed, hash.loadC ? 1 : 0);
  seed = combine_32(seed, hash.attentionSinks ? 1 : 0);
  seed = combine_32(seed, uint32_t(std::hash<float>{}(hash.scale)));
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
    constants->setConstantValue(&R, MTL::DataTypeUInt, NS::UInteger(7));
    constants->setConstantValue(&isCausal, MTL::DataTypeBool, NS::UInteger(8));
    constants->setConstantValue(&reductionSIMDGroups, MTL::DataTypeUInt, NS::UInteger(9));
    constants->setConstantValue(&headsPerSIMD, MTL::DataTypeUInt, NS::UInteger(10));
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
  if (mode == Mode::cooperative) {
    output->pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "attention_r1_cooperative"));
  } else if (mode == Mode::direct) {
    output->pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "attention_r1_direct"));
  } else {
    output->pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "attention_r1_split_partials"));
    output->second = NS::TransferPtr(createPipeline(kernel->library.get(), "attention_r1_split_reduce"));
  }
  return std::make_pair(kernelDesc, output);
}
