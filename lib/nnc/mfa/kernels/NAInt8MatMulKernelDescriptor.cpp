#include "NAInt8MatMulKernelDescriptor.hpp"
#include "NAInt8MatMulDescriptor.hpp"
#include <algorithm>
#include <climits>
#include "../ccv_nnc_mfa_hash.hpp"

bool NAInt8MatMulKernelDescriptor::operator==(const NAInt8MatMulKernelDescriptor& rhs) const {
  return
      simd_all(blockDimensions == rhs.blockDimensions) &&
      executionSIMDGroups == rhs.executionSIMDGroups &&
      ioPrecision == rhs.ioPrecision &&
      useRegisterOperands == rhs.useRegisterOperands &&
      useBias == rhs.useBias &&
      loadM == rhs.loadM &&
      useLeadingDimensions == rhs.useLeadingDimensions &&
      activationQuantizeThreads == rhs.activationQuantizeThreads &&
      activationHadamard256 == rhs.activationHadamard256 &&
      groupM == rhs.groupM &&
      groupN == rhs.groupN;
}

std::size_t std::hash<NAInt8MatMulKernelDescriptor>::operator()(const NAInt8MatMulKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  seed = combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  seed = combine_32(seed, pack_32(simd::ushort2 { hash.executionSIMDGroups, (uint16_t)hash.ioPrecision.value }));
  seed = combine_32(seed, pack_32(simd::ushort2 { hash.activationQuantizeThreads, (uint16_t)hash.useBias }));
  seed = combine_32(seed, pack_32(simd::ushort2 { (uint16_t)hash.loadM, (uint16_t)hash.useLeadingDimensions }));
  seed = combine_32(seed, hash.groupM);
  seed = combine_32(seed, hash.groupN);
  seed = combine_32(seed, hash.activationHadamard256 ? 1 : 0);
  seed = combine_32(seed, hash.useRegisterOperands);
  return seed;
}

NAInt8MatMulKernelDescriptor::NAInt8MatMulKernelDescriptor(
    simd::ushort3 blockDimensions,
    uint16_t executionSIMDGroups,
    GEMMOperandPrecision ioPrecision,
    bool useBias,
    bool loadM,
    uint16_t activationQuantizeThreads,
    uint32_t groupM,
    uint32_t groupN,
    bool useLeadingDimensions,
    bool activationHadamard256,
    bool useRegisterOperands) noexcept
{
  this->blockDimensions = blockDimensions;
  this->executionSIMDGroups = executionSIMDGroups;
  this->ioPrecision = ioPrecision;
  this->useRegisterOperands = useRegisterOperands;
  this->useBias = useBias;
  this->loadM = loadM;
  this->useLeadingDimensions = useLeadingDimensions;
  this->activationQuantizeThreads = activationQuantizeThreads;
  this->activationHadamard256 = activationHadamard256;
  this->groupM = groupM;
  this->groupN = groupN;
}

NAInt8MatMulKernelDescriptor::NAInt8MatMulKernelDescriptor(
    const NAInt8MatMulDescriptor& descriptor,
    const DeviceProperties& dprops) noexcept
{
  useRegisterOperands = [&]() {
    // The register kernel consumes contiguous, unbatched rows and writes output
    // after activation quantization. Scalar IO precision and rotation are
    // shared with the native path and do not restrict this choice.
    if (descriptor.batchDimension != 1 || descriptor.leadingDimensions.has_value())
      return false;

    const uint64_t M = descriptor.matrixDimensions[0], N = descriptor.matrixDimensions[1], K = descriptor.matrixDimensions[2];
    if (M > INT32_MAX - 63 || N > INT32_MAX - 127 || M * K > UINT32_MAX)
      return false;

    // Keep short reductions eligible: tail gains outweigh small aligned-case
    // losses without needing a separate core threshold or alignment rule.
    // Smaller GPUs retain native tiles unless B or C needs wider offsets.
    const uint64_t cores = dprops.coreCount;
    const bool needsWideOffsets = N * K > UINT32_MAX || M * N > UINT32_MAX;
    if (!needsWideOffsets && cores < 36)
      return false;

    // Keep the measured width / reduction range and the limit on deep
    // reductions with fewer than 4096 rows.
    if (N < 1536 || K < 4096 || K > 32768 || K > 4 * N || (M < 4096 && K > 20480))
      return false;

    // Large outputs amortize the register traversal directly.
    if (M >= 2048 && M * N >= uint64_t(4096) * 1536)
      return true;
    if (M < 512 || K <= 8192)
      return false;

    // Medium-row reductions need either two register groups per core or
    // more independent groups reaching cores than the native grid provides.
    // Count useful tiles, excluding empty slots in the native Morton grid.
    const uint64_t columns = (N + 127) / 128;
    const uint64_t registerGroups = ((M + 63) / 64) * columns;
    const uint64_t nativeGroups = ((M + 127) / 128) * columns;
    return registerGroups >= cores * 2 ||
        std::min(registerGroups, cores) > std::min(nativeGroups, cores);
  }();
  const bool moreNumberOfTiles = dprops.coreCount >= 36 && descriptor.matrixDimensions[2] <= 8192;
  blockDimensions = simd::ushort3 {
    uint16_t(useRegisterOperands || moreNumberOfTiles ? 64 : 128), 128,
    uint16_t(useRegisterOperands ? 32 : 128),
  };
  executionSIMDGroups = useRegisterOperands ? 8 : (moreNumberOfTiles ? 4 : 8);
  ioPrecision = descriptor.ioPrecision;
  useBias = descriptor.useBias;
  loadM = descriptor.loadM;
  useLeadingDimensions = descriptor.leadingDimensions.has_value();
  activationQuantizeThreads = 256;
  activationHadamard256 = descriptor.activationHadamard256;
  groupM = !useRegisterOperands && descriptor.matrixDimensions[0] >= 4096 ? 4096 : 0;
  groupN = !useRegisterOperands && descriptor.matrixDimensions[1] >= 4096 ? 4096 : 0;
}
