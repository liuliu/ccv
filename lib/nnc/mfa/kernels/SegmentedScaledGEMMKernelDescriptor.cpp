#include "SegmentedScaledGEMMKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool SegmentedScaledGEMMKernelDescriptor::operator==(const SegmentedScaledGEMMKernelDescriptor& rhs) const
{
  return
    simd_all(blockDimensions == rhs.blockDimensions) &&
    executionSIMDGroups == rhs.executionSIMDGroups &&
    ioPrecision == rhs.ioPrecision &&
    useBias == rhs.useBias &&
    loadM == rhs.loadM;
}

std::size_t std::hash<SegmentedScaledGEMMKernelDescriptor>::operator()(const SegmentedScaledGEMMKernelDescriptor& hash) const noexcept
{
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd::uint2 {
    ((uint32_t)hash.blockDimensions[0] << 16) | hash.blockDimensions[1],
    ((uint32_t)hash.blockDimensions[2] << 16) | hash.executionSIMDGroups,
  }));
  combine_32(seed, pack_32(simd::ushort2 { (uint16_t)hash.ioPrecision.value, (uint16_t)hash.useBias }));
  combine_32(seed, hash.loadM ? 1 : 0);
  return seed;
}

SegmentedScaledGEMMKernelDescriptor::SegmentedScaledGEMMKernelDescriptor(
    simd::ushort3 blockDimensions,
    uint16_t executionSIMDGroups,
    GEMMOperandPrecision ioPrecision,
    bool useBias,
    bool loadM) noexcept
{
  this->blockDimensions = blockDimensions;
  this->executionSIMDGroups = executionSIMDGroups;
  this->ioPrecision = ioPrecision;
  this->useBias = useBias;
  this->loadM = loadM;
}
