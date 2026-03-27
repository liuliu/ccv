#include "SegmentedScaledGEMMPrologueKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool SegmentedScaledGEMMPrologueKernelDescriptor::operator==(const SegmentedScaledGEMMPrologueKernelDescriptor& rhs) const
{
  return ioPrecision == rhs.ioPrecision && useBias == rhs.useBias;
}

std::size_t std::hash<SegmentedScaledGEMMPrologueKernelDescriptor>::operator()(const SegmentedScaledGEMMPrologueKernelDescriptor& hash) const noexcept
{
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, pack_32(simd::ushort2 { (uint16_t)hash.ioPrecision.value, (uint16_t)hash.useBias }));
  return seed;
}

SegmentedScaledGEMMPrologueKernelDescriptor::SegmentedScaledGEMMPrologueKernelDescriptor(GEMMOperandPrecision ioPrecision, bool useBias) noexcept
{
  this->ioPrecision = ioPrecision;
  this->useBias = useBias;
}
