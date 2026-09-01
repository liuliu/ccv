#include "ANERowwiseTransformKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool ANERowwiseTransformKernelDescriptor::operator==(const ANERowwiseTransformKernelDescriptor& rhs) const
{
  return
      memoryPrecision == rhs.memoryPrecision &&
      supportsApple10 == rhs.supportsApple10;
}

std::size_t std::hash<ANERowwiseTransformKernelDescriptor>::operator()(const ANERowwiseTransformKernelDescriptor& hash) const noexcept
{
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, (uint32_t)hash.memoryPrecision.value);
  combine_32(seed, (uint32_t)hash.supportsApple10);
  return seed;
}

ANERowwiseTransformKernelDescriptor::ANERowwiseTransformKernelDescriptor(
    GEMMOperandPrecision memoryPrecision,
    bool supportsApple10) noexcept
{
  this->memoryPrecision = memoryPrecision;
  this->supportsApple10 = supportsApple10;
}
