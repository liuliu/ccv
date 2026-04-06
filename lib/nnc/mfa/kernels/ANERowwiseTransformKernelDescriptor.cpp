#include "ANERowwiseTransformKernelDescriptor.hpp"

bool ANERowwiseTransformKernelDescriptor::operator==(const ANERowwiseTransformKernelDescriptor& rhs) const
{
  return memoryPrecision == rhs.memoryPrecision;
}

std::size_t std::hash<ANERowwiseTransformKernelDescriptor>::operator()(const ANERowwiseTransformKernelDescriptor& hash) const noexcept
{
  return std::hash<int>()((int)hash.memoryPrecision.value);
}

ANERowwiseTransformKernelDescriptor::ANERowwiseTransformKernelDescriptor(GEMMOperandPrecision memoryPrecision) noexcept
{
  this->memoryPrecision = memoryPrecision;
}
