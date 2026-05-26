#ifndef MFA_SCALEDDOTPRODUCTARGPARTITIONKERNELDESCRIPTOR_HPP_
#define MFA_SCALEDDOTPRODUCTARGPARTITIONKERNELDESCRIPTOR_HPP_

#include "GEMMOperandPrecision.hpp"

struct ScaledDotProductArgPartitionKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;
  uint32_t kth = 0;
  uint16_t scoreBlockM = 16;
  uint16_t scoreBlockN = 32;
  uint16_t scoreSIMDGroups = 4;

  bool operator==(const ScaledDotProductArgPartitionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ScaledDotProductArgPartitionKernelDescriptor>
{
  std::size_t operator()(const ScaledDotProductArgPartitionKernelDescriptor& hash) const noexcept;
};

#endif
