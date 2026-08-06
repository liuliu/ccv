#ifndef MFA_NASCALEDDOTPRODUCTARGPARTITIONKERNELDESCRIPTOR_HPP_
#define MFA_NASCALEDDOTPRODUCTARGPARTITIONKERNELDESCRIPTOR_HPP_

#include "GEMMOperandPrecision.hpp"

struct NAScaledDotProductArgPartitionKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;
  uint32_t kth = 0;
  uint16_t scoreBlockM = 16;
  uint16_t scoreBlockN = 32;
  uint16_t scoreSIMDGroups = 4;
  bool loadC = false;
  bool isCausal = false;

  bool operator==(const NAScaledDotProductArgPartitionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAScaledDotProductArgPartitionKernelDescriptor>
{
  std::size_t operator()(const NAScaledDotProductArgPartitionKernelDescriptor& hash) const noexcept;
};

#endif
