#ifndef MFA_SCALEDDOTPRODUCTARGPARTITIONKERNELDESCRIPTOR_HPP_
#define MFA_SCALEDDOTPRODUCTARGPARTITIONKERNELDESCRIPTOR_HPP_

#include "GEMMOperandPrecision.hpp"

struct ScaledDotProductArgPartitionKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;
  uint32_t kth = 0;
  uint8_t scoreMode = 0; // 0: dense, 1: candidate rows, 2: block maxima, 3: block bitset, 4: dense with index utilities.
  uint16_t scoreBlockM = 16;
  uint16_t scoreBlockN = 32;
  uint16_t scoreSIMDGroups = 4;
  bool loadC = false;
  bool loadM = false;

  bool operator==(const ScaledDotProductArgPartitionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ScaledDotProductArgPartitionKernelDescriptor>
{
  std::size_t operator()(const ScaledDotProductArgPartitionKernelDescriptor& hash) const noexcept;
};

#endif
