#ifndef MFA_NAMATMULSMALLMKERNELDESCRIPTOR_HPP_
#define MFA_NAMATMULSMALLMKERNELDESCRIPTOR_HPP_

#include "GEMMOperandPrecision.hpp"
#include <cstdint>
#include <functional>
#include <simd/simd.h>

struct NAMatMulSmallMKernelDescriptor {
  simd::ushort3 blockDimensions;
  GEMMOperandPrecisions memoryPrecisions;
  uint16_t pack;
  uint16_t executionSIMDGroups;
  bool useBias;
  bool loadM;

  NAMatMulSmallMKernelDescriptor() = delete;
  NAMatMulSmallMKernelDescriptor(
      simd::ushort3 blockDimensions,
      GEMMOperandPrecisions memoryPrecisions,
      uint16_t pack,
      uint16_t executionSIMDGroups,
      bool useBias,
      bool loadM) noexcept;

  bool operator==(const NAMatMulSmallMKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAMatMulSmallMKernelDescriptor>
{
  std::size_t operator()(const NAMatMulSmallMKernelDescriptor& hash) const noexcept;
};

#endif
