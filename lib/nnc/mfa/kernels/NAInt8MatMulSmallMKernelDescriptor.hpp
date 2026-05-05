#ifndef MFA_NAINT8MATMULSMALLMKERNELDESCRIPTOR_HPP_
#define MFA_NAINT8MATMULSMALLMKERNELDESCRIPTOR_HPP_

#include "GEMMOperandPrecision.hpp"
#include <cstdint>
#include <functional>
#include <simd/simd.h>

struct NAInt8MatMulSmallMKernelDescriptor {
  simd::ushort3 blockDimensions;
  uint16_t pack;
  uint16_t executionSIMDGroups;
  GEMMOperandPrecision ioPrecision;
  bool useBias;

  NAInt8MatMulSmallMKernelDescriptor() = delete;
  NAInt8MatMulSmallMKernelDescriptor(
      simd::ushort3 blockDimensions,
      uint16_t pack,
      uint16_t executionSIMDGroups,
      GEMMOperandPrecision ioPrecision,
      bool useBias) noexcept;

  bool operator==(const NAInt8MatMulSmallMKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAInt8MatMulSmallMKernelDescriptor>
{
  std::size_t operator()(const NAInt8MatMulSmallMKernelDescriptor& hash) const noexcept;
};

#endif
