#ifndef SEGMENTEDSCALEDGEMMKERNELDESCRIPTOR_HPP
#define SEGMENTEDSCALEDGEMMKERNELDESCRIPTOR_HPP

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct SegmentedScaledGEMMKernelDescriptor {
  simd::ushort3 blockDimensions;
  uint16_t executionSIMDGroups;
  GEMMOperandPrecision ioPrecision;
  bool useBias;

  SegmentedScaledGEMMKernelDescriptor() = delete;
  SegmentedScaledGEMMKernelDescriptor(
      simd::ushort3 blockDimensions,
      uint16_t executionSIMDGroups,
      GEMMOperandPrecision ioPrecision,
      bool useBias) noexcept;

  bool operator==(const SegmentedScaledGEMMKernelDescriptor& rhs) const;
};

template<>
struct std::hash<SegmentedScaledGEMMKernelDescriptor>
{
  std::size_t operator()(const SegmentedScaledGEMMKernelDescriptor& hash) const noexcept;
};

#endif
