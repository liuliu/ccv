#ifndef SEGMENTEDSCALEDGEMMPROLOGUEKERNELDESCRIPTOR_HPP
#define SEGMENTEDSCALEDGEMMPROLOGUEKERNELDESCRIPTOR_HPP

#include "GEMMOperandPrecision.hpp"

struct SegmentedScaledGEMMPrologueKernelDescriptor {
  GEMMOperandPrecision ioPrecision;
  bool useBias;

  SegmentedScaledGEMMPrologueKernelDescriptor() = delete;
  SegmentedScaledGEMMPrologueKernelDescriptor(GEMMOperandPrecision ioPrecision, bool useBias) noexcept;

  bool operator==(const SegmentedScaledGEMMPrologueKernelDescriptor& rhs) const;
};

template<>
struct std::hash<SegmentedScaledGEMMPrologueKernelDescriptor>
{
  std::size_t operator()(const SegmentedScaledGEMMPrologueKernelDescriptor& hash) const noexcept;
};

#endif
