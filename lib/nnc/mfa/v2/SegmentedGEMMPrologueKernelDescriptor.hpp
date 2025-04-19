#ifndef SEGMENTEDGEMMPROLOGUEKernelDescriptor_hpp
#define SEGMENTEDGEMMPROLOGUEKernelDescriptor_hpp

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct SegmentedGEMMPrologueDescriptor;

/// A configuration for a SegmentedGEMMPrologue kernel.
struct SegmentedGEMMPrologueKernelDescriptor {
  GEMMOperandPrecisions memoryPrecisions;
  /// Required. Whether it contains the bias.
  bool useBias;
  
  // MARK: - Functionality from SegmentedGEMMPrologueDescriptor
  
  SegmentedGEMMPrologueKernelDescriptor() = delete;
  
  /// Initialize the kernel descriptor.
  SegmentedGEMMPrologueKernelDescriptor(GEMMOperandPrecisions memoryPrecisions, bool useBias) noexcept;
  
  bool operator==(const SegmentedGEMMPrologueKernelDescriptor& rhs) const;
};

template<>
struct std::hash<SegmentedGEMMPrologueKernelDescriptor>
{
  std::size_t operator()(const SegmentedGEMMPrologueKernelDescriptor& hash) const noexcept;
};

#endif /* SegmentedGEMMPrologueKernelDescriptor_hpp */
