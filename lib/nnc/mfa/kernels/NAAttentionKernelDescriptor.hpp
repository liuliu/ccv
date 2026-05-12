#ifndef NAAttentionKernelDescriptor_hpp
#define NAAttentionKernelDescriptor_hpp

#include "GEMMOperandPrecision.hpp"
#include "AttentionOperand.hpp"
#include "AttentionKernelType.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct NAAttentionDescriptor;

/// A configuration for a Attention kernel.
struct NAAttentionKernelDescriptor {
  /// parallelization, traversal, head
  simd::ushort3 blockDimensions;

  /// Required. The problem size along the head dimension.
  unsigned short headDimension;

  unsigned short Hq;

  unsigned short Hk;

  uint16_t executionSIMDGroups;

  bool checkCEdge1;

  bool bypassThreadgroupMemory;

  bool isCausal;

  bool masked;

  bool isVarlen;

  bool loadC;

  // Selected split count. A value <= 1 disables splitKV.
  uint16_t splitKV;

  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;

  AttentionKernelType type;

  float scale;

  // MARK: - Functionality from AttentionDescriptor
  
  NAAttentionKernelDescriptor() = delete;
  
  /// Initialize the kernel descriptor.
  NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale) noexcept;
  NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale, bool bypassThreadgroupMemory) noexcept;
  NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale, bool bypassThreadgroupMemory, bool isCausal, bool masked) noexcept;
  NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale, bool bypassThreadgroupMemory, bool isCausal, bool masked, bool isVarlen, uint16_t splitKV = 1, bool loadC = false) noexcept;

  bool operator==(const NAAttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAAttentionKernelDescriptor>
{
  std::size_t operator()(const NAAttentionKernelDescriptor& hash) const noexcept;
};

#endif /* NAAttentionKernelDescriptor_hpp */
