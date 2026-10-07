#ifndef NAInt8AttentionKernelDescriptor_hpp
#define NAInt8AttentionKernelDescriptor_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "GEMMOperandPrecision.hpp"
#include "AttentionKernelType.hpp"
#include <simd/simd.h>

struct NAInt8AttentionDescriptor;
struct DeviceProperties;

struct NAInt8AttentionKernelDescriptor {
  simd::ushort3 blockDimensions;
  unsigned short headDimension;
  unsigned short Hq;
  unsigned short Hk;
  uint16_t qScaleTileSize;
  uint16_t kvScaleTileSize;
  uint16_t executionSIMDGroups;
  uint16_t vMeanThreads;
  bool hasCRemainder;
  uint16_t threadBarrierEveryC;
  GEMMOperandPrecision ioPrecision;
  bool lowPrecisionIntermediates;
  AttentionKernelType type;
  float scale;
  bool isCausal;
  bool masked;
  bool isVarlen;
  bool hasCausalEmptyRows;
  bool loadR = false;
  bool hasRRemainder = true;
  bool loadC = false;
  bool attentionSinks;
  // Output channels owned by each threadgroup; QK still reduces over the full head.
  unsigned short outputTileSize;
  // Otherwise visit all row groups for one head before the next head.
  bool mortonTraversal = false;

  NAInt8AttentionKernelDescriptor() = delete;
  NAInt8AttentionKernelDescriptor(const NAInt8AttentionDescriptor& descriptor,
      const DeviceProperties& dprops) noexcept;
  NAInt8AttentionKernelDescriptor(
      simd::ushort3 blockDimensions,
      unsigned short headDimension,
      unsigned short Hq,
      unsigned short Hk,
      uint16_t qScaleTileSize,
      uint16_t kvScaleTileSize,
      uint16_t executionSIMDGroups,
      uint16_t vMeanThreads,
      bool hasCRemainder,
      uint16_t threadBarrierEveryC,
      GEMMOperandPrecision ioPrecision,
      bool lowPrecisionIntermediates,
      AttentionKernelType type,
      float scale,
      bool isCausal,
      bool masked,
      bool hasCausalEmptyRows,
      bool isVarlen,
      bool mortonTraversal,
      bool attentionSinks = false) noexcept;

  bool operator==(const NAInt8AttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAInt8AttentionKernelDescriptor>
{
  std::size_t operator()(const NAInt8AttentionKernelDescriptor& hash) const noexcept;
};

#endif
