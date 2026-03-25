#ifndef NAInt8AttentionKernelDescriptor_hpp
#define NAInt8AttentionKernelDescriptor_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "GEMMOperandPrecision.hpp"
#include <simd/simd.h>

struct NAInt8AttentionKernelDescriptor {
  simd::ushort3 blockDimensions;
  unsigned short headDimension;
  unsigned short Hq;
  unsigned short Hk;
  uint16_t executionSIMDGroups;
  bool hasCRemainder;
  bool threadBarrierOverC;
  GEMMOperandPrecision ioPrecision;
  float scale;

  NAInt8AttentionKernelDescriptor() = delete;
  NAInt8AttentionKernelDescriptor(
      simd::ushort3 blockDimensions,
      unsigned short headDimension,
      unsigned short Hq,
      unsigned short Hk,
      uint16_t executionSIMDGroups,
      bool hasCRemainder,
      bool threadBarrierOverC,
      GEMMOperandPrecision ioPrecision,
      float scale) noexcept;

  bool operator==(const NAInt8AttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAInt8AttentionKernelDescriptor>
{
  std::size_t operator()(const NAInt8AttentionKernelDescriptor& hash) const noexcept;
};

#endif
