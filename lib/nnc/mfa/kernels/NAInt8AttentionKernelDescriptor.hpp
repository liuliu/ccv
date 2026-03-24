#ifndef NAInt8AttentionKernelDescriptor_hpp
#define NAInt8AttentionKernelDescriptor_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "GEMMOperandPrecision.hpp"
#include <simd/simd.h>

enum class NAInt8AttentionKernelMode : uint16_t {
  full = 0,
  qk_only = 1,
  pv_only = 2,
  qk_pv_raw = 3,
  softmax_stats = 4,
};

struct NAInt8AttentionKernelDescriptor {
  simd::ushort3 blockDimensions;
  unsigned short headDimension;
  unsigned short Hq;
  unsigned short Hk;
  uint16_t executionSIMDGroups;
  bool checkCEdge1;
  bool useInt8QK;
  bool useQKScales;
  bool threadBarrierOverC;
  bool mortonOrder;
  GEMMOperandPrecision ioPrecision;
  NAInt8AttentionKernelMode mode;
  float scale;

  NAInt8AttentionKernelDescriptor() = delete;
  NAInt8AttentionKernelDescriptor(
      simd::ushort3 blockDimensions,
      unsigned short headDimension,
      unsigned short Hq,
      unsigned short Hk,
      uint16_t executionSIMDGroups,
      bool checkCEdge1,
      bool useInt8QK,
      bool useQKScales,
      bool threadBarrierOverC,
      bool mortonOrder,
      GEMMOperandPrecision ioPrecision,
      NAInt8AttentionKernelMode mode,
      float scale) noexcept;

  bool operator==(const NAInt8AttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAInt8AttentionKernelDescriptor>
{
  std::size_t operator()(const NAInt8AttentionKernelDescriptor& hash) const noexcept;
};

#endif
