#ifndef Conv3DKernelDescriptor_hpp
#define Conv3DKernelDescriptor_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct Conv3DDescriptor;

struct Conv3DKernelDescriptor {
  simd::ushort2 blockDimensions;
  simd::ushort3 kernelDimensions;
  uint64_t dataType;
  uint32_t inputChannels;
  uint32_t outputChannels;
  uint32_t paddingLeft;
  uint32_t paddingRight;
  uint32_t paddingTop;
  uint32_t paddingBottom;
  bool useBias;

  Conv3DKernelDescriptor() = delete;
  Conv3DKernelDescriptor(simd::ushort2 blockDimensions, simd::ushort3 kernelDimensions, uint64_t dataType, uint32_t inputChannels, uint32_t outputChannels, uint32_t paddingLeft, uint32_t paddingRight, uint32_t paddingTop, uint32_t paddingBottom, bool useBias) noexcept;

  bool operator==(const Conv3DKernelDescriptor& rhs) const;
};

template<>
struct std::hash<Conv3DKernelDescriptor>
{
  std::size_t operator()(const Conv3DKernelDescriptor& hash) const noexcept;
};

#endif
