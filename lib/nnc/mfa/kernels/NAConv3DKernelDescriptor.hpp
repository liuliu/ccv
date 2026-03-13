#ifndef NAConv3DKernelDescriptor_hpp
#define NAConv3DKernelDescriptor_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct NAConv3DDescriptor;

struct NAConv3DKernelDescriptor {
  simd::ushort2 blockDimensions;
  simd::ushort3 kernelDimensions;
  uint64_t dataType;
  uint32_t inputChannels;
  uint32_t outputChannels;
  bool useBias;

  NAConv3DKernelDescriptor() = delete;
  NAConv3DKernelDescriptor(simd::ushort2 blockDimensions, simd::ushort3 kernelDimensions, uint64_t dataType, uint32_t inputChannels, uint32_t outputChannels, bool useBias) noexcept;

  bool operator==(const NAConv3DKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAConv3DKernelDescriptor>
{
  std::size_t operator()(const NAConv3DKernelDescriptor& hash) const noexcept;
};

#endif
