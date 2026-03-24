#ifndef NAInt8MatMulKernelDescriptor_hpp
#define NAInt8MatMulKernelDescriptor_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct NAInt8MatMulKernelDescriptor {
  simd::ushort3 blockDimensions;
  uint16_t executionSIMDGroups;
  bool outputFloat;
  uint32_t groupM;
  uint32_t groupN;

  NAInt8MatMulKernelDescriptor() = delete;
  NAInt8MatMulKernelDescriptor(
      simd::ushort3 blockDimensions,
      uint16_t executionSIMDGroups,
      bool outputFloat,
      uint32_t groupM,
      uint32_t groupN) noexcept;

  bool operator==(const NAInt8MatMulKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAInt8MatMulKernelDescriptor>
{
  std::size_t operator()(const NAInt8MatMulKernelDescriptor& hash) const noexcept;
};

#endif
