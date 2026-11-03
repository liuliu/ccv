#ifndef MFA_INT8SWIGLUKERNEL_HPP_
#define MFA_INT8SWIGLUKERNEL_HPP_

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

constexpr uint32_t kInt8SwiGLURowsPerThreadgroup = 2;
constexpr uint32_t kInt8SwiGLUSIMDGroupsPerThreadgroup = 4;
// Q5_K and Q6_K weights give each half SIMD group a row of its own.
constexpr uint32_t kInt8SwiGLUPackedRowsPerThreadgroup = kInt8SwiGLUSIMDGroupsPerThreadgroup * 2;

struct Int8SwiGLUKernelDescriptor {
  bool clamp;
  GEMMOperandPrecision memoryPrecision;
  uint32_t activationRows;
  uint32_t format;

  constexpr bool operator==(const Int8SwiGLUKernelDescriptor& rhs) const
  {
    return clamp == rhs.clamp && memoryPrecision == rhs.memoryPrecision &&
      activationRows == rhs.activationRows && format == rhs.format;
  }
};

template<>
struct std::hash<Int8SwiGLUKernelDescriptor>
{
  std::size_t operator()(const Int8SwiGLUKernelDescriptor& value) const noexcept
  {
    return std::hash<uint64_t>()(
      (uint64_t)value.memoryPrecision.value | ((uint64_t)value.clamp << 8) |
      ((uint64_t)value.activationRows << 16) | ((uint64_t)value.format << 32));
  }
};

struct Int8SwiGLUKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;

  Int8SwiGLUKernel(Int8SwiGLUKernelDescriptor descriptor, MTL::Device* device);
};

#endif
