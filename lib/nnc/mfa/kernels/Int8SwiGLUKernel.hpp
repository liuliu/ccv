#ifndef MFA_INT8SWIGLUKERNEL_HPP_
#define MFA_INT8SWIGLUKERNEL_HPP_

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

constexpr uint32_t kInt8SwiGLURowsPerThreadgroup = 2;
constexpr uint32_t kInt8SwiGLUSIMDGroupsPerThreadgroup = 4;

struct Int8SwiGLUKernelDescriptor {
  bool clamp;
  GEMMOperandPrecision memoryPrecision;

  constexpr bool operator==(const Int8SwiGLUKernelDescriptor& rhs) const
  {
    return clamp == rhs.clamp && memoryPrecision == rhs.memoryPrecision;
  }
};

template<>
struct std::hash<Int8SwiGLUKernelDescriptor>
{
  std::size_t operator()(const Int8SwiGLUKernelDescriptor& value) const noexcept
  {
    return std::hash<uint32_t>()(
      (uint32_t)value.memoryPrecision.value | ((uint32_t)value.clamp << 8));
  }
};

struct Int8SwiGLUKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;

  Int8SwiGLUKernel(Int8SwiGLUKernelDescriptor descriptor, MTL::Device* device);
};

#endif
