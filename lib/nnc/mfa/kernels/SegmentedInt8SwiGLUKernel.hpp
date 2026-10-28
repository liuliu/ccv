#ifndef MFA_SEGMENTEDINT8SWIGLUKERNEL_HPP_
#define MFA_SEGMENTEDINT8SWIGLUKERNEL_HPP_

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

struct SegmentedInt8SwiGLUKernelDescriptor {
  uint32_t format;
  bool clamp;
  GEMMOperandPrecision memoryPrecision;
  // Also evaluate a shared expert's int8 SwiGLU in the same dispatch.
  bool sharedFused = false;

  constexpr bool operator==(const SegmentedInt8SwiGLUKernelDescriptor& rhs) const
  {
    return format == rhs.format && clamp == rhs.clamp &&
      memoryPrecision == rhs.memoryPrecision && sharedFused == rhs.sharedFused;
  }
};

template<>
struct std::hash<SegmentedInt8SwiGLUKernelDescriptor>
{
  std::size_t operator()(const SegmentedInt8SwiGLUKernelDescriptor& value) const noexcept
  {
    return std::hash<uint64_t>()(
      (uint64_t)value.format | ((uint64_t)value.memoryPrecision.value << 32) |
      ((uint64_t)value.clamp << 40) | ((uint64_t)value.sharedFused << 41));
  }
};

struct SegmentedInt8SwiGLUKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;

  SegmentedInt8SwiGLUKernel(
    SegmentedInt8SwiGLUKernelDescriptor descriptor,
    MTL::Device* device);
};

#endif
