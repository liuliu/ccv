#ifndef MFA_SEGMENTEDINT8SWIGLUKERNEL_HPP_
#define MFA_SEGMENTEDINT8SWIGLUKERNEL_HPP_

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

// Segmented int8 SwiGLU threadgroups are four SIMD groups (the fused shared-expert rows assume it
// too), except TMCQ without a fused shared expert. A TMCQ threadgroup covers one 16-row tile row and
// splits K eight ways: one split per SIMD group over eight SIMD groups, or two per SIMD group over
// the four of a dispatch with a fused shared expert, with the same summation order.
constexpr uint32_t kSegmentedInt8SwiGLUTMCQSIMDGroups = 8;
constexpr uint32_t kSegmentedInt8SwiGLUTMCQSharedSIMDGroups = 4;

// SIMD groups per threadgroup of a TMCQ segmented SwiGLU dispatch.
constexpr uint32_t segmentedInt8SwiGLUTMCQSIMDGroups(const bool sharedFused)
{
  return sharedFused ? kSegmentedInt8SwiGLUTMCQSharedSIMDGroups : kSegmentedInt8SwiGLUTMCQSIMDGroups;
}

struct SegmentedInt8SwiGLUKernelDescriptor {
  uint32_t format;
  bool clamp;
  GEMMOperandPrecision memoryPrecision;
  // Also evaluate a shared expert's int8 SwiGLU in the same dispatch, with Q8_0 (0), Q5_K or
  // Q6_K weights.
  bool sharedFused = false;
  uint32_t sharedFormat = 0;

  constexpr bool operator==(const SegmentedInt8SwiGLUKernelDescriptor& rhs) const
  {
    return format == rhs.format && clamp == rhs.clamp &&
      memoryPrecision == rhs.memoryPrecision && sharedFused == rhs.sharedFused &&
      sharedFormat == rhs.sharedFormat;
  }
};

template<>
struct std::hash<SegmentedInt8SwiGLUKernelDescriptor>
{
  std::size_t operator()(const SegmentedInt8SwiGLUKernelDescriptor& value) const noexcept
  {
    return std::hash<uint64_t>()(
      (uint64_t)value.format | ((uint64_t)value.memoryPrecision.value << 32) |
      ((uint64_t)value.clamp << 40) | ((uint64_t)value.sharedFused << 41) |
      ((uint64_t)value.sharedFormat << 48));
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
