#ifndef MFA_SEGMENTEDINT8GEMVKERNEL_HPP_
#define MFA_SEGMENTEDINT8GEMVKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "SegmentedInt8GemvDescriptor.hpp"

struct SegmentedInt8GemvKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  uint8_t fusedBias;

  uint8_t mrows;

  uint32_t format;

  GEMMOperandPrecision memoryPrecision;

  SegmentedInt8GemvKernel(
    SegmentedInt8GemvKernelDescriptor descriptor,
    MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
