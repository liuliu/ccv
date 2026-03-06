#ifndef SigmoidKernel_hpp
#define SigmoidKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "SigmoidDescriptor.hpp"

struct SigmoidKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size threadgroupSize;

  uint8_t gradient;

  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  SigmoidKernel(SigmoidKernelDescriptor descriptor, MTL::Device* const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
