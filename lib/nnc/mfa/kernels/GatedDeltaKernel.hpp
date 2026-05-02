#ifndef GatedDeltaKernel_hpp
#define GatedDeltaKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "GatedDeltaDescriptor.hpp"
#include <simd/simd.h>

struct GatedDeltaKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size threadgroupSize;

  uint8_t stateElementsPerLane;

  GatedDeltaKernel(GatedDeltaKernelDescriptor descriptor, MTL::Device* const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
};

#endif
