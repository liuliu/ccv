#ifndef SwishKernel_hpp
#define SwishKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "SwishDescriptor.hpp"

struct SwishKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  MTL::Size threadgroupSize;

  uint8_t gradient;

  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  SwishKernel(SwishKernelDescriptor descriptor, MTL::Device *const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
  std::string createErf() const noexcept;
};

#endif /* SwishKernel_hpp */

