#ifndef GeluKernel_hpp
#define GeluKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "GeluDescriptor.hpp"

struct GeluKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  MTL::Size threadgroupSize;

  uint8_t gradient;

  uint8_t tanh;

  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  GeluKernel(GeluKernelDescriptor descriptor, MTL::Device *const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
  std::string createErf() const noexcept;
};

#endif /* GeluKernel_hpp */

