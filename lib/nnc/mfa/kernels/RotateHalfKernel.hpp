#ifndef RotateHalfKernel_hpp
#define RotateHalfKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "RotateHalfDescriptor.hpp"

struct RotateHalfKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size threadgroupSize;

  uint8_t value;

  bool loadM;

  GEMMOperandPrecision memoryPrecision;

  RotateHalfKernel(RotateHalfKernelDescriptor descriptor, MTL::Device *const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
