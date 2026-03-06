#ifndef DepalettizeKernel_hpp
#define DepalettizeKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "DepalettizeDescriptor.hpp"

struct DepalettizeKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size threadgroupSize;

  uint8_t qbits;

  uint8_t partial;

  GEMMOperandPrecision memoryPrecision;

  DepalettizeKernel(DepalettizeKernelDescriptor descriptor, MTL::Device* const device);

  MTL::Size gridSize(uint32_t length, uint32_t numberInBlocks) const noexcept;

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
