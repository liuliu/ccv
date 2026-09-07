#ifndef MulKernel_hpp
#define MulKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "MulDescriptor.hpp"

struct MulKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  uint8_t value;

  bool loadM;

  GEMMOperandPrecision memoryPrecision;

  MulKernel(MulKernelDescriptor descriptor, MTL::Device *const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif /* MulKernel_hpp */
