#ifndef MFA_RMSNORMGATEDKERNEL_HPP_
#define MFA_RMSNORMGATEDKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "RMSNormGatedDescriptor.hpp"

struct RMSNormGatedKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size groupSize;

  float epsilon;

  GEMMOperandPrecision aPrecision;

  GEMMOperandPrecision gatePrecision;

  GEMMOperandPrecision scalePrecision;

  uint32_t columnCount;

  RMSNormGatedKernel(RMSNormGatedKernelDescriptor descriptor, MTL::Device* const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
