#ifndef SwishMulKernel_hpp
#define SwishMulKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "SwishMulDescriptor.hpp"

struct SwishMulKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size threadgroupSize;

  uint8_t gradient;

  uint8_t outputMask;

  uint8_t value;

  float beta;

  float scale;

  GEMMOperandPrecision gPrecision;

  GEMMOperandPrecision aPrecision;

  GEMMOperandPrecision bPrecision;

  GEMMOperandPrecision daPrecision;

  GEMMOperandPrecision dbPrecision;

  SwishMulKernel(SwishMulKernelDescriptor descriptor, MTL::Device* const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
