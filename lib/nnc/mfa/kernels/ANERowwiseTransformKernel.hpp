#ifndef ANERowwiseTransformKernel_hpp
#define ANERowwiseTransformKernel_hpp

#include "ANERowwiseTransformKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct ANERowwiseTransformKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;

  GEMMOperandPrecision memoryPrecision;
  uint16_t activationScaleThreads;
  uint16_t quantTileDimension;
  uint16_t quantBlockRows;
  uint16_t quantTilePad;
  simd::ushort2 outputTileDimensions;

  ANERowwiseTransformKernel(ANERowwiseTransformKernelDescriptor descriptor, MTL::Device* const device);

  MTL::Size activationScaleThreadgroupSize() const noexcept;
  MTL::Size activationQuantizeThreadgroupSize() const noexcept;
  MTL::Size outputDequantizeThreadgroupSize() const noexcept;
  MTL::Size activationScaleGridSize(uint32_t paddedM) const noexcept;
  MTL::Size activationQuantizeGridSize(uint32_t paddedM, uint32_t K) const noexcept;
  MTL::Size outputDequantizeGridSize(uint32_t paddedM, uint32_t N) const noexcept;

private:
  std::string createSource() const noexcept;
};

#endif
