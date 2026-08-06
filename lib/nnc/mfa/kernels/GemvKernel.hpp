#ifndef GemvKernel_hpp
#define GemvKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "GemvDescriptor.hpp"

struct GemvKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  uint8_t fusedBias;

  uint8_t mrows;

  uint8_t batched;

  GEMMOperandPrecision memoryPrecision;

  GemvKernel(GemvKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
