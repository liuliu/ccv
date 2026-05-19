#ifndef WalshHadamardTransformKernel_hpp
#define WalshHadamardTransformKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "WalshHadamardTransformDescriptor.hpp"

struct WalshHadamardTransformKernel {
  NS::SharedPtr<MTL::Library> library;

  GEMMOperandPrecision memoryPrecision;

  WalshHadamardTransformKernel(WalshHadamardTransformKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
