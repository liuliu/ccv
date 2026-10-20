#ifndef MFA_TANHKERNEL_HPP_
#define MFA_TANHKERNEL_HPP_

#include "TanhDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <string>

struct TanhKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  MTL::Size threadgroupSize;

  bool gradient;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;

  TanhKernel(TanhKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
