#ifndef MFA_SIGNEDSQRTKERNEL_HPP_
#define MFA_SIGNEDSQRTKERNEL_HPP_

#include "SignedSqrtDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <string>

struct SignedSqrtKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  MTL::Size threadgroupSize;

  bool gradient;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;

  SignedSqrtKernel(SignedSqrtKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
