#ifndef MFA_ARGMAXKERNEL_HPP_
#define MFA_ARGMAXKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ArgmaxDescriptor.hpp"
#include <string>

struct ArgmaxKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  MTL::Size groupSize = MTL::Size(256, 1, 1);
  GEMMOperandPrecision memoryPrecision;

  ArgmaxKernel(ArgmaxKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
};

#endif
