#ifndef MFA_CAUSALCONVOLUTIONKERNEL_HPP_
#define MFA_CAUSALCONVOLUTIONKERNEL_HPP_

#include "CausalConvolutionDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <string>

struct CausalConvolutionKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  MTL::Size threadgroupSize;

  bool silu;
  GEMMOperandPrecision memoryPrecision;

  CausalConvolutionKernel(CausalConvolutionKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
};

#endif
