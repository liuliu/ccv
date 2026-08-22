#ifndef MFA_REDUCELOGSUMEXPKERNEL_HPP_
#define MFA_REDUCELOGSUMEXPKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ReduceLogSumExpDescriptor.hpp"
#include <string>

struct ReduceLogSumExpKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  GEMMOperandPrecision memoryPrecision;

  ReduceLogSumExpKernel(ReduceLogSumExpKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
};

#endif
