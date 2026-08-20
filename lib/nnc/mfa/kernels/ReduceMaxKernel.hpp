#ifndef MFA_REDUCEMAXKERNEL_HPP_
#define MFA_REDUCEMAXKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ReduceMaxDescriptor.hpp"
#include <string>

struct ReduceMaxKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  GEMMOperandPrecision memoryPrecision;

  ReduceMaxKernel(ReduceMaxKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
};

#endif
