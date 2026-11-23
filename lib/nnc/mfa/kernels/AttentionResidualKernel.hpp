#ifndef MFA_ATTENTIONRESIDUALKERNEL_HPP_
#define MFA_ATTENTIONRESIDUALKERNEL_HPP_

#include "AttentionResidualDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <string>

struct AttentionResidualKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  MTL::Size threadgroupSize;

  uint32_t queryCount;
  GEMMOperandPrecision memoryPrecision;

  AttentionResidualKernel(AttentionResidualKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
