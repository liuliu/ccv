#ifndef MFA_ADAMKERNEL_HPP_
#define MFA_ADAMKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "AdamDescriptor.hpp"

struct AdamKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size threadgroupSize;

  uint8_t adamw;

  uint8_t amsgrad;

  GEMMOperandPrecision memoryPrecision;

  AdamKernel(AdamKernelDescriptor descriptor, MTL::Device* const device);

  MTL::Size gridSize(uint32_t length) const noexcept;

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
