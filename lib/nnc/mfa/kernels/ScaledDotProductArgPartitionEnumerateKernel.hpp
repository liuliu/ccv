#ifndef MFA_SCALEDDOTPRODUCTARGPARTITIONENUMERATEKERNEL_HPP_
#define MFA_SCALEDDOTPRODUCTARGPARTITIONENUMERATEKERNEL_HPP_

#include <cstdint>
#include <string>
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ScaledDotProductArgPartitionEnumerateKernelDescriptor.hpp"

struct ScaledDotProductArgPartitionEnumerateKernel {
  MTL::Size threadgroupSize;
  std::string source;
  NS::SharedPtr<MTL::Library> library;

  ScaledDotProductArgPartitionEnumerateKernel(ScaledDotProductArgPartitionEnumerateKernelDescriptor descriptor, MTL::Device* const device);

  MTL::Size gridSize(uint32_t T, uint32_t kth) const noexcept;

private:
  std::string createSource() const noexcept;
};

#endif
