#ifndef MFA_NASCALEDDOTPRODUCTARGPARTITIONKERNEL_HPP_
#define MFA_NASCALEDDOTPRODUCTARGPARTITIONKERNEL_HPP_

#include <string>
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "NAScaledDotProductArgPartitionKernelDescriptor.hpp"

struct NAScaledDotProductArgPartitionKernel {
  GEMMOperandPrecision memoryPrecision;
  uint32_t kth;
  uint16_t scoreBlockM;
  uint16_t scoreBlockN;
  uint16_t scoreSIMDGroups;
  bool loadC;
  bool isCausal;
  MTL::Size scoreThreadgroupSize;
  MTL::Size topKThreadgroupSize;
  MTL::Size topKTileThreadgroupSize;
  MTL::Size topKMergeThreadgroupSize;
  std::string source;
  NS::SharedPtr<MTL::Library> library;

  NAScaledDotProductArgPartitionKernel(NAScaledDotProductArgPartitionKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string createSource() const noexcept;
};

#endif
