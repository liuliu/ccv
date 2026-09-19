#ifndef MFA_SCALEDDOTPRODUCTARGPARTITIONKERNEL_HPP_
#define MFA_SCALEDDOTPRODUCTARGPARTITIONKERNEL_HPP_

#include <string>
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ScaledDotProductArgPartitionKernelDescriptor.hpp"

struct ScaledDotProductArgPartitionKernel {
  GEMMOperandPrecision memoryPrecision;
  uint32_t kth;
  uint8_t scoreMode = 0; // 0: dense, 1: candidate rows, 2: block maxima, 3: block bitset, 4: dense with index utilities.
  uint16_t scoreBlockM;
  uint16_t scoreBlockN;
  uint16_t scoreSIMDGroups;
  bool loadC;
  bool loadM = false;
  MTL::Size scoreThreadgroupSize;
  MTL::Size topKThreadgroupSize;
  MTL::Size topKTileThreadgroupSize;
  MTL::Size topKMergeThreadgroupSize;
  std::string source;
  NS::SharedPtr<MTL::Library> library;

  ScaledDotProductArgPartitionKernel(ScaledDotProductArgPartitionKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string createSource() const noexcept;
};

#endif
