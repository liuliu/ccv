#ifndef MFA_SPARSEINDEXEDATTENTIONKERNEL_HPP_
#define MFA_SPARSEINDEXEDATTENTIONKERNEL_HPP_

#include <string>
#include "SparseIndexedAttentionKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

struct SparseIndexedAttentionKernel {
  static constexpr uint16_t simdGroupSize = 32;
  static constexpr uint16_t simdGroupsPerThreadgroup = 8;
  static constexpr uint16_t threadsPerThreadgroup = simdGroupSize * simdGroupsPerThreadgroup;
  static constexpr uint16_t maxHeadDimension = 512;
  static constexpr uint16_t valuesPerLane = (maxHeadDimension + simdGroupSize - 1) / simdGroupSize;
  static constexpr uint16_t rowsPerBlock = 4;

  GEMMOperandPrecision memoryPrecision;
  bool attentionSinks;
  bool loadRows;
  std::string source;
  NS::SharedPtr<MTL::Library> library;

  uint32_t threadgroupMemoryAllocation() const noexcept;
  MTL::Size threadgroupSize() const noexcept;
  MTL::Size threadgroupsPerGrid(uint32_t T, uint32_t H) const noexcept;

  SparseIndexedAttentionKernel(SparseIndexedAttentionKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string createSource() const noexcept;
};

#endif
