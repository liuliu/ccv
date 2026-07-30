#ifndef MFA_SPARSEINDEXEDATTENTIONR1KERNEL_HPP_
#define MFA_SPARSEINDEXEDATTENTIONR1KERNEL_HPP_

#include <string>

#include "SparseIndexedAttentionR1Descriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

struct SparseIndexedAttentionR1Kernel {
  static constexpr uint16_t maxHeadDimension = 512;

  GEMMOperandPrecision memoryPrecision;

  bool loadK;

  bool attentionSinks;

  std::string source;

  NS::SharedPtr<MTL::Library> library;

  SparseIndexedAttentionR1Kernel(
      SparseIndexedAttentionR1KernelDescriptor descriptor,
      MTL::Device* const device);

  uint32_t threadgroupMemoryAllocation(
      const SparseIndexedAttentionR1Descriptor& descriptor) const noexcept;

  uint32_t threadgroupSize(
      const SparseIndexedAttentionR1Descriptor& descriptor) const noexcept;

private:
  std::string createSource() const noexcept;

  std::string createConstants() const noexcept;
};

#endif
