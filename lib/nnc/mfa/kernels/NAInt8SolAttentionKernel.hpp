#ifndef MFA_NAINT8SOLATTENTIONKERNEL_HPP_
#define MFA_NAINT8SOLATTENTIONKERNEL_HPP_

#include "NAInt8SolAttentionDescriptor.hpp"

class CodeWriter;

struct NAInt8SolAttentionKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;

  NAInt8SolAttentionKernel(NAInt8SolAttentionKernelDescriptor descriptor, MTL::Device* const device);

  uint32_t threadgroupSize(const NAInt8SolAttentionDescriptor& descriptor) const noexcept;
  MTL::Size threadgroupsPerGrid(const NAInt8SolAttentionDescriptor& descriptor) const noexcept;

  static uint32_t vMeanVectorsPerTile(uint32_t T, uint32_t H) noexcept;
  static uint32_t vMeanThreadgroupSize(uint32_t T, uint32_t H) noexcept;
  static MTL::Size vMeanThreadgroupsPerGrid(uint32_t N, uint32_t T, uint32_t H) noexcept;

private:
  uint32_t blockSize;

  std::string createSource() const noexcept;
  void createConstants(CodeWriter& source) const noexcept;
  void createVMean(CodeWriter& source) const noexcept;
  void createQuantize(CodeWriter& source) const noexcept;
  void createPool(CodeWriter& source) const noexcept;
  void createPrepareSummaries(CodeWriter& source) const noexcept;
  void createRoute(CodeWriter& source) const noexcept;
  void createAttention(CodeWriter& source) const noexcept;
  void loopAttention(CodeWriter& source, bool summary) const noexcept;
  void accumulateAttention(CodeWriter& source, bool summary) const noexcept;
};

#endif
