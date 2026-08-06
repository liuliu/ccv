#ifndef MFA_NASPARSEINDEXEDATTENTIONKERNEL_HPP_
#define MFA_NASPARSEINDEXEDATTENTIONKERNEL_HPP_

#include <string>
#include "NASparseIndexedAttentionKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

class CodeWriter;

struct NASparseIndexedAttentionKernel {
  static constexpr uint16_t headDimension = 512;
  static constexpr uint16_t simdGroupSize = 32;
  static constexpr uint16_t headGroup = 32;
  static constexpr uint16_t denseBlockColumns = 64;
  static constexpr uint16_t denseExecutionSIMDGroups = 8;
  static constexpr uint16_t threadgroupRowBlock = 32;
  static constexpr uint16_t threadgroupHeadDimensionD128 = 128;
  static constexpr uint16_t threadgroupRowBlockD128 = 64;
  static constexpr uint16_t dimBlock = 128;

  GEMMOperandPrecision memoryPrecision;
  bool attentionSinks;
  bool denseOnly;
  bool loadRows;
  NASparseIndexedAttentionVariant variant;
  std::string source;
  NS::SharedPtr<MTL::Library> library;

  uint32_t threadgroupMemoryAllocation() const noexcept;
  MTL::Size threadgroupSize() const noexcept;
  MTL::Size threadgroupsPerGrid(uint32_t T, uint32_t H) const noexcept;

  NASparseIndexedAttentionKernel(NASparseIndexedAttentionKernelDescriptor descriptor, MTL::Device *const device);

private:
  uint16_t sparseHeadGroup() const noexcept;
  uint16_t sparseExecutionSIMDGroups() const noexcept;
  std::string createSource() const noexcept;
  std::string createDenseOnlySource() const noexcept;
  std::string createThreadgroupSource() const noexcept;
  std::string createThreadgroupD128Source() const noexcept;
  void createThreadgroupAttendBlock(CodeWriter& source) const noexcept;
  void createThreadgroupD128AttendBlock(CodeWriter& source) const noexcept;
};

#endif
