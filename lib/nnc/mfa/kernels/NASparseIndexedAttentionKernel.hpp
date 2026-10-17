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
  // Threadgroup64TwoPass: both passes multiply groups of 64 heads, which must tile
  // H, with 4 cooperating SIMD groups. A token chunk's probabilities are capped at
  // twoPassScratchBytes, and a chunk should give every GPU core this many
  // workgroups in the first pass.
  static constexpr uint16_t twoPassHeadGroup = 64;
  static constexpr uint16_t twoPassSIMDGroups = 4;
  static constexpr uint64_t twoPassScratchBytes = 256ull << 20;
  static constexpr uint32_t twoPassMinimumTokensPerCore = 12;

  GEMMOperandPrecision memoryPrecision;
  bool attentionSinks;
  bool denseOnly;
  bool loadRows;
  bool loadM = false;
  bool loadK = false;
  NASparseIndexedAttentionVariant variant;
  std::string source;
  NS::SharedPtr<MTL::Library> library;

  uint32_t threadgroupMemoryAllocation() const noexcept;
  MTL::Size threadgroupSize() const noexcept;
  MTL::Size threadgroupsPerGrid(uint32_t T, uint32_t H) const noexcept;

  // The two-pass values kernel computes one dimBlock-wide output slice per workgroup.
  uint32_t valuesThreadgroupMemoryAllocation() const noexcept;

  MTL::Size valuesThreadgroupSize() const noexcept;

  MTL::Size valuesThreadgroupsPerGrid(uint32_t T, uint32_t H) const noexcept;

  NASparseIndexedAttentionKernel(NASparseIndexedAttentionKernelDescriptor descriptor, MTL::Device *const device);

private:
  uint16_t sparseHeadGroup() const noexcept;
  uint16_t sparseExecutionSIMDGroups() const noexcept;
  std::string createSource() const noexcept;
  std::string createDenseOnlySource() const noexcept;
  std::string createThreadgroupSource() const noexcept;
  std::string createThreadgroupD128Source() const noexcept;
  std::string createThreadgroupTwoPassSource() const noexcept;
  std::string sparsePreamble() const noexcept;
  static std::string probabilityPeak() noexcept;
  static std::string causalDenseRange() noexcept;
  static std::string twoPassScores(const char* row) noexcept;
  void createThreadgroupAttendBlock(CodeWriter& source) const noexcept;
  void createThreadgroupD128AttendBlock(CodeWriter& source) const noexcept;
};

#endif
