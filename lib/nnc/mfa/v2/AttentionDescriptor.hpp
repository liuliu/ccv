#ifndef MFA_ATTENTIONDESCRIPTOR_HPP_
#define MFA_ATTENTIONDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "AttentionKernelType.hpp"
#include "AttentionOperand.hpp"

struct AttentionKernelDescriptor;
struct AttentionKernel;

struct AttentionParameterRow {
  unsigned short maximumHeadDimension;
  unsigned short parallelization;
  unsigned short traversal;
  unsigned short head;
  std::vector<AttentionOperand> cachedOperands;
  AttentionParameterRow() = delete;
  AttentionParameterRow(unsigned short maximumHeadDimension, unsigned short parallelization, unsigned short traversal, unsigned short head, std::vector<AttentionOperand> cachedOperands) noexcept : maximumHeadDimension(maximumHeadDimension), parallelization(parallelization), traversal(traversal), head(head), cachedOperands(cachedOperands) {}
};

struct AttentionDescriptor {
  /// The number of equally sized attention per sequence that run in parallel.
  uint32_t batchDimension = 1;

  /// The number of query heads per sequence that run in parallel.
  unsigned short Hq = 1;

  /// The number of key / value heads per sequence that run in parallel.
  unsigned short Hk = 1;

  /// Q, K, V, dO
  bool lowPrecisionInputs;

  /// Similar to flash_attn.
  bool isBF16;

  /// S, P, L, D, dP, dS
  bool lowPrecisionIntermediates;
  
  /// row:    Output sequence length; rows of the attention matrix.
  /// column: Input sequence length; columns of the attention matrix.
  /// head:   Head dimension, typically 32 - 256.
  simd::uint3 matrixDimensions;

  /// Q, K, V, O
  simd::uchar4 transposeState;

  /// The leading dimensions after transposed (if applied).
  /// Q, K, V, O
  std::optional<simd::uint4> leadingDimensions;

  AttentionOperands<unsigned int> batchStrides;

  AttentionKernelType type;

  float scale;

  bool operator==(const AttentionDescriptor& rhs) const;

  std::pair<AttentionKernelDescriptor, PipelineValue<AttentionKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<AttentionKernelDescriptor, std::unique_ptr<AttentionKernel>> *const libraryCache) const noexcept;

private:
  AttentionKernelDescriptor kernelDescriptor(MTL::Device *const device, const DeviceProperties &dprops) const noexcept;
  /// AttentionDescriptor+Precisions
  AttentionOperands<GEMMOperandPrecision> createMemoryPrecisions() const noexcept;
  AttentionOperands<GEMMOperandPrecision> createRegisterPrecisions(MTL::Device *const device) const noexcept;
  /// AttentionDescriptor+Parameters
  std::vector<AttentionParameterRow> parameterFile(AttentionKernelType type, MTL::Device *const device) const noexcept;
  AttentionParameterRow row(const std::vector<AttentionParameterRow>& table) const noexcept;
  std::vector<AttentionParameterRow> defaultParameters(MTL::Device *const device) const noexcept;
  std::vector<AttentionParameterRow> forwardMixed(MTL::Device *const device) const noexcept;
  std::vector<AttentionParameterRow> forward(MTL::Device *const device) const noexcept;
  std::vector<AttentionParameterRow> backwardQueryMixed(MTL::Device *const device) const noexcept;
  std::vector<AttentionParameterRow> backwardQuery(MTL::Device *const device) const noexcept;
  std::vector<AttentionParameterRow> backwardKeyValueMixed(MTL::Device *const device) const noexcept;
  std::vector<AttentionParameterRow> backwardKeyValue(MTL::Device *const device) const noexcept;
};

template<>
struct std::hash<AttentionDescriptor>
{
  std::size_t operator()(const AttentionDescriptor& hash) const noexcept;
};

#endif

