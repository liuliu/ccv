#ifndef MFA_NAATTENTIONDESCRIPTOR_HPP_
#define MFA_NAATTENTIONDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "AttentionKernelType.hpp"
#include "AttentionOperand.hpp"

struct NAAttentionKernelDescriptor;
struct NAAttentionKernel;

struct NAAttentionDescriptor {
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

  AttentionOperands<unsigned int> batchStrides;

  AttentionKernelType type;

  float scale;

  bool operator==(const NAAttentionDescriptor& rhs) const;

  std::pair<NAAttentionKernelDescriptor, PipelineValue<NAAttentionKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAAttentionKernelDescriptor, std::unique_ptr<NAAttentionKernel>> *const libraryCache) const noexcept;

private:
  NAAttentionKernelDescriptor kernelDescriptor(MTL::Device *const device, const DeviceProperties &dprops) const noexcept;
  AttentionOperands<GEMMOperandPrecision> createMemoryPrecisions() const noexcept;
};

template<>
struct std::hash<NAAttentionDescriptor>
{
  std::size_t operator()(const NAAttentionDescriptor& hash) const noexcept;
};

#endif

