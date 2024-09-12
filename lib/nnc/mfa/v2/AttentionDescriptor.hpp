#ifndef MFA_ATTENTIONDESCRIPTOR_HPP_
#define MFA_ATTENTIONDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct AttentionKernelDescriptor;
struct AttentionKernel;

struct AttentionDescriptor {
  /// Q, K, V, dO
  bool lowPrecisionInputs;

  /// S, P, L, D, dP, dS
  bool lowPrecisionIntermediates;
  
  /// row:    Output sequence length; rows of the attention matrix.
  /// column: Input sequence length; columns of the attention matrix.
  /// head:   Head dimension, typically 32 - 256.
  simd::uint3 matrixDimensions;

  /// Q, K, V, O
  simd::uchar4 transposeState;

  bool operator==(const AttentionDescriptor& rhs) const;

  // std::pair<AttentionKernelDescriptor, PipelineValue<AttentionKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, std::unordered_map<AttentionKernelDescriptor, std::unique_ptr<AttentionKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<AttentionDescriptor>
{
  std::size_t operator()(const AttentionDescriptor& hash) const noexcept;
};

#endif

