#ifndef AttentionKernelDescriptor_hpp
#define AttentionKernelDescriptor_hpp

#include "GEMMOperandPrecision.hpp"
#include "AttentionOperand.hpp"
#include "AttentionKernelType.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct AttentionDescriptor;

/// A configuration for a Attention kernel.
struct AttentionKernelDescriptor {
  /// parallelization, traversal, head
  simd::ushort3 blockDimensions;

  /// Whether each operand is cached in registers.
  AttentionOperands<bool> cacheState;

  /// Required. The problem size along the head dimension.
  unsigned short headDimension;

  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;

  /// Reads with a one-to-one mapping to threads (like GEMM store) and writes.
  bool preferAsyncCache;

  /// Reads that are shared among threads (like GEMM load).
  bool preferAsyncLoad;

  AttentionOperands<GEMMOperandPrecision> registerPrecisions;

  /// Whether each operand is transposed in RAM.
  ///
  /// If the layout is row-major, where a row spans D contiguous elements in
  /// memory, enter `false`. If the layout is column-major, where a row spans
  /// D widely separated elements in memory, enter `true`.
  ///
  /// The transpose state of a derivative (e.g. dQ for Q) must match the
  /// corresponding input from the forward pass.
  ///
  /// > NOTE: To implement multi-head attention, clients may need to modify
  /// the stride of matrix elements in memory. If and only if the transpose
  /// state is `false`, change the stride from `D` to `D * H`. Ensure the
  /// value of H is known at compile time, so the product `D * H` can be
  /// embedded into the GPU assembly code.
  AttentionOperands<bool> transposeState;

  /// The leading dimensions after transposed if applied.
  AttentionOperands<bool> leadingDimensions;

  AttentionKernelType type;

  // MARK: - Functionality from AttentionDescriptor
  
  AttentionKernelDescriptor() = delete;
  
  /// Initialize the kernel descriptor.
  AttentionKernelDescriptor(simd::ushort3 blockDimensions, AttentionOperands<bool> cacheState, unsigned short headDimension, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, bool preferAsyncCache, bool preferAsyncLoad, AttentionOperands<GEMMOperandPrecision> registerPrecisions, AttentionOperands<bool> transposeState, AttentionOperands<bool> leadingDimensions, AttentionKernelType type) noexcept;

  bool operator==(const AttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<AttentionKernelDescriptor>
{
  std::size_t operator()(const AttentionKernelDescriptor& hash) const noexcept;
};

#endif /* AttentionKernelDescriptor_hpp */
