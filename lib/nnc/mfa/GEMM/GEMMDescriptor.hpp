#ifndef GEMMDescriptor_hpp
#define GEMMDescriptor_hpp

#include "GEMMKernelDescriptor.hpp"
#include <optional>
#include <simd/simd.h>

struct GEMMDescriptor {
  /// The number of equally sized multiplications that run in parallel.
  /// Batching is out of scope for the reference implementation. However, there
  /// should be a guide for clients that wish to modify the shader, in ways
  /// that increase the compute workload. For example, by batching the
  /// multiplication of (sub)matrices located at arbitrary pointers in memory
  /// (with potentially nonuniform stride or noncontiguous padding).
  int64_t batchDimension = 1;
  
  /// The dimensions of the input and output matrices.
  /// - Parameter M: Number of output columns.
  /// - Parameter N: Number of output rows.
  /// - Parameter K: Number of loop iterations for the dot products.
  ///
  /// For all practical purposes, one can assume matrix dimensions are 32-bit.
  /// I use this quite often in other code. The pointers themselves are 64-bit,
  /// but the offsets between different elements are 32-bit. With 4-byte words,
  /// this scheme could access up to 16 GB of memory - larger than any array
  /// in any reasonable application. Handling larger allocations likely
  /// requires consideration of more failure points than just integer
  /// overflows.
  std::optional<simd::uint3> matrixDimensions;
  
  std::optional<GEMMOperandPrecisions> memoryPrecisions;
  
  std::optional<simd::uchar2> transposeState;
};

#endif /* GEMMDescriptor_hpp */

