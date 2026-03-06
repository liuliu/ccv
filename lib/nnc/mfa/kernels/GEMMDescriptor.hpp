#ifndef MFA_GEMMDESCRIPTOR_HPP_
#define MFA_GEMMDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct GEMMKernelDescriptor;
struct GEMMKernel;

struct GEMMDescriptor {
  /// The number of equally sized multiplications that run in parallel.
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
  simd::uint3 matrixDimensions;
  
  GEMMOperandPrecisions memoryPrecisions;

  std::optional<GEMMOperandPrecision> registerPrecisionC;

  std::optional<simd::uint3> leadingDimensions;

  std::optional<simd::uint4> batchStrides;
  
  simd::uchar3 transposeState;

  bool loadPreviousC;

  bool useBias;

  /// Whether load M from a buffer.
  bool loadM;

  /// Whether the compiled pipeline will support indirect command buffers.
  bool supportIndirectCommandBuffers;

  bool operator==(const GEMMDescriptor& rhs) const;

  std::pair<GEMMKernelDescriptor, PipelineValue<GEMMKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GEMMKernelDescriptor, std::unique_ptr<GEMMKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<GEMMDescriptor>
{
  std::size_t operator()(const GEMMDescriptor& hash) const noexcept;
};

#endif

