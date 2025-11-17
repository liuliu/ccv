#ifndef MFA_SEGMENTEDGEMMPROLOGUEDESCRIPTOR_HPP_
#define MFA_SEGMENTEDGEMMPROLOGUEDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SegmentedGEMMPrologueKernelDescriptor;
struct SegmentedGEMMPrologueKernel;

struct SegmentedGEMMPrologueDescriptor {
  /// The dimensions of the input and output matrices.
  /// - Parameter M: Number of output columns (ignored).
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

  simd::ushort3 blockDimensions;
  
  GEMMOperandPrecisions memoryPrecisions;

  bool useBias;

  /// The number of threads per group for the GEMM kernel.
  uint16_t threadgroupSize;

  /// The threadgroup memory for the GEMM kernel.
  uint32_t threadgroupMemoryAllocation;

  /// Whether to dispatch with M or N for the GEMM kernel.
  bool dispatchMMajor;

  /// Whether this GEMM kernel has a separate splitK reduction kernel.
  uint16_t splitK;

  bool operator==(const SegmentedGEMMPrologueDescriptor& rhs) const;

  std::pair<SegmentedGEMMPrologueKernelDescriptor, PipelineValue<SegmentedGEMMPrologueKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SegmentedGEMMPrologueKernelDescriptor, std::unique_ptr<SegmentedGEMMPrologueKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<SegmentedGEMMPrologueDescriptor>
{
  std::size_t operator()(const SegmentedGEMMPrologueDescriptor& hash) const noexcept;
};

#endif

