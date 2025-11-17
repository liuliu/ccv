#ifndef MFA_NAMATMULDESCRIPTOR_HPP_
#define MFA_NAMATMULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct NAMatMulKernelDescriptor;
struct NAMatMulKernel;

struct NAMatMulDescriptor {
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

  std::optional<simd::uint4> batchStrides;
  
  simd::uchar3 transposeState;

  bool useBias;

  // Whether to use M as the major dispatch axis (x-axis). When M is large, this is beneficial.
  bool dispatchMMajor;

  /// Whether load M from a buffer.
  bool loadM;

  /// Whether the compiled pipeline will support indirect command buffers.
  bool supportIndirectCommandBuffers;

  bool operator==(const NAMatMulDescriptor& rhs) const;

  uint16_t splitK() const noexcept;

  std::pair<NAMatMulKernelDescriptor, PipelineValue<NAMatMulKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAMatMulKernelDescriptor, std::unique_ptr<NAMatMulKernel>> *const libraryCache) const noexcept;

  static bool preferDispatchMMajor(const uint32_t M, const uint32_t N, const uint32_t K) noexcept;
};

template<>
struct std::hash<NAMatMulDescriptor>
{
  std::size_t operator()(const NAMatMulDescriptor& hash) const noexcept;
};

#endif

