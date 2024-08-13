#ifndef GEMMKernelDescriptor_hpp
#define GEMMKernelDescriptor_hpp

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct GEMMDescriptor;

/// A configuration for a GEMM kernel.
///
/// The information in this data structure is enough to uniquely identify the
/// kernel. It can be used as a key in a key-value cache.
///
/// ## Usage
///
/// The code for generating the GEMM kernel does not include any assumptions
/// about performance. It should only be responsible for correctly generating
/// a shader source, provided a configuration. The user is responsible for
/// choosing that configuration.
struct GEMMKernelDescriptor {
  /// Required. The number of matrix elements spanned by each threadgroup.
  /// - Parameter M: Number of output columns spanned.
  /// - Parameter N: Number of output rows spanned.
  /// - Parameter K: Number of loop iterations unrolled.
  ///
  /// Optimal values:
  /// - Apple7 and Apple8: 48x48x24
  /// - Apple9 and later: 32x32x8
  ///
  /// To reach optimal performance on Apple7 and Apple8, the recommended default
  /// value needs to be modified conditionally. When all three operands have
  /// 16-bit memory precisions, change `K` to 32. When the matrix is too small
  /// to saturate all of the GPU cores, change all dimensions to 32x32x32. Even
  /// smaller blocks can be exploited in low-occupancy cases, but 32x32 and
  /// 48x48 are sufficient for general use.
  ///
  /// For simplicity or an out-of-the-box performance test, one can assume
  /// occupancy is always high. But to match the performance of MPS, one must
  /// optimize for small problem sizes on large GPUs.
  ///
  /// ## Choosing Block Size by Precision
  ///
  /// Legend:
  /// - memA: precision for left input matrix, in memory
  /// - memB: precision for right input matrix, in memory
  /// - memC: precision for output matrix, in memory
  /// - regA: precision for left input matrix, in registers
  /// - regB: precision for right input matrix, in registers
  /// - regC: precision for output matrix, in registers
  /// - M1: optimal block size on Apple7 and Apple8
  /// - M3: optimal block size on Apple9 and later
  ///
  /// memA | memB | memC | regA | regB | regC | M1       | M3      |
  /// ---- | ---- | ---- | ---- | ---- | ---- | -------- | ------- |
  /// FP16 | FP16 | FP16 | any  | any  | any  | 48x48x32 | 32x32x8 |
  /// BF16 | BF16 | BF16 | any  | any  | any  | 48x48x32 | 32x32x8 |
  /// FP16 | FP16 | FP32 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// BF16 | BF16 | FP32 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// FP16 | FP32 | FP16 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// BF16 | FP32 | BF16 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// FP32 | FP32 | FP32 | any  | any  | any  | 48x48x24 | 32x32x8 |
  ///
  /// ## Detecting Low-Occupancy Cases
  ///
  /// To determine whether the matrix saturates the GPU, divide the output
  /// matrix's dimensions by 48x48. Round up to the nearest integer. Then,
  /// multiply the number of row blocks by the number of column blocks. The
  /// result is the number of threadgroups dispatched. For example, a C matrix
  /// with dimensions 768x768 would dispatch 256 threadgroups. If you are
  /// batching multiple matrix multiplications into one shader call, multiply
  /// the number of threadgroups by the batch count.
  ///
  /// Next, calculate the target occupancy. Start by finding the GPU core count.
  /// This can be accomplished in many ways; there is a heavily tested reference
  /// implementation [here](https://github.com/philipturner/applegpuinfo). On
  /// macOS, you can query the core count through IORegistry. On iOS, go with a
  /// conservative (meaning more likely to overestimate) estimate of 5 cores on
  /// A14 - A16, 10 cores on M1 - M2.
  ///
  /// When one of the operands is 32-bit, the target occupancy is 6 threadgroups
  /// per core. When all three operands are 16-bit, the target increases to 9
  /// per core. Multiply the number of cores by the number of threadgroups per
  /// core. If the total GPU occupancy is greater than or equal to the number of
  /// matrix blocks, use the smaller blocking scheme.
  ///
  /// For example, the following decision tree would be used on an M1 Max
  /// (32 cores).
  ///
  /// ```
  /// is device Apple9 or later?
  /// yes: use block size 32x32x8
  /// no: continue decision tree [selected decision]
  /// unsure: use block size 48x48x24-32
  ///
  /// compute number of matrix blocks
  /// 768x768 / 48x48 = 16.0 x 16.0
  ///   round floating point (16.0 x 16.0)
  ///   to next greatest integer (16 x 16)
  ///  16 x 16 x (batch size of 1) = 256 threadgroups
  ///
  /// compute target occupancies with 48x48 scheme
  /// 32 x 6 = 192 [selected when A, B, or C is FP32]
  /// 32 x 9 = 288 [selected when every matrix is FP16/BF16]
  ///
  /// prefer 32x32 when 48x48 has low occupancy
  /// if 256 ≤ 192
  ///    choose small block size (32x32x32xFP32)
  /// else
  ///    choose large block size (48x48x24xFP32) [selected]
  /// if 256 ≤ 288
  ///   choose small block size (32x32x32xFP16) [selected]
  /// else
  ///   choose large block size (48x48x32xFP16)
  /// ```
  ///
  /// ## C++ Adaptation
  ///
  /// Mapping from the Swift implementation:
  /// - M -> blockDimensions[0]
  /// - N -> blockDimensions[1]
  /// - K -> blockDimensions[2]
  simd::ushort3 blockDimensions;
  
  GEMMOperandPrecisions memoryPrecisions;
  
  /// Optional. The layout of elements in threadgroup memory.
  ///
  /// If not specified, the default value matches the actual block dimensions.
  ///
  /// This property can be used to avoid bank conflicts. For example, of one
  /// operand will have 16 FP32 elements per row, there is good chance of
  /// increased bank conflicts on M1. One may pad that threadgroup memory
  /// allocation to 20 FP32 elements per row.
  ///
  /// Note that the assignment of M/N/K to row dimensions varies based on which
  /// operand is discussed, and what its transpose state is.
  ///
  /// ## C++ Adaptation
  ///
  /// Mapping from the Swift implementation:
  /// - A.M -> paddedBlockDimensions[0]
  /// - A.K -> paddedBlockDimensions[1]
  /// - B.K -> paddedBlockDimensions[2]
  /// - B.N -> paddedBlockDimensions[3]
  /// - C.M -> paddedBlockDimensions[4]
  /// - C.N -> paddedBlockDimensions[5]
  std::optional<simd::ushort8> paddedBlockDimensions;
  
  /// Required. Whether async copies will improve performance during the
  /// matrix multiplication loop.
  ///
  /// The default value is `true`. Async copies improve performance on Apple7
  /// and Apple8, but harm performance on Apple9 and later. However, they are
  /// essential for correctness when reading from the edges of unaligned
  /// matrices. Setting the value to `false` means skipping async copies when
  /// doing so will not change the final result.
  bool preferAsyncLoad;
  
  /// Required. Whether async copies will improve performance when storing the
  /// accumulator to main memory.
  ///
  /// There is no default value that will reliably yield consistent performance.
  bool preferAsyncStore;
  
  /// Set the register precision based on the GPU architecture, and your choice
  /// for memory precision. The following set of logic statements should provide
  /// optimal performance for all permutations of operand precisions.
  ///
  /// ```
  /// regA is identical to memA
  /// regB is identical to memB
  /// If memA, memB, and memC are FP16,
  ///   regC is FP16
  /// else
  ///   regC is FP32
  ///
  /// If earlier than M3
  ///   If memA is BF16,
  ///     regA is FP32
  ///   If memB is BF16,
  ///     regB is FP32
  /// ```
  GEMMOperandPrecisions registerPrecisions;
  
  /// Required. The array of SIMDs to divide the threadgroup into.
  ///
  /// Optimal values:
  /// - Apple7 and Apple8: 2x2
  /// - Apple9 and later: 1x1
  ///
  /// ## C++ Adaptation
  ///
  /// Mapping from the Swift implementation:
  /// - M -> splits[0]
  /// - N -> splits[1]
  simd::ushort2 splits;
  
  /// Required. Whether each of the inputs deviates from row-major order.
  ///
  /// ## C++ Adaptation
  ///
  /// Mapping from the Swift implementation:
  /// - A -> transposeState[0]
  /// - B -> transposeState[1]
  /// - bias -> transposeState[2]
  simd::uchar3 transposeState;

  /// Required. Whether it contains the bias.
  bool useBias;
  
  // MARK: - Functionality from GEMMDescriptor
  
  GEMMKernelDescriptor() = delete;
  
  /// Initialize the kernel descriptor.
  GEMMKernelDescriptor(simd::ushort3 blockDimensions, GEMMOperandPrecisions memoryPrecisions, std::optional<simd::ushort8> paddedBlockDimensions, bool preferAsyncLoad, bool preferAsyncStore, GEMMOperandPrecisions registerPrecisions, simd::ushort2 splits, simd::uchar3 transposeState, bool useBias) noexcept;
  
  /// Implementation of the block size selection heuristic.
  ///
  /// This function initializes the 'blockDimensions' and
  /// 'paddedBlockDimensions' properties.
  static std::pair<simd::ushort3, std::optional<simd::ushort8>> getBlockDimensions(MTL::Device* const mtlDevice, const uint32_t coreCount, const simd::uint3 matrixDimensions, const int64_t batchDimension, const GEMMOperandPrecisions memoryPrecisions, const simd::uchar3 transposeState) noexcept;

  bool operator==(const GEMMKernelDescriptor& rhs) const;
};

template<>
struct std::hash<GEMMKernelDescriptor>
{
  std::size_t operator()(const GEMMKernelDescriptor& hash) const noexcept;
};

#endif /* GEMMKernelDescriptor_hpp */
