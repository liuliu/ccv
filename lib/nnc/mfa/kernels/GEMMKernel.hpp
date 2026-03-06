#ifndef GEMMKernel_hpp
#define GEMMKernel_hpp

#include "GEMMKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct GEMMKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  /// A copy of the block dimensions from the descriptor.
  ///
  /// ## C++ Adaptation
  ///
  /// Mapping from the Swift implementation:
  /// - M -> blockDimensions[0]
  /// - N -> blockDimensions[1]
  /// - K -> blockDimensions[2]
  simd::ushort3 blockDimensions;

  /// These properties are copied from GEMMKernelDescriptor for other helper functions to use.
  simd::ushort3 leadingBlockDimensions;

  GEMMOperandPrecisions memoryPrecisions;

  GEMMOperandPrecisions registerPrecisions;

  simd::ushort2 splits;

  simd::uchar3 transposeState;

  bool preferAsyncLoad;

  bool preferAsyncStore;

  bool useBias;

  bool loadM;

  bool disableAsyncCopy;

  uint16_t registerM;

  uint16_t registerN;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  uint16_t threadgroupSize;

  GEMMKernel(GEMMKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string memoryName(char operand) const noexcept;
  std::string registerName(char operand) const noexcept;
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  bool transposed(char operand) const noexcept;
  std::string leadingDimension(char operand) const noexcept;
  unsigned short leadingBlockDimension(char operand) const noexcept;
  unsigned short trailingBlockDimension(char operand) const noexcept;
  unsigned short blockBytes(char operand) const noexcept;

  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
  void createUtilities(CodeWriter *source) const noexcept;
  void createInitializeC(CodeWriter *source) const noexcept;
  void createLoadC(CodeWriter *source) const noexcept;
  void createMultiplyIterations(CodeWriter *source) const noexcept;
  void createStoreC(CodeWriter *source) const noexcept;
  MTL::Library* findPrecompiledLibrary(GEMMKernelDescriptor descriptor, MTL::Device *const device, NS::Error **error) const noexcept;
};

#endif /* GEMMKernel_hpp */

