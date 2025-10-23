#ifndef NAMatMulKernel_hpp
#define NAMatMulKernel_hpp

#include "NAMatMulKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct NAMatMulKernel {
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

  GEMMOperandPrecisions memoryPrecisions;

  GEMMOperandPrecisions registerPrecisions;

  uint16_t splitK;

  uint16_t executionSIMDGroups;

  simd::uchar3 transposeState;

  bool useBias;

  bool loadM;

  /// The number of threads per group.
  uint16_t threadgroupSize(MTL::ComputePipelineState *const pipelineState, const NAMatMulDescriptor &descriptor) const noexcept;

  MTL::Size threadgroupsPerGrid(const NAMatMulDescriptor &descriptor) const noexcept;

  NAMatMulKernel(NAMatMulKernelDescriptor descriptor, MTL::Device *const device);

private:
  bool transposed(char operand) const noexcept;

  std::string memoryName(char operand) const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
  void createInitializeC(CodeWriter *source) const noexcept;
};

#endif /* NAMatMulKernel_hpp */

