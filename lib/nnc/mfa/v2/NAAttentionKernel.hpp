#ifndef NAAttentionKernel_hpp
#define NAAttentionKernel_hpp

#include "NAAttentionKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct NAAttentionKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;

  AttentionKernelType type;

  float scale;

  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;

  /// parallelization, traversal, head
  simd::ushort3 blockDimensions;

  unsigned short headDimension;

  unsigned short Hq;

  unsigned short Hk;

  uint16_t executionSIMDGroups;

  bool bypassThreadgroupMemory;

  bool checkCEdge1;

  unsigned short threadgroupMemoryAllocation(MTL::ComputePipelineState *const pipelineState, const NAAttentionDescriptor &descriptor) const noexcept;

  /// The number of threads per group.
  uint16_t threadgroupSize(MTL::ComputePipelineState *const pipelineState, const NAAttentionDescriptor &descriptor) const noexcept;

  MTL::Size threadgroupsPerGrid(const NAAttentionDescriptor &descriptor) const noexcept;

  NAAttentionKernel(NAAttentionKernelDescriptor descriptor, MTL::Device *const device);

private:
  /// AttentionKernel.
  std::string memoryName(AttentionOperand operand) const noexcept;
  std::string sequenceLength(AttentionOperand operand) const noexcept;
  unsigned short blockSequenceLength(AttentionOperand operand) const noexcept;

  std::string operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept;

  /// AttentionKernel+Source
  std::string createSource() const noexcept;
  void createConstants(CodeWriter &source) const noexcept;
  void loopForward(CodeWriter &source) const noexcept;
  std::string createAdjustOffsets() const noexcept;
  std::string createBufferBindings() const noexcept;
};

#endif /* NAAttentionKernel_hpp */

