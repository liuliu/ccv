#ifndef NAInt8AttentionKernel_hpp
#define NAInt8AttentionKernel_hpp

#include "NAInt8AttentionKernelDescriptor.hpp"
#include "AttentionKernelType.hpp"
#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct NAInt8AttentionKernel {
  static constexpr uint16_t qQuantizeThreads = 128;
  static constexpr uint16_t kvQuantizeThreads = 256;
  static constexpr uint16_t blockMaskThreads = 256;
  static constexpr uint16_t smallSequenceVMeanThreads = 256;
  static constexpr uint16_t largeSequenceVMeanThreads = 128;
  // KV rows summed by one mean threadgroup. Longer sequences add a fixed-order
  // second pass over the per-chunk sums.
  static constexpr uint32_t vMeanChunkRows = 512;
  static constexpr uint16_t vMeanFinalizeThreads = 64;
  static constexpr uint16_t computeDThreads = 32;

  NS::SharedPtr<MTL::Library> library;
  std::string source;

  simd::ushort3 blockDimensions;
  AttentionKernelType type;
  unsigned short headDimension;
  unsigned short Hq;
  unsigned short Hk;
  uint16_t qScaleTileSize;
  uint16_t kvScaleTileSize;
  uint16_t executionSIMDGroups;
  uint16_t vMeanThreads;
  bool hasCRemainder;
  uint16_t threadBarrierEveryC;
  GEMMOperandPrecision ioPrecision;
  // Output channels owned by each threadgroup; QK still reduces over the full head.
  unsigned short outputTileSize;
  bool mortonTraversal;
  bool lowPrecisionIntermediates;
  float scale;
  bool isCausal;
  bool masked;
  bool isVarlen;
  bool hasCausalEmptyRows;
  bool loadR = false;
  bool hasRRemainder = true;
  bool loadC = false;
  bool attentionSinks;
  bool qkHadamard;
  bool qkMeanCorrection;

  NAInt8AttentionKernel(NAInt8AttentionKernelDescriptor descriptor, MTL::Device *const device);

  uint16_t vMeanThreadgroupSize() const noexcept;
  uint32_t vMeanChunks(uint32_t sequenceLength) const noexcept;
  // Device scratch for per-chunk sums; zero when one chunk covers the sequence.
  size_t vMeanPartialBytes(uint32_t batchDimension, uint32_t sequenceLength) const noexcept;
  // Encodes compute_v_mean and, for multiple chunks, finalize_v_mean. The caller
  // binds V (0), V mean (1), K (2) and K mean (3) for Hadamard, and any varlen or
  // runtime-dimension buffers. Partial sums bind at index 4.
  void encodeVMean(MTL::ComputeCommandEncoder* encoder,
                   MTL::ComputePipelineState* reducePipeline,
                   MTL::ComputePipelineState* finalizePipeline,
                   MTL::Buffer* partials, size_t partialsOffset,
                   uint32_t batchDimension, uint32_t sequenceLength) const noexcept;

  uint32_t threadgroupMemoryAllocation() const noexcept;
  uint16_t threadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept;
  MTL::Size threadgroupsPerGrid(uint32_t batchDimension, uint32_t rowDimension) const noexcept;

private:
  void createVMean(CodeWriter& source) const noexcept;
  std::string createSource() const noexcept;
  void createConstants(CodeWriter& source) const noexcept;
  std::string createRuntimeConstants(bool quantize) const noexcept;
  std::string createBufferBindings() const noexcept;
  std::string createAdjustOffsets() const noexcept;
  std::string createComputeD() const noexcept;
  void loopForward(CodeWriter& source) const noexcept;
  void loopBackwardQuery(CodeWriter& source) const noexcept;
  void loopBackwardKeyValue(CodeWriter& source) const noexcept;
};

#endif
