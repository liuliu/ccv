#ifndef NAInt8AttentionKernel_hpp
#define NAInt8AttentionKernel_hpp

#include "NAInt8AttentionKernelDescriptor.hpp"
#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct NAInt8AttentionKernel {
  static constexpr uint16_t qQuantizeThreads = 128;
  static constexpr uint16_t kvQuantizeThreads = 256;
  // Debug note: keep the production kernel full-only and Morton-ordered.
  // If stripped debug modes are needed again, reintroduce them in the bench
  // harness and source generator together instead of widening this surface.

  NS::SharedPtr<MTL::Library> library;
  std::string source;

  simd::ushort3 blockDimensions;
  unsigned short headDimension;
  unsigned short Hq;
  unsigned short Hk;
  uint16_t executionSIMDGroups;
  bool checkCEdge1;
  bool useInt8QK;
  bool useQKScales;
  bool threadBarrierOverC;
  GEMMOperandPrecision ioPrecision;
  float scale;

  NAInt8AttentionKernel(NAInt8AttentionKernelDescriptor descriptor, MTL::Device *const device);

  unsigned short threadgroupMemoryAllocation() const noexcept;
  uint16_t threadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept;
  MTL::Size threadgroupsPerGrid(uint32_t batchDimension, uint32_t rowDimension) const noexcept;

private:
  std::string createSource() const noexcept;
  void createConstants(CodeWriter& source) const noexcept;
  std::string createBufferBindings() const noexcept;
  std::string createAdjustOffsets() const noexcept;
  void loopForward(CodeWriter& source) const noexcept;
};

#endif
