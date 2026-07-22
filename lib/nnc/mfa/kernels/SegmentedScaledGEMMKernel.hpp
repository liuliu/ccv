#ifndef SEGMENTEDSCALEDGEMMKERNEL_HPP
#define SEGMENTEDSCALEDGEMMKERNEL_HPP

#include <string>
#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct SegmentedScaledGEMMKernelDescriptor;

struct SegmentedScaledGEMMKernel {
  simd::ushort3 blockDimensions;
  uint16_t executionSIMDGroups;
  GEMMOperandPrecision ioPrecision;
  bool useBias;
  bool loadM;

  NS::SharedPtr<MTL::Library> library;
  std::string source;

  SegmentedScaledGEMMKernel(SegmentedScaledGEMMKernelDescriptor descriptor, MTL::Device* const device);

  uint16_t threadgroupSize(MTL::ComputePipelineState* const pipelineState) const noexcept;
  uint32_t maxTileRecords(uint32_t originalM, uint32_t segments) const noexcept;

  std::string createSource() const noexcept;
};

#endif
