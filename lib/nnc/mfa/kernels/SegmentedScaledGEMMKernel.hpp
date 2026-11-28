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
  uint32_t format = 0;

  NS::SharedPtr<MTL::Library> library;
  std::string source;

  SegmentedScaledGEMMKernel(SegmentedScaledGEMMKernelDescriptor descriptor, MTL::Device* const device);

  uint16_t threadgroupSize(MTL::ComputePipelineState* const pipelineState) const noexcept;
  uint32_t maxTileRecords(uint32_t originalM, uint32_t bincount) const noexcept;

  // Whether packed weights in this format should be decoded per tile rather than expanded into scratch, for N x K
  // weights per expert: TMCQ's register tiles (32 x 64 x 64) need N % 64 and K % 64, every other packed format
  // N % 256 and K % 256. Other shapes expand the selected experts to int8 rows instead.
  static bool fusesPackedFormat(uint32_t format, uint64_t M, uint32_t binCount, uint32_t N, uint32_t K) noexcept;

  std::string createSource() const noexcept;
};

#endif
