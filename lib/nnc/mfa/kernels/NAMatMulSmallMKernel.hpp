#ifndef MFA_NAMATMULSMALLMKERNEL_HPP_
#define MFA_NAMATMULSMALLMKERNEL_HPP_

#include "NAMatMulSmallMKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include <string>

struct NAMatMulSmallMDescriptor;

struct NAMatMulSmallMKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;

  simd::ushort3 blockDimensions;
  GEMMOperandPrecisions memoryPrecisions;
  uint16_t pack;
  uint16_t executionSIMDGroups;
  bool useBias;

  uint16_t threadgroupSize(MTL::ComputePipelineState* const pipelineState) const noexcept;
  MTL::Size threadgroupsPerGrid(const NAMatMulSmallMDescriptor& descriptor) const noexcept;

  NAMatMulSmallMKernel(NAMatMulSmallMKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string memoryName(char operand) const noexcept;
  std::string createSource() const noexcept;
};

#endif
