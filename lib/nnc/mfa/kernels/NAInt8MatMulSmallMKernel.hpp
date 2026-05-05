#ifndef MFA_NAINT8MATMULSMALLMKERNEL_HPP_
#define MFA_NAINT8MATMULSMALLMKERNEL_HPP_

#include "NAInt8MatMulSmallMKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <string>

struct NAInt8MatMulSmallMDescriptor;

struct NAInt8MatMulSmallMKernel {
  simd::ushort3 blockDimensions;
  uint16_t pack;
  uint16_t executionSIMDGroups;
  GEMMOperandPrecision ioPrecision;
  bool useBias;
  std::string source;
  NS::SharedPtr<MTL::Library> library;

  uint16_t threadgroupSize(MTL::ComputePipelineState* const pipelineState) const noexcept;
  MTL::Size threadgroupsPerGrid(const NAInt8MatMulSmallMDescriptor& descriptor) const noexcept;

  NAInt8MatMulSmallMKernel(NAInt8MatMulSmallMKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
};

#endif
