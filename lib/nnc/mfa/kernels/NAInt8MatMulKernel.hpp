#ifndef NAInt8MatMulKernel_hpp
#define NAInt8MatMulKernel_hpp

#include "NAInt8MatMulKernelDescriptor.hpp"
#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct NAInt8MatMulKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;

  simd::ushort3 blockDimensions;
  uint16_t executionSIMDGroups;
  GEMMOperandPrecision ioPrecision;
  bool useBias;
  uint16_t activationQuantizeThreads;
  uint32_t groupM;
  uint32_t groupN;

  NAInt8MatMulKernel(NAInt8MatMulKernelDescriptor descriptor, MTL::Device *const device);

  uint16_t threadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept;
  MTL::Size threadgroupsPerGrid(uint32_t M, uint32_t N) const noexcept;

private:
  std::string createSource() const noexcept;
};

#endif
