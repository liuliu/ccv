#ifndef NAConv3DKernel_hpp
#define NAConv3DKernel_hpp

#include "NAConv3DKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

struct NAConv3DDescriptor;

struct NAConv3DKernel {
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::ComputePipelineState> permutationPipeline;

  std::string source;

  simd::ushort2 blockDimensions;
  simd::ushort3 kernelDimensions;
  uint64_t dataType;
  uint32_t inputChannels;
  uint32_t outputChannels;
  uint32_t paddingLeft;
  uint32_t paddingRight;
  uint32_t paddingTop;
  uint32_t paddingBottom;
  bool useBias;
  uint16_t executionSIMDGroups;

  uint16_t permutationThreadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept;

  uint16_t threadgroupSize(MTL::ComputePipelineState *const pipelineState, const NAConv3DDescriptor &descriptor) const noexcept;

  MTL::Size threadgroupsPerGrid(const NAConv3DDescriptor &descriptor) const noexcept;

  NAConv3DKernel(NAConv3DKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string createSource() const noexcept;
};

#endif
