#ifndef MFA_CAUSALCONVOLUTIONDESCRIPTOR_HPP_
#define MFA_CAUSALCONVOLUTIONDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <simd/simd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct CausalConvolutionKernelDescriptor {
  bool silu;
  bool dynamicWeights;
  GEMMOperandPrecision memoryPrecision;
  bool operator==(const CausalConvolutionKernelDescriptor& rhs) const { return silu == rhs.silu && dynamicWeights == rhs.dynamicWeights && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<CausalConvolutionKernelDescriptor> {
  std::size_t operator()(const CausalConvolutionKernelDescriptor& descriptor) const noexcept {
    return std::hash<int>()((int)descriptor.silu | ((int)descriptor.dynamicWeights << 1) | ((int)descriptor.memoryPrecision.value << 8));
  }
};

struct CausalConvolutionKernel;

struct CausalConvolutionDescriptor {
  bool silu;
  // Per-token dynamic weights add to the weights of each channel group.
  bool dynamicWeights;
  GEMMOperandPrecision memoryPrecision;
  // Function constants of the pipeline. The tokens and the batches come from the grid.
  uint32_t channels;
  uint32_t kernelSize;
  uint32_t nextRows;
  uint32_t dynamicWeightGroups;

  bool operator==(const CausalConvolutionDescriptor& rhs) const;
  std::pair<CausalConvolutionKernelDescriptor, PipelineValue<CausalConvolutionKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CausalConvolutionKernelDescriptor, std::unique_ptr<CausalConvolutionKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<CausalConvolutionDescriptor> {
  std::size_t operator()(const CausalConvolutionDescriptor& descriptor) const noexcept;
};

#endif
