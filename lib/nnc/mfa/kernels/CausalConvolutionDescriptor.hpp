#ifndef MFA_CAUSALCONVOLUTIONDESCRIPTOR_HPP_
#define MFA_CAUSALCONVOLUTIONDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct CausalConvolutionKernelDescriptor {
  bool silu;
  GEMMOperandPrecision memoryPrecision;
  bool operator==(const CausalConvolutionKernelDescriptor& rhs) const { return silu == rhs.silu && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<CausalConvolutionKernelDescriptor> {
  std::size_t operator()(const CausalConvolutionKernelDescriptor& descriptor) const noexcept {
    return std::hash<int>()((int)descriptor.silu | ((int)descriptor.memoryPrecision.value << 8));
  }
};

struct CausalConvolutionKernel;

// Sizes are bound at encode time, so one pipeline serves every shape.
struct CausalConvolutionDescriptor {
  bool silu;
  GEMMOperandPrecision memoryPrecision;

  bool operator==(const CausalConvolutionDescriptor& rhs) const;
  std::pair<CausalConvolutionKernelDescriptor, PipelineValue<CausalConvolutionKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CausalConvolutionKernelDescriptor, std::unique_ptr<CausalConvolutionKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<CausalConvolutionDescriptor> {
  std::size_t operator()(const CausalConvolutionDescriptor& descriptor) const noexcept;
};

#endif
