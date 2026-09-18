#ifndef MFA_SIGNEDSQRTDESCRIPTOR_HPP_
#define MFA_SIGNEDSQRTDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct SignedSqrtKernelDescriptor {
  bool gradient;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  bool operator==(const SignedSqrtKernelDescriptor& rhs) const { return gradient == rhs.gradient && value == rhs.value && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<SignedSqrtKernelDescriptor> {
  std::size_t operator()(const SignedSqrtKernelDescriptor& descriptor) const noexcept {
    return std::hash<int>()((int)descriptor.value | ((int)descriptor.gradient << 8) | ((int)descriptor.memoryPrecision.value << 16) | ((int)descriptor.loadM << 24));
  }
};

struct SignedSqrtKernel;

// Length and the floor specialize the pipeline, not the kernel source.
struct SignedSqrtDescriptor {
  bool gradient;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  uint32_t length;
  float minimum_magnitude;

  bool operator==(const SignedSqrtDescriptor& rhs) const;
  std::pair<SignedSqrtKernelDescriptor, PipelineValue<SignedSqrtKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SignedSqrtKernelDescriptor, std::unique_ptr<SignedSqrtKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<SignedSqrtDescriptor> {
  std::size_t operator()(const SignedSqrtDescriptor& descriptor) const noexcept;
};

#endif
