#ifndef MFA_TANHDESCRIPTOR_HPP_
#define MFA_TANHDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct TanhKernelDescriptor {
  bool gradient;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  bool operator==(const TanhKernelDescriptor& rhs) const { return gradient == rhs.gradient && value == rhs.value && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<TanhKernelDescriptor> {
  std::size_t operator()(const TanhKernelDescriptor& descriptor) const noexcept {
    return std::hash<int>()((int)descriptor.value | ((int)descriptor.gradient << 8) | ((int)descriptor.memoryPrecision.value << 16) | ((int)descriptor.loadM << 24));
  }
};

struct TanhKernel;

// Length and the soft cap specialize the pipeline, not the kernel source.
struct TanhDescriptor {
  bool gradient;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  uint32_t length;
  float soft_cap;

  bool operator==(const TanhDescriptor& rhs) const;
  std::pair<TanhKernelDescriptor, PipelineValue<TanhKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<TanhKernelDescriptor, std::unique_ptr<TanhKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<TanhDescriptor> {
  std::size_t operator()(const TanhDescriptor& descriptor) const noexcept;
};

#endif
