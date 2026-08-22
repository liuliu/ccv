#ifndef MFA_CLAMPDESCRIPTOR_HPP_
#define MFA_CLAMPDESCRIPTOR_HPP_

#include <functional>
#include <simd/simd.h>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct ClampKernelDescriptor {
  uint8_t value;
  uint8_t bounds;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const ClampKernelDescriptor& rhs) const { return value == rhs.value && bounds == rhs.bounds && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<ClampKernelDescriptor> {
  std::size_t operator()(const ClampKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.bounds << 4) | ((int)hash.memoryPrecision.value << 8) | ((int)hash.loadM << 16));
  }
};

struct ClampKernel;

struct ClampDescriptor {
  uint8_t value;
  uint8_t bounds;
  GEMMOperandPrecision memoryPrecision;
  uint32_t length;
  bool loadM;

  bool operator==(const ClampDescriptor& rhs) const;
  std::pair<ClampKernelDescriptor, PipelineValue<ClampKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ClampKernelDescriptor, std::unique_ptr<ClampKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<ClampDescriptor> {
  std::size_t operator()(const ClampDescriptor& hash) const noexcept;
};

#endif
