#ifndef MFA_LOGDESCRIPTOR_HPP_
#define MFA_LOGDESCRIPTOR_HPP_

#include <functional>
#include <simd/simd.h>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct LogKernelDescriptor {
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const LogKernelDescriptor& rhs) const { return value == rhs.value && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<LogKernelDescriptor> {
  std::size_t operator()(const LogKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.memoryPrecision.value << 8) | ((int)hash.loadM << 16));
  }
};

struct LogKernel;

struct LogDescriptor {
  uint8_t value;
  GEMMOperandPrecision memoryPrecision;
  uint32_t length;
  bool loadM;

  bool operator==(const LogDescriptor& rhs) const;
  std::pair<LogKernelDescriptor, PipelineValue<LogKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<LogKernelDescriptor, std::unique_ptr<LogKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<LogDescriptor> {
  std::size_t operator()(const LogDescriptor& hash) const noexcept;
};

#endif
