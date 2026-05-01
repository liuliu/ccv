#ifndef MFA_SOFTPLUSDESCRIPTOR_HPP_
#define MFA_SOFTPLUSDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <functional>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SoftplusKernelDescriptor {
  uint8_t value;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const SoftplusKernelDescriptor& rhs) const { return value == rhs.value && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<SoftplusKernelDescriptor>
{
  std::size_t operator()(const SoftplusKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.memoryPrecision.value << 8));
  }
};

struct SoftplusKernel;

struct SoftplusDescriptor {
  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool operator==(const SoftplusDescriptor& rhs) const;

  std::pair<SoftplusKernelDescriptor, PipelineValue<SoftplusKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SoftplusKernelDescriptor, std::unique_ptr<SoftplusKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<SoftplusDescriptor>
{
  std::size_t operator()(const SoftplusDescriptor& hash) const noexcept;
};

#endif
