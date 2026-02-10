#ifndef MFA_SWISHDESCRIPTOR_HPP_
#define MFA_SWISHDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <functional>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SwishKernelDescriptor {
  uint8_t gradient;
  uint8_t value;
  float beta;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const SwishKernelDescriptor &rhs) const { return value == rhs.value && memoryPrecision == rhs.memoryPrecision && gradient == rhs.gradient && beta == rhs.beta; }
};

template<>
struct std::hash<SwishKernelDescriptor>
{
  std::size_t operator()(const SwishKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.gradient << 8) | ((int)hash.memoryPrecision.value << 16)) ^ std::hash<float>()(hash.beta);
  }
};

struct SwishKernel;

struct SwishDescriptor {
  uint8_t gradient;

  uint8_t value;

  float beta;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool operator==(const SwishDescriptor& rhs) const;

  std::pair<SwishKernelDescriptor, PipelineValue<SwishKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SwishKernelDescriptor, std::unique_ptr<SwishKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<SwishDescriptor>
{
  std::size_t operator()(const SwishDescriptor& hash) const noexcept;
};

#endif
