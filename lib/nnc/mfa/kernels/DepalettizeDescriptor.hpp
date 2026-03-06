#ifndef MFA_DEPALETTIZEDESCRIPTOR_HPP_
#define MFA_DEPALETTIZEDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct DepalettizeKernelDescriptor {
  uint8_t qbits;
  uint8_t partial;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const DepalettizeKernelDescriptor& rhs) const { return qbits == rhs.qbits && partial == rhs.partial && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<DepalettizeKernelDescriptor>
{
  std::size_t operator()(const DepalettizeKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.qbits | ((int)hash.partial << 8) | ((int)hash.memoryPrecision.value << 16));
  }
};

struct DepalettizeKernel;

struct DepalettizeDescriptor {
  uint8_t qbits;

  GEMMOperandPrecision memoryPrecision;

  uint32_t numberInBlocks;

  uint32_t length;

  bool operator==(const DepalettizeDescriptor& rhs) const;

  bool partial() const noexcept;

  std::pair<DepalettizeKernelDescriptor, PipelineValue<DepalettizeKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<DepalettizeKernelDescriptor, std::unique_ptr<DepalettizeKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<DepalettizeDescriptor>
{
  std::size_t operator()(const DepalettizeDescriptor& hash) const noexcept;
};

#endif
