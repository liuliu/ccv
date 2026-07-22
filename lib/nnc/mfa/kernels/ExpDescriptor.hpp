#ifndef MFA_EXPDESCRIPTOR_HPP_
#define MFA_EXPDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <functional>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct ExpKernelDescriptor {
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const ExpKernelDescriptor& rhs) const { return value == rhs.value && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<ExpKernelDescriptor>
{
  std::size_t operator()(const ExpKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.memoryPrecision.value << 8) | ((int)hash.loadM << 16));
  }
};

struct ExpKernel;

struct ExpDescriptor {
  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool loadM;

  bool operator==(const ExpDescriptor& rhs) const;

  std::pair<ExpKernelDescriptor, PipelineValue<ExpKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ExpKernelDescriptor, std::unique_ptr<ExpKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<ExpDescriptor>
{
  std::size_t operator()(const ExpDescriptor& hash) const noexcept;
};

#endif
