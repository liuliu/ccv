#ifndef MFA_SIGMOIDDESCRIPTOR_HPP_
#define MFA_SIGMOIDDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <functional>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SigmoidKernelDescriptor {
  uint8_t gradient;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const SigmoidKernelDescriptor& rhs) const { return value == rhs.value && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision && gradient == rhs.gradient; }
};

template<>
struct std::hash<SigmoidKernelDescriptor>
{
  std::size_t operator()(const SigmoidKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.gradient << 8) | ((int)hash.memoryPrecision.value << 16) | ((int)hash.loadM << 24));
  }
};

struct SigmoidKernel;

struct SigmoidDescriptor {
  uint8_t gradient;

  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool loadM;

  bool operator==(const SigmoidDescriptor& rhs) const;

  std::pair<SigmoidKernelDescriptor, PipelineValue<SigmoidKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SigmoidKernelDescriptor, std::unique_ptr<SigmoidKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<SigmoidDescriptor>
{
  std::size_t operator()(const SigmoidDescriptor& hash) const noexcept;
};

#endif
