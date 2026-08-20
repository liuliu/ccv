#ifndef MFA_FILLIFLESSTHANDESCRIPTOR_HPP_
#define MFA_FILLIFLESSTHANDESCRIPTOR_HPP_

#include <functional>
#include <simd/simd.h>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct FillIfLessThanKernelDescriptor {
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const FillIfLessThanKernelDescriptor& rhs) const { return value == rhs.value && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<FillIfLessThanKernelDescriptor>
{
  std::size_t operator()(const FillIfLessThanKernelDescriptor& hash) const noexcept { return (size_t)hash.value | ((size_t)hash.memoryPrecision.value << 8) | ((size_t)hash.loadM << 16); }
};

struct FillIfLessThanKernel;

struct FillIfLessThanDescriptor {
  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool loadM;

  bool operator==(const FillIfLessThanDescriptor& rhs) const;

  std::pair<FillIfLessThanKernelDescriptor, PipelineValue<FillIfLessThanKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<FillIfLessThanKernelDescriptor, std::unique_ptr<FillIfLessThanKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<FillIfLessThanDescriptor>
{
  std::size_t operator()(const FillIfLessThanDescriptor& hash) const noexcept;
};

#endif
