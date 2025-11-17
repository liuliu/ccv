#ifndef MFA_CASTDESCRIPTOR_HPP_
#define MFA_CASTDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct CastKernelDescriptor {
  uint8_t value;
  GEMMOperandPrecision fromMemoryPrecision;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const CastKernelDescriptor &rhs) const { return value == rhs.value && fromMemoryPrecision == rhs.fromMemoryPrecision && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<CastKernelDescriptor>
{
  std::size_t operator()(const CastKernelDescriptor& hash) const noexcept { return (size_t)hash.value; }
};

struct CastKernel;

struct CastDescriptor {
  uint8_t value;

  GEMMOperandPrecision fromMemoryPrecision;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool operator==(const CastDescriptor& rhs) const;

  std::pair<CastKernelDescriptor, PipelineValue<CastKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CastKernelDescriptor, std::unique_ptr<CastKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<CastDescriptor>
{
  std::size_t operator()(const CastDescriptor& hash) const noexcept;
};

#endif

