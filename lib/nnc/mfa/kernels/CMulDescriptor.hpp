#ifndef MFA_CMULDESCRIPTOR_HPP_
#define MFA_CMULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct CMulKernelDescriptor {
  uint8_t conjugate;
  uint8_t value;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const CMulKernelDescriptor &rhs) const { return value == rhs.value && memoryPrecision == rhs.memoryPrecision && conjugate == rhs.conjugate; }
};

template<>
struct std::hash<CMulKernelDescriptor>
{
  std::size_t operator()(const CMulKernelDescriptor& hash) const noexcept { return (size_t)hash.value; }
};

struct CMulKernel;

struct CMulDescriptor {
  uint8_t conjugate;

  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  simd::uint3 stridesA;

  simd::uint3 stridesB;

  simd::uint3 stridesC;

  simd::uint3 dimensions;

  bool operator==(const CMulDescriptor& rhs) const;

  std::pair<CMulKernelDescriptor, PipelineValue<CMulKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CMulKernelDescriptor, std::unique_ptr<CMulKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<CMulDescriptor>
{
  std::size_t operator()(const CMulDescriptor& hash) const noexcept;
};

#endif

