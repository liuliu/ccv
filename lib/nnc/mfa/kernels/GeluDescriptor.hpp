#ifndef MFA_GELUDESCRIPTOR_HPP_
#define MFA_GELUDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct GeluKernelDescriptor {
  uint8_t gradient;
  uint8_t tanh;
  uint8_t value;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const GeluKernelDescriptor &rhs) const { return value == rhs.value && memoryPrecision == rhs.memoryPrecision && tanh == rhs.tanh && gradient == rhs.gradient; }
};

template<>
struct std::hash<GeluKernelDescriptor>
{
  std::size_t operator()(const GeluKernelDescriptor& hash) const noexcept { return (size_t)hash.value; }
};

struct GeluKernel;

struct GeluDescriptor {
  uint8_t gradient;

  uint8_t tanh;

  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool operator==(const GeluDescriptor& rhs) const;

  std::pair<GeluKernelDescriptor, PipelineValue<GeluKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GeluKernelDescriptor, std::unique_ptr<GeluKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<GeluDescriptor>
{
  std::size_t operator()(const GeluDescriptor& hash) const noexcept;
};

#endif

