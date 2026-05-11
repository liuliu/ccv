#ifndef MFA_INT8GEMVDESCRIPTOR_HPP_
#define MFA_INT8GEMVDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

constexpr uint32_t kInt8GemvRowsPerThreadgroup = 2;
constexpr uint32_t kInt8GemvSIMDGroupsPerThreadgroup = 4;

struct Int8GemvKernelDescriptor {
  uint8_t fusedBias;
  uint8_t mrows;
  uint32_t format;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const Int8GemvKernelDescriptor& rhs) const { return fusedBias == rhs.fusedBias && mrows == rhs.mrows && format == rhs.format && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<Int8GemvKernelDescriptor>
{
  std::size_t operator()(const Int8GemvKernelDescriptor& hash) const noexcept {
    return std::hash<uint64_t>()((uint64_t)hash.fusedBias | ((uint64_t)hash.mrows << 8) | ((uint64_t)hash.memoryPrecision.value << 16) | ((uint64_t)hash.format << 32));
  }
};

struct Int8GemvKernel;

struct Int8GemvDescriptor {
  uint8_t fusedBias;

  uint8_t mrows;

  uint32_t format;

  GEMMOperandPrecision memoryPrecision;

  uint32_t nrows;

  uint32_t ncols;

  bool operator==(const Int8GemvDescriptor& rhs) const;

  uint32_t groupSize() const noexcept;
  uint32_t groupsPerRow() const noexcept;
  uint32_t groupBits() const noexcept;
  uint32_t inputScaleOffset() const noexcept;

  std::pair<Int8GemvKernelDescriptor, PipelineValue<Int8GemvKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Int8GemvKernelDescriptor, std::unique_ptr<Int8GemvKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<Int8GemvDescriptor>
{
  std::size_t operator()(const Int8GemvDescriptor& hash) const noexcept;
};

#endif
