#ifndef MFA_GEMVDESCRIPTOR_HPP_
#define MFA_GEMVDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct GemvKernelDescriptor {
  uint8_t fusedBias;
  uint8_t mrows;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const GemvKernelDescriptor& rhs) const { return fusedBias == rhs.fusedBias && mrows == rhs.mrows && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<GemvKernelDescriptor>
{
  std::size_t operator()(const GemvKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.fusedBias | ((int)hash.mrows << 8) | ((int)hash.memoryPrecision.value << 16));
  }
};

struct GemvKernel;

struct GemvDescriptor {
  uint8_t fusedBias;

  uint8_t mrows;

  GEMMOperandPrecision memoryPrecision;

  uint32_t nrows;

  uint32_t ncols;

  bool operator==(const GemvDescriptor& rhs) const;

  static uint32_t rowsPerThreadgroup(MTL::Device* const device) noexcept;

  std::pair<GemvKernelDescriptor, PipelineValue<GemvKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GemvKernelDescriptor, std::unique_ptr<GemvKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<GemvDescriptor>
{
  std::size_t operator()(const GemvDescriptor& hash) const noexcept;
};

#endif
