#ifndef MFA_GEMVDESCRIPTOR_HPP_
#define MFA_GEMVDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <optional>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct GemvKernelDescriptor {
  uint8_t fusedBias;
  uint8_t mrows;
  uint8_t batched;
  uint8_t cooperative = 0;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const GemvKernelDescriptor& rhs) const { return fusedBias == rhs.fusedBias && mrows == rhs.mrows && batched == rhs.batched && cooperative == rhs.cooperative && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<GemvKernelDescriptor>
{
  std::size_t operator()(const GemvKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.fusedBias | ((int)hash.mrows << 8) | ((int)hash.memoryPrecision.value << 16) | ((int)hash.batched << 24) | ((int)hash.cooperative << 25));
  }
};

struct GemvKernel;

struct GemvDescriptor {
  uint8_t fusedBias;

  uint8_t mrows;

  uint8_t cooperative = 0;

  GEMMOperandPrecision memoryPrecision;

  uint32_t nrows;

  uint32_t ncols;

  std::optional<simd::uint3> batchStrides;

  bool operator==(const GemvDescriptor& rhs) const;

  static uint32_t rowsPerThreadgroup(MTL::Device* const device) noexcept;

  static uint32_t cooperativeSIMDGroups(MTL::Device* const device, const uint32_t ncols) noexcept;

  std::pair<GemvKernelDescriptor, PipelineValue<GemvKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GemvKernelDescriptor, std::unique_ptr<GemvKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<GemvDescriptor>
{
  std::size_t operator()(const GemvDescriptor& hash) const noexcept;
};

#endif
