#ifndef MFA_SWISHMULDESCRIPTOR_HPP_
#define MFA_SWISHMULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <functional>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SwishMulKernelDescriptor {
  uint8_t value;
  float beta;
  float scale;
  GEMMOperandPrecision aPrecision;
  GEMMOperandPrecision bPrecision;
  constexpr bool operator==(const SwishMulKernelDescriptor& rhs) const { return value == rhs.value && beta == rhs.beta && scale == rhs.scale && aPrecision == rhs.aPrecision && bPrecision == rhs.bPrecision; }
};

template<>
struct std::hash<SwishMulKernelDescriptor>
{
  std::size_t operator()(const SwishMulKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.aPrecision.value << 8) | ((int)hash.bPrecision.value << 16)) ^ std::hash<float>()(hash.beta) ^ (std::hash<float>()(hash.scale) << 1);
  }
};

struct SwishMulKernel;

struct SwishMulDescriptor {
  uint8_t value;

  float beta;

  float scale;

  GEMMOperandPrecision aPrecision;

  GEMMOperandPrecision bPrecision;

  uint32_t length;

  bool operator==(const SwishMulDescriptor& rhs) const;

  std::pair<SwishMulKernelDescriptor, PipelineValue<SwishMulKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SwishMulKernelDescriptor, std::unique_ptr<SwishMulKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<SwishMulDescriptor>
{
  std::size_t operator()(const SwishMulDescriptor& hash) const noexcept;
};

#endif
