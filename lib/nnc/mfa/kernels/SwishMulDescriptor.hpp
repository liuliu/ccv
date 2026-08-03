#ifndef MFA_SWISHMULDESCRIPTOR_HPP_
#define MFA_SWISHMULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <functional>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SwishMulKernelDescriptor {
  uint8_t gradient;
  uint8_t outputMask;
  uint8_t value;
  uint8_t weighted;
  bool loadM;
  float beta;
  float scale;
  uint8_t clamped;
  GEMMOperandPrecision gPrecision;
  GEMMOperandPrecision aPrecision;
  GEMMOperandPrecision bPrecision;
  GEMMOperandPrecision weightPrecision;
  GEMMOperandPrecision daPrecision;
  GEMMOperandPrecision dbPrecision;
  constexpr bool operator==(const SwishMulKernelDescriptor& rhs) const { return gradient == rhs.gradient && outputMask == rhs.outputMask && value == rhs.value && weighted == rhs.weighted && loadM == rhs.loadM && beta == rhs.beta && scale == rhs.scale && clamped == rhs.clamped && gPrecision == rhs.gPrecision && aPrecision == rhs.aPrecision && bPrecision == rhs.bPrecision && weightPrecision == rhs.weightPrecision && daPrecision == rhs.daPrecision && dbPrecision == rhs.dbPrecision; }
};

template<>
struct std::hash<SwishMulKernelDescriptor>
{
  std::size_t operator()(const SwishMulKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.value | ((int)hash.gradient << 8) | ((int)hash.outputMask << 16) | ((int)hash.gPrecision.value << 24)) ^ std::hash<int>()((int)hash.aPrecision.value | ((int)hash.bPrecision.value << 8) | ((int)hash.daPrecision.value << 16) | ((int)hash.dbPrecision.value << 24)) ^ std::hash<int>()((int)hash.weightPrecision.value | ((int)hash.weighted << 8) | ((int)hash.clamped << 16)) ^ std::hash<float>()(hash.beta) ^ (std::hash<float>()(hash.scale) << 1) ^ std::hash<bool>()(hash.loadM);
  }
};

struct SwishMulKernel;

struct SwishMulDescriptor {
  uint8_t gradient;

  uint8_t outputMask;

  uint8_t value;
  uint8_t weighted;

  float beta;

  float scale;

  float clamp;

  GEMMOperandPrecision gPrecision;

  GEMMOperandPrecision aPrecision;

  GEMMOperandPrecision bPrecision;
  GEMMOperandPrecision weightPrecision;

  GEMMOperandPrecision daPrecision;

  GEMMOperandPrecision dbPrecision;

  uint32_t length;
  uint32_t weightCount;

  bool loadM;

  bool operator==(const SwishMulDescriptor& rhs) const;

  std::pair<SwishMulKernelDescriptor, PipelineValue<SwishMulKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SwishMulKernelDescriptor, std::unique_ptr<SwishMulKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<SwishMulDescriptor>
{
  std::size_t operator()(const SwishMulDescriptor& hash) const noexcept;
};

#endif
