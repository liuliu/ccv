#ifndef MFA_RMSNORMGATEDDESCRIPTOR_HPP_
#define MFA_RMSNORMGATEDDESCRIPTOR_HPP_

#include <functional>
#include <simd/simd.h>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct RMSNormGatedKernelDescriptor {
  float epsilon;
  GEMMOperandPrecision aPrecision;
  GEMMOperandPrecision gatePrecision;
  GEMMOperandPrecision scalePrecision;
  uint32_t columnCount;

  bool operator==(const RMSNormGatedKernelDescriptor& rhs) const;
};

template<>
struct std::hash<RMSNormGatedKernelDescriptor>
{
  std::size_t operator()(const RMSNormGatedKernelDescriptor& hash) const noexcept;
};

struct RMSNormGatedKernel;

struct RMSNormGatedDescriptor {
  float epsilon;

  GEMMOperandPrecision aPrecision;

  GEMMOperandPrecision gatePrecision;

  GEMMOperandPrecision scalePrecision;

  uint32_t columnCount;

  bool operator==(const RMSNormGatedDescriptor& rhs) const;

  std::pair<RMSNormGatedKernelDescriptor, PipelineValue<RMSNormGatedKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<RMSNormGatedKernelDescriptor, std::unique_ptr<RMSNormGatedKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<RMSNormGatedDescriptor>
{
  std::size_t operator()(const RMSNormGatedDescriptor& hash) const noexcept;
};

#endif
