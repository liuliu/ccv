#ifndef MFA_NAMATMULSMALLMDESCRIPTOR_HPP_
#define MFA_NAMATMULSMALLMDESCRIPTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <simd/simd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct NAMatMulSmallMKernelDescriptor;
struct NAMatMulSmallMKernel;

struct NAMatMulSmallMScratchOffsets {
  size_t partials;
  size_t total;
};

struct NAMatMulSmallMDescriptor {
  int64_t batchDimension = 1;
  simd::uint3 matrixDimensions;
  GEMMOperandPrecisions memoryPrecisions;
  bool useBias;
  bool loadM = false;

  bool operator==(const NAMatMulSmallMDescriptor& rhs) const;

  uint16_t pack() const noexcept;
  simd::ushort3 blockDimensions() const noexcept;
  uint16_t executionSIMDGroups() const noexcept;
  uint16_t splitK() const noexcept;
  NAMatMulSmallMScratchOffsets scratchOffsets() const noexcept;

  std::pair<NAMatMulSmallMKernelDescriptor, PipelineValue<NAMatMulSmallMKernel>*> findKernel(
      MTL::Device* const device,
      const DeviceProperties& dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<NAMatMulSmallMKernelDescriptor, std::unique_ptr<NAMatMulSmallMKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<NAMatMulSmallMDescriptor>
{
  std::size_t operator()(const NAMatMulSmallMDescriptor& hash) const noexcept;
};

#endif
