#ifndef MFA_NAINT8MATMULSMALLMDESCRIPTOR_HPP_
#define MFA_NAINT8MATMULSMALLMDESCRIPTOR_HPP_

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

struct NAInt8MatMulSmallMKernelDescriptor;
struct NAInt8MatMulSmallMKernel;

struct NAInt8MatMulSmallMScratchOffsets {
  size_t partials;
  size_t total;
};

struct NAInt8MatMulSmallMDescriptor {
  int64_t batchDimension = 1;
  simd::uint3 matrixDimensions;
  GEMMOperandPrecision ioPrecision = GEMMOperandPrecision::FP16;
  bool useBias = false;
  bool loadM = false;

  bool operator==(const NAInt8MatMulSmallMDescriptor& rhs) const;

  uint16_t pack() const noexcept;
  simd::ushort3 blockDimensions() const noexcept;
  uint16_t executionSIMDGroups() const noexcept;
  uint16_t splitK() const noexcept;
  NAInt8MatMulSmallMScratchOffsets scratchOffsets() const noexcept;

  std::pair<NAInt8MatMulSmallMKernelDescriptor, PipelineValue<NAInt8MatMulSmallMKernel>*> findKernel(
      MTL::Device* const device,
      const DeviceProperties& dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<NAInt8MatMulSmallMKernelDescriptor, std::unique_ptr<NAInt8MatMulSmallMKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<NAInt8MatMulSmallMDescriptor>
{
  std::size_t operator()(const NAInt8MatMulSmallMDescriptor& hash) const noexcept;
};

#endif
