#ifndef MFA_ROTATEHALFDESCRIPTOR_HPP_
#define MFA_ROTATEHALFDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct RotateHalfKernelDescriptor {
  uint8_t value;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const RotateHalfKernelDescriptor &rhs) const { return value == rhs.value && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<RotateHalfKernelDescriptor>
{
  std::size_t operator()(const RotateHalfKernelDescriptor& hash) const noexcept { return (size_t)hash.value; }
};

struct RotateHalfKernel;

struct RotateHalfDescriptor {
  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t rowCount;

  uint32_t dim;

  bool operator==(const RotateHalfDescriptor& rhs) const;

  std::pair<RotateHalfKernelDescriptor, PipelineValue<RotateHalfKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<RotateHalfKernelDescriptor, std::unique_ptr<RotateHalfKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<RotateHalfDescriptor>
{
  std::size_t operator()(const RotateHalfDescriptor& hash) const noexcept;
};

#endif
