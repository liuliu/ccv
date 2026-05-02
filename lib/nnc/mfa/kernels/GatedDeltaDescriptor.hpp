#ifndef MFA_GATEDDELTADESCRIPTOR_HPP_
#define MFA_GATEDDELTADESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct GatedDeltaKernelDescriptor {
  uint8_t stateElementsPerLane;
  constexpr bool operator==(const GatedDeltaKernelDescriptor& rhs) const { return stateElementsPerLane == rhs.stateElementsPerLane; }
};

template<>
struct std::hash<GatedDeltaKernelDescriptor>
{
  std::size_t operator()(const GatedDeltaKernelDescriptor& hash) const noexcept { return (size_t)hash.stateElementsPerLane; }
};

struct GatedDeltaKernel;

struct GatedDeltaDescriptor {
  uint32_t batchSize;

  uint32_t sequenceLength;

  uint32_t keyHeadCount;

  uint32_t valueHeadCount;

  uint32_t keyDim;

  uint32_t valueDim;

  bool operator==(const GatedDeltaDescriptor& rhs) const;

  std::pair<GatedDeltaKernelDescriptor, PipelineValue<GatedDeltaKernel> *> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GatedDeltaKernelDescriptor, std::unique_ptr<GatedDeltaKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<GatedDeltaDescriptor>
{
  std::size_t operator()(const GatedDeltaDescriptor& hash) const noexcept;
};

#endif
