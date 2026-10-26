#ifndef MFA_GATEDDELTADESCRIPTOR_HPP_
#define MFA_GATEDDELTADESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct GatedDeltaKernelDescriptor {
  uint8_t stateElementsPerLane;
  bool stateCheckpointing;
  GEMMOperandPrecision inputMemoryPrecision;
  GEMMOperandPrecision betaMemoryPrecision;
  bool loadM;
  // Blocked kernel: SIMD groups of four value columns (eight lanes each) over timestep
  // blocks staged in threadgroup memory. stateElementsPerLane is then keyDim / 8.
  bool blocked;
  bool channelDecay;
  uint8_t timeBlock;
  uint8_t simdGroups;
  constexpr bool operator==(const GatedDeltaKernelDescriptor& rhs) const { return stateElementsPerLane == rhs.stateElementsPerLane && stateCheckpointing == rhs.stateCheckpointing && inputMemoryPrecision == rhs.inputMemoryPrecision && betaMemoryPrecision == rhs.betaMemoryPrecision && loadM == rhs.loadM && blocked == rhs.blocked && channelDecay == rhs.channelDecay && timeBlock == rhs.timeBlock && simdGroups == rhs.simdGroups; }
};

template<>
struct std::hash<GatedDeltaKernelDescriptor>
{
  std::size_t operator()(const GatedDeltaKernelDescriptor& hash) const noexcept { return (size_t)hash.stateElementsPerLane | ((size_t)hash.inputMemoryPrecision.value << 8) | ((size_t)hash.betaMemoryPrecision.value << 16) | ((size_t)hash.stateCheckpointing << 24) | ((size_t)hash.loadM << 25) | ((size_t)hash.blocked << 26) | ((size_t)hash.channelDecay << 27) | ((size_t)hash.timeBlock << 32) | ((size_t)hash.simdGroups << 40); }
};

struct GatedDeltaKernel;

struct GatedDeltaDescriptor {
  uint32_t batchSize;

  uint32_t sequenceLength;

  uint32_t keyHeadCount;

  uint32_t valueHeadCount;

  uint32_t keyDim;

  uint32_t valueDim;

  uint32_t stateCheckpointCount;

  GEMMOperandPrecision inputMemoryPrecision;

  GEMMOperandPrecision betaMemoryPrecision;

  bool logDecay;

  bool channelDecay;

  bool loadM;

  // Chosen at encode time from the actual sequence length, which loadM leaves out of the
  // pipeline key: the blocked kernel for prefill, the per-column kernel for short sequences.
  bool blocked;

  bool operator==(const GatedDeltaDescriptor& rhs) const;

  std::pair<GatedDeltaKernelDescriptor, PipelineValue<GatedDeltaKernel> *> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GatedDeltaKernelDescriptor, std::unique_ptr<GatedDeltaKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<GatedDeltaDescriptor>
{
  std::size_t operator()(const GatedDeltaDescriptor& hash) const noexcept;
};

#endif
