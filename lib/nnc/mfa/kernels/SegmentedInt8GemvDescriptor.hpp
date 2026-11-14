#ifndef MFA_SEGMENTEDINT8GEMVDESCRIPTOR_HPP_
#define MFA_SEGMENTEDINT8GEMVDESCRIPTOR_HPP_

#include <cstdint>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"
#include <simd/simd.h>
#include <utility>

constexpr uint32_t kSegmentedInt8GemvRowsPerSIMDGroup = 2;
constexpr uint32_t kSegmentedInt8GemvSIMDGroupsPerThreadgroup = 2;
constexpr uint32_t kSegmentedInt8GemvRowsPerThreadgroup =
  kSegmentedInt8GemvRowsPerSIMDGroup *
  kSegmentedInt8GemvSIMDGroupsPerThreadgroup;

// The Q2_K kernel shares each activation group across a threadgroup's rows, with its SIMD groups
// splitting the groups, so it takes more SIMD groups (and rows) per threadgroup.
uint32_t segmentedInt8GemvSIMDGroupsPerThreadgroup(const uint32_t format);

// The TMCQ rate n (n / 16 bits per weight) of a CCV_NNC_QX_8I_ROWWISE_TMCQ_N* format, or 0 for any
// other format (including a TMCQ codec with modifier bits, which no kernel supports).
uint32_t segmentedTMCQRate(const uint32_t format);

// Rows a threadgroup of the segmented GEMV covers. TMCQ tiles span 16 rows, so each of its SIMD
// groups takes one 16-row tile row and shares the activation loads across those rows.
uint32_t segmentedInt8GemvRowsPerThreadgroup(const uint32_t format);

struct SegmentedInt8GemvKernelDescriptor {
  uint8_t fusedBias;
  uint8_t mrows;
  uint8_t sharedFused;
  uint32_t format;
  GEMMOperandPrecision memoryPrecision;

  constexpr bool operator==(const SegmentedInt8GemvKernelDescriptor& rhs) const
  {
    return
      fusedBias == rhs.fusedBias &&
      mrows == rhs.mrows &&
      sharedFused == rhs.sharedFused &&
      format == rhs.format &&
      memoryPrecision == rhs.memoryPrecision;
  }
};

template<>
struct std::hash<SegmentedInt8GemvKernelDescriptor>
{
  std::size_t operator()(const SegmentedInt8GemvKernelDescriptor& hash) const noexcept
  {
    return std::hash<uint64_t>()(
      (uint64_t)hash.fusedBias |
      ((uint64_t)hash.mrows << 8) |
      ((uint64_t)hash.sharedFused << 12) |
      ((uint64_t)hash.memoryPrecision.value << 16) |
      ((uint64_t)hash.format << 32));
  }
};

struct SegmentedInt8GemvKernel;

struct SegmentedInt8GemvDescriptor {
  simd::uint2 matrixDimensions; // N, K.

  uint32_t expertCount;
  uint32_t binCount;
  uint32_t format;

  GEMMOperandPrecision memoryPrecision;

  bool useBias;
  bool broadcastInput;

  // Shared expert rows interleaved with the routed threadgroups (Q2_K only), and how many routed
  // threadgroups the grid starts with.
  uint32_t sharedRows;
  uint32_t routedThreadgroups;

  bool operator==(const SegmentedInt8GemvDescriptor& rhs) const;

  uint32_t groupSize() const noexcept;
  uint32_t groupsPerRow() const noexcept;
  uint32_t groupBits() const noexcept;
  uint64_t inputScaleOffset() const noexcept;
  uint64_t weightExpertStride() const noexcept;

  std::pair<
    SegmentedInt8GemvKernelDescriptor,
    PipelineValue<SegmentedInt8GemvKernel>*> findKernel(
      MTL::Device* const device,
      const DeviceProperties& dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<
        SegmentedInt8GemvKernelDescriptor,
        std::unique_ptr<SegmentedInt8GemvKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<SegmentedInt8GemvDescriptor>
{
  std::size_t operator()(const SegmentedInt8GemvDescriptor& hash) const noexcept;
};

#endif
