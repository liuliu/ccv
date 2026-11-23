#ifndef AttentionR1Descriptor_hpp
#define AttentionR1Descriptor_hpp

#include <simd/simd.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct AttentionR1KernelDescriptor {
  GEMMOperandPrecision memoryPrecision;

  bool loadC;

  bool attentionSinks;

  // One sink for each query and head, [batch, R, Hq], rather than one for each head.
  bool sinksPerQuery = false;

  // Each query adds its own bias to the scores of the keys within an extent before it.
  bool relativeBias = false;

  // An additive mask [1, 1, R, C] in the input precision; -inf closes a key.
  bool masked = false;

  constexpr bool operator==(const AttentionR1KernelDescriptor& rhs) const {
    return memoryPrecision == rhs.memoryPrecision &&
        loadC == rhs.loadC &&
        attentionSinks == rhs.attentionSinks &&
        sinksPerQuery == rhs.sinksPerQuery &&
        relativeBias == rhs.relativeBias &&
        masked == rhs.masked;
  }
};

template<>
struct std::hash<AttentionR1KernelDescriptor>
{
  std::size_t operator()(const AttentionR1KernelDescriptor& hash) const noexcept {
    return std::hash<int>()(((int)hash.memoryPrecision.value << 5) | ((hash.sinksPerQuery ? 1 : 0) << 4) | ((hash.masked ? 1 : 0) << 3) | ((hash.relativeBias ? 1 : 0) << 2) | ((hash.loadC ? 1 : 0) << 1) | (hash.attentionSinks ? 1 : 0));
  }
};

struct AttentionR1Kernel;

struct AttentionR1Descriptor {
  enum class Mode : uint8_t {
    direct = 0,
    splitReduce = 1,
    cooperative = 2,
  };

  GEMMOperandPrecision memoryPrecision;

  uint32_t C;

  uint32_t Hq;

  uint32_t Hk;

  uint32_t D;

  uint32_t R;

  bool isCausal;

  float scale;

  bool loadC;

  bool attentionSinks;

  /// One sink for each query and head, [batch, R, Hq], rather than one for each head.
  bool sinksPerQuery = false;

  /// A relative-position bias [batch, R, Hq, extent], with causal attention. 0 is no bias.
  uint32_t relativeBiasExtent = 0;

  /// Each causal row reads only its last this many keys. 0 reads every key up to the row's own.
  uint32_t slidingWindow = 0;

  /// An additive mask [1, 1, R, C] of a single batch, without a sliding window.
  bool masked = false;

  uint32_t simdgroups;

  /// Query heads of a workgroup, which share a key head.
  uint32_t headsPerGroup;

  uint32_t headsPerSIMD;

  /// Consecutive rows of a SIMD group, which reads each key and value once for all of them.
  uint32_t rowsPerSIMD;

  uint32_t reductionSIMDGroups;

  uint32_t workgroups;

  Mode mode;

  bool operator==(const AttentionR1Descriptor& rhs) const;

  static AttentionR1Descriptor select(
      GEMMOperandPrecision memoryPrecision,
      uint32_t C,
      uint32_t Hq,
      uint32_t Hk,
      uint32_t D,
      float scale,
      bool loadC,
      bool attentionSinks,
      uint32_t R,
      bool isCausal,
      uint32_t batchDimension,
      uint32_t coreCount) noexcept;

  std::pair<AttentionR1KernelDescriptor, PipelineValue<AttentionR1Kernel>*> findKernel(
      MTL::Device* const device,
      const DeviceProperties& dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<AttentionR1KernelDescriptor, std::unique_ptr<AttentionR1Kernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<AttentionR1Descriptor>
{
  std::size_t operator()(const AttentionR1Descriptor& hash) const noexcept;
};

#endif
