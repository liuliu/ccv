#ifndef MFA_SPARSEINDEXEDATTENTIONR1DESCRIPTOR_HPP_
#define MFA_SPARSEINDEXEDATTENTIONR1DESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct SparseIndexedAttentionR1KernelDescriptor {
  GEMMOperandPrecision memoryPrecision;

  bool loadK; // Load denseRows, sparseRows, and K from the runtime shape buffer.

  bool attentionSinks;

  constexpr bool operator==(const SparseIndexedAttentionR1KernelDescriptor& rhs) const {
    return memoryPrecision == rhs.memoryPrecision &&
        loadK == rhs.loadK &&
        attentionSinks == rhs.attentionSinks;
  }
};

template<>
struct std::hash<SparseIndexedAttentionR1KernelDescriptor>
{
  std::size_t operator()(const SparseIndexedAttentionR1KernelDescriptor& hash) const noexcept {
    return std::hash<int>()(((int)hash.memoryPrecision.value << 2) | ((hash.loadK ? 1 : 0) << 1) | (hash.attentionSinks ? 1 : 0));
  }
};

struct SparseIndexedAttentionR1Kernel;

struct SparseIndexedAttentionR1Descriptor {
  enum class Mode : uint8_t {
    direct = 0,
    splitReduce = 1,
  };

  GEMMOperandPrecision memoryPrecision;

  uint32_t denseRows;

  uint32_t sparseRows;

  uint32_t K;

  uint32_t H;

  uint32_t D;

  float scale;

  bool loadK;

  bool attentionSinks;

  uint32_t slidingWindow;

  uint32_t simdgroups;

  uint32_t workgroups;

  Mode mode;

  bool operator==(const SparseIndexedAttentionR1Descriptor& rhs) const;

  static SparseIndexedAttentionR1Descriptor select(
      GEMMOperandPrecision memoryPrecision,
      uint32_t denseRows,
      uint32_t sparseRows,
      uint32_t K,
      uint32_t H,
      uint32_t D,
      float scale,
      bool loadK,
      bool attentionSinks,
      uint32_t slidingWindow) noexcept;

  std::pair<SparseIndexedAttentionR1KernelDescriptor, PipelineValue<SparseIndexedAttentionR1Kernel>*> findKernel(
      MTL::Device* const device,
      const DeviceProperties& dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<SparseIndexedAttentionR1KernelDescriptor, std::unique_ptr<SparseIndexedAttentionR1Kernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<SparseIndexedAttentionR1Descriptor>
{
  std::size_t operator()(const SparseIndexedAttentionR1Descriptor& hash) const noexcept;
};

#endif
