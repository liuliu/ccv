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

  constexpr bool operator==(const AttentionR1KernelDescriptor& rhs) const {
    return memoryPrecision == rhs.memoryPrecision &&
        loadC == rhs.loadC;
  }
};

template<>
struct std::hash<AttentionR1KernelDescriptor>
{
  std::size_t operator()(const AttentionR1KernelDescriptor& hash) const noexcept {
    return std::hash<int>()(((int)hash.memoryPrecision.value << 1) | (hash.loadC ? 1 : 0));
  }
};

struct AttentionR1Kernel;

struct AttentionR1Descriptor {
  enum class Mode : uint8_t {
    direct = 0,
    splitReduce = 1,
  };

  GEMMOperandPrecision memoryPrecision;

  uint32_t C;

  uint32_t Hq;

  uint32_t Hk;

  uint32_t D;

  float scale;

  bool loadC;

  uint32_t simdgroups;

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
      bool loadC) noexcept;

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
