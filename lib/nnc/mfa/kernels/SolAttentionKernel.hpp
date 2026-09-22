#ifndef SolAttentionKernel_hpp
#define SolAttentionKernel_hpp

#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include <simd/simd.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

// Separate descriptor types keep Sol libraries out of the native attention cache.
enum class SolAttentionKernelKey : uint32_t {};

struct SolAttentionKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  simd::ushort3 blockDimensions;
  uint16_t threadgroupSize;
  uint16_t threadgroupMemoryAllocation;
  uint16_t denseThreadgroupMemoryAllocation;

  SolAttentionKernel(MTL::Device* device, uint32_t blockSize, bool useRouteBits = false);
};

struct SolAttentionDescriptor {
  uint32_t N, T, H, blockSize, queryBlockSize;
  float scale;
  bool useRouteBits;
  bool operator==(const SolAttentionDescriptor& rhs) const;
  std::pair<SolAttentionKernelKey, PipelineValue<SolAttentionKernel>*>
  findKernel(MTL::Device*, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&,
             std::unordered_map<SolAttentionKernelKey, std::unique_ptr<SolAttentionKernel>>*) const noexcept;
};

template <> struct std::hash<SolAttentionDescriptor> {
  size_t operator()(const SolAttentionDescriptor& d) const noexcept;
};

struct SolAttentionPreparationDescriptor {
  uint32_t entry, blockSize, N, T, H, queryBlockSize;
  bool useRouteBits;
  bool operator==(const SolAttentionPreparationDescriptor& rhs) const;
  std::pair<SolAttentionKernelKey, PipelineValue<SolAttentionKernel>*>
  findKernel(MTL::Device*, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&,
             std::unordered_map<SolAttentionKernelKey, std::unique_ptr<SolAttentionKernel>>*) const noexcept;
};

template <> struct std::hash<SolAttentionPreparationDescriptor> {
  size_t operator()(const SolAttentionPreparationDescriptor& d) const noexcept;
};
#endif
