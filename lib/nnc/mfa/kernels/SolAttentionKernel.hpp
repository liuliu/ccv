#ifndef SolAttentionKernel_hpp
#define SolAttentionKernel_hpp

#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include <simd/simd.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

// Source-generation properties only; shapes specialize pipeline constants.
struct SolAttentionKernelDescriptor {
  uint32_t blockSize;
  bool useRouteBits;
  // Retain Q for the entire traversal in registers instead of threadgroup memory.
  bool cacheQueryInRegisters;
  // Share each K tile across SIMD groups instead of loading K directly.
  bool stageKeyInThreadgroup;
  bool preferAsyncCache, preferAsyncLoad;
  bool operator==(const SolAttentionKernelDescriptor& rhs) const;
};

template <> struct std::hash<SolAttentionKernelDescriptor> {
  size_t operator()(const SolAttentionKernelDescriptor& d) const noexcept;
};

struct SolAttentionKernel {
  const SolAttentionKernelDescriptor descriptor;
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  simd::ushort3 blockDimensions;
  uint16_t threadgroupSize;
  uint16_t threadgroupMemoryAllocation;
  uint16_t denseThreadgroupMemoryAllocation;

  SolAttentionKernel(MTL::Device* device, const SolAttentionKernelDescriptor& descriptor);
};

struct SolAttentionDescriptor {
  uint32_t N, T, H, blockSize, queryBlockSize;
  float scale;
  bool useRouteBits;
  bool operator==(const SolAttentionDescriptor& rhs) const;
  SolAttentionKernelDescriptor kernelDescriptor(MTL::Device* device) const noexcept;
  std::pair<SolAttentionKernelDescriptor, PipelineValue<SolAttentionKernel>*>
  findKernel(MTL::Device*, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&,
             std::unordered_map<SolAttentionKernelDescriptor, std::unique_ptr<SolAttentionKernel>>*) const noexcept;
};

template <> struct std::hash<SolAttentionDescriptor> {
  size_t operator()(const SolAttentionDescriptor& d) const noexcept;
};

struct SolAttentionPreparationDescriptor {
  uint32_t entry, blockSize, N, T, H, queryBlockSize;
  bool useRouteBits;
  bool operator==(const SolAttentionPreparationDescriptor& rhs) const;
  std::pair<SolAttentionKernelDescriptor, PipelineValue<SolAttentionKernel>*>
  findKernel(MTL::Device*, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&,
             std::unordered_map<SolAttentionKernelDescriptor, std::unique_ptr<SolAttentionKernel>>*) const noexcept;
};

template <> struct std::hash<SolAttentionPreparationDescriptor> {
  size_t operator()(const SolAttentionPreparationDescriptor& d) const noexcept;
};
#endif
