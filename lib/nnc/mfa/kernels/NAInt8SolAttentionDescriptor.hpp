#ifndef MFA_NAINT8SOLATTENTIONDESCRIPTOR_HPP_
#define MFA_NAINT8SOLATTENTIONDESCRIPTOR_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include "DeviceProperties.hpp"
#include "PipelineValue.hpp"

// Only source-generation properties belong in the library cache key.
struct NAInt8SolAttentionKernelDescriptor {
  uint32_t blockSize = 64;
  bool operator==(const NAInt8SolAttentionKernelDescriptor& rhs) const { return blockSize == rhs.blockSize; }
};

template<> struct std::hash<NAInt8SolAttentionKernelDescriptor> {
  size_t operator()(const NAInt8SolAttentionKernelDescriptor& d) const noexcept { return d.blockSize; }
};

enum class NAInt8SolAttentionStage : uint32_t {
  VMean, Quantize, Pool, PrepareSummaries, Route, Attention
};

struct NAInt8SolAttentionKernel;

struct NAInt8SolAttentionDescriptor {
  NAInt8SolAttentionStage stage;
  uint32_t blockSize, N, T, H, queryBlockSize;

  bool operator==(const NAInt8SolAttentionDescriptor& rhs) const;
  std::pair<NAInt8SolAttentionKernelDescriptor, PipelineValue<NAInt8SolAttentionKernel>*> findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&, std::unordered_map<NAInt8SolAttentionKernelDescriptor, std::unique_ptr<NAInt8SolAttentionKernel>>* cache) const noexcept;
};

template<> struct std::hash<NAInt8SolAttentionDescriptor> {
  size_t operator()(const NAInt8SolAttentionDescriptor& d) const noexcept;
};

#endif
