#ifndef MFA_SPARSEINDEXEDATTENTIONDESCRIPTOR_HPP_
#define MFA_SPARSEINDEXEDATTENTIONDESCRIPTOR_HPP_

#include <memory>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"
#include "SparseIndexedAttentionKernelDescriptor.hpp"

struct SparseIndexedAttentionKernel;

struct SparseIndexedAttentionDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP16;
  bool attentionSinks = false;
  uint32_t T = 0;
  uint32_t denseRows = 0;
  uint32_t sparseRows = 0;
  uint32_t H = 0;
  uint32_t D = 0;
  uint32_t K = 0;
  bool isCausal = false;
  uint32_t slidingWindow = 0;
  uint32_t sinkHeadStride = 0;
  float scale = 1;
  bool loadRows = false;

  bool operator==(const SparseIndexedAttentionDescriptor& rhs) const;

  std::pair<SparseIndexedAttentionKernelDescriptor, PipelineValue<SparseIndexedAttentionKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SparseIndexedAttentionKernelDescriptor, std::unique_ptr<SparseIndexedAttentionKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<SparseIndexedAttentionDescriptor>
{
  std::size_t operator()(const SparseIndexedAttentionDescriptor& hash) const noexcept;
};

#endif
