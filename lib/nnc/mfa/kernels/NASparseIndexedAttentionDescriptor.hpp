#ifndef MFA_NASPARSEINDEXEDATTENTIONDESCRIPTOR_HPP_
#define MFA_NASPARSEINDEXEDATTENTIONDESCRIPTOR_HPP_

#include <memory>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "NASparseIndexedAttentionKernelDescriptor.hpp"
#include "PipelineValue.hpp"

struct NASparseIndexedAttentionKernel;

struct NASparseIndexedAttentionDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP16;
  bool attentionSinks = false;
  uint32_t T = 0;
  uint32_t denseRows = 0;
  uint32_t sparseRows = 0;
  uint32_t H = 0;
  uint32_t K = 0;
  bool isCausal = false;
  uint32_t slidingWindow = 0;
  uint32_t sinkHeadStride = 0;
  float scale = 1;
  NASparseIndexedAttentionVariant variant = NASparseIndexedAttentionVariant::Threadgroup16;

  bool operator==(const NASparseIndexedAttentionDescriptor& rhs) const;

  std::pair<NASparseIndexedAttentionKernelDescriptor, PipelineValue<NASparseIndexedAttentionKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NASparseIndexedAttentionKernelDescriptor, std::unique_ptr<NASparseIndexedAttentionKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<NASparseIndexedAttentionDescriptor>
{
  std::size_t operator()(const NASparseIndexedAttentionDescriptor& hash) const noexcept;
};

#endif
