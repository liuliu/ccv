#ifndef MFA_ATTENTIONRESIDUALDESCRIPTOR_HPP_
#define MFA_ATTENTIONRESIDUALDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <simd/simd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct AttentionResidualKernelDescriptor {
  // Each query has its own output buffer.
  uint32_t queryCount;
  GEMMOperandPrecision memoryPrecision;
  bool operator==(const AttentionResidualKernelDescriptor& rhs) const { return queryCount == rhs.queryCount && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<AttentionResidualKernelDescriptor> {
  std::size_t operator()(const AttentionResidualKernelDescriptor& descriptor) const noexcept {
    return std::hash<int>()((int)descriptor.queryCount | ((int)descriptor.memoryPrecision.value << 8));
  }
};

struct AttentionResidualKernel;

struct AttentionResidualDescriptor {
  uint32_t queryCount;
  GEMMOperandPrecision memoryPrecision;
  // Function constants of the pipeline. The rows come from the grid.
  uint32_t channels;
  uint32_t blockCount;
  float epsilon;
  float scale;

  bool operator==(const AttentionResidualDescriptor& rhs) const;
  std::pair<AttentionResidualKernelDescriptor, PipelineValue<AttentionResidualKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<AttentionResidualKernelDescriptor, std::unique_ptr<AttentionResidualKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<AttentionResidualDescriptor> {
  std::size_t operator()(const AttentionResidualDescriptor& descriptor) const noexcept;
};

#endif
