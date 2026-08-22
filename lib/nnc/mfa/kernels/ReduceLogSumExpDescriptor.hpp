#ifndef MFA_REDUCELOGSUMEXPDESCRIPTOR_HPP_
#define MFA_REDUCELOGSUMEXPDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct ReduceLogSumExpKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;

  bool operator==(const ReduceLogSumExpKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ReduceLogSumExpKernelDescriptor>
{
  std::size_t operator()(const ReduceLogSumExpKernelDescriptor& value) const noexcept;
};

struct ReduceLogSumExpKernel;

struct ReduceLogSumExpDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;
  uint32_t columnCount = 0;
  uint32_t partitionSize = 0;
  uint32_t partitionCount = 0;
  float scale = 1;
  bool partitioned = false;

  bool operator==(const ReduceLogSumExpDescriptor& rhs) const;

  std::pair<ReduceLogSumExpKernelDescriptor, PipelineValue<ReduceLogSumExpKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ReduceLogSumExpKernelDescriptor, std::unique_ptr<ReduceLogSumExpKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<ReduceLogSumExpDescriptor>
{
  std::size_t operator()(const ReduceLogSumExpDescriptor& value) const noexcept;
};

#endif
