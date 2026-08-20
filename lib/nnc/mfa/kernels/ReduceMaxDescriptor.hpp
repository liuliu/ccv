#ifndef MFA_REDUCEMAXDESCRIPTOR_HPP_
#define MFA_REDUCEMAXDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct ReduceMaxKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;

  bool operator==(const ReduceMaxKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ReduceMaxKernelDescriptor>
{
  std::size_t operator()(const ReduceMaxKernelDescriptor& value) const noexcept;
};

struct ReduceMaxKernel;

struct ReduceMaxDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;
  uint32_t columnCount = 0;
  uint32_t partitionSize = 0;
  uint32_t partitionCount = 0;
  bool partitioned = false;

  bool operator==(const ReduceMaxDescriptor& rhs) const;

  std::pair<ReduceMaxKernelDescriptor, PipelineValue<ReduceMaxKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ReduceMaxKernelDescriptor, std::unique_ptr<ReduceMaxKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<ReduceMaxDescriptor>
{
  std::size_t operator()(const ReduceMaxDescriptor& value) const noexcept;
};

#endif
