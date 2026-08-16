#ifndef MFA_ARGMAXDESCRIPTOR_HPP_
#define MFA_ARGMAXDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct ArgmaxKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;

  bool operator==(const ArgmaxKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ArgmaxKernelDescriptor>
{
  std::size_t operator()(const ArgmaxKernelDescriptor& value) const noexcept;
};

struct ArgmaxKernel;

struct ArgmaxDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;
  uint32_t columnCount = 0;
  uint32_t partitionSize = 0;
  uint32_t partitionCount = 0;
  bool gumbel = false;
  bool partitioned = false;

  bool operator==(const ArgmaxDescriptor& rhs) const;

  std::pair<ArgmaxKernelDescriptor, PipelineValue<ArgmaxKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ArgmaxKernelDescriptor, std::unique_ptr<ArgmaxKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<ArgmaxDescriptor>
{
  std::size_t operator()(const ArgmaxDescriptor& value) const noexcept;
};

#endif
