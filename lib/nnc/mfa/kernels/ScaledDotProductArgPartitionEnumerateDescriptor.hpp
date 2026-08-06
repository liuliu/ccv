#ifndef MFA_SCALEDDOTPRODUCTARGPARTITIONENUMERATEDESCRIPTOR_HPP_
#define MFA_SCALEDDOTPRODUCTARGPARTITIONENUMERATEDESCRIPTOR_HPP_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "PipelineValue.hpp"

struct ScaledDotProductArgPartitionEnumerateKernelDescriptor;
struct ScaledDotProductArgPartitionEnumerateKernel;

struct ScaledDotProductArgPartitionEnumerateDescriptor {
  uint32_t T = 0;
  uint32_t C = 0;
  uint32_t kth = 0;
  uint32_t compressionRatio = 1;
  int32_t queryOffset = 0;
  bool isCausal = false;

  bool operator==(const ScaledDotProductArgPartitionEnumerateDescriptor& rhs) const;

  std::pair<ScaledDotProductArgPartitionEnumerateKernelDescriptor, PipelineValue<ScaledDotProductArgPartitionEnumerateKernel> *> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ScaledDotProductArgPartitionEnumerateKernelDescriptor, std::unique_ptr<ScaledDotProductArgPartitionEnumerateKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<ScaledDotProductArgPartitionEnumerateDescriptor>
{
  std::size_t operator()(const ScaledDotProductArgPartitionEnumerateDescriptor& hash) const noexcept;
};

#endif
