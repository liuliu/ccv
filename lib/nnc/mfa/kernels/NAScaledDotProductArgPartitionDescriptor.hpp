#ifndef MFA_NASCALEDDOTPRODUCTARGPARTITIONDESCRIPTOR_HPP_
#define MFA_NASCALEDDOTPRODUCTARGPARTITIONDESCRIPTOR_HPP_

#include <unordered_map>
#include <utility>
#include <memory>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct NAScaledDotProductArgPartitionKernelDescriptor;
struct NAScaledDotProductArgPartitionKernel;

struct NAScaledDotProductArgPartitionDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP32;
  uint32_t T = 0;
  uint32_t C = 0;
  uint32_t H = 0;
  uint32_t D = 0;
  uint32_t kth = 0;
  uint32_t compressionRatio = 1;
  int32_t queryOffset = 0;
  float scale = 1;
  bool isCausal = false;
  uint16_t scoreBlockM = 16;
  uint16_t scoreBlockN = 32;
  uint16_t scoreSIMDGroups = 4;

  bool operator==(const NAScaledDotProductArgPartitionDescriptor& rhs) const;

  std::pair<NAScaledDotProductArgPartitionKernelDescriptor, PipelineValue<NAScaledDotProductArgPartitionKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAScaledDotProductArgPartitionKernelDescriptor, std::unique_ptr<NAScaledDotProductArgPartitionKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<NAScaledDotProductArgPartitionDescriptor>
{
  std::size_t operator()(const NAScaledDotProductArgPartitionDescriptor& hash) const noexcept;
};

#endif
