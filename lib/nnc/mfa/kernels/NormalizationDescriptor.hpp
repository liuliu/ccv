#ifndef MFA_NORMALIZATIONDESCRIPTOR_HPP_
#define MFA_NORMALIZATIONDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct NormalizationKernelDescriptor {
  uint64_t dataType;
  uint32_t channelCount;
  uint32_t channelGroups;
  uint32_t sequenceCount;
  float epsilon;
  uint8_t elementwiseAffine;
  uint8_t scaleTranslationBatched;
  uint8_t normalizationType;
  uint8_t reuseSavedStatistics;
  uint32_t srcBatchStride;
  uint32_t dstBatchStride;

  bool operator==(const NormalizationKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NormalizationKernelDescriptor>
{
  std::size_t operator()(const NormalizationKernelDescriptor& hash) const noexcept;
};

struct NormalizationKernel;

struct NormalizationDescriptor {
  uint64_t dataType;
  uint32_t channelCount;
  uint32_t channelGroups;
  uint32_t sequenceCount;
  float epsilon;
  uint8_t elementwiseAffine;
  uint8_t scaleTranslationBatched;
  uint8_t normalizationType;
  uint8_t reuseSavedStatistics;
  uint32_t srcBatchStride;
  uint32_t dstBatchStride;

  bool operator==(const NormalizationDescriptor& rhs) const;

  std::pair<NormalizationKernelDescriptor, PipelineValue<NormalizationKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NormalizationKernelDescriptor, std::unique_ptr<NormalizationKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<NormalizationDescriptor>
{
  std::size_t operator()(const NormalizationDescriptor& hash) const noexcept;
};

#endif
