#ifndef MFA_ANEROWWISETRANSFORMDESCRIPTOR_HPP_
#define MFA_ANEROWWISETRANSFORMDESCRIPTOR_HPP_

#include "ANERowwiseTransformKernelDescriptor.hpp"
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct ANERowwiseTransformKernel;

struct ANERowwiseTransformDescriptor {
  GEMMOperandPrecision memoryPrecision;
  uint32_t M;
  uint32_t paddedM;
  uint32_t batchDimension;
  uint32_t N;
  uint32_t K;
  uint32_t batchStrideA;
  uint32_t batchStrideC;
  uint32_t sourceRowOffset;
  uint32_t outputRowOffset;
  uint32_t activationScaleBatchStride;
  uint32_t activationScaleRowOffset;

  bool operator==(const ANERowwiseTransformDescriptor& rhs) const;

  std::pair<ANERowwiseTransformKernelDescriptor, PipelineValue<ANERowwiseTransformKernel>*> findKernel(
      MTL::Device* const device,
      const DeviceProperties& dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<ANERowwiseTransformKernelDescriptor, std::unique_ptr<ANERowwiseTransformKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<ANERowwiseTransformDescriptor>
{
  std::size_t operator()(const ANERowwiseTransformDescriptor& hash) const noexcept;
};

#endif
