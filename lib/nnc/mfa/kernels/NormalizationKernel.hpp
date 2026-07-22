#ifndef NormalizationKernel_hpp
#define NormalizationKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "NormalizationDescriptor.hpp"

struct NormalizationKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  MTL::Size gridSize;

  MTL::Size groupSize;

  uint64_t dataType;
  uint32_t channelCount;
  uint32_t channelGroups;
  uint32_t sequenceCount;
  float epsilon;
  float scale;
  uint8_t elementwiseAffine;
  uint8_t scaleTranslationBatched;
  uint8_t normalizationType;
  uint8_t reuseSavedStatistics;
  bool loadM;
  uint32_t srcBatchStride;
  uint32_t dstBatchStride;

  NormalizationKernel(NormalizationKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
};

#endif
