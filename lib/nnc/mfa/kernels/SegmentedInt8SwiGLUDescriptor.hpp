#ifndef MFA_SEGMENTEDINT8SWIGLUDESCRIPTOR_HPP_
#define MFA_SEGMENTEDINT8SWIGLUDESCRIPTOR_HPP_

#include <cstdint>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"
#include "SegmentedInt8SwiGLUKernel.hpp"
#include <simd/simd.h>

struct SegmentedInt8SwiGLUDescriptor {
  simd::uint2 matrixDimensions; // N, K.
  uint32_t expertCount;
  uint32_t routeCount;
  uint32_t format;
  uint32_t broadcastInput;
  GEMMOperandPrecision memoryPrecision;

  bool operator==(const SegmentedInt8SwiGLUDescriptor& rhs) const;
  uint32_t groupSize() const noexcept;
  uint32_t groupsPerRow() const noexcept;
  uint64_t inputScaleOffset() const noexcept;
  uint64_t weightExpertStride() const noexcept;

  std::pair<
    SegmentedInt8SwiGLUKernelDescriptor,
    PipelineValue<SegmentedInt8SwiGLUKernel>*> findKernel(
      MTL::Device* device,
      const DeviceProperties& dprops,
      NS::Array* binaryArchivesToRead,
      MTL::BinaryArchive* binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<
        SegmentedInt8SwiGLUKernelDescriptor,
        std::unique_ptr<SegmentedInt8SwiGLUKernel>>* libraryCache) const noexcept;
};

template<>
struct std::hash<SegmentedInt8SwiGLUDescriptor>
{
  std::size_t operator()(const SegmentedInt8SwiGLUDescriptor& value) const noexcept;
};

#endif
