#ifndef MFA_SEGMENTEDSCALEDGEMMDESCRIPTOR_HPP_
#define MFA_SEGMENTEDSCALEDGEMMDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SegmentedScaledGEMMKernelDescriptor;
struct SegmentedScaledGEMMKernel;

struct SegmentedScaledGEMMDescriptor {
  GEMMOperandPrecision ioPrecision;
  simd::uint4 matrixDimensions; // M, N, K, segments.
  bool useBias;

  bool operator==(const SegmentedScaledGEMMDescriptor& rhs) const;

  std::pair<SegmentedScaledGEMMKernelDescriptor, PipelineValue<SegmentedScaledGEMMKernel>*> findKernel(
      MTL::Device* const device,
      const DeviceProperties& dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<SegmentedScaledGEMMKernelDescriptor, std::unique_ptr<SegmentedScaledGEMMKernel>>* const libraryCache) const noexcept;

private:
  SegmentedScaledGEMMKernelDescriptor kernelDescriptor() const noexcept;
};

template<>
struct std::hash<SegmentedScaledGEMMDescriptor>
{
  std::size_t operator()(const SegmentedScaledGEMMDescriptor& hash) const noexcept;
};

#endif
