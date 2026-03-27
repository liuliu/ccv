#ifndef MFA_SEGMENTEDSCALEDGEMMPROLOGUEDESCRIPTOR_HPP_
#define MFA_SEGMENTEDSCALEDGEMMPROLOGUEDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct SegmentedScaledGEMMPrologueKernelDescriptor;
struct SegmentedScaledGEMMPrologueKernel;

struct SegmentedScaledGEMMPrologueDescriptor {
  simd::uint3 matrixDimensions;
  simd::ushort3 blockDimensions;
  GEMMOperandPrecision ioPrecision;
  bool useBias;
  uint16_t threadgroupSize;

  bool operator==(const SegmentedScaledGEMMPrologueDescriptor& rhs) const;

  std::pair<SegmentedScaledGEMMPrologueKernelDescriptor, PipelineValue<SegmentedScaledGEMMPrologueKernel> *> findKernel(
      MTL::Device* const device,
      const DeviceProperties &dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<SegmentedScaledGEMMPrologueKernelDescriptor, std::unique_ptr<SegmentedScaledGEMMPrologueKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<SegmentedScaledGEMMPrologueDescriptor>
{
  std::size_t operator()(const SegmentedScaledGEMMPrologueDescriptor& hash) const noexcept;
};

#endif
