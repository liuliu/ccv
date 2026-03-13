#ifndef MFA_CONV3DDESCRIPTOR_HPP_
#define MFA_CONV3DDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct Conv3DKernelDescriptor;
struct Conv3DKernel;

struct Conv3DDescriptor {
  uint64_t dataType;
  uint32_t batchDimension = 1;
  uint32_t inputChannels;
  uint32_t outputChannels;
  uint32_t paddingLeft;
  uint32_t paddingRight;
  uint32_t paddingTop;
  uint32_t paddingBottom;
  simd::uint3 matrixDimensions;
  simd::uint3 kernelDimensions;
  bool useBias;

  bool operator==(const Conv3DDescriptor& rhs) const;

  std::pair<Conv3DKernelDescriptor, PipelineValue<Conv3DKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Conv3DKernelDescriptor, std::unique_ptr<Conv3DKernel>> *const libraryCache) const noexcept;

private:
  Conv3DKernelDescriptor kernelDescriptor() const noexcept;
};

template<>
struct std::hash<Conv3DDescriptor>
{
  std::size_t operator()(const Conv3DDescriptor& hash) const noexcept;
};

#endif
