#ifndef MFA_NACONV3DDESCRIPTOR_HPP_
#define MFA_NACONV3DDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct NAConv3DKernelDescriptor;
struct NAConv3DKernel;

struct NAConv3DDescriptor {
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

  bool operator==(const NAConv3DDescriptor& rhs) const;

  std::pair<NAConv3DKernelDescriptor, PipelineValue<NAConv3DKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAConv3DKernelDescriptor, std::unique_ptr<NAConv3DKernel>> *const libraryCache) const noexcept;

private:
  NAConv3DKernelDescriptor kernelDescriptor() const noexcept;
};

template<>
struct std::hash<NAConv3DDescriptor>
{
  std::size_t operator()(const NAConv3DDescriptor& hash) const noexcept;
};

#endif
