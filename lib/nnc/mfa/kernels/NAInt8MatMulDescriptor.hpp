#ifndef MFA_NAINT8MATMULDESCRIPTOR_HPP_
#define MFA_NAINT8MATMULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct NAInt8MatMulKernelDescriptor;
struct NAInt8MatMulKernel;

struct NAInt8MatMulDescriptor {
  GEMMOperandPrecision ioPrecision = GEMMOperandPrecision::FP16;
  simd::uint3 matrixDimensions;
  bool useBias = false;

  bool operator==(const NAInt8MatMulDescriptor& rhs) const;

  std::pair<NAInt8MatMulKernelDescriptor, PipelineValue<NAInt8MatMulKernel> *> findKernel(
      MTL::Device* const device,
      const DeviceProperties &dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<NAInt8MatMulKernelDescriptor, std::unique_ptr<NAInt8MatMulKernel>> *const libraryCache) const noexcept;

private:
  NAInt8MatMulKernelDescriptor kernelDescriptor() const noexcept;
};

template<>
struct std::hash<NAInt8MatMulDescriptor>
{
  std::size_t operator()(const NAInt8MatMulDescriptor& hash) const noexcept;
};

#endif
