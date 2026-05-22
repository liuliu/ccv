#ifndef MFA_NAINT8ATTENTIONDESCRIPTOR_HPP_
#define MFA_NAINT8ATTENTIONDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "AttentionOperand.hpp"
#include "AttentionKernelType.hpp"

struct NAInt8AttentionKernelDescriptor;
struct NAInt8AttentionKernel;

struct NAInt8AttentionDescriptor {
  uint32_t batchDimension = 1;
  unsigned short Hq = 1;
  unsigned short Hk = 1;
  GEMMOperandPrecision ioPrecision = GEMMOperandPrecision::FP16;
  bool lowPrecisionIntermediates = false;
  simd::uint3 matrixDimensions;
  AttentionOperands<unsigned int> batchStrides;
  AttentionKernelType type = AttentionKernelType::forward;
  float scale;
  bool isCausal = false;
  bool masked = false;
  bool isVarlen = false;
  bool attentionSinks = false;
  uint32_t maskBatchStride = 0;

  bool operator==(const NAInt8AttentionDescriptor& rhs) const;

  std::pair<NAInt8AttentionKernelDescriptor, PipelineValue<NAInt8AttentionKernel> *> findKernel(
      MTL::Device* const device,
      const DeviceProperties &dprops,
      NS::Array* const binaryArchivesToRead,
      MTL::BinaryArchive* const binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<NAInt8AttentionKernelDescriptor, std::unique_ptr<NAInt8AttentionKernel>> *const libraryCache) const noexcept;

private:
  NAInt8AttentionKernelDescriptor kernelDescriptor() const noexcept;
};

template<>
struct std::hash<NAInt8AttentionDescriptor>
{
  std::size_t operator()(const NAInt8AttentionDescriptor& hash) const noexcept;
};

#endif
