#ifndef MFA_ADAMDESCRIPTOR_HPP_
#define MFA_ADAMDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct AdamKernelDescriptor {
  uint8_t adamw;
  uint8_t amsgrad;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const AdamKernelDescriptor& rhs) const { return adamw == rhs.adamw && amsgrad == rhs.amsgrad && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<AdamKernelDescriptor>
{
  std::size_t operator()(const AdamKernelDescriptor& hash) const noexcept {
    return std::hash<int>()((int)hash.adamw | ((int)hash.amsgrad << 8) | ((int)hash.memoryPrecision.value << 16));
  }
};

struct AdamKernel;

struct AdamDescriptor {
  uint8_t adamw;

  uint8_t amsgrad;

  GEMMOperandPrecision memoryPrecision;

  float rate;

  float scale;

  float beta1;

  float beta2;

  float decay;

  float epsilon;

  uint32_t length;

  bool operator==(const AdamDescriptor& rhs) const;

  std::pair<AdamKernelDescriptor, PipelineValue<AdamKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<AdamKernelDescriptor, std::unique_ptr<AdamKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<AdamDescriptor>
{
  std::size_t operator()(const AdamDescriptor& hash) const noexcept;
};

#endif
