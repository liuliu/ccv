#ifndef MFA_MULDESCRIPTOR_HPP_
#define MFA_MULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct MulKernelDescriptor {
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const MulKernelDescriptor &rhs) const { return value == rhs.value && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<MulKernelDescriptor>
{
  std::size_t operator()(const MulKernelDescriptor& hash) const noexcept { return (size_t)hash.value | ((size_t)hash.loadM << 8) | ((size_t)hash.memoryPrecision.value << 9); }
};

struct MulKernel;

struct MulDescriptor {
  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool loadM;

  bool operator==(const MulDescriptor& rhs) const;

  std::pair<MulKernelDescriptor, PipelineValue<MulKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<MulKernelDescriptor, std::unique_ptr<MulKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<MulDescriptor>
{
  std::size_t operator()(const MulDescriptor& hash) const noexcept;
};

#endif
