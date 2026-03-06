#ifndef MFA_ADDDESCRIPTOR_HPP_
#define MFA_ADDDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct AddKernelDescriptor {
  uint8_t args;
  uint8_t value;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const AddKernelDescriptor &rhs) const { return args == rhs.args && value == rhs.value && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<AddKernelDescriptor>
{
  std::size_t operator()(const AddKernelDescriptor& hash) const noexcept { return (size_t)hash.value; }
};

struct AddKernel;

struct AddDescriptor {
  uint8_t args;

  uint8_t value;

  GEMMOperandPrecision memoryPrecision;

  uint32_t length;

  bool operator==(const AddDescriptor& rhs) const;

  std::pair<AddKernelDescriptor, PipelineValue<AddKernel> *> findKernel(MTL::Device* const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<AddKernelDescriptor, std::unique_ptr<AddKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<AddDescriptor>
{
  std::size_t operator()(const AddDescriptor& hash) const noexcept;
};

#endif

