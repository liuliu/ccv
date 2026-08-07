#ifndef MFA_CONFORMDATAFORMATDESCRIPTOR_HPP_
#define MFA_CONFORMDATAFORMATDESCRIPTOR_HPP_

#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct ConformDataFormatKernelDescriptor {
  bool loadM;
  GEMMOperandPrecision memoryPrecision;
  constexpr bool operator==(const ConformDataFormatKernelDescriptor& rhs) const { return loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<ConformDataFormatKernelDescriptor>
{
  std::size_t operator()(const ConformDataFormatKernelDescriptor& hash) const noexcept { return (hash.loadM ? 1 : 0) | ((std::size_t)hash.memoryPrecision.value << 1); }
};

struct ConformDataFormatKernel;

struct ConformDataFormatDescriptor {
  uint32_t rowCount;
  uint32_t headDim;
  uint32_t preservedTail;

  bool loadM;

  GEMMOperandPrecision memoryPrecision;

  bool operator==(const ConformDataFormatDescriptor& rhs) const;

  std::pair<ConformDataFormatKernelDescriptor, PipelineValue<ConformDataFormatKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ConformDataFormatKernelDescriptor, std::unique_ptr<ConformDataFormatKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<ConformDataFormatDescriptor>
{
  std::size_t operator()(const ConformDataFormatDescriptor& hash) const noexcept;
};

#endif
