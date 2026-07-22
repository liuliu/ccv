#ifndef MFA_WALSHHADAMARDTRANSFORMDESCRIPTOR_HPP_
#define MFA_WALSHHADAMARDTRANSFORMDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct WalshHadamardTransformKernelDescriptor {
  GEMMOperandPrecision memoryPrecision;
  uint8_t loadM;
  constexpr bool operator==(const WalshHadamardTransformKernelDescriptor& rhs) const { return memoryPrecision == rhs.memoryPrecision && loadM == rhs.loadM; }
};

template<>
struct std::hash<WalshHadamardTransformKernelDescriptor>
{
  std::size_t operator()(const WalshHadamardTransformKernelDescriptor& hash) const noexcept { return (size_t)hash.memoryPrecision.value | ((size_t)hash.loadM << 8); }
};

struct WalshHadamardTransformKernel;

struct WalshHadamardTransformDescriptor {
  GEMMOperandPrecision memoryPrecision;

  uint32_t rowCount;

  uint32_t dim;

  float scale;

  bool loadM;

  bool operator==(const WalshHadamardTransformDescriptor& rhs) const;

  std::pair<WalshHadamardTransformKernelDescriptor, PipelineValue<WalshHadamardTransformKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<WalshHadamardTransformKernelDescriptor, std::unique_ptr<WalshHadamardTransformKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<WalshHadamardTransformDescriptor>
{
  std::size_t operator()(const WalshHadamardTransformDescriptor& hash) const noexcept;
};

#endif
