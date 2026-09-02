#ifndef MFA_INT8SWIGLUDESCRIPTOR_HPP_
#define MFA_INT8SWIGLUDESCRIPTOR_HPP_

#include <cstdint>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "Int8SwiGLUKernel.hpp"
#include "PipelineValue.hpp"

struct Int8SwiGLUDescriptor {
  uint32_t N;
  uint32_t K;
  float clamp;
  GEMMOperandPrecision memoryPrecision;

  bool operator==(const Int8SwiGLUDescriptor& rhs) const;

  std::pair<
    Int8SwiGLUKernelDescriptor,
    PipelineValue<Int8SwiGLUKernel>*> findKernel(
      MTL::Device* device,
      const DeviceProperties& dprops,
      NS::Array* binaryArchivesToRead,
      MTL::BinaryArchive* binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<
        Int8SwiGLUKernelDescriptor,
        std::unique_ptr<Int8SwiGLUKernel>>* libraryCache) const noexcept;
};

template<>
struct std::hash<Int8SwiGLUDescriptor>
{
  std::size_t operator()(const Int8SwiGLUDescriptor& value) const noexcept;
};

#endif
