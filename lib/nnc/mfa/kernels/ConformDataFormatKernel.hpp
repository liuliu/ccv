#ifndef ConformDataFormatKernel_hpp
#define ConformDataFormatKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ConformDataFormatDescriptor.hpp"

struct ConformDataFormatKernel {
  NS::SharedPtr<MTL::Library> library;

  MTL::Size threadgroupSize;

  bool loadM;

  GEMMOperandPrecision memoryPrecision;

  ConformDataFormatKernel(ConformDataFormatKernelDescriptor descriptor, MTL::Device* const device);

  MTL::Size gridSize(uint32_t rowCount, uint32_t headDim, uint32_t preservedTail) const noexcept;

  std::string createSource() const noexcept;
};

#endif
