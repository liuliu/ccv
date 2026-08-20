#ifndef FillIfLessThanKernel_hpp
#define FillIfLessThanKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "FillIfLessThanDescriptor.hpp"

struct FillIfLessThanKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  unsigned short threadgroupMemoryAllocation;

  MTL::Size threadgroupSize;

  uint8_t value;

  bool loadM;

  GEMMOperandPrecision memoryPrecision;

  FillIfLessThanKernel(FillIfLessThanKernelDescriptor descriptor, MTL::Device* const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
