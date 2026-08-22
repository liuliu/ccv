#ifndef ClampKernel_hpp
#define ClampKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ClampDescriptor.hpp"
#include <string>

struct ClampKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  MTL::Size threadgroupSize;
  uint8_t value;
  uint8_t bounds;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;

  ClampKernel(ClampKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
