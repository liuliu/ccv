#ifndef LogKernel_hpp
#define LogKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "LogDescriptor.hpp"
#include <string>

struct LogKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  MTL::Size threadgroupSize;
  uint8_t value;
  bool loadM;
  GEMMOperandPrecision memoryPrecision;

  LogKernel(LogKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
