#ifndef AttentionR1Kernel_hpp
#define AttentionR1Kernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

#include <simd/simd.h>
#include <string>

#include "AttentionR1Descriptor.hpp"

struct AttentionR1Kernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  GEMMOperandPrecision memoryPrecision;

  bool loadC;

  AttentionR1Kernel(AttentionR1KernelDescriptor descriptor, MTL::Device* const device);

  uint32_t threadgroupMemoryAllocation(const AttentionR1Descriptor& descriptor) const noexcept;

  uint32_t threadgroupSize(const AttentionR1Descriptor& descriptor) const noexcept;

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
