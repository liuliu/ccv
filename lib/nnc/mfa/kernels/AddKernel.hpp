#ifndef AddKernel_hpp
#define AddKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "AddDescriptor.hpp"

struct AddKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  MTL::Size threadgroupSize;

  uint8_t args;

  uint8_t value;

  bool loadM;

  uint8_t negative_mask;

  uint8_t broadcast;

  GEMMOperandPrecision memoryPrecision;

  AddKernel(AddKernelDescriptor descriptor, MTL::Device *const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif /* AddKernel_hpp */
