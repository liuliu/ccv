#ifndef MFA_INT8GEMVKERNEL_HPP_
#define MFA_INT8GEMVKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "Int8GemvDescriptor.hpp"

struct Int8GemvKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  uint8_t fusedBias;

  uint8_t mrows;

  uint8_t batched;

  uint32_t format;

  GEMMOperandPrecision memoryPrecision;

  Int8GemvKernel(Int8GemvKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
  void specializeBatchedSource(std::string& shader) const noexcept;
};

#endif
