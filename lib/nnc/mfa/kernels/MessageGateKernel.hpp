#ifndef MFA_MESSAGEGATEKERNEL_HPP_
#define MFA_MESSAGEGATEKERNEL_HPP_

#include "MessageGateDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <string>

struct MessageGateKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  // Every kernel's threadgroups.
  MTL::Size threadgroupSize;
  // The gate's tiles.
  uint32_t rowsPerThreadgroup;
  uint32_t lanesPerChannel;
  uint32_t channelsPerThreadgroup;
  // The few-rows mix: a threadgroup takes one rank of mixRows rows.
  uint32_t mixRows;

  GEMMOperandPrecision memoryPrecision;
  GEMMOperandPrecision downPrecision;
  GEMMOperandPrecision weightPrecision;
  bool downQuantized;
  bool quantized;
  uint32_t rank;

  MessageGateKernel(MessageGateKernelDescriptor descriptor, MTL::Device* const device);

private:
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif
