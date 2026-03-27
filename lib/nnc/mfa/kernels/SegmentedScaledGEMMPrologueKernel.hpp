#ifndef SEGMENTEDSCALEDGEMMPROLOGUEKERNEL_HPP
#define SEGMENTEDSCALEDGEMMPROLOGUEKERNEL_HPP

#include "SegmentedScaledGEMMPrologueKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

class CodeWriter;

struct SegmentedScaledGEMMPrologueKernel {
  NS::SharedPtr<MTL::Library> library;
  std::string source;
  GEMMOperandPrecision ioPrecision;
  bool useBias;

  SegmentedScaledGEMMPrologueKernel(SegmentedScaledGEMMPrologueKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string createSource() const noexcept;
};

#endif
