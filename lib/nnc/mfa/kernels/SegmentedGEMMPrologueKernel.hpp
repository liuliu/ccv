#ifndef SEGMENTEDGEMMPROLOGUEKernel_hpp
#define SEGMENTEDGEMMPROLOGUEKernel_hpp

#include "SegmentedGEMMPrologueKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct SegmentedGEMMPrologueKernel {
  NS::SharedPtr<MTL::Library> library;

  std::string source;

  GEMMOperandPrecisions memoryPrecisions;

  bool useBias;

  /// Whether this GEMM kernel has a separate splitK reduction kernel.
  uint16_t splitK;

  SegmentedGEMMPrologueKernel(SegmentedGEMMPrologueKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string createSource() const noexcept;
};

#endif /* SegmentedGEMMPrologueKernel_hpp */

