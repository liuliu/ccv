#ifndef SEGMENTEDGEMMPROLOGUEKernel_hpp
#define SEGMENTEDGEMMPROLOGUEKernel_hpp

#include "SegmentedGEMMPrologueKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct SegmentedGEMMPrologueKernel {
  NS::SharedPtr<MTL::Library> library;

  NS::SharedPtr<MTL::Function> function;
  
  std::string source;

  GEMMOperandPrecisions memoryPrecisions;

  bool useBias;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  uint16_t threadgroupSize;

  SegmentedGEMMPrologueKernel(SegmentedGEMMPrologueKernelDescriptor descriptor, MTL::Device *const device);

private:
  std::string createSource() const noexcept;
};

#endif /* SegmentedGEMMPrologueKernel_hpp */

