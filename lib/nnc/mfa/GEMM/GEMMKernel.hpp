#ifndef GEMMKernel_hpp
#define GEMMKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp" // I guess this imports <string>.
#include <simd/simd.h>

#include "GEMMKernelDescriptor.hpp"

struct GEMMKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;
  
  /// A copy of the block dimensions from the descriptor.
  ///
  /// ## C++ Adaptation
  ///
  /// Mapping from the Swift implementation:
  /// - M -> blockDimensions[0]
  /// - N -> blockDimensions[1]
  /// - K -> blockDimensions[2]
  simd::ushort3 blockDimensions;
  
  // If you allocate threadgroup memory after compiling the kernel, the code
    // has higher performance.
  uint16_t threadgroupMemoryAllocation;
    
    // The number of threads per group.
  uint16_t threadgroupSize;
  
  GEMMKernel(GEMMKernelDescriptor descriptor);
};

#endif /* GEMMKernel_hpp */

