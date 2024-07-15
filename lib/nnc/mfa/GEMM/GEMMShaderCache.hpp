#ifndef GEMMShaderCache_hpp
#define GEMMShaderCache_hpp

#include "GEMMDescriptor.hpp"
#include "GEMMKernelDescriptor.hpp"
#include "GEMMKernel.hpp"
#include <unordered_map>

struct GEMMPipelineValue {
  GEMMKernel* kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

/// A reference implementation of shader caching.
///
/// One good design for a shader caching mechanism:
/// - Two key-value caches.
/// - The first caches `MTLLibrary` objects.
///   - Large latency
///   - Small number of combinatorial possibilities, likely to be shared by
///     matrices with a different size.
///   - Don't bother with serializing Metal binary archives to disk. You are
///     already utilizing the system-wide Metal shader cache.
/// - The second caches `MTLComputePipelineState` objects.
///   - Instantiations of the `MTLLibrary` with different function constants.
///   - Less latency than compiling from source, but still non-negligible. You
///     can't spawn a new PSO during every call to a matrix multiplication.
struct GEMMShaderCache {
  /// WARNING: Not thread safe. But will the DSL interpreter even use
  /// multithreading?
  static std::unordered_map<GEMMKernelKey, GEMMKernel*> libraryCache;
  
  /// WARNING: Not thread safe. But will the DSL interpreter even use
  /// multithreading?
  static std::unordered_map<GEMMKey, GEMMPipelineValue*> pipelineCache;
  
  /// Implementation of the logic for choosing between 'device' and
  /// 'threadgroup' store.
  ///
  /// ## C++ Adaptation
  ///
  /// Wrap every call to this function in an autoreleasepool.
  static GEMMPipelineValue* fetchKernel(GEMMDescriptor descriptor);
};

#endif /* GEMMShaderCache_hpp */
