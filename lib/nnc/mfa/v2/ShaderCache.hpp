#ifndef MFA_SHADER_CACHE_HPP_
#define MFA_SHADER_CACHE_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

using TypeInfoRef = std::reference_wrapper<const std::type_info>;

struct Hasher {
  std::size_t operator()(TypeInfoRef code) const {
    return code.get().hash_code();
  }
};

struct EqualTo {
  bool operator()(TypeInfoRef lhs, TypeInfoRef rhs) const {
    return lhs.get() == rhs.get();
  }
};

struct TypeErasedUnorderedMap {
	virtual ~TypeErasedUnorderedMap() = default;
};

template<typename Key, typename Value>
struct UnorderedMapWrapper: public TypeErasedUnorderedMap {
	std::unordered_map<Key, Value> map;
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
struct ShaderCache {
private:
  /// WARNING: Not thread safe. But will the DSL interpreter even use
  /// multithreading?
  std::unordered_map<TypeInfoRef, std::unique_ptr<TypeErasedUnorderedMap>, Hasher, EqualTo> libraryCache;
  
  /// WARNING: Not thread safe. But will the DSL interpreter even use
  /// multithreading?
  std::unordered_map<TypeInfoRef, std::unique_ptr<TypeErasedUnorderedMap>, Hasher, EqualTo> pipelineCache;
public:
  /// Implementation of the logic for choosing between 'device' and
  /// 'threadgroup' store.
  ///
  /// ## C++ Adaptation
  ///
  /// Wrap every call to this function in an autoreleasepool.
  template<typename Kernel, typename Descriptor, typename KernelDescriptor>
  PipelineValue<Kernel>* findKernel(Descriptor descriptor, MTL::Device *const device, const DeviceProperties &dprops) noexcept {
    UnorderedMapWrapper<Descriptor, std::unique_ptr<PipelineValue<Kernel>>> *pipelineCache = static_cast<UnorderedMapWrapper<Descriptor, std::unique_ptr<PipelineValue<Kernel>>> *>(this->pipelineCache.try_emplace(typeid(Descriptor), std::make_unique<UnorderedMapWrapper<Descriptor, std::unique_ptr<PipelineValue<Kernel>>>>()).first->second.get());
    auto iterator = pipelineCache->map.find(descriptor);
    if (iterator != pipelineCache->map.end()) {
      return iterator->second.get();
    }
    UnorderedMapWrapper<KernelDescriptor, std::unique_ptr<Kernel>> *libraryCache = static_cast<UnorderedMapWrapper<KernelDescriptor, std::unique_ptr<Kernel>> *>(this->libraryCache.try_emplace(typeid(KernelDescriptor), std::make_unique<UnorderedMapWrapper<KernelDescriptor, std::unique_ptr<Kernel>>>()).first->second.get());
    auto result = descriptor.findKernel(device, dprops, &libraryCache->map);
    pipelineCache->map[descriptor] = std::unique_ptr<PipelineValue<Kernel>>(result.second);
    return result.second;
  }
};

#endif
