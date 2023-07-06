#include "ccv_nnc_mfa.hpp"
using namespace ccv::nnc;

#include <iostream>

// MARK: - C

mfa::context* ccv_nnc_init_mfa_context(MTL::Device* device) {
  return new mfa::context(device);
}

void ccv_nnc_deinit_mfa_context(mfa::context* context) {
  delete context;
}

uint8_t ccv_nnc_mfa_context_supported(mfa::context* context) {
  return context->supported ? 1 : 0;
}

uint16_t ccv_nnc_mfa_context_log_level(mfa::context* context) {
  return context->log_level;
}

void ccv_nnc_mfa_log_message(const char* message) {
  std::cerr << METAL_LOG_HEADER << message << std::endl;
}

MTL::CommandBatch* ccv_nnc_start_command_batch(MTL::CommandQueue* command_queue) {
  return new MTL::CommandBatch(command_queue);
}

void ccv_nnc_finish_command_batch(MTL::CommandBatch* command_batch) {
  delete command_batch;
}

// MARK: - C++

template <typename T, typename U>
mfa::cache<T, U>::cache()
{
  this->map = {};
}

template <typename T, typename U>
mfa::cache<T, U>::~cache()
{
  for (auto it = map->begin(); it != map->end(); ++it) {
    delete it->second;
  }
}

// This is a workaround. If we use a template member function directly, the
// symbols won't link.
template <typename T, typename U>
inline void _mfa_cache_prepare(std::unordered_map<T, U*>* map, mfa::context* context, T hash, bool async)
{
  if (map->find(hash) == map->end()) {
    if (METAL_LOG_LEVEL(context) >= 2) {
      std::cout << METAL_LOG_HEADER << "PSO cache miss." << std::endl;
      std::cout << METAL_LOG_HEADER << "  Creating new PSO asynchronously: " << async << std::endl;
      std::cout << METAL_LOG_HEADER << "  Contents of map (before):" << std::endl;
      for (auto it = map->begin(); it != map->end(); ++it) {
        std::cout << METAL_LOG_HEADER << "    " << it->first << ": " << it->second << std::endl;
      }
    }
    
    auto* pipeline = new mfa::gemm::pipeline(context, hash, async);
    (*map)[hash] = pipeline;
    
    if (METAL_LOG_LEVEL(context) >= 2) {
      std::cout << METAL_LOG_HEADER << "  Contents of map (after):" << std::endl;
      for (auto it = map->begin(); it != map->end(); ++it) {
        std::cout << METAL_LOG_HEADER << "    " << it->first << ": " << it->second << std::endl;
      }
    }
  }
}

template <>
void mfa::cache<mfa::gemm::hash, mfa::gemm::pipeline>::prepare(mfa::context* context, mfa::gemm::hash hash, bool async)
{
  _mfa_cache_prepare(&map, context, hash, async);
}

mfa::context::context(MTL::Device* device)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  this->log_level = 0;
#if CCV_METAL_LOGGING_ENABLE
  const char* log_level_repr = getenv("CCV_METAL_LOG_LEVEL");
  if (log_level_repr) {
    int log_level_raw = atoi(log_level_repr);
    std::cerr << METAL_LOG_HEADER << "Using log level: " << log_level_raw << std::endl;
    CCV_NNC_MFA_PRECONDITION(log_level_raw >= 0 && log_level_raw <= 3)
    
    this->log_level = uint16_t(log_level_raw);
  }
#endif
  
  // Example: /usr/local/MetalFlashAttention/lib/libMetalFlashAttention.metallib
  // We need to have two different variants based on the operating system. macOS
  // will not accept a metallib compiled for iOS/tvOS/visionOS and vice versa.
  const char* metallib_path = getenv("CCV_NNC_MFA_METALLIB_PATH");
  if (!metallib_path) {
    // If a metallib was bundled with the Bazel build, you can hard-code the
    // metallib's path into the source code. Choose this path if the user hasn't
    // already set the `CCV_NNC_MFA_METALLIB_PATH` environment variable.
    constexpr const char* bundled_path = nullptr;
    
    if (bundled_path) {
      metallib_path = bundled_path;
    } else {
      this->supported = false;
      return;
    }
  }
  if (METAL_LOG_LEVEL(this) >= 1) {
    std::cerr << METAL_LOG_HEADER << "Started loading 'libMetalFlashAttention.metallib'." << std::endl;
  }
  
  // Check whether the device architecture is supported.
  this->supported = device->supportsFamily(MTL::GPUFamilyApple7);
  if (!supported) {
    if (METAL_LOG_LEVEL(this) >= 1) {
      std::cerr << METAL_LOG_HEADER << "Device architecture not supported by Metal FlashAttention." << std::endl;
    }
    return;
  }
  
  this->device = NS::RetainPtr(device);
#if TARGET_OS_MAC
  // This method is only available on macOS 13.3+. To make the code compatible
  // with macOS 12, we need to call ObjC runtime functions that check whether
  // the selector actually exists.
  device->setShouldMaximizeConcurrentCompilation(true);
#endif
  
  // Create a URL out of the path string.
  auto c_path = metallib_path;
  auto swift_path = NS::String::string(c_path, NS::UTF8StringEncoding);
  auto url = NS::URL::fileURLWithPath(swift_path);
  
  // Attempt to load the library, otherwise crash with a detailed log message.
  NS::Error* error;
  this->library = NS::TransferPtr(device->newLibrary(url, &error));
  CCV_NNC_MFA_CHECK_ERROR(error)
  
  // Notify that this finished successfully, and is not just stalling on one of
  // the previous lines of code.
  if (METAL_LOG_LEVEL(this) >= 1) {
    std::cerr << METAL_LOG_HEADER << "Finished loading 'libMetalFlashAttention.metallib'." << std::endl;
  }
  
  pool->drain();
}

MTL::CommandBatch::CommandBatch(MTL::CommandQueue* command_queue) {
  command_buffer = command_queue->commandBuffer();
  command_encoder = command_buffer->computeCommandEncoder();
}

MTL::ComputeCommandEncoder* MTL::CommandBatch::start_command(MTL::ComputePipelineState* pso) {
  CCV_NNC_MFA_PRECONDITION(command_active == 0)
  command_active = 1;
  command_encoder->setComputePipelineState(pso);
  return command_encoder;
}

void MTL::CommandBatch::finish_command(MTL::ComputeCommandEncoder* command_encoder) {
  CCV_NNC_MFA_PRECONDITION(command_active == 1)
  command_active = 0;
  batched_command_count += 1;
}

MTL::CommandBatch::~CommandBatch() {
  CCV_NNC_MFA_PRECONDITION(command_active == 0)
  command_encoder->endEncoding();
  command_buffer->commit();
}
