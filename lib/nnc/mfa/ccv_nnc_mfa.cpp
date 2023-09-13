#include "ccv_nnc_mfa.hpp"
#include "libmfa.inc"
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

mtl_buffer_t* ccv_nnc_mfa_request_scratch(ccv_nnc_mfa_context_t* context, const uint64_t size) {
  return context->request_scratch(size);
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
  map = {};
}

template <typename T, typename U>
mfa::cache<T, U>::~cache()
{
  for (auto it = map.begin(); it != map.end(); ++it) {
    delete it->second;
  }
}

// This is a workaround. If we use a template member function directly, the
// symbols won't link.
template <typename T, typename U>
inline void _mfa_cache_prepare(std::unordered_map<T, U*>* map, mfa::context* context, T hash)
{
  if (map->find(hash) == map->end()) {
    if (METAL_LOG_LEVEL(context) >= 2) {
      std::cout << METAL_LOG_HEADER << "PSO cache miss." << std::endl;
      std::cout << METAL_LOG_HEADER << "  Contents of map (before):" << std::endl;
      for (auto it = map->begin(); it != map->end(); ++it) {
        std::cout << METAL_LOG_HEADER << "    " << it->first << ": " << it->second << std::endl;
      }
    }
    
    auto* pipeline = new U(context, hash);
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
void mfa::cache<mfa::attention::hash, mfa::attention::pipeline>::prepare(mfa::context* context, mfa::attention::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::gemm::hash, mfa::gemm::pipeline>::prepare(mfa::context* context, mfa::gemm::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::normalization::hash, mfa::normalization::pipeline>::prepare(mfa::context* context, mfa::normalization::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::depalettize::hash, mfa::depalettize::pipeline>::prepare(mfa::context* context, mfa::depalettize::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

mfa::context::context(MTL::Device* device)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  this->log_level = 0;
#if CCV_METAL_LOGGING_ENABLE
  const char* log_level_repr = getenv("CCV_METAL_LOG_LEVEL");
  if (log_level_repr) {
    int log_level_raw = atoi(log_level_repr);
    std::cerr << std::endl;
    std::cerr << METAL_LOG_HEADER << "Using log level: " << log_level_raw << std::endl;
    CCV_NNC_MFA_PRECONDITION(log_level_raw >= 0 && log_level_raw <= 4)
    
    this->log_level = uint16_t(log_level_raw);
  }
#endif
  
  // Example: /usr/local/MetalFlashAttention/lib/libMetalFlashAttention.metallib
  // We need to have two different variants based on the operating system. macOS
  // will not accept a metallib compiled for iOS/tvOS/visionOS and vice versa.
  if (METAL_LOG_LEVEL(this) >= 1) {
    std::cerr << METAL_LOG_HEADER << "Started loading 'libMetalFlashAttention.metallib'." << std::endl;
  }

  this->device = NS::RetainPtr(device);

  this->scratch = NS::TransferPtr(device->newBuffer(65536, 0));

  // Check whether the device architecture is supported.
  this->supported = device->supportsFamily(MTL::GPUFamilyApple7);
  if (!supported) {
    if (METAL_LOG_LEVEL(this) >= 1) {
      std::cerr << METAL_LOG_HEADER << "Device architecture not supported by Metal FlashAttention." << std::endl;
    }
    pool->drain();
    return;
  }
  
  const char *external_metallib_path = nullptr;
#if CCV_NNC_MFA_EXTERNAL_METALLIB_ENABLE
  external_metallib_path = getenv("CCV_NNC_MFA_METALLIB_PATH");
#endif
  if (METAL_LOG_LEVEL(this) >= 1) {
    if (external_metallib_path) {
      std::cerr << METAL_LOG_HEADER << "Loading from path '" << external_metallib_path << "'." << std::endl;
    } else {
      std::cerr << METAL_LOG_HEADER << "Loading from embedded string." << std::endl;
    }
  }
  
  // Attempt to load the library, otherwise crash with a detailed log message.
  NS::Error* error = nullptr;
  if (external_metallib_path) {
    auto string = NS::String::string(external_metallib_path, NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    this->library = NS::TransferPtr(device->newLibrary(url, &error));
  } else {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(libmfaios16_v1_0_1_metallib, sizeof(libmfaios16_v1_0_1_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(libmfamacos13_v1_0_1_metallib, sizeof(libmfamacos13_v1_0_1_metallib), NULL, 0);
#endif
    this->library = NS::TransferPtr(device->newLibrary(data, &error));
    dispatch_release(data);
  }
  if (!this->library) {
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
  
  // Notify that this finished successfully, and is not just stalling on one of
  // the previous lines of code.
  if (METAL_LOG_LEVEL(this) >= 1) {
    std::cerr << METAL_LOG_HEADER << "Finished loading 'libMetalFlashAttention.metallib'." << std::endl;
  }
  
  pool->drain();
}

MTL::Buffer* mfa::context::request_scratch(uint64_t size) {
  if (size > scratch->length()) {
    uint64_t padded_size = std::max(int64_t(0), int64_t(size) - 1);
    uint64_t leading_zeroes = __builtin_clzll(padded_size);
    uint64_t rounded_size = 1 << uint64_t(64 - leading_zeroes);
    
    auto buffer = device->newBuffer(rounded_size, MTL::ResourceStorageModePrivate);
    CCV_NNC_MFA_PRECONDITION(buffer != nullptr);
    this->scratch = NS::TransferPtr(buffer);
  }
  return scratch.get();
}

MTL::CommandBatch::CommandBatch(MTL::CommandQueue* commandQueue) {
  commandBuffer = commandQueue->commandBuffer();
  commandEncoder = commandBuffer->computeCommandEncoder();
}

MTL::ComputeCommandEncoder* MTL::CommandBatch::startCommand() {
  CCV_NNC_MFA_PRECONDITION(commandActive == 0)
  commandActive = 1;
  return commandEncoder;
}

void MTL::CommandBatch::finishCommand(MTL::ComputeCommandEncoder* commandEncoder) {
  CCV_NNC_MFA_PRECONDITION(commandActive == 1)
  commandActive = 0;
  batchedCommandCount += 1;
}

MTL::CommandBatch::~CommandBatch() {
  CCV_NNC_MFA_PRECONDITION(commandActive == 0)
  commandEncoder->endEncoding();
  if (commandBuffer) {
    commandBuffer->commit();
  }
}
