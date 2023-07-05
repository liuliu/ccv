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

mfa::context::context(MTL::Device* device)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  // Example: /usr/local/MetalFlashAttention/lib/libMetalFlashAttention.metallib
  // We need to have two different variants based on the operating system. macOS
  // will not accept a metallib compiled for iOS/tvOS/visionOS and vice versa.
  const char* metallib_path = getenv("CCV_NNC_MFA_METALLIB_PATH");
  if (!metallib_path) {
    this->supported = false;
    return;
  }
  std::cerr << METAL_LOG_HEADER << "Started loading 'libMetalFlashAttention.metallib'." << std::endl;
  
  // Check whether the device architecture is supported.
  this->supported = device->supportsFamily(MTL::GPUFamilyApple7);
  if (!supported) {
    std::cerr << METAL_LOG_HEADER << "Device architecture not supported by Metal FlashAttention." << std::endl;
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
  std::cerr << METAL_LOG_HEADER << "Finished loading 'libMetalFlashAttention.metallib'." << std::endl;
  
  pool->drain();
}
