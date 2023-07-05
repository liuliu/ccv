#include "ccv_nnc_mfa.hpp"
#include <iostream>

void* ccv_nnc_init_mfa_context(void* device) {
  return new ccv_nnc_mfa_context((MTL::Device*)device);
}

void ccv_nnc_deinit_mfa_context(void* mfa_context) {
  delete (ccv_nnc_mfa_context*)mfa_context;
}

int ccv_nnc_mfa_context_supported(void* mfa_context) {
  return ((ccv_nnc_mfa_context*)mfa_context)->supported ? 1 : 0;
}

#define METAL_LOG_HEADER "\e[0;36m[Metal]\e[0m "

void ccv_nnc_mfa_log_source_location(int line, const char *file_name, const char *function_name) {
  std::cerr << METAL_LOG_HEADER << "Encountered unexpected error in: " << function_name << std::endl;
  std::cerr << "\e[0;1m" << file_name << ":" << line << ":\e[0m ";
  std::cerr << "\e[0;31m" << "error:" << "\e[0m ";
}

void ccv_nnc_mfa_fatal_error(NS::Error* error, int line, const char *file_name, const char *function_name) {
  auto description = error->localizedDescription();
  auto recovery_suggestion = error->localizedRecoverySuggestion();
  auto failure_reason = error->localizedFailureReason();
  
  ccv_nnc_mfa_log_source_location(line, file_name, function_name);
  std::cerr << "\e[0;1m";
  if (description) {
    std::cerr << description;
  } else {
    std::cerr << "[description not available]";
  }
  std::cerr << "\e[0m" << std::endl;
  if (recovery_suggestion) {
    std::cerr << METAL_LOG_HEADER << "Recovery suggestion: " << recovery_suggestion << std::endl;
  }
  if (failure_reason) {
    std::cerr << METAL_LOG_HEADER << "Failure reason: " << failure_reason << std::endl;
  }
  std::cerr << METAL_LOG_HEADER << "Quitting now." << std::endl;
  exit(-1);
}

ccv_nnc_mfa_context::ccv_nnc_mfa_context(MTL::Device* device)
{
  // Example: /usr/local/MetalFlashAttention/lib/libMetalFlashAttention.metallib
  // We need to have two different variants based on the operating system. macOS
  // will not accept a metallib compiled for iOS/tvOS/visionOS and vice versa.
  const char* metallib_path = getenv("CCV_NNC_MFA_METALLIB_PATH");
  if (!metallib_path) {
    this->supported = false;
    return;
  }
  std::cerr << METAL_LOG_HEADER << "Loading libMetalFlashAttention.metallib." << std::endl;
  
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
  CCV_NNC_MFA_ASSERT(error);
  
  // Notify that this finished successfully, and is not just stalling on one of
  // the previous lines of code.
  std::cerr << METAL_LOG_HEADER << "Finished loading libMetalFlashAttention." << std::endl;
}
