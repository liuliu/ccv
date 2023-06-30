#include "ccv_nnc_mfa.h"
#include <nnc/mps/ccv_nnc_mps.h>
#include <iostream>

void* ccv_nnc_init_mfa_context(void* device, void* command_queue) {
  auto* mfa_context = new ccv_nnc_mfa_context((MTL::Device*)device, (MTL::CommandQueue*)command_queue);
  return (void*)mfa_context;
}

bool ccv_nnc_mfa_context_supported(void* mfa_context) {
  return ((ccv_nnc_mfa_context*)mfa_context)->get_supported();
}

void ccv_nnc_deinit_mfa_context(void* mfa_context) {
  delete (ccv_nnc_mfa_context*)mfa_context;
}

#define METAL_LOG_HEADER "\e[0;36m[Metal]\e[0m "

void ccv_nnc_mfa_log_source_location(const char* line, const char *column, const char *file_name, const char *function_name) {
  std::cerr << METAL_LOG_HEADER << "Encountered unexpected error in: " << function << std::endl;
  std::cerr << "\e[0;1m" << file_name << ":" << line << ":" << column << ":\e[0m ";
  std::cerr << "\e[0;31m << "error:"" << "\e[0m ";
}

void ccv_nnc_mfa_fatal_error(NS::Error* error, const char* line, const char *column, const char *file_name, const char *function_name) {
  auto description = error->localizedDescription();
  auto recovery_suggestion = error->localizedRecoverySuggestion();
  auto failure_reason = error->localizedFailureReason();
  
  ccv_nnc_mfa_log_source_location(line, column, file_name, function_name);
  std::cerr << "\e[0;1m"
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

ccv_nnc_mfa_context::ccv_nnc_mfa_context(MTL::Device* device, MTL::CommandQueue* command_queue)
{
  this->_supported = device->supportsFamily(MTL::GPUFamilyApple7);
  if (!_supported) {
    return;
  }
  
  this->device = NS::RetainPtr(device);
  this->command_queue = NS::RetainPtr(command_queue);
  
  // Temporary hard-coded path, until we determine how to package libMFA with
  // the rest of the build process.
  const char *c_path = "/usr/local/MetalFlashAttention/lib/libMetalFlashAttention.metallib";
  auto swift_path = NS::String::string(path, NS::UTF8StringEncoding);
  auto url = NS::URL::fileURLWithPath(swift_path);
  
  NS::Error* error;
  this->library = NS::TransferPtr(device->newLibrary(url, &error));
  CCV_NNC_MFA_ASSERT(error);
  
  // Temporary means to evaluate whether MFA loaded correctly.
  std::cout << METAL_LOG_HEADER << "libMetalFlashAttention initialized." << std::endl;
}
