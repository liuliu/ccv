#include "ccv_nnc_mfa.hpp"
using namespace ccv::nnc;

#include <iostream>

inline void log_source_location(int line, const char *file_name, const char *function_name) {
  std::cerr << METAL_LOG_HEADER << "Encountered unexpected error in: " << function_name << std::endl;
  std::cerr << "\e[0;1m" << file_name << ":" << line << ":\e[0m ";
  std::cerr << "\e[0;31m" << "error:" << "\e[0m ";
}

void mfa::fatal_error(NS::Error* error, int line, const char *file_name, const char *function_name) {
  auto description = error->localizedDescription();
  auto recovery_suggestion = error->localizedRecoverySuggestion();
  auto failure_reason = error->localizedFailureReason();
  
  log_source_location(line, file_name, function_name);
  std::cerr << "\e[0;1m";
  if (description) {
    std::cerr << description->cString(NS::UTF8StringEncoding);
  } else {
    std::cerr << "[description not available]";
  }
  std::cerr << "\e[0m" << std::endl;
  if (recovery_suggestion) {
    std::cerr << METAL_LOG_HEADER << "Recovery suggestion: " << recovery_suggestion->cString(NS::UTF8StringEncoding) << std::endl;
  }
  if (failure_reason) {
    std::cerr << METAL_LOG_HEADER << "Failure reason: " << failure_reason->cString(NS::UTF8StringEncoding) << std::endl;
  }
  std::cerr << METAL_LOG_HEADER << "Quitting now." << std::endl;
  __builtin_trap();
}

void mfa::precondition_failure(const char *message, int line, const char *file_name, const char *function_name) {
  log_source_location(line, file_name, function_name);
  std::cerr << "\e[0;1m";
  if (message) {
    std::cerr << message;
  } else {
    std::cerr << "[precondition failure]";
  }
  std::cerr << "\e[0m" << std::endl;
  std::cerr << METAL_LOG_HEADER << "Quitting now." << std::endl;
  __builtin_trap();
}
