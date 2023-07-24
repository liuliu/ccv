#ifndef GUARD_ccv_nnc_mfa_error_hpp
#define GUARD_ccv_nnc_mfa_error_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

// `std::cout` and `CACurrentMediaTime()` for profiling.
#include <iostream>
#include <QuartzCore/QuartzCore.h>

namespace ccv {
namespace nnc {
namespace mfa {

#define METAL_LOG_HEADER "\e[0;36m[Metal]\e[0m "

#define CCV_NNC_MFA_CHECK_ERROR(error) \
if (error) { ccv::nnc::mfa::fatal_error(error, __LINE__, __FILE__, __FUNCTION__); } \

void fatal_error(NS::Error* error, int line, const char *file_name, const char *function_name);

#define CCV_NNC_MFA_PRECONDITION(expr) \
if (!(expr)) { ccv::nnc::mfa::precondition_failure(nullptr, __LINE__, __FILE__, __FUNCTION__); } \

void precondition_failure(const char *message, int line, const char *file_name, const char *function_name);

} // namespace mfa
} // namespace nnc
} // namespace ccv

#endif
