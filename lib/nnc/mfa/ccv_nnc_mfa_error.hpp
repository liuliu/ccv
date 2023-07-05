#ifndef GUARD_ccv_nnc_mfa_error_hpp
#define GUARD_ccv_nnc_mfa_error_hpp

#include "3rdparty/metal-cpp/Metal.hpp"

namespace ccv {
namespace nnc {
namespace mfa {

#define METAL_LOG_HEADER "\e[0;36m[Metal]\e[0m "

#define CCV_NNC_MFA_ASSERT(error) \
if (error) { ccv::nnc::mfa::fatal_error(error, __LINE__, __FILE__, __FUNCTION__); } \

void fatal_error(NS::Error* error, int line, const char *file_name, const char *function_name);

#define CCV_NNC_MFA_PRECONDITION(expr) \
if (!expr) { ccv::nnc::mfa::precondition_failure(__LINE__, __FILE__, __FUNCTION__); } \

void precondition_failure(int line, const char *file_name, const char *function_name);

} // namespace mfa
} // namespace nnc
} // namespace ccv

#endif
