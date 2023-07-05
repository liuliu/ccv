#ifndef GUARD_ccv_nnc_mfa_defines_hpp
#define GUARD_ccv_nnc_mfa_defines_hpp

#ifdef __cplusplus
#include "3rdparty/metal-cpp/Metal.hpp"
namespace ccv {
namespace nnc {
namespace mfa {
class context;
} // namespace mfa
} // namespace nnc
} // namespace ccv

typedef ccv::nnc::mfa::context ccv_nnc_mfa_context_t;
typedef MTL::Buffer mtl_buffer_t;
typedef MTL::ComputeCommandEncoder mtl_compute_command_encoder_t;
typedef MTL::Device mtl_device_t;
#else
typedef void ccv_nnc_mfa_context_t;
typedef void mtl_buffer_t;
typedef void mtl_compute_command_encoder_t;
typedef void mtl_device_t;
#endif // __cplusplus

#endif
