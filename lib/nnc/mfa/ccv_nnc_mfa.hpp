#ifndef GUARD_ccv_nnc_mfa_hpp
#define GUARD_ccv_nnc_mfa_hpp

#include "nnc/ccv_nnc.h"
#include "ccv_nnc_mfa_defines.hpp"

#include "ccv_nnc_mfa_gemm.hpp"
#ifdef __cplusplus
#include "3rdparty/metal-cpp/Dispatch.hpp"
#include "3rdparty/metal-cpp/Metal.hpp"
#include "ccv_nnc_mfa_error.hpp"

namespace ccv {
namespace nnc {
namespace mfa {

class context {
public:
  bool supported;
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Library> library;
  
  context(MTL::Device* device);
  
  // It's okay to async prepare an op multiple times for the same set of
  // parameters; MFA keeps an internal cache. What's not okay is calling it
  // eagerly, just before dispatching the command. Doing so is technically
  // supported for debugging, but only for debugging - shift such calls to NNC
  // graph 'autotune' ASAP.
  //
  // `encode_*` C functions require that you already called `async_prepare_*`,
  // forcing you to create them asynchronously several milliseconds beforehand.
  // Tensors entered into such functions should be a null-terminated list of
  // `MTL::Buffer` references.
  std::unordered_map<gemm::hash, gemm::pipeline*> gemm_cache;
};

} // namespace mfa
} // namespace nnc
} // namespace ccv

extern "C" {
#endif // __cplusplus

ccv_nnc_mfa_context_t* ccv_nnc_init_mfa_context(mtl_device_t* context);
void ccv_nnc_deinit_mfa_context(ccv_nnc_mfa_context_t* context);
uint8_t ccv_nnc_mfa_context_supported(ccv_nnc_mfa_context_t* context);
void ccv_nnc_mfa_log_message(const char* message);

mtl_command_batch_t* ccv_nnc_start_command_batch(mtl_command_queue_t* command_queue);
void ccv_nnc_finish_command_batch(mtl_command_batch_t* command_batch);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
