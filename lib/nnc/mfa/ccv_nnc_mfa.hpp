#ifndef GUARD_ccv_nnc_mfa_hpp
#define GUARD_ccv_nnc_mfa_hpp

#include "nnc/ccv_nnc.h"
#include "ccv_nnc_mfa_defines.hpp"
#include "ccv_nnc_mfa_gemm.hpp"

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ccv_nnc_mfa_error.hpp"

namespace ccv {
namespace nnc {
namespace mfa {

class context;

template <typename T, typename U>
class cache {
public:
  std::unordered_map<T, U*> map;
  
  cache();
  ~cache();
  
  void prepare(context* context, T hash, bool async);
};

class context {
public:
  bool supported;
  uint16_t log_level;
  
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Library> library;
  
  context(MTL::Device* device);
  
  // MFA keeps internal caches of pipeline state objects. If you're eagerly
  // executing a command, call `sync_prepare_*` just before encoding it. This
  // incurs non-negligible latency, which can be removed by compiling during
  // graph compilation. Use `async_prepare_*` during graph compilation, which
  // will transform the subsequent `sync_prepare_*` into a NOP. The async
  // version has more latency but utilizes multicore CPU parallelism.
  //
  // After preparing the pipeline, call `encode_*`. Pass each tensor's backing
  // `MTL::Buffer*` through a null-terminated list.
  cache<gemm::hash, gemm::pipeline> gemm_cache;
};

} // namespace mfa
} // namespace nnc
} // namespace ccv

extern "C" {
#endif // __cplusplus

ccv_nnc_mfa_context_t* ccv_nnc_init_mfa_context(mtl_device_t* context);
void ccv_nnc_deinit_mfa_context(ccv_nnc_mfa_context_t* context);
uint8_t ccv_nnc_mfa_context_supported(ccv_nnc_mfa_context_t* context);
uint16_t ccv_nnc_mfa_context_log_level(ccv_nnc_mfa_context_t* context);
void ccv_nnc_mfa_log_message(const char* message);

mtl_command_batch_t* ccv_nnc_start_command_batch(mtl_command_queue_t* command_queue);
void ccv_nnc_finish_command_batch(mtl_command_batch_t* command_batch);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
