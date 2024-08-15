#ifndef GUARD_ccv_nnc_mfa_hpp
#define GUARD_ccv_nnc_mfa_hpp

#include "nnc/ccv_nnc.h"
#include "ccv_nnc_mfa_defines.hpp"
#include "ccv_nnc_mfa_attention.hpp"
#include "ccv_nnc_mfa_gemm.hpp"
#include "ccv_nnc_mfa_normalization.hpp"
#include "ccv_nnc_mfa_depalettize.hpp"
#include "ccv_nnc_mfa_adam.hpp"
#include "ccv_nnc_mfa_cmul.hpp"
#include "ccv_nnc_mfa_gemv.hpp"
#include "ccv_nnc_mfa_cast.hpp"
#include "ccv_nnc_mfa_add.hpp"

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ccv_nnc_mfa_error.hpp"
#include "v2/ShaderCache.hpp"
#include <unordered_map>

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
  
  void prepare(context* context, T hash);
};

class context {
public:
  bool supported;
  uint16_t log_level;
  
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::Buffer> scratch;
  
  context(MTL::Device* device);
  
  cache<attention::hash, attention::pipeline> attention_cache;
  cache<gemm::hash, gemm::pipeline> gemm_cache;
  cache<normalization::hash, normalization::pipeline> normalization_cache;
  cache<depalettize::hash, depalettize::pipeline> depalettize_cache;
  cache<adam::hash, adam::pipeline> adam_cache;
  cache<cmul::hash, cmul::pipeline> cmul_cache;
  cache<gemv::hash, gemv::pipeline> gemv_cache;
  cache<cast::hash, cast::pipeline> cast_cache;
  cache<add::hash, add::pipeline> add_cache;

  ShaderCache v2_cache;
  
  MTL::Buffer* request_scratch(uint64_t size);
};

} // namespace mfa
} // namespace nnc
} // namespace ccv

extern "C" {
#endif // __cplusplus

ccv_nnc_mfa_context_t* ccv_nnc_init_mfa_context(mtl_device_t* context);
void ccv_nnc_mfa_clear_pipeline_cache(ccv_nnc_mfa_context_t* context);
void ccv_nnc_deinit_mfa_context(ccv_nnc_mfa_context_t* context);
uint8_t ccv_nnc_mfa_context_supported(ccv_nnc_mfa_context_t* context);
uint16_t ccv_nnc_mfa_context_log_level(ccv_nnc_mfa_context_t* context);
void ccv_nnc_mfa_log_message(const char* message);

mtl_command_batch_t* ccv_nnc_start_command_batch(mtl_command_queue_t* command_queue);
void ccv_nnc_finish_command_batch(mtl_command_batch_t* command_batch);
mtl_buffer_t* ccv_nnc_mfa_request_scratch(ccv_nnc_mfa_context_t* context, const uint64_t size);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
