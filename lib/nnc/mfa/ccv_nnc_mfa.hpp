#ifndef GUARD_ccv_nnc_mfa_hpp
#define GUARD_ccv_nnc_mfa_hpp

#ifdef __cplusplus
extern "C" {
#endif
#include "ccv.h"
#include "nnc/ccv_nnc.h"
#ifdef __cplusplus
}
#endif
#include "ccv_nnc_mfa_defines.hpp"
#include "ccv_nnc_mfa_attention.hpp"
#include "ccv_nnc_mfa_normalization.hpp"
#include "ccv_nnc_mfa_rmsnorm_gated.hpp"
#include "ccv_nnc_mfa_rmsnorm_cmul.hpp"
#include "ccv_nnc_mfa_depalettize.hpp"
#include "ccv_nnc_mfa_dequantize_8i_rowwise.hpp"
#include "ccv_nnc_mfa_dequantize_8i_rowwise_x.hpp"
#include "ccv_nnc_mfa_dequantize_8i_rowwise_x_fp.hpp"
#include "ccv_nnc_mfa_index_select_8i_rowwise.hpp"
#include "ccv_nnc_mfa_index_select_8i_rowwise_x.hpp"
#include "ccv_nnc_mfa_adam.hpp"
#include "ccv_nnc_mfa_cmul.hpp"
#include "ccv_nnc_mfa_conform_data_format.hpp"
#include "ccv_nnc_mfa_gelu.hpp"
#include "ccv_nnc_mfa_gemm.hpp"
#include "ccv_nnc_mfa_scaled_gemm.hpp"
#include "ccv_nnc_mfa_scaled_gemv.hpp"
#include "ccv_nnc_mfa_ane_rowwise_gemm.hpp"
#include "ccv_nnc_mfa_segmented_scaled_gemm.hpp"
#include "ccv_nnc_mfa_segmented_int8_gemv.hpp"
#include "ccv_nnc_mfa_segmented_int8_swiglu.hpp"
#include "ccv_nnc_mfa_conv3d.hpp"
#include "ccv_nnc_mfa_gemv.hpp"
#include "ccv_nnc_mfa_cast.hpp"
#include "ccv_nnc_mfa_strided_copy.hpp"
#include "ccv_nnc_mfa_sigmoid.hpp"
#include "ccv_nnc_mfa_swish.hpp"
#include "ccv_nnc_mfa_swish_mul.hpp"
#include "ccv_nnc_mfa_exp.hpp"
#include "ccv_nnc_mfa_softplus.hpp"
#include "ccv_nnc_mfa_add.hpp"
#include "ccv_nnc_mfa_fast_fence.hpp"
#include "ccv_nnc_mfa_rotate_half.hpp"
#include "ccv_nnc_mfa_scaled_dot_product_arg_partition.hpp"
#include "ccv_nnc_mfa_sparse_indexed_attention.hpp"
#include "ccv_nnc_mfa_walsh_hadamard_transform.hpp"
#include "ccv_nnc_mfa_hyper_connection.hpp"
#include "ccv_nnc_mfa_segmented_gemm.hpp"
#include "ccv_nnc_mfa_gated_delta.hpp"
#include "ccv_nnc_mfa_moe_routing.hpp"
#include "ccv_nnc_mfa_scatter_add.hpp"

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "ccv_nnc_mfa_error.hpp"
#include "kernels/ShaderCache.hpp"

namespace ccv {
namespace nnc {
namespace mfa {

class context;

class context {
public:
  bool supported;
  uint16_t log_level;
  
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Buffer> scratch;
  void* ane_rowwise_gemm_cache;
  
  context(MTL::Device* device);
  
  ShaderCache kernel_cache;
  
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
uint8_t ccv_nnc_mfa_supports_int8_ane(ccv_nnc_mfa_context_t* context);
uint8_t ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_mfa_context_t* context);
uint8_t ccv_nnc_mfa_neural_accelerators_support_bfloat(ccv_nnc_mfa_context_t* context);
uint16_t ccv_nnc_mfa_context_log_level(ccv_nnc_mfa_context_t* context);
void ccv_nnc_mfa_log_message(const char* message);

mtl_command_batch_t* ccv_nnc_start_command_batch(mtl_command_queue_t* command_queue);
mtl_command_batch_t* ccv_nnc_start_command_batch_from_command_buffer(mtl_command_buffer_t* command_buffer, int commit_on_finish);
void ccv_nnc_finish_command_batch(mtl_command_batch_t* command_batch);
mtl_buffer_t* ccv_nnc_mfa_request_scratch(ccv_nnc_mfa_context_t* context, const uint64_t size);
void ccv_nnc_mfa_set_binary_archives(ccv_nnc_mfa_context_t* context, const char** paths_to_read, const int paths_to_read_size, const char* path_to_write);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
