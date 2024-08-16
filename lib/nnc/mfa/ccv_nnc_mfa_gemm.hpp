#ifndef GUARD_ccv_nnc_mfa_gemm_hpp
#define GUARD_ccv_nnc_mfa_gemm_hpp

typedef struct {
  uint64_t data_type;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t A_trans;
  uint8_t B_trans;
  uint8_t D_trans;
  uint8_t batched;
  uint8_t fused_bias;
  uint8_t register_float;
  
  // Fill these in the same order as the original shape, but null-terminated.
  // Both arrays must have the same length.
  uint32_t batch_dims_a[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t batch_dims_b[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t batch_dims_d[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_mfa_gemm_params_t;

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemm_params_t params);
void ccv_nnc_mfa_encode_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemm_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
