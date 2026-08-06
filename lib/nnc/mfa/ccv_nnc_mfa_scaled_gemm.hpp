#ifndef GUARD_ccv_nnc_mfa_scaled_gemm_hpp
#define GUARD_ccv_nnc_mfa_scaled_gemm_hpp

typedef struct {
  uint64_t data_type;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t fused_bias;
  uint8_t use_neural_accelerators;
  uint32_t batch_dimension;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_c;
  uint32_t batch_stride_d;
  uint32_t leading_dimension_a;
  uint32_t leading_dimension_c;
  uint8_t loadM;
} ccv_nnc_mfa_scaled_gemm_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_scaled_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_gemm_params_t params);
size_t ccv_nnc_mfa_scaled_gemm_reserved_scratch_size(ccv_nnc_mfa_scaled_gemm_params_t params);
void ccv_nnc_mfa_encode_scaled_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_gemm_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
