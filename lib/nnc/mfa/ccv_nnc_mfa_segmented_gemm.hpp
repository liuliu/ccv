#ifndef GUARD_ccv_nnc_mfa_segmented_gemm_hpp
#define GUARD_ccv_nnc_mfa_segmented_gemm_hpp

typedef struct {
  uint64_t data_type;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t originalM;
  uint8_t A_trans;
  uint8_t B_trans;
  uint8_t D_trans;
  uint8_t fused_bias;
  uint8_t register_float;
  uint8_t use_neural_accelerators;
  
  uint32_t segments;
} ccv_nnc_mfa_segmented_gemm_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

size_t ccv_nnc_mfa_segmented_gemm_reserved_scratch_size(ccv_nnc_mfa_segmented_gemm_params_t params);
void ccv_nnc_mfa_encode_segmented_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_segmented_gemm_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
