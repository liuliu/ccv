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
  uint8_t fused_bias;
  uint8_t register_float;
  uint8_t use_neural_accelerators;
  
  // Fill these in the same order as the original shape, but null-terminated.
  // Both arrays must have the same length.
  uint32_t batch_dimension;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_c;
  uint32_t batch_stride_d;
} ccv_nnc_mfa_gemm_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemm_params_t params);
size_t ccv_nnc_mfa_gemm_reserved_scratch_size(ccv_nnc_mfa_gemm_params_t params);
void ccv_nnc_mfa_encode_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemm_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
