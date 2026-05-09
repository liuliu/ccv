#ifndef GUARD_ccv_nnc_mfa_scaled_gemv_hpp
#define GUARD_ccv_nnc_mfa_scaled_gemv_hpp

typedef struct {
  uint64_t data_type;
  uint32_t mrows;
  uint32_t nrows;
  uint32_t ncols;
  uint8_t fused_bias;
} ccv_nnc_mfa_scaled_gemv_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_scaled_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_gemv_params_t params);
void ccv_nnc_mfa_encode_scaled_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_gemv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
