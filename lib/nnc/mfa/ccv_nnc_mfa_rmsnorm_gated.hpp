#ifndef GUARD_ccv_nnc_mfa_rmsnorm_gated_hpp
#define GUARD_ccv_nnc_mfa_rmsnorm_gated_hpp

typedef struct {
  float epsilon;
  uint64_t a_data_type;
  uint64_t gate_data_type;
  uint64_t scale_data_type;
  uint32_t row_count;
  uint32_t column_count;
} ccv_nnc_mfa_rmsnorm_gated_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_rmsnorm_gated(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rmsnorm_gated_params_t params);
void ccv_nnc_mfa_encode_rmsnorm_gated(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rmsnorm_gated_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
