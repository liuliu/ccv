#ifndef GUARD_ccv_nnc_mfa_gated_delta_hpp
#define GUARD_ccv_nnc_mfa_gated_delta_hpp

typedef struct {
  uint32_t batch_size;
  uint32_t sequence_length;
  uint32_t key_head_count;
  uint32_t value_head_count;
  uint32_t key_dim;
  uint32_t value_dim;
  uint32_t data_type;
  uint32_t beta_data_type;
  uint32_t state_checkpoint_count;
  uint8_t log_decay;
  uint8_t loadM;
} ccv_nnc_mfa_gated_delta_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_gated_delta(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gated_delta_params_t params);
void ccv_nnc_mfa_encode_gated_delta(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gated_delta_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
