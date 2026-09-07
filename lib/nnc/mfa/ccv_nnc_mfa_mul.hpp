#ifndef GUARD_ccv_nnc_mfa_mul_hpp
#define GUARD_ccv_nnc_mfa_mul_hpp

typedef struct {
  uint64_t data_type;
  uint32_t length;
  uint8_t loadM;
  // [M, 1, D] op [1, H, 1]; bit 0 / 1 identifies the channel-weight input.
  uint8_t channel_broadcast;
  uint32_t channel_count;
  uint32_t channel_length;
} ccv_nnc_mfa_mul_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_mul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_mul_params_t params);
void ccv_nnc_mfa_encode_mul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_mul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
