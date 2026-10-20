#ifndef GUARD_ccv_nnc_mfa_tanh_hpp
#define GUARD_ccv_nnc_mfa_tanh_hpp

typedef struct {
  uint8_t gradient;
  float soft_cap;
  uint64_t data_type;
  uint32_t length;
  uint8_t loadM;
} ccv_nnc_mfa_tanh_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_tanh(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_tanh_params_t params);
void ccv_nnc_mfa_encode_tanh(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_tanh_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
