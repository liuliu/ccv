#ifndef GUARD_ccv_nnc_mfa_gelu_hpp
#define GUARD_ccv_nnc_mfa_gelu_hpp

typedef struct {
  uint8_t gradient;
  uint8_t tanh;
  uint64_t data_type;
  uint32_t length;
  uint8_t loadM;
} ccv_nnc_mfa_gelu_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_gelu(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gelu_params_t params);
void ccv_nnc_mfa_encode_gelu(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gelu_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
