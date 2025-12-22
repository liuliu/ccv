#ifndef GUARD_ccv_nnc_mfa_swish_hpp
#define GUARD_ccv_nnc_mfa_swish_hpp

typedef struct {
  uint8_t gradient;
  uint64_t data_type;
  uint32_t length;
} ccv_nnc_mfa_swish_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_swish(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_swish_params_t params);
void ccv_nnc_mfa_encode_swish(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_swish_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
