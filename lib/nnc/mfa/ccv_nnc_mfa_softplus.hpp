#ifndef GUARD_ccv_nnc_mfa_softplus_hpp
#define GUARD_ccv_nnc_mfa_softplus_hpp

typedef struct {
  uint64_t data_type;
  uint32_t length;
} ccv_nnc_mfa_softplus_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_softplus(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_softplus_params_t params);
void ccv_nnc_mfa_encode_softplus(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_softplus_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
