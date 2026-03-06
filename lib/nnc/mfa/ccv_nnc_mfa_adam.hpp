#ifndef GUARD_ccv_nnc_mfa_adam_hpp
#define GUARD_ccv_nnc_mfa_adam_hpp

typedef struct {
  uint64_t data_type;
  int adamw;
  int amsgrad;
  int step;
  float rate;
  float scale;
  float beta1;
  float beta2;
  float decay;
  float epsilon;
  uint64_t length;
} ccv_nnc_mfa_adam_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_adam(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_adam_params_t params);
void ccv_nnc_mfa_encode_adam(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_adam_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
