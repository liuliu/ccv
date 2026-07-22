#ifndef GUARD_ccv_nnc_mfa_add_hpp
#define GUARD_ccv_nnc_mfa_add_hpp

typedef struct {
  uint64_t data_type;
  uint8_t args;
  uint32_t length;
  uint8_t loadM;
} ccv_nnc_mfa_add_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_add_params_t params);
void ccv_nnc_mfa_encode_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_add_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
