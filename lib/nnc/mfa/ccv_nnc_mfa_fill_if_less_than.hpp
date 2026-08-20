#ifndef GUARD_ccv_nnc_mfa_fill_if_less_than_hpp
#define GUARD_ccv_nnc_mfa_fill_if_less_than_hpp

typedef struct {
  uint64_t data_type;
  uint32_t length;
  float fill;
  uint8_t loadM;
} ccv_nnc_mfa_fill_if_less_than_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_fill_if_less_than(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_fill_if_less_than_params_t params);
void ccv_nnc_mfa_encode_fill_if_less_than(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_fill_if_less_than_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
