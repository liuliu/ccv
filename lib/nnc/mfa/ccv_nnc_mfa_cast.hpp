#ifndef GUARD_ccv_nnc_mfa_cast_hpp
#define GUARD_ccv_nnc_mfa_cast_hpp

typedef struct {
  uint64_t original_data_type;
  uint64_t data_type;
  uint32_t length;
  uint8_t loadM;
} ccv_nnc_mfa_cast_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_cast(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_cast_params_t params);
void ccv_nnc_mfa_encode_cast(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_cast_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
