#ifndef GUARD_ccv_nnc_mfa_signed_sqrt_hpp
#define GUARD_ccv_nnc_mfa_signed_sqrt_hpp

typedef struct {
  uint8_t gradient;
  float minimum_magnitude;
  uint64_t data_type;
  uint32_t length;
  uint8_t loadM;
} ccv_nnc_mfa_signed_sqrt_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_signed_sqrt(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_signed_sqrt_params_t params);
void ccv_nnc_mfa_encode_signed_sqrt(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_signed_sqrt_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
