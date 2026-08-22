#ifndef GUARD_ccv_nnc_mfa_clamp_hpp
#define GUARD_ccv_nnc_mfa_clamp_hpp

typedef struct {
  uint64_t data_type;
  uint32_t length;
  float min;
  float max;
  uint8_t bounds;
  uint8_t loadM;
} ccv_nnc_mfa_clamp_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_clamp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_clamp_params_t params);
void ccv_nnc_mfa_encode_clamp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_clamp_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
