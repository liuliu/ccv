#ifndef GUARD_ccv_nnc_mfa_argmax_hpp
#define GUARD_ccv_nnc_mfa_argmax_hpp

typedef struct {
  uint64_t data_type;
  uint32_t row_count;
  uint32_t column_count;
  uint32_t state[7];
  uint8_t gumbel;
} ccv_nnc_mfa_argmax_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_argmax(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_argmax_params_t params);
void ccv_nnc_mfa_encode_argmax(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_argmax_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
