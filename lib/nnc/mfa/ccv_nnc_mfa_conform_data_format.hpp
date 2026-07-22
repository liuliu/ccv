#ifndef GUARD_ccv_nnc_mfa_conform_data_format_hpp
#define GUARD_ccv_nnc_mfa_conform_data_format_hpp

typedef struct {
  uint32_t row_count;
  uint32_t head_dim;
  uint32_t preserved_tail;
  uint8_t loadM;
} ccv_nnc_mfa_conform_data_format_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_conform_data_format(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_conform_data_format_params_t params);
void ccv_nnc_mfa_encode_conform_data_format(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_conform_data_format_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
