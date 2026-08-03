#ifndef GUARD_ccv_nnc_mfa_swish_mul_hpp
#define GUARD_ccv_nnc_mfa_swish_mul_hpp

typedef struct {
  float beta;
  float scale;
  float clamp;
  uint64_t a_data_type;
  uint64_t b_data_type;
  uint64_t weight_data_type;
  uint32_t length;
  uint32_t weight_count;
  uint8_t gradient;
  uint8_t weighted;
  uint8_t output_mask;
  uint64_t g_data_type;
  uint64_t da_data_type;
  uint64_t db_data_type;
  uint8_t loadM;
} ccv_nnc_mfa_swish_mul_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_swish_mul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_swish_mul_params_t params);
void ccv_nnc_mfa_encode_swish_mul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_swish_mul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
