#ifndef GUARD_ccv_nnc_mfa_segmented_int8_gemv_hpp
#define GUARD_ccv_nnc_mfa_segmented_int8_gemv_hpp

typedef struct {
  uint64_t data_type;
  uint32_t format;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t fused_bias;
  uint8_t broadcast_input;
  uint32_t expert_count;
  uint32_t bincount;
} ccv_nnc_mfa_segmented_int8_gemv_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_segmented_int8_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_segmented_int8_gemv_params_t params);
void ccv_nnc_mfa_encode_segmented_int8_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_segmented_int8_gemv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
