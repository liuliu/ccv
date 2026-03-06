#ifndef GUARD_ccv_nnc_mfa_normalization_hpp
#define GUARD_ccv_nnc_mfa_normalization_hpp

typedef struct {
  uint64_t data_type;
  uint32_t channel_count;
  uint32_t channel_groups;
  uint32_t sequence_count;
  float epsilon;
  uint8_t elementwise_affine;
  uint8_t scale_translation_batched;
  uint8_t normalization_type;
  uint8_t reuse_saved_statistics;

  uint32_t batch_dims_data[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t batch_dims_scale_translation[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t src_batch_stride;
  uint32_t dst_batch_stride;
} ccv_nnc_mfa_normalization_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_normalization(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_normalization_params_t params);
void ccv_nnc_mfa_encode_normalization(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_normalization_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
