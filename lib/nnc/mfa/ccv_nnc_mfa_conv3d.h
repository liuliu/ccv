#ifndef GUARD_ccv_nnc_mfa_conv3d_h
#define GUARD_ccv_nnc_mfa_conv3d_h

typedef struct {
  uint64_t data_type;
  uint32_t batch_size;
  uint32_t input_channels;
  uint32_t output_channels;
  uint32_t groups;
  uint32_t input_dimensions[3]; // D, H, W
  uint32_t output_dimensions[3]; // D, H, W
  uint32_t filter_dimensions[3]; // D, H, W
  uint32_t stride_dimensions[3]; // D, H, W
  uint32_t dilation_dimensions[3]; // D, H, W
  uint32_t padding_begin[3]; // D, H, W
  uint32_t padding_end[3]; // D, H, W
  uint8_t format;
  uint8_t fused_bias;
  uint8_t use_neural_accelerators;
} ccv_nnc_mfa_conv3d_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_conv3d(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_conv3d_params_t params);
size_t ccv_nnc_mfa_conv3d_reserved_scratch_size(ccv_nnc_mfa_conv3d_params_t params);
void ccv_nnc_mfa_encode_conv3d(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_conv3d_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
