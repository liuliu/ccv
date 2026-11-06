#ifndef GUARD_ccv_nnc_mfa_causal_convolution_hpp
#define GUARD_ccv_nnc_mfa_causal_convolution_hpp

typedef struct {
  uint64_t data_type;
  uint32_t batch_size;
  uint32_t sequence_length;
  uint32_t channels;
  uint32_t kernel_size;
  uint32_t next_rows;
  uint8_t silu;
} ccv_nnc_mfa_causal_convolution_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_causal_convolution(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_causal_convolution_params_t params);
void ccv_nnc_mfa_encode_causal_convolution(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_causal_convolution_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
