#ifndef GUARD_ccv_nnc_mfa_sparse_indexed_attention_hpp
#define GUARD_ccv_nnc_mfa_sparse_indexed_attention_hpp

typedef struct {
  uint8_t is_causal;
  uint8_t attention_sinks;
  uint8_t use_neural_accelerators;
  uint32_t sink_head_stride;
  uint32_t T;
  uint32_t dense_rows;
  uint32_t sparse_rows;
  uint32_t H;
  uint32_t D;
  uint32_t K;
  uint32_t variant;
  float scale;
  uint64_t data_type;
} ccv_nnc_mfa_sparse_indexed_attention_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_sparse_indexed_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_sparse_indexed_attention_params_t params);
void ccv_nnc_mfa_encode_sparse_indexed_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_sparse_indexed_attention_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
