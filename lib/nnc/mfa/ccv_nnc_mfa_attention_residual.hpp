#ifndef GUARD_ccv_nnc_mfa_attention_residual_hpp
#define GUARD_ccv_nnc_mfa_attention_residual_hpp

// The block states and the partial state a mixture may have, and the queries, each with an output.
#define CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_STATES (16)
#define CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_QUERIES (4)

typedef struct {
  uint64_t data_type;
  uint32_t rows;
  uint32_t channels;
  uint32_t block_count;
  uint32_t query_count;
  float epsilon;
  float scale;
} ccv_nnc_mfa_attention_residual_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_encode_attention_residual(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_attention_residual_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
