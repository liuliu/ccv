#ifndef GUARD_ccv_nnc_mfa_reduce_logsumexp_hpp
#define GUARD_ccv_nnc_mfa_reduce_logsumexp_hpp

typedef struct {
  uint64_t data_type;
  uint32_t row_count;
  uint32_t column_count;
  float scale;
} ccv_nnc_mfa_reduce_logsumexp_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_reduce_logsumexp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_reduce_logsumexp_params_t params);
void ccv_nnc_mfa_encode_reduce_logsumexp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_reduce_logsumexp_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
