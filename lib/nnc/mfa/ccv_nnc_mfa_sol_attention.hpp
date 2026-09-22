#ifndef GUARD_ccv_nnc_mfa_sol_attention_hpp
#define GUARD_ccv_nnc_mfa_sol_attention_hpp

typedef struct {
  uint32_t N, T, H, block_size, approximation_start, approximation_end;
  float scale, tau;
  uint32_t query_block_size;
  uint32_t local_block_radius;
} ccv_nnc_mfa_sol_attention_params_t;

#ifdef __cplusplus
extern "C" {
#endif
void ccv_nnc_mfa_encode_sol_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_sol_attention_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);
#ifdef __cplusplus
}
#endif
#endif
