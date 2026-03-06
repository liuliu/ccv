#ifndef GUARD_ccv_nnc_mfa_depalettize_hpp
#define GUARD_ccv_nnc_mfa_depalettize_hpp

typedef struct {
  uint64_t data_type;
  int qbits;
  int number_in_blocks;
  uint64_t length;
} ccv_nnc_mfa_depalettize_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_depalettize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depalettize_params_t params);
void ccv_nnc_mfa_encode_depalettize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depalettize_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
