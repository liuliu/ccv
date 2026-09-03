#ifndef GUARD_ccv_nnc_mfa_fast_fence_hpp
#define GUARD_ccv_nnc_mfa_fast_fence_hpp

typedef struct {
	uint32_t word_offset;
	uint32_t word_count;
	uint32_t pending;
	uint32_t complete;
} ccv_nnc_mfa_fast_fence_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int ccv_nnc_mfa_prepare_fast_fence(ccv_nnc_mfa_context_t* context);
int ccv_nnc_mfa_encode_fast_fence(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_fast_fence_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);
int ccv_nnc_mfa_encode_fast_fence_wait(ccv_nnc_mfa_context_t* context, uint32_t value, mtl_command_batch_t* command_batch, mtl_buffer_t* timestamp, size_t timestamp_offset);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
