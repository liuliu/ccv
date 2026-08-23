#ifndef GUARD_ccv_nnc_mfa_transpose_hpp
#define GUARD_ccv_nnc_mfa_transpose_hpp

typedef struct {
	uint64_t data_type;
	uint32_t batch_size;
	uint32_t rows;
	uint32_t cols;
	uint32_t source_batch_stride;
	uint32_t source_row_stride;
	uint32_t destination_batch_stride;
	uint32_t destination_row_stride;
} ccv_nnc_mfa_transpose_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_encode_transpose(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_transpose_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
