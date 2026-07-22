#ifndef GUARD_ccv_nnc_mfa_strided_copy_hpp
#define GUARD_ccv_nnc_mfa_strided_copy_hpp

typedef struct {
	uint64_t data_type;
	uint32_t rows;
	uint32_t cols;
	uint32_t source_row_stride;
	uint32_t destination_row_stride;
	uint8_t loadM;
} ccv_nnc_mfa_strided_copy_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_encode_strided_copy(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_strided_copy_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
