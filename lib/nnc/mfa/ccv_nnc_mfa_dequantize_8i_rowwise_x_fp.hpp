#ifndef GUARD_ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_hpp
#define GUARD_ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_hpp

typedef struct {
	uint64_t data_type;
	uint32_t format;
	uint64_t row_length;
	uint64_t length;
	// The external codebook of a TMCQ format whose bits 16 - 23 hold its log2 entry count (see
	// ccv_nnc_mfa_codebook_buffer); bound at buffer(3). NULL for the built-in table.
	mtl_buffer_t* codebook;
} ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x_fp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_params_t params);
void ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_fp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
