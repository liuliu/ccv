#ifndef GUARD_ccv_nnc_mfa_dequantize_8i_rowwise_x_hpp
#define GUARD_ccv_nnc_mfa_dequantize_8i_rowwise_x_hpp

typedef struct {
	uint64_t data_type;
	uint32_t format;
	uint64_t row_length;
	uint64_t length;
} ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t;

typedef struct {
	uint64_t data_type;
	uint32_t format;
	uint64_t row_length;
	uint64_t rows_per_expert;
	uint64_t expert_count;
	uint64_t segment_count;
} ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params);
void ccv_nnc_mfa_encode_dequantize_8i_rowwise_x(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);
size_t ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params);
void ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
