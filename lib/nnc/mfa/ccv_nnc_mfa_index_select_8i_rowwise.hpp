#ifndef GUARD_ccv_nnc_mfa_index_select_8i_rowwise_hpp
#define GUARD_ccv_nnc_mfa_index_select_8i_rowwise_hpp

typedef struct {
	uint64_t data_type;
	uint64_t row_length;
	uint64_t input_length;
	uint64_t output_length;
	uint8_t loadM;
} ccv_nnc_mfa_index_select_8i_rowwise_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_index_select_8i_rowwise(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_index_select_8i_rowwise_params_t params);
void ccv_nnc_mfa_encode_index_select_8i_rowwise(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_index_select_8i_rowwise_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
