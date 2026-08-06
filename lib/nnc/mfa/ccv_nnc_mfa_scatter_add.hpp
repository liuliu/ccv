#ifndef GUARD_ccv_nnc_mfa_scatter_add_hpp
#define GUARD_ccv_nnc_mfa_scatter_add_hpp

typedef struct {
	uint64_t data_type;
	uint32_t input_rows;
	uint32_t output_rows;
	uint32_t columns;
	uint32_t count_per_output;
} ccv_nnc_mfa_scatter_add_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_scatter_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scatter_add_params_t params);
void ccv_nnc_mfa_encode_scatter_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scatter_add_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
