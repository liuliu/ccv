#ifndef GUARD_ccv_nnc_mfa_hyper_connection_hpp
#define GUARD_ccv_nnc_mfa_hyper_connection_hpp

typedef struct {
	uint32_t row_count;
	uint32_t count;
	uint32_t hidden;
	uint32_t sinkhorn_iterations;
	float epsilon;
	uint32_t operation;
	uint8_t block_fp16;
	uint8_t loadM;
} ccv_nnc_mfa_hyper_connection_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_hyper_connection(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_hyper_connection_params_t params);
void ccv_nnc_mfa_encode_hyper_connection(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_hyper_connection_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
