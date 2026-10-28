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
	// Expansion adds an FP32 addend, bound as the second output, to the block.
	uint8_t expand_addend;
	// Mix stages: 3 decode partials, 4 split, 5 decode split and collapse normalization, 6 collapse
	// and normalization per row.
	float norm_epsilon;
	uint32_t dot_splits;
	uint32_t collapse_splits;
} ccv_nnc_mfa_hyper_connection_params_t;

// Threads and flattened residual columns per decode mix partial threadgroup.
#define CCV_NNC_MFA_HYPER_CONNECTION_MIX_THREADS (256)
#define CCV_NNC_MFA_HYPER_CONNECTION_MIX_CHUNK (1024)

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_hyper_connection(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_hyper_connection_params_t params);
void ccv_nnc_mfa_encode_hyper_connection(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_hyper_connection_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
