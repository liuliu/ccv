#ifndef GUARD_ccv_nnc_mfa_rmsnorm_cmul_hpp
#define GUARD_ccv_nnc_mfa_rmsnorm_cmul_hpp

typedef struct {
	float epsilon;
	uint64_t a_data_type;
	uint64_t rotation_data_type;
	uint64_t scale_data_type;
	uint32_t row_count;
	uint32_t column_count;
	uint32_t broadcast_ratio;
	uint32_t elementwise_affine;
} ccv_nnc_mfa_rmsnorm_cmul_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_rmsnorm_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rmsnorm_cmul_params_t params);
void ccv_nnc_mfa_encode_rmsnorm_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rmsnorm_cmul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
