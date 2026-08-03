#ifndef GUARD_ccv_nnc_mfa_moe_routing_hpp
#define GUARD_ccv_nnc_mfa_moe_routing_hpp

typedef struct {
	uint32_t data_type;
	uint32_t expert_count;
	uint32_t kth;
	uint32_t hidden;
	float weight_scale;
	uint32_t preselected;
	uint32_t compact_single_token_activation;
} ccv_nnc_mfa_moe_routing_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_moe_routing(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_moe_routing_params_t params);
void ccv_nnc_mfa_encode_moe_routing(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_moe_routing_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
