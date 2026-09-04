#ifndef GUARD_ccv_nnc_mfa_moe_weights_streaming_hpp
#define GUARD_ccv_nnc_mfa_moe_weights_streaming_hpp

typedef struct {
	uint32_t generation;
	uint32_t index_count;
	uint32_t expert_count;
	uint32_t resident_slots;
	uint32_t routing_width;
	uint32_t route_weight_count;
	uint32_t route_weight_bytes;
} ccv_nnc_mfa_moe_weights_streaming_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_moe_weights_streaming(ccv_nnc_mfa_context_t* context);
void ccv_nnc_mfa_encode_moe_weights_streaming(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_moe_weights_streaming_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
