#include "MoERoutingKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

MoERoutingKernel::MoERoutingKernel(MoERoutingKernelDescriptor descriptor, MTL::Device* const device)
{
	const char* activation_type = nullptr;
	switch (descriptor.dataType)
	{
		case 3:
			activation_type = "float";
			break;
		case 16:
			activation_type = "half";
			break;
		case 121:
			activation_type = "bfloat";
			break;
		default:
			CCV_NNC_MFA_PRECONDITION(false);
	}
	std::string source = R"(
#include <metal_stdlib>
using namespace metal;

typedef ACTIVATION_TYPE activation_t;

struct moe_routing_params {
	uint expert_count;
	uint kth;
	uint hidden;
	float weight_scale;
	uint preselected;
	uint compact_single_token_activation;
};

inline float stable_log1p(float x)
{
	const float xp1 = 1.0f + x;
	if (xp1 == 1.0f)
		return x;
	return x * (precise::log(xp1) / (xp1 - 1.0f));
}

inline float routing_probability(float x)
{
	const float positive = x > 0.0f ? x : 0.0f;
	return sqrt(positive + stable_log1p(precise::exp(-abs(x))));
}

kernel void moe_routing_t1(
	device const float* logits [[buffer(0)]],
	device const char* route [[buffer(1)]],
	device const activation_t* activation [[buffer(2)]],
	device activation_t* gathered [[buffer(3)]],
	device float* route_weights [[buffer(4)]],
	device int* token_indices [[buffer(5)]],
	device int* expert_indices [[buffer(6)]],
	device int* expert_counts [[buffer(7)]],
	constant moe_routing_params& params [[buffer(8)]],
	threadgroup float* probabilities [[threadgroup(0)]],
	uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]],
	uint simd_group [[simdgroup_index_in_threadgroup]],
	uint ntg [[threads_per_threadgroup]])
{
	threadgroup float group_scores[8];
	threadgroup uint group_experts[8];
	threadgroup uint top_experts[32];
	for (uint expert = tid; expert < params.expert_count; expert += ntg)
		probabilities[expert] = routing_probability(logits[expert]);
	threadgroup_barrier(mem_flags::mem_threadgroup);

	if (params.preselected)
	{
		if (tid == 0)
		{
			device const int* selected = (device const int*)route;
			for (uint slot = 0; slot < params.kth; slot++)
				top_experts[slot] = (uint)selected[slot];
		}
	} else {
		float candidate_score = tid < params.expert_count ? probabilities[tid] + ((device const float*)route)[tid] : -INFINITY;
		for (uint slot = 0; slot < params.kth; slot++)
		{
			const float simd_score = simd_max(candidate_score);
			const uint candidate_expert = candidate_score == simd_score ? tid : 0xffffffffu;
			const uint simd_expert = simd_min(candidate_expert);
			if (lane == 0)
			{
				group_scores[simd_group] = simd_score;
				group_experts[simd_group] = simd_expert;
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			if (simd_group == 0)
			{
				const uint simd_groups = (ntg + 31) / 32;
				const float group_score = lane < simd_groups ? group_scores[lane] : -INFINITY;
				const float best_score = simd_max(group_score);
				const uint group_expert = group_score == best_score ? group_experts[lane] : 0xffffffffu;
				const uint best_expert = simd_min(group_expert);
				if (lane == 0)
					top_experts[slot] = best_expert;
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			if (tid == top_experts[slot])
				candidate_score = -INFINITY;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	if (tid == 0)
	{
		float selected_sum = 0.0f;
		for (uint slot = 0; slot < params.kth; slot++)
			selected_sum += probabilities[top_experts[slot]];
		const float selected_scale = params.weight_scale / max(selected_sum, 6.103515625e-5f);
		for (uint slot = 0; slot < params.kth; slot++)
		{
			const uint expert = top_experts[slot];
			route_weights[slot] = probabilities[expert] * selected_scale;
			token_indices[slot] = 0;
			expert_indices[slot] = (int)expert;
			expert_counts[slot] = 1;
		}
	}

	if (params.compact_single_token_activation)
	{
		for (uint index = tid; index < params.hidden; index += ntg)
			gathered[index] = activation[index];
	} else {
		const uint gathered_count = params.kth * params.hidden;
		for (uint index = tid; index < gathered_count; index += ntg)
			gathered[index] = activation[index % params.hidden];
	}
}
)";
	const std::string activation_token = "ACTIVATION_TYPE";
	const std::string::size_type activation_position = source.find(activation_token);
	CCV_NNC_MFA_PRECONDITION(activation_position != std::string::npos);
	source.replace(activation_position, activation_token.size(), activation_type);
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}
