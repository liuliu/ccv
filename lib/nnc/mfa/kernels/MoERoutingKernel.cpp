#include "MoERoutingKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "CodeWriter.hpp"

static const char* _ccv_nnc_moe_routing_type_name(const uint32_t data_type)
{
	switch (data_type)
	{
		case 3:
			return "float";
		case 16:
			return "half";
		case 121:
			return "bfloat";
		default:
			CCV_NNC_MFA_PRECONDITION(false);
			return nullptr;
	}
}

MoERoutingKernel::MoERoutingKernel(MoERoutingKernelDescriptor descriptor, MTL::Device* const device)
{
	const char* const activation_type = _ccv_nnc_moe_routing_type_name(descriptor.activationDataType);
	const char* const routing_type = _ccv_nnc_moe_routing_type_name(descriptor.routingDataType);
	const char* const bias_type = _ccv_nnc_moe_routing_type_name(descriptor.biasDataType);
	CodeWriter source;
	source.SetValue("ACTIVATION_TYPE", activation_type);
	source.SetValue("ROUTING_TYPE", routing_type);
	source.SetValue("BIAS_TYPE", bias_type);
	source.SetValue("SIMD_GROUPS", std::to_string(descriptor.executionSIMDGroups));
	source += R"(
#include <metal_stdlib>
using namespace metal;

typedef {{ACTIVATION_TYPE}} activation_t;
typedef {{ROUTING_TYPE}} routing_t;
typedef {{BIAS_TYPE}} bias_t;

constant uint expert_count [[function_constant(0)]];
constant uint kth [[function_constant(1)]];
constant uint hidden [[function_constant(2)]];
constant float weight_scale [[function_constant(3)]];
constant bool preselected [[function_constant(4)]];
constant bool single_input_token [[function_constant(5)]];
constant float normalization_epsilon [[function_constant(6)]];
constant bool sigmoid_scoring [[function_constant(7)]];
constant uint token_count [[function_constant(8)]];

inline float stable_log1p(float x)
{
	const float xp1 = 1.0f + x;
	if (xp1 == 1.0f)
		return x;
	return x * (precise::log(xp1) / (xp1 - 1.0f));
}

inline float routing_probability(float x)
{
	if (sigmoid_scoring)
		return 1.0f / (1.0f + precise::exp(-x));
	const float positive = x > 0.0f ? x : 0.0f;
	return sqrt(positive + stable_log1p(precise::exp(-abs(x))));
}

kernel void moe_routing_t1(
	device const routing_t* logits [[buffer(0)]],
	device const char* route [[buffer(1)]],
	device const activation_t* activation [[buffer(2)]],
	device activation_t* gathered [[buffer(3)]],
	device float* route_weights [[buffer(4)]],
	device int* token_indices [[buffer(5)]],
	device int* expert_indices [[buffer(6)]],
	device int* expert_counts [[buffer(7)]],
	threadgroup float* probabilities [[threadgroup(0)]],
	uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]],
	uint simd_group [[simdgroup_index_in_threadgroup]],
	uint ntg [[threads_per_threadgroup]])
{
	threadgroup float group_scores[{{SIMD_GROUPS}}];
	threadgroup uint group_experts[{{SIMD_GROUPS}}];
	threadgroup uint top_experts[32];
	for (uint expert = tid; expert < expert_count; expert += ntg)
		probabilities[expert] = routing_probability((float)logits[expert]);
	threadgroup_barrier(mem_flags::mem_threadgroup);

	if (preselected)
	{
		if (tid == 0)
		{
			device const int* selected = (device const int*)route;
			for (uint slot = 0; slot < kth; slot++)
				top_experts[slot] = (uint)selected[slot];
		}
	} else if (expert_count <= 32 * 16) {
		// One SIMD group selects without threadgroup barriers: lane l holds experts l, l + 32, ...
		// Ties go to the lowest expert index, as in the threadgroup-wide search below. Experts out
		// of range or already selected are masked rather than scored -INFINITY, so experts whose
		// score is -INFINITY (or NaN, scored as -INFINITY) stay selectable, as on the CPU.
		if (simd_group == 0)
		{
			float scores[16];
			uint taken = 0;
			for (uint i = 0; i < 16; i++)
			{
				const uint expert = i * 32 + lane;
				const float score = expert < expert_count ? probabilities[expert] + (float)((device const bias_t*)route)[expert] : -INFINITY;
				scores[i] = isnan(score) ? -INFINITY : score;
				if (expert >= expert_count)
					taken |= 1u << i;
			}
			for (uint slot = 0; slot < kth; slot++)
			{
				float best_score = -INFINITY;
				uint best_expert = 0xffffffffu;
				for (uint i = 0; i < 16; i++)
					if (!(taken & (1u << i)) && (best_expert == 0xffffffffu || scores[i] > best_score))
					{
						best_score = scores[i];
						best_expert = i * 32 + lane;
					}
				const float simd_score = simd_max(best_score);
				const uint selected = simd_min(best_score == simd_score ? best_expert : 0xffffffffu);
				if (lane == 0)
					top_experts[slot] = selected;
				if (selected % 32 == lane)
					taken |= 1u << (selected / 32);
			}
		}
	} else {
		float candidate_score = tid < expert_count ? probabilities[tid] + (float)((device const bias_t*)route)[tid] : -INFINITY;
		for (uint slot = 0; slot < kth; slot++)
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
		for (uint slot = 0; slot < kth; slot++)
			selected_sum += probabilities[top_experts[slot]];
		const float denominator = normalization_epsilon > 0 ? selected_sum + normalization_epsilon : max(selected_sum, 6.103515625e-5f);
		const float selected_scale = weight_scale / denominator;
		for (uint slot = 0; slot < kth; slot++)
		{
			const uint expert = top_experts[slot];
			route_weights[slot] = probabilities[expert] * selected_scale;
			token_indices[slot] = 0;
			expert_indices[slot] = (int)expert;
			expert_counts[slot] = 1;
		}
	}

	if (single_input_token)
	{
		for (uint index = tid; index < hidden; index += ntg)
			gathered[index] = activation[index];
	} else {
		const uint gathered_count = kth * hidden;
		for (uint index = tid; index < gathered_count; index += ntg)
			gathered[index] = activation[index % hidden];
	}
}

// Up to 16 tokens of up to 8 experts each, as a speculative verify routes: SIMD group t selects
// token t's experts as moe_routing_t1's single SIMD group does, then the pairs sort by (expert,
// token, slot) and group by expert, as the CPU reference orders them.
kernel void moe_routing_tn(
	device const routing_t* logits [[buffer(0)]],
	device const char* route [[buffer(1)]],
	device const activation_t* activation [[buffer(2)]],
	device activation_t* gathered [[buffer(3)]],
	device float* route_weights [[buffer(4)]],
	device int* token_indices [[buffer(5)]],
	device int* expert_indices [[buffer(6)]],
	device int* expert_counts [[buffer(7)]],
	uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]],
	uint simd_group [[simdgroup_index_in_threadgroup]],
	uint ntg [[threads_per_threadgroup]])
{
	threadgroup uint pair_keys[128];
	threadgroup float pair_weights[128];
	threadgroup uint sorted_keys[128];
	threadgroup float sorted_weights[128];
	const uint pair_count = token_count * kth;
	if (simd_group < token_count)
	{
		const uint token = simd_group;
		float probabilities[16];
		float scores[16];
		uint taken = 0;
		for (uint i = 0; i < 16; i++)
		{
			const uint expert = i * 32 + lane;
			const float probability = expert < expert_count ? routing_probability((float)logits[token * expert_count + expert]) : 0.0f;
			probabilities[i] = probability;
			const float score = expert < expert_count ? probability + (preselected ? 0.0f : (float)((device const bias_t*)route)[expert]) : -INFINITY;
			scores[i] = isnan(score) ? -INFINITY : score;
			if (expert >= expert_count)
				taken |= 1u << i;
		}
		uint selected_experts[8];
		float selected_probabilities[8];
		float selected_sum = 0.0f;
		for (uint slot = 0; slot < kth; slot++)
		{
			uint selected;
			if (preselected)
				selected = (uint)((device const int*)route)[token * kth + slot];
			else {
				float best_score = -INFINITY;
				uint best_expert = 0xffffffffu;
				for (uint i = 0; i < 16; i++)
					if (!(taken & (1u << i)) && (best_expert == 0xffffffffu || scores[i] > best_score))
					{
						best_score = scores[i];
						best_expert = i * 32 + lane;
					}
				const float simd_score = simd_max(best_score);
				selected = simd_min(best_score == simd_score ? best_expert : 0xffffffffu);
				if (selected % 32 == lane)
					taken |= 1u << (selected / 32);
			}
			const float probability = simd_shuffle(probabilities[selected / 32], (ushort)(selected % 32));
			selected_experts[slot] = selected;
			selected_probabilities[slot] = probability;
			selected_sum += probability;
		}
		const float denominator = normalization_epsilon > 0 ? selected_sum + normalization_epsilon : max(selected_sum, 6.103515625e-5f);
		const float selected_scale = weight_scale / denominator;
		if (lane == 0)
			for (uint slot = 0; slot < kth; slot++)
			{
				pair_keys[token * kth + slot] = (selected_experts[slot] << 16) | (token << 8) | slot;
				pair_weights[token * kth + slot] = selected_probabilities[slot] * selected_scale;
			}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	// The keys are distinct, so a key's rank is its sorted position.
	if (tid < pair_count)
	{
		const uint key = pair_keys[tid];
		uint rank = 0;
		for (uint j = 0; j < pair_count; j++)
			rank += pair_keys[j] < key ? 1 : 0;
		sorted_keys[rank] = key;
		sorted_weights[rank] = pair_weights[tid];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const uint group_count = min(pair_count, expert_count);
	if (tid < group_count)
	{
		uint groups = 1;
		for (uint j = 1; j < pair_count; j++)
			groups += (sorted_keys[j] >> 16) != (sorted_keys[j - 1] >> 16) ? 1 : 0;
		if (tid >= groups)
		{
			expert_indices[tid] = -1;
			expert_counts[tid] = 0;
		}
	}
	if (tid < pair_count)
	{
		const uint key = sorted_keys[tid];
		const uint expert = key >> 16;
		route_weights[tid] = sorted_weights[tid];
		token_indices[tid] = (int)((key >> 8) & 0xffu);
		if (tid == 0 || (sorted_keys[tid - 1] >> 16) != expert)
		{
			uint group = 0;
			for (uint j = 1; j <= tid; j++)
				group += (sorted_keys[j] >> 16) != (sorted_keys[j - 1] >> 16) ? 1 : 0;
			uint count = 1;
			for (uint j = tid + 1; j < pair_count && (sorted_keys[j] >> 16) == expert; j++)
				count++;
			expert_indices[group] = (int)expert;
			expert_counts[group] = (int)count;
		}
	}
	const uint gathered_count = pair_count * hidden;
	for (uint index = tid; index < gathered_count; index += ntg)
	{
		const uint row = index / hidden;
		const uint token = (sorted_keys[row] >> 8) & 0xffu;
		gathered[index] = activation[token * hidden + (index - row * hidden)];
	}
}
)";
	const std::string shader = source.ToString();
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(NS::String::string(shader.c_str(), NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}
