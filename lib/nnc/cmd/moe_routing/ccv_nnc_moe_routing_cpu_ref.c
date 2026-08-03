#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>

typedef struct {
	int expert;
	int token;
	int slot;
	float weight;
} ccv_nnc_moe_routing_pair_t;

#define less_than(a, b, aux) (((a).expert < (b).expert) || ((a).expert == (b).expert && ((a).token < (b).token || ((a).token == (b).token && (a).slot < (b).slot))))
static CCV_IMPLEMENT_QSORT(_ccv_nnc_moe_routing_pair_sort, ccv_nnc_moe_routing_pair_t, less_than)
#undef less_than

static float _ccv_nnc_moe_routing_probability(const float logit)
{
	return sqrtf(ccv_max(logit, 0) + log1pf(expf(-fabsf(logit))));
}

static int _ccv_nnc_moe_routing_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 3 || output_size != 5)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const logits = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const route = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const activation = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const gathered = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const route_weights = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const token_indices = (ccv_nnc_tensor_view_t*)outputs[2];
	ccv_nnc_tensor_view_t* const expert_indices = (ccv_nnc_tensor_view_t*)outputs[3];
	ccv_nnc_tensor_view_t* const expert_counts = (ccv_nnc_tensor_view_t*)outputs[4];
	const int kth = cmd.info.moe_routing.kth;
	const int token_count = logits->info.dim[0];
	const int expert_count = logits->info.dim[1];
	const int hidden = activation->info.dim[1];
	const int pair_count = token_count * kth;
	const int group_count = ccv_min(pair_count, expert_count);
	const int compact_single_token_activation = token_count == 1 && (cmd.info.moe_routing.flags & CCV_NNC_MOE_ROUTING_COMPACT_SINGLE_TOKEN_ACTIVATION);
	const int gathered_rows = compact_single_token_activation ? 1 : pair_count;
	if (kth <= 0 || expert_count < kth || token_count <= 0 || hidden <= 0 || cmd.info.moe_routing.weight_scale <= 0 ||
		(cmd.info.moe_routing.preselected != 0 && cmd.info.moe_routing.preselected != 1) ||
		ccv_nnc_tensor_nd(logits->info.dim) != 2 || ccv_nnc_tensor_nd(activation->info.dim) != 2 || activation->info.dim[0] != token_count ||
		logits->info.datatype != CCV_32F || route_weights->info.datatype != CCV_32F ||
		token_indices->info.datatype != CCV_32S || expert_indices->info.datatype != CCV_32S ||
		expert_counts->info.datatype != CCV_32S || activation->info.datatype != gathered->info.datatype ||
		(activation->info.datatype != CCV_32F && activation->info.datatype != CCV_16F && activation->info.datatype != CCV_16BF) ||
		gathered->info.dim[0] != gathered_rows || gathered->info.dim[1] != hidden ||
		ccv_nnc_tensor_count(route_weights->info) != pair_count || ccv_nnc_tensor_count(token_indices->info) != pair_count ||
		ccv_nnc_tensor_count(expert_indices->info) != group_count || ccv_nnc_tensor_count(expert_counts->info) != group_count ||
		!CCV_IS_TENSOR_CONTIGUOUS(logits) || !CCV_IS_TENSOR_CONTIGUOUS(route) ||
		!CCV_IS_TENSOR_CONTIGUOUS(activation) || !CCV_IS_TENSOR_CONTIGUOUS(gathered) ||
		!CCV_IS_TENSOR_CONTIGUOUS(route_weights) || !CCV_IS_TENSOR_CONTIGUOUS(token_indices) ||
		!CCV_IS_TENSOR_CONTIGUOUS(expert_indices) || !CCV_IS_TENSOR_CONTIGUOUS(expert_counts))
		return CCV_NNC_EXEC_INVALID;
	if (cmd.info.moe_routing.preselected)
	{
		if (route->info.datatype != CCV_32S || ccv_nnc_tensor_nd(route->info.dim) != 2 || route->info.dim[0] != token_count || route->info.dim[1] != kth)
			return CCV_NNC_EXEC_INVALID;
	} else if (route->info.datatype != CCV_32F || ccv_nnc_tensor_nd(route->info.dim) != 1 || route->info.dim[0] != expert_count) {
		return CCV_NNC_EXEC_INVALID;
	}
	const size_t workspace_size = sizeof(float) * expert_count + sizeof(ccv_nnc_moe_routing_pair_t) * pair_count + sizeof(int) * kth;
	float* const probabilities = (float*)ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_CPU_MEMORY);
	ccv_nnc_moe_routing_pair_t* const pairs = (ccv_nnc_moe_routing_pair_t*)(probabilities + expert_count);
	int* const selected = (int*)(pairs + pair_count);
	int token;
	for (token = 0; token < token_count; token++)
	{
		const float* const token_logits = logits->data.f32 + token * expert_count;
		int expert;
		for (expert = 0; expert < expert_count; expert++)
			probabilities[expert] = _ccv_nnc_moe_routing_probability(token_logits[expert]);
		int slot;
		if (cmd.info.moe_routing.preselected)
		{
			for (slot = 0; slot < kth; slot++)
				selected[slot] = route->data.i32[token * kth + slot];
		} else {
			for (slot = 0; slot < kth; slot++)
				selected[slot] = -1;
			for (expert = 0; expert < expert_count; expert++)
			{
				const float score = probabilities[expert] + route->data.f32[expert];
				for (slot = 0; slot < kth; slot++)
					if (selected[slot] < 0 || score > probabilities[selected[slot]] + route->data.f32[selected[slot]])
					{
						int j;
						for (j = kth - 1; j > slot; j--)
							selected[j] = selected[j - 1];
						selected[slot] = expert;
						break;
					}
			}
		}
		float sum = 0;
		for (slot = 0; slot < kth; slot++)
		{
			if (selected[slot] < 0 || selected[slot] >= expert_count)
				return CCV_NNC_EXEC_INVALID;
			sum += probabilities[selected[slot]];
		}
		sum = ccv_max(sum, 6.103515625e-5f);
		for (slot = 0; slot < kth; slot++)
			pairs[token * kth + slot] = (ccv_nnc_moe_routing_pair_t){
				.expert = selected[slot],
				.token = token,
				.slot = slot,
				.weight = probabilities[selected[slot]] / sum * cmd.info.moe_routing.weight_scale,
			};
	}
	if (token_count > 1)
		_ccv_nnc_moe_routing_pair_sort(pairs, pair_count, 0);
	const size_t element_size = CCV_GET_DATA_TYPE_SIZE(activation->info.datatype);
	if (compact_single_token_activation)
		memcpy(gathered->data.u8, activation->data.u8, (size_t)hidden * element_size);
	int i;
	for (i = 0; i < pair_count; i++)
	{
		route_weights->data.f32[i] = pairs[i].weight;
		token_indices->data.i32[i] = pairs[i].token;
		if (!compact_single_token_activation)
			memcpy(gathered->data.u8 + (size_t)i * hidden * element_size,
				activation->data.u8 + (size_t)pairs[i].token * hidden * element_size,
				(size_t)hidden * element_size);
	}
	if (token_count == 1)
	{
		for (i = 0; i < kth; i++)
		{
			expert_indices->data.i32[i] = pairs[i].expert;
			expert_counts->data.i32[i] = 1;
		}
	} else {
		for (i = 0; i < group_count; i++)
		{
			expert_indices->data.i32[i] = -1;
			expert_counts->data.i32[i] = 0;
		}
		int group = -1;
		int last_expert = -1;
		for (i = 0; i < pair_count; i++)
		{
			if (pairs[i].expert != last_expert)
			{
				++group;
				assert(group < group_count);
				last_expert = pairs[i].expert;
				expert_indices->data.i32[group] = last_expert;
			}
			++expert_counts->data.i32[group];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_moe_routing_forw;
}
