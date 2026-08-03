#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static float _moe_routing_probability(const float logit)
{
	return sqrtf(ccv_max(logit, 0) + log1pf(expf(-fabsf(logit))));
}

TEST_CASE("moe routing infers outputs and groups preselected routes on CPU")
{
	const int token_count = 4;
	const int expert_count = 8;
	const int kth = 3;
	const int hidden = 5;
	const int pair_count = token_count * kth;
	const float weight_scale = 1.25f;
	const ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD(kth, weight_scale, 1);
	ccv_nnc_tensor_param_t input_params[] = {
		CPU_TENSOR_NHWC(32F, token_count, expert_count),
		CPU_TENSOR_NHWC(32S, token_count, kth),
		CPU_TENSOR_NHWC(16F, token_count, hidden),
	};
	ccv_nnc_tensor_param_t output_params[5];
	ccv_nnc_hint_tensor_auto(cmd, input_params, 3, ccv_nnc_no_hint, output_params, 5);
	REQUIRE_EQ(output_params[0].datatype, CCV_16F, "gathered activations should preserve the activation datatype");
	REQUIRE_EQ(output_params[0].dim[0], pair_count, "gathered activations should contain one row per route");
	REQUIRE_EQ(output_params[0].dim[1], hidden, "gathered activations should preserve hidden width");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[1]), pair_count, "weights should contain one value per route");
	REQUIRE_EQ(output_params[1].datatype, CCV_32F, "weights should be float");
	REQUIRE_EQ(output_params[2].datatype, CCV_32S, "source tokens should be int32");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[3]), expert_count, "group metadata should be bounded by the expert count");

	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const selected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, token_count, kth), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, hidden), 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, pair_count, hidden), 0);
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, pair_count), 0);
	ccv_nnc_tensor_t* const tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, pair_count), 0);
	ccv_nnc_tensor_t* const experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, expert_count), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, expert_count), 0);
	int i;
	for (i = 0; i < token_count * expert_count; i++)
		logits->data.f32[i] = (float)(i - 11) / 8.0f;
	const int selected_values[] = {
		5, 1, 3,
		1, 6, 3,
		7, 1, 5,
		3, 7, 1,
	};
	memcpy(selected->data.i32, selected_values, sizeof(selected_values));
	for (i = 0; i < token_count; i++)
	{
		int j;
		for (j = 0; j < hidden; j++)
			activation->data.f32[i * hidden + j] = (float)(i * 100 + j);
	}
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0,
		TENSOR_LIST(logits, selected, activation),
		TENSOR_LIST(gathered, weights, tokens, experts, counts), 0),
		CCV_NNC_EXEC_SUCCESS, "preselected CPU routing should execute");
	const int expected_experts[] = { 1, 3, 5, 6, 7, -1, -1, -1 };
	const int expected_counts[] = { 4, 3, 2, 1, 2, 0, 0, 0 };
	const int expected_route_experts[] = { 1, 1, 1, 1, 3, 3, 3, 5, 5, 6, 7, 7 };
	const int expected_tokens[] = { 0, 1, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3 };
	REQUIRE_ARRAY_EQ(int, experts->data.i32, expected_experts, expert_count, "expert groups should be sorted and padded");
	REQUIRE_ARRAY_EQ(int, counts->data.i32, expected_counts, expert_count, "expert counts should describe every grouped route");
	REQUIRE_ARRAY_EQ(int, tokens->data.i32, expected_tokens, pair_count, "source tokens should stay aligned after grouping");
	float token_weight_sums[4];
	memset(token_weight_sums, 0, sizeof(token_weight_sums));
	for (i = 0; i < pair_count; i++)
	{
		const int token = expected_tokens[i];
		const int expert = expected_route_experts[i];
		float denominator = 0;
		int slot;
		for (slot = 0; slot < kth; slot++)
			denominator += _moe_routing_probability(logits->data.f32[token * expert_count + selected_values[token * kth + slot]]);
		const float expected_weight = _moe_routing_probability(logits->data.f32[token * expert_count + expert]) / denominator * weight_scale;
		REQUIRE_EQ_WITH_TOLERANCE(weights->data.f32[i], expected_weight, 1e-6, "route weights should use the unbiased probabilities");
		token_weight_sums[token] += weights->data.f32[i];
		int j;
		for (j = 0; j < hidden; j++)
			REQUIRE_EQ(gathered->data.f32[i * hidden + j], activation->data.f32[token * hidden + j], "gathered activations should follow source-token metadata");
	}
	for (i = 0; i < token_count; i++)
		REQUIRE_EQ_WITH_TOLERANCE(token_weight_sums[i], weight_scale, 1e-6, "each token's selected weights should sum to weight_scale");
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(experts);
	ccv_nnc_tensor_free(tokens);
	ccv_nnc_tensor_free(weights);
	ccv_nnc_tensor_free(gathered);
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(selected);
	ccv_nnc_tensor_free(logits);
}

TEST_CASE("moe routing selects biased top-k and rejects invalid preselected experts on CPU")
{
	const int expert_count = 8;
	const int kth = 3;
	const int hidden = 4;
	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, expert_count), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, expert_count), 0);
	ccv_nnc_tensor_t* const selected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, kth), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, hidden), 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, kth, hidden), 0);
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, kth), 0);
	ccv_nnc_tensor_t* const tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	ccv_nnc_tensor_t* const experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	int i;
	for (i = 0; i < expert_count; i++)
	{
		logits->data.f32[i] = 0;
		bias->data.f32[i] = (float)i;
	}
	for (i = 0; i < hidden; i++)
		activation->data.f32[i] = (float)(i + 1);
	const ccv_nnc_cmd_t standard_cmd = CMD_MOE_ROUTING_FORWARD(kth, 1.5f, 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(standard_cmd, ccv_nnc_no_hint, 0,
		TENSOR_LIST(logits, bias, activation), TENSOR_LIST(gathered, weights, tokens, experts, counts), 0),
		CCV_NNC_EXEC_SUCCESS, "standard CPU routing should execute");
	const int expected_experts[] = { 7, 6, 5 };
	const int expected_counts[] = { 1, 1, 1 };
	const int expected_tokens[] = { 0, 0, 0 };
	REQUIRE_ARRAY_EQ(int, experts->data.i32, expected_experts, kth, "selection bias should determine top-k experts");
	REQUIRE_ARRAY_EQ(int, counts->data.i32, expected_counts, kth, "single-token routes should each have count one");
	REQUIRE_ARRAY_EQ(int, tokens->data.i32, expected_tokens, kth, "single-token source indices should be zero");
	for (i = 0; i < kth; i++)
		REQUIRE_EQ_WITH_TOLERANCE(weights->data.f32[i], 0.5f, 1e-6, "equal unbiased probabilities should normalize equally");
	const int invalid_selected[] = { 1, expert_count, 2 };
	memcpy(selected->data.i32, invalid_selected, sizeof(invalid_selected));
	const ccv_nnc_cmd_t preselected_cmd = CMD_MOE_ROUTING_FORWARD(kth, 1.0f, 1);
	REQUIRE_EQ(ccv_nnc_cmd_exec(preselected_cmd, ccv_nnc_no_hint, 0,
		TENSOR_LIST(logits, selected, activation), TENSOR_LIST(gathered, weights, tokens, experts, counts), 0),
		CCV_NNC_EXEC_INVALID, "preselected routing should reject an out-of-range expert ID");
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(experts);
	ccv_nnc_tensor_free(tokens);
	ccv_nnc_tensor_free(weights);
	ccv_nnc_tensor_free(gathered);
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(selected);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(logits);
}

TEST_CASE("moe routing optionally keeps a single-token activation compact on CPU")
{
	const int expert_count = 8;
	const int kth = 3;
	const int hidden = 4;
	const ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 1.5f, 0, CCV_NNC_MOE_ROUTING_COMPACT_SINGLE_TOKEN_ACTIVATION);
	ccv_nnc_tensor_param_t single_token_inputs[] = {
		CPU_TENSOR_NHWC(32F, 1, expert_count),
		CPU_TENSOR_NHWC(32F, expert_count),
		CPU_TENSOR_NHWC(32F, 1, hidden),
	};
	ccv_nnc_tensor_param_t output_params[5];
	ccv_nnc_hint_tensor_auto(cmd, single_token_inputs, 3, ccv_nnc_no_hint, output_params, 5);
	REQUIRE_EQ(output_params[0].dim[0], 1, "compact single-token routing should preserve one activation row");
	REQUIRE_EQ(output_params[0].dim[1], hidden, "compact single-token routing should preserve hidden width");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[1]), kth, "compact routing should retain one weight per selected expert");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[2]), kth, "compact routing should retain one token index per selected expert");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[3]), kth, "compact routing should retain one expert ID per selected expert");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[4]), kth, "compact routing should retain one expert count per selected expert");
	const ccv_nnc_tensor_param_t compact_gathered_params = output_params[0];
	ccv_nnc_tensor_param_t multi_token_inputs[] = {
		CPU_TENSOR_NHWC(32F, 2, expert_count),
		CPU_TENSOR_NHWC(32F, expert_count),
		CPU_TENSOR_NHWC(32F, 2, hidden),
	};
	ccv_nnc_hint_tensor_auto(cmd, multi_token_inputs, 3, ccv_nnc_no_hint, output_params, 5);
	REQUIRE_EQ(output_params[0].dim[0], 2 * kth, "the compact flag should retain grouped activation rows for multiple tokens");

	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, single_token_inputs[0], 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, single_token_inputs[1], 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, single_token_inputs[2], 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, compact_gathered_params, 0);
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, kth), 0);
	ccv_nnc_tensor_t* const tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	ccv_nnc_tensor_t* const experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	int i;
	for (i = 0; i < expert_count; i++)
	{
		logits->data.f32[i] = 0;
		bias->data.f32[i] = (float)i;
	}
	for (i = 0; i < hidden; i++)
		activation->data.f32[i] = (float)(i + 1);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0,
		TENSOR_LIST(logits, bias, activation), TENSOR_LIST(gathered, weights, tokens, experts, counts), 0),
		CCV_NNC_EXEC_SUCCESS, "compact single-token CPU routing should execute");
	REQUIRE_ARRAY_EQ(float, gathered->data.f32, activation->data.f32, hidden, "compact routing should copy the activation only once");
	const int expected_experts[] = { 7, 6, 5 };
	const int expected_counts[] = { 1, 1, 1 };
	const int expected_tokens[] = { 0, 0, 0 };
	REQUIRE_ARRAY_EQ(int, experts->data.i32, expected_experts, kth, "compact routing should preserve selected expert IDs");
	REQUIRE_ARRAY_EQ(int, counts->data.i32, expected_counts, kth, "compact routing should preserve expert counts");
	REQUIRE_ARRAY_EQ(int, tokens->data.i32, expected_tokens, kth, "compact routing should preserve token indices");
	for (i = 0; i < kth; i++)
		REQUIRE_EQ_WITH_TOLERANCE(weights->data.f32[i], 0.5f, 1e-6, "compact routing should preserve normalized route weights");
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(experts);
	ccv_nnc_tensor_free(tokens);
	ccv_nnc_tensor_free(weights);
	ccv_nnc_tensor_free(gathered);
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(logits);
}

typedef struct {
	int expert;
	int token;
	int index;
} moe_routing_record_t;

typedef struct {
	int status;
	int metadata_match;
	int gathered_match;
	float max_weight_difference;
} moe_routing_mps_result_t;

static int _moe_routing_record_compare(const void* const a, const void* const b)
{
	const moe_routing_record_t* const ap = (const moe_routing_record_t*)a;
	const moe_routing_record_t* const bp = (const moe_routing_record_t*)b;
	if (ap->expert != bp->expert)
		return ap->expert < bp->expert ? -1 : 1;
	if (ap->token != bp->token)
		return ap->token < bp->token ? -1 : 1;
	return ap->index - bp->index;
}

static int _moe_routing_records(const ccv_nnc_tensor_t* const tokens, const ccv_nnc_tensor_t* const experts, const ccv_nnc_tensor_t* const counts, const int token_count, const int expert_count, const int pair_count, const int group_count, moe_routing_record_t* const records)
{
	int position = 0;
	int group;
	for (group = 0; group < group_count; group++)
	{
		const int count = counts->data.i32[group];
		const int expert = experts->data.i32[group];
		if (count < 0 || position + count > pair_count || (count == 0 && expert != -1) || (count > 0 && (expert < 0 || expert >= expert_count)))
			return 0;
		int j;
		for (j = 0; j < count; j++)
		{
			const int token = tokens->data.i32[position];
			if (token < 0 || token >= token_count)
				return 0;
			records[position] = (moe_routing_record_t){ .expert = expert, .token = token, .index = position };
			++position;
		}
	}
	return position == pair_count;
}

static moe_routing_mps_result_t _moe_routing_mps_case(const int token_count, const int expert_count, const int kth, const int hidden, const int preselected, const int disable_mfa, const int activation_datatype, const int command_flags)
{
	const int pair_count = token_count * kth;
	const int group_count = ccv_min(pair_count, expert_count);
	const int compact_single_token_activation = token_count == 1 && (command_flags & CCV_NNC_MOE_ROUTING_COMPACT_SINGLE_TOKEN_ACTIVATION);
	const int gathered_rows = compact_single_token_activation ? 1 : pair_count;
	moe_routing_mps_result_t result = {
		.status = CCV_NNC_EXEC_SUCCESS,
		.metadata_match = 1,
		.gathered_match = 1,
		.max_weight_difference = 0,
	};
	ccv_nnc_tensor_param_t hroute_params = CPU_TENSOR_NHWC(32F, expert_count);
	ccv_nnc_tensor_param_t route_params = GPU_TENSOR_NHWC(000, 32F, expert_count);
	if (preselected)
	{
		hroute_params = CPU_TENSOR_NHWC(32S, token_count, kth);
		route_params = GPU_TENSOR_NHWC(000, 32S, token_count, kth);
	}
	ccv_nnc_tensor_param_t hactivation_params = CPU_TENSOR_NHWC(32F, token_count, hidden);
	ccv_nnc_tensor_param_t activation_params = GPU_TENSOR_NHWC(000, 32F, token_count, hidden);
	ccv_nnc_tensor_param_t hgathered_params = CPU_TENSOR_NHWC(32F, gathered_rows, hidden);
	ccv_nnc_tensor_param_t gathered_params = GPU_TENSOR_NHWC(000, 32F, gathered_rows, hidden);
	hactivation_params.datatype = activation_datatype;
	activation_params.datatype = activation_datatype;
	hgathered_params.datatype = activation_datatype;
	gathered_params.datatype = activation_datatype;
	ccv_nnc_tensor_t* const hlogits = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const hroute = ccv_nnc_tensor_new(0, hroute_params, 0);
	ccv_nnc_tensor_t* const hactivation = ccv_nnc_tensor_new(0, hactivation_params, 0);
	ccv_nnc_tensor_t* const expected_gathered = ccv_nnc_tensor_new(0, hgathered_params, 0);
	ccv_nnc_tensor_t* const expected_weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, pair_count), 0);
	ccv_nnc_tensor_t* const expected_tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, pair_count), 0);
	ccv_nnc_tensor_t* const expected_experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
	ccv_nnc_tensor_t* const expected_counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, route_params, 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, activation_params, 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, gathered_params, 0);
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, pair_count), 0);
	ccv_nnc_tensor_t* const tokens = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, pair_count), 0);
	ccv_nnc_tensor_t* const experts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, group_count), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, group_count), 0);
	ccv_nnc_tensor_t* const actual_gathered = ccv_nnc_tensor_new(0, hgathered_params, 0);
	ccv_nnc_tensor_t* const actual_weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, pair_count), 0);
	ccv_nnc_tensor_t* const actual_tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, pair_count), 0);
	ccv_nnc_tensor_t* const actual_experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
	ccv_nnc_tensor_t* const actual_counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
	int i;
	for (i = 0; i < token_count; i++)
	{
		int expert;
		for (expert = 0; expert < expert_count; expert++)
		{
			const int rank = (expert * 37 + i * 53) % expert_count;
			hlogits->data.f32[i * expert_count + expert] = (float)(rank - expert_count / 2) * 0.03125f;
		}
	}
	if (preselected)
	{
		for (i = 0; i < token_count; i++)
		{
			int slot;
			for (slot = 0; slot < kth; slot++)
				hroute->data.i32[i * kth + slot] = (i * 29 + slot * 43 + 7) % expert_count;
		}
	} else {
		for (i = 0; i < expert_count; i++)
			hroute->data.f32[i] = (float)i * 0.00001f;
	}
	const int activation_count = token_count * hidden;
	float* const activation_values = (float*)malloc(sizeof(float) * activation_count);
	for (i = 0; i < activation_count; i++)
		activation_values[i] = (float)(((i * 13) % 127) - 63) / 64.0f;
	if (activation_datatype == CCV_32F)
		memcpy(hactivation->data.f32, activation_values, sizeof(float) * activation_count);
	else if (activation_datatype == CCV_16F)
		ccv_float_to_half_precision(activation_values, (uint16_t*)hactivation->data.f16, activation_count);
	else
		ccv_float_to_bfloat(activation_values, (uint16_t*)hactivation->data.f16, activation_count);
	free(activation_values);
	ccv_nnc_cmd_t cpu_cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 1.25f, preselected, command_flags);
	cpu_cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	result.status = ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0,
		TENSOR_LIST(hlogits, hroute, hactivation),
		TENSOR_LIST(expected_gathered, expected_weights, expected_tokens, expected_experts, expected_counts), 0);
	const uint64_t old_flags = ccv_nnc_flags();
	if (disable_mfa)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	if (result.status == CCV_NNC_EXEC_SUCCESS)
		result.status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(hlogits, hroute, hactivation), TENSOR_LIST(logits, route, activation), stream);
	ccv_nnc_cmd_t mps_cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 1.25f, preselected, command_flags);
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	if (result.status == CCV_NNC_EXEC_SUCCESS)
		result.status = ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0,
			TENSOR_LIST(logits, route, activation), TENSOR_LIST(gathered, weights, tokens, experts, counts), stream);
	if (result.status == CCV_NNC_EXEC_SUCCESS)
		result.status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(gathered, weights, tokens, experts, counts),
			TENSOR_LIST(actual_gathered, actual_weights, actual_tokens, actual_experts, actual_counts), stream);
	ccv_nnc_stream_context_wait(stream);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	moe_routing_record_t* const expected_records = (moe_routing_record_t*)malloc(sizeof(moe_routing_record_t) * pair_count);
	moe_routing_record_t* const actual_records = (moe_routing_record_t*)malloc(sizeof(moe_routing_record_t) * pair_count);
	if (result.status == CCV_NNC_EXEC_SUCCESS)
	{
		const size_t row_size = (size_t)hidden * CCV_GET_DATA_TYPE_SIZE(activation_datatype);
		if (compact_single_token_activation && memcmp(expected_gathered->data.u8, actual_gathered->data.u8, row_size) != 0)
			result.gathered_match = 0;
		result.metadata_match = _moe_routing_records(expected_tokens, expected_experts, expected_counts, token_count, expert_count, pair_count, group_count, expected_records) &&
			_moe_routing_records(actual_tokens, actual_experts, actual_counts, token_count, expert_count, pair_count, group_count, actual_records);
		if (result.metadata_match)
		{
			qsort(expected_records, pair_count, sizeof(moe_routing_record_t), _moe_routing_record_compare);
			qsort(actual_records, pair_count, sizeof(moe_routing_record_t), _moe_routing_record_compare);
			for (i = 0; i < pair_count; i++)
			{
				if (expected_records[i].expert != actual_records[i].expert || expected_records[i].token != actual_records[i].token)
				{
					result.metadata_match = 0;
					break;
				}
				const float difference = fabsf(expected_weights->data.f32[expected_records[i].index] - actual_weights->data.f32[actual_records[i].index]);
				result.max_weight_difference = ccv_max(result.max_weight_difference, difference);
				if (!compact_single_token_activation && memcmp(expected_gathered->data.u8 + (size_t)expected_records[i].index * row_size,
					actual_gathered->data.u8 + (size_t)actual_records[i].index * row_size, row_size) != 0)
					result.gathered_match = 0;
			}
		}
	}
	free(actual_records);
	free(expected_records);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual_counts);
	ccv_nnc_tensor_free(actual_experts);
	ccv_nnc_tensor_free(actual_tokens);
	ccv_nnc_tensor_free(actual_weights);
	ccv_nnc_tensor_free(actual_gathered);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(experts);
	ccv_nnc_tensor_free(tokens);
	ccv_nnc_tensor_free(weights);
	ccv_nnc_tensor_free(gathered);
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(logits);
	ccv_nnc_tensor_free(expected_counts);
	ccv_nnc_tensor_free(expected_experts);
	ccv_nnc_tensor_free(expected_tokens);
	ccv_nnc_tensor_free(expected_weights);
	ccv_nnc_tensor_free(expected_gathered);
	ccv_nnc_tensor_free(hactivation);
	ccv_nnc_tensor_free(hroute);
	ccv_nnc_tensor_free(hlogits);
	return result;
}

TEST_CASE("single-token moe routing MFA and MPSGraph paths match CPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const moe_routing_mps_result_t standard_mfa = _moe_routing_mps_case(1, 256, 6, 4096, 0, 0, CCV_16F, 0);
	const moe_routing_mps_result_t standard_graph = _moe_routing_mps_case(1, 256, 6, 4096, 0, 1, CCV_16F, 0);
	const moe_routing_mps_result_t selected_mfa = _moe_routing_mps_case(1, 256, 6, 1024, 1, 0, CCV_16BF, 0);
	const moe_routing_mps_result_t selected_graph = _moe_routing_mps_case(1, 256, 6, 1024, 1, 1, CCV_16BF, 0);
	const moe_routing_mps_result_t compact_mfa = _moe_routing_mps_case(1, 256, 6, 4096, 0, 0, CCV_16F, CCV_NNC_MOE_ROUTING_COMPACT_SINGLE_TOKEN_ACTIVATION);
	const moe_routing_mps_result_t compact_graph = _moe_routing_mps_case(1, 256, 6, 4096, 0, 1, CCV_16F, CCV_NNC_MOE_ROUTING_COMPACT_SINGLE_TOKEN_ACTIVATION);
	REQUIRE_EQ(standard_mfa.status, CCV_NNC_EXEC_SUCCESS, "standard single-token MFA routing should execute");
	REQUIRE(standard_mfa.metadata_match, "standard single-token MFA metadata should match CPU");
	REQUIRE(standard_mfa.gathered_match, "standard single-token MFA gathering should match CPU exactly");
	REQUIRE(standard_mfa.max_weight_difference <= 5e-4f, "standard single-token MFA weights should match CPU");
	REQUIRE_EQ(standard_graph.status, CCV_NNC_EXEC_SUCCESS, "standard single-token MPSGraph routing should execute");
	REQUIRE(standard_graph.metadata_match, "standard single-token MPSGraph metadata should match CPU");
	REQUIRE(standard_graph.gathered_match, "standard single-token MPSGraph gathering should match CPU exactly");
	REQUIRE(standard_graph.max_weight_difference <= 5e-4f, "standard single-token MPSGraph weights should match CPU");
	REQUIRE_EQ(selected_mfa.status, CCV_NNC_EXEC_SUCCESS, "preselected single-token MFA routing should execute");
	REQUIRE(selected_mfa.metadata_match, "preselected single-token MFA metadata should match CPU");
	REQUIRE(selected_mfa.gathered_match, "preselected single-token MFA gathering should match CPU exactly");
	REQUIRE(selected_mfa.max_weight_difference <= 5e-4f, "preselected single-token MFA weights should match CPU");
	REQUIRE_EQ(selected_graph.status, CCV_NNC_EXEC_SUCCESS, "preselected single-token MPSGraph routing should execute");
	REQUIRE(selected_graph.metadata_match, "preselected single-token MPSGraph metadata should match CPU");
	REQUIRE(selected_graph.gathered_match, "preselected single-token MPSGraph gathering should match CPU exactly");
	REQUIRE(selected_graph.max_weight_difference <= 5e-4f, "preselected single-token MPSGraph weights should match CPU");
	REQUIRE_EQ(compact_mfa.status, CCV_NNC_EXEC_SUCCESS, "compact single-token MFA routing should execute");
	REQUIRE(compact_mfa.metadata_match, "compact single-token MFA metadata should match CPU");
	REQUIRE(compact_mfa.gathered_match, "compact single-token MFA activation should match CPU exactly");
	REQUIRE(compact_mfa.max_weight_difference <= 5e-4f, "compact single-token MFA weights should match CPU");
	REQUIRE_EQ(compact_graph.status, CCV_NNC_EXEC_SUCCESS, "compact single-token MPSGraph routing should execute");
	REQUIRE(compact_graph.metadata_match, "compact single-token MPSGraph metadata should match CPU");
	REQUIRE(compact_graph.gathered_match, "compact single-token MPSGraph activation should match CPU exactly");
	REQUIRE(compact_graph.max_weight_difference <= 5e-4f, "compact single-token MPSGraph weights should match CPU");
}

TEST_CASE("multi-token moe routing ignores the compact activation flag and matches CPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const moe_routing_mps_result_t result = _moe_routing_mps_case(7, 16, 3, 65, 0, 0, CCV_32F, CCV_NNC_MOE_ROUTING_COMPACT_SINGLE_TOKEN_ACTIVATION);
	REQUIRE_EQ(result.status, CCV_NNC_EXEC_SUCCESS, "multi-token MPSGraph routing should execute");
	REQUIRE(result.metadata_match, "multi-token grouping metadata should match CPU independent of equal-key sort order");
	REQUIRE(result.gathered_match, "multi-token gathered activations should match CPU exactly");
	REQUIRE(result.max_weight_difference <= 5e-4f, "multi-token normalized weights should match CPU");
}

#include "case_main.h"
