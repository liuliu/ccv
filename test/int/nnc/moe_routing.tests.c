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
		CPU_TENSOR_NHWC(16F, token_count, expert_count),
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
	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, expert_count), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, expert_count), 0);
	ccv_nnc_tensor_t* const selected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, kth), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, hidden), 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, kth, hidden), 0);
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, kth), 0);
	ccv_nnc_tensor_t* const tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	ccv_nnc_tensor_t* const experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, kth), 0);
	int i;
	float routing_values[16];
	for (i = 0; i < expert_count; i++)
	{
		routing_values[i] = 0;
		routing_values[expert_count + i] = (float)i;
	}
	ccv_float_to_half_precision(routing_values, (uint16_t*)logits->data.f16, expert_count);
	ccv_float_to_half_precision(routing_values + expert_count, (uint16_t*)bias->data.f16, expert_count);
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

TEST_CASE("moe routing single input token flag preserves one activation row on CPU")
{
	const int expert_count = 8;
	const int kth = 3;
	const int hidden = 4;
	const ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 1.5f, 0, CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN);
	ccv_nnc_tensor_param_t single_token_inputs[] = {
		CPU_TENSOR_NHWC(32F, 1, expert_count),
		CPU_TENSOR_NHWC(32F, expert_count),
		CPU_TENSOR_NHWC(32F, 1, hidden),
	};
	ccv_nnc_tensor_param_t output_params[5];
	ccv_nnc_hint_tensor_auto(cmd, single_token_inputs, 3, ccv_nnc_no_hint, output_params, 5);
	REQUIRE_EQ(output_params[0].dim[0], 1, "single-input-token routing should preserve one activation row");
	REQUIRE_EQ(output_params[0].dim[1], hidden, "single-input-token routing should preserve hidden width");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[1]), kth, "single-input-token routing should retain one weight per selected expert");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[2]), kth, "single-input-token routing should retain one token index per selected expert");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[3]), kth, "single-input-token routing should retain one expert ID per selected expert");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[4]), kth, "single-input-token routing should retain one expert count per selected expert");
	const ccv_nnc_tensor_param_t single_input_gathered_params = output_params[0];
	ccv_nnc_tensor_param_t multi_token_inputs[] = {
		CPU_TENSOR_NHWC(32F, 2, expert_count),
		CPU_TENSOR_NHWC(32F, expert_count),
		CPU_TENSOR_NHWC(32F, 2, hidden),
	};
	ccv_nnc_hint_tensor_auto(cmd, multi_token_inputs, 3, ccv_nnc_no_hint, output_params, 5);
	REQUIRE_EQ(output_params[0].dim[0], 2 * kth, "the single input token flag should retain grouped activation rows for multiple tokens");

	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, single_token_inputs[0], 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, single_token_inputs[1], 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, single_token_inputs[2], 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, single_input_gathered_params, 0);
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
		CCV_NNC_EXEC_SUCCESS, "single-input-token CPU routing should execute");
	REQUIRE_ARRAY_EQ(float, gathered->data.f32, activation->data.f32, hidden, "single-input-token routing should copy the activation only once");
	const int expected_experts[] = { 7, 6, 5 };
	const int expected_counts[] = { 1, 1, 1 };
	const int expected_tokens[] = { 0, 0, 0 };
	REQUIRE_ARRAY_EQ(int, experts->data.i32, expected_experts, kth, "single-input-token routing should preserve selected expert IDs");
	REQUIRE_ARRAY_EQ(int, counts->data.i32, expected_counts, kth, "single-input-token routing should preserve expert counts");
	REQUIRE_ARRAY_EQ(int, tokens->data.i32, expected_tokens, kth, "single-input-token routing should preserve token indices");
	for (i = 0; i < kth; i++)
		REQUIRE_EQ_WITH_TOLERANCE(weights->data.f32[i], 0.5f, 1e-6, "single-input-token routing should preserve normalized route weights");
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(experts);
	ccv_nnc_tensor_free(tokens);
	ccv_nnc_tensor_free(weights);
	ccv_nnc_tensor_free(gathered);
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(logits);
}

TEST_CASE("moe routing sigmoid scoring selects and normalizes with sigmoid on CPU")
{
	const int token_count = 2;
	const int expert_count = 16;
	const int kth = 4;
	const int hidden = 3;
	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, expert_count), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, hidden), 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count * kth, hidden), 0);
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count * kth), 0);
	ccv_nnc_tensor_t* const tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, token_count * kth), 0);
	ccv_nnc_tensor_t* const experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, ccv_min(token_count * kth, expert_count)), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, ccv_min(token_count * kth, expert_count)), 0);
	int i, j;
	for (i = 0; i < token_count * expert_count; i++)
		logits->data.f32[i] = (float)(((i * 7) % 23) - 11) * 0.37f;
	// The bias changes which experts are selected, but not their weights.
	for (i = 0; i < expert_count; i++)
		bias->data.f32[i] = (i % 3 == 0) ? 0.3f : 0;
	for (i = 0; i < token_count * hidden; i++)
		activation->data.f32[i] = (float)i;
	ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 2.5f, 0, CCV_NNC_MOE_ROUTING_SIGMOID);
	cmd.info.moe_routing.normalization_epsilon = 1e-20f;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0,
		TENSOR_LIST(logits, bias, activation), TENSOR_LIST(gathered, weights, tokens, experts, counts), 0),
		CCV_NNC_EXEC_SUCCESS, "sigmoid CPU routing should execute");
	// Independent reference: per token, pick the top-k of sigmoid + bias, then normalize the unbiased sigmoid.
	float expected_weight[2][16];
	for (i = 0; i < token_count; i++)
	{
		float score[16];
		int chosen[16] = {0};
		for (j = 0; j < expert_count; j++)
			score[j] = 1.0f / (1.0f + expf(-logits->data.f32[i * expert_count + j]));
		float sum = 0;
		int slot;
		for (slot = 0; slot < kth; slot++)
		{
			int best = -1;
			for (j = 0; j < expert_count; j++)
				if (!chosen[j] && (best < 0 || score[j] + bias->data.f32[j] > score[best] + bias->data.f32[best]))
					best = j;
			chosen[best] = 1;
			sum += score[best];
		}
		for (j = 0; j < expert_count; j++)
			expected_weight[i][j] = chosen[j] ? score[j] / (sum + 1e-20f) * 2.5f : 0;
	}
	int position = 0;
	int matched = 0;
	for (i = 0; i < ccv_min(token_count * kth, expert_count); i++)
	{
		int c;
		for (c = 0; c < counts->data.i32[i]; c++, position++)
		{
			const int token = tokens->data.i32[position];
			const int expert = experts->data.i32[i];
			REQUIRE(expected_weight[token][expert] > 0, "routed expert should be in the sigmoid top-k");
			REQUIRE_EQ_WITH_TOLERANCE(weights->data.f32[position], expected_weight[token][expert], 1e-6, "route weight should be the normalized sigmoid score");
			++matched;
		}
	}
	REQUIRE_EQ(matched, token_count * kth, "every token should route to kth experts");
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

static moe_routing_mps_result_t _moe_routing_mps_case(const int token_count, const int expert_count, const int kth, const int hidden, const int preselected, const int disable_mfa, const int routing_datatype, const int activation_datatype, const int command_flags, const float normalization_epsilon, const float logit_shift)
{
	const int pair_count = token_count * kth;
	const int group_count = ccv_min(pair_count, expert_count);
	const int single_input_token = token_count == 1 && (command_flags & CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN);
	const int gathered_rows = single_input_token ? 1 : pair_count;
	moe_routing_mps_result_t result = {
		.status = CCV_NNC_EXEC_SUCCESS,
		.metadata_match = 1,
		.gathered_match = 1,
		.max_weight_difference = 0,
	};
	ccv_nnc_tensor_param_t hroute_params = CPU_TENSOR_NHWC(32F, expert_count);
	ccv_nnc_tensor_param_t route_params = GPU_TENSOR_NHWC(000, 32F, expert_count);
	hroute_params.datatype = routing_datatype;
	route_params.datatype = routing_datatype;
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
	ccv_nnc_tensor_param_t hlogits_params = CPU_TENSOR_NHWC(32F, token_count, expert_count);
	ccv_nnc_tensor_param_t logits_params = GPU_TENSOR_NHWC(000, 32F, token_count, expert_count);
	hlogits_params.datatype = routing_datatype;
	logits_params.datatype = routing_datatype;
	ccv_nnc_tensor_t* const hlogits = ccv_nnc_tensor_new(0, hlogits_params, 0);
	ccv_nnc_tensor_t* const hroute = ccv_nnc_tensor_new(0, hroute_params, 0);
	ccv_nnc_tensor_t* const hactivation = ccv_nnc_tensor_new(0, hactivation_params, 0);
	ccv_nnc_tensor_t* const expected_gathered = ccv_nnc_tensor_new(0, hgathered_params, 0);
	ccv_nnc_tensor_t* const expected_weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, pair_count), 0);
	ccv_nnc_tensor_t* const expected_tokens = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, pair_count), 0);
	ccv_nnc_tensor_t* const expected_experts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
	ccv_nnc_tensor_t* const expected_counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, logits_params, 0);
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
	const int logits_count = token_count * expert_count;
	float* const logits_values = (float*)malloc(sizeof(float) * logits_count);
	int i;
	for (i = 0; i < token_count; i++)
	{
		int expert;
		for (expert = 0; expert < expert_count; expert++)
		{
			const int rank = (expert * 37 + i * 53) % expert_count;
			logits_values[i * expert_count + expert] = (float)(rank - expert_count / 2) * 0.03125f + logit_shift;
		}
	}
	if (routing_datatype == CCV_32F)
		memcpy(hlogits->data.f32, logits_values, sizeof(float) * logits_count);
	else
		ccv_float_to_half_precision(logits_values, (uint16_t*)hlogits->data.f16, logits_count);
	free(logits_values);
	if (preselected)
	{
		for (i = 0; i < token_count; i++)
		{
			int slot;
			for (slot = 0; slot < kth; slot++)
				hroute->data.i32[i * kth + slot] = (i * 29 + slot * 43 + 7) % expert_count;
		}
	} else {
		float* const bias_values = (float*)malloc(sizeof(float) * expert_count);
		for (i = 0; i < expert_count; i++)
			bias_values[i] = (float)i * 0.00001f;
		if (routing_datatype == CCV_32F)
			memcpy(hroute->data.f32, bias_values, sizeof(float) * expert_count);
		else
			ccv_float_to_half_precision(bias_values, (uint16_t*)hroute->data.f16, expert_count);
		free(bias_values);
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
	cpu_cmd.info.moe_routing.normalization_epsilon = normalization_epsilon;
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
	mps_cmd.info.moe_routing.normalization_epsilon = normalization_epsilon;
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
		if (single_input_token && memcmp(expected_gathered->data.u8, actual_gathered->data.u8, row_size) != 0)
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
				result.max_weight_difference = isfinite(difference) ? ccv_max(result.max_weight_difference, difference) : INFINITY;
				if (!single_input_token && memcmp(expected_gathered->data.u8 + (size_t)expected_records[i].index * row_size,
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
	const moe_routing_mps_result_t standard_mfa = _moe_routing_mps_case(1, 256, 6, 4096, 0, 0, CCV_16F, CCV_16F, 0, 0, 0);
	const moe_routing_mps_result_t standard_graph = _moe_routing_mps_case(1, 256, 6, 4096, 0, 1, CCV_16F, CCV_16F, 0, 0, 0);
	const moe_routing_mps_result_t selected_mfa = _moe_routing_mps_case(1, 256, 6, 1024, 1, 0, CCV_16F, CCV_16BF, 0, 0, 0);
	const moe_routing_mps_result_t selected_graph = _moe_routing_mps_case(1, 256, 6, 1024, 1, 1, CCV_16F, CCV_16BF, 0, 0, 0);
	const moe_routing_mps_result_t single_input_mfa = _moe_routing_mps_case(1, 256, 6, 4096, 0, 0, CCV_32F, CCV_16F, CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN, 0, 0);
	const moe_routing_mps_result_t single_input_graph = _moe_routing_mps_case(1, 256, 6, 4096, 0, 1, CCV_32F, CCV_16F, CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN, 0, 0);
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
	REQUIRE_EQ(single_input_mfa.status, CCV_NNC_EXEC_SUCCESS, "single-input-token MFA routing should execute");
	REQUIRE(single_input_mfa.metadata_match, "single-input-token MFA metadata should match CPU");
	REQUIRE(single_input_mfa.gathered_match, "single-input-token MFA activation should match CPU exactly");
	REQUIRE(single_input_mfa.max_weight_difference <= 5e-4f, "single-input-token MFA weights should match CPU");
	REQUIRE_EQ(single_input_graph.status, CCV_NNC_EXEC_SUCCESS, "single-input-token MPSGraph routing should execute");
	REQUIRE(single_input_graph.metadata_match, "single-input-token MPSGraph metadata should match CPU");
	REQUIRE(single_input_graph.gathered_match, "single-input-token MPSGraph activation should match CPU exactly");
	REQUIRE(single_input_graph.max_weight_difference <= 5e-4f, "single-input-token MPSGraph weights should match CPU");
}

TEST_CASE("moe routing sigmoid scoring MFA and MPSGraph paths match CPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const int flags = CCV_NNC_MOE_ROUTING_SIGMOID;
	const moe_routing_mps_result_t single_mfa = _moe_routing_mps_case(1, 288, 8, 4096, 0, 0, CCV_32F, CCV_16BF, flags | CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN, 1e-20f, 0);
	// Same MFA shape without the flag, so the cached sigmoid pipeline must not be reused for sqrt(softplus) scoring.
	const moe_routing_mps_result_t softplus_mfa = _moe_routing_mps_case(1, 288, 8, 4096, 0, 0, CCV_32F, CCV_16BF, CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN, 1e-20f, 0);
	const moe_routing_mps_result_t single_graph = _moe_routing_mps_case(1, 288, 8, 4096, 0, 1, CCV_32F, CCV_16BF, flags | CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN, 1e-20f, 0);
	// A few tokens take the MFA kernel unless it is disabled.
	const moe_routing_mps_result_t multi_mfa = _moe_routing_mps_case(7, 288, 8, 65, 0, 0, CCV_32F, CCV_32F, flags, 1e-20f, 0);
	const moe_routing_mps_result_t multi_graph = _moe_routing_mps_case(7, 288, 8, 65, 0, 1, CCV_32F, CCV_32F, flags, 1e-20f, 0);
	const moe_routing_mps_result_t shifted_mfa = _moe_routing_mps_case(1, 288, 8, 128, 0, 0, CCV_16F, CCV_16F, flags, 0, -4);
	// Large logits saturate sigmoid, so FP16 scores would collapse and change the selected experts without FP32 promotion.
	const moe_routing_mps_result_t saturated_mfa = _moe_routing_mps_case(2, 288, 8, 128, 0, 0, CCV_16F, CCV_16F, flags, 0, 4);
	const moe_routing_mps_result_t saturated_graph = _moe_routing_mps_case(2, 288, 8, 128, 0, 1, CCV_16F, CCV_16F, flags, 0, 4);
	REQUIRE_EQ(single_mfa.status, CCV_NNC_EXEC_SUCCESS, "sigmoid single-token MFA routing should execute");
	REQUIRE(single_mfa.metadata_match, "sigmoid single-token MFA metadata should match CPU");
	REQUIRE(single_mfa.gathered_match, "sigmoid single-token MFA activation should match CPU exactly");
	REQUIRE(single_mfa.max_weight_difference <= 5e-4f, "sigmoid single-token MFA weights should match CPU");
	REQUIRE_EQ(softplus_mfa.status, CCV_NNC_EXEC_SUCCESS, "sqrt(softplus) single-token MFA routing should execute after sigmoid");
	REQUIRE(softplus_mfa.metadata_match, "sqrt(softplus) single-token MFA metadata should match CPU after sigmoid");
	REQUIRE(softplus_mfa.max_weight_difference <= 5e-4f, "sqrt(softplus) single-token MFA weights should match CPU after sigmoid");
	REQUIRE_EQ(single_graph.status, CCV_NNC_EXEC_SUCCESS, "sigmoid single-token MPSGraph routing should execute");
	REQUIRE(single_graph.metadata_match, "sigmoid single-token MPSGraph metadata should match CPU");
	REQUIRE(single_graph.gathered_match, "sigmoid single-token MPSGraph activation should match CPU exactly");
	REQUIRE(single_graph.max_weight_difference <= 5e-4f, "sigmoid single-token MPSGraph weights should match CPU");
	REQUIRE_EQ(multi_mfa.status, CCV_NNC_EXEC_SUCCESS, "sigmoid multi-token MFA routing should execute");
	REQUIRE(multi_mfa.metadata_match, "sigmoid multi-token MFA metadata should match CPU");
	REQUIRE(multi_mfa.gathered_match, "sigmoid multi-token MFA gathered activations should match CPU exactly");
	REQUIRE(multi_mfa.max_weight_difference <= 5e-4f, "sigmoid multi-token MFA weights should match CPU");
	REQUIRE_EQ(multi_graph.status, CCV_NNC_EXEC_SUCCESS, "sigmoid multi-token MPSGraph routing should execute");
	REQUIRE(multi_graph.metadata_match, "sigmoid multi-token metadata should match CPU");
	REQUIRE(multi_graph.gathered_match, "sigmoid multi-token gathered activations should match CPU exactly");
	REQUIRE(multi_graph.max_weight_difference <= 5e-4f, "sigmoid multi-token weights should match CPU");
	REQUIRE_EQ(shifted_mfa.status, CCV_NNC_EXEC_SUCCESS, "sigmoid half-precision MFA routing should execute");
	REQUIRE(shifted_mfa.metadata_match, "sigmoid half-precision MFA metadata should match CPU");
	REQUIRE(shifted_mfa.max_weight_difference <= 5e-4f, "sigmoid half-precision MFA weights should match CPU");
	REQUIRE_EQ(saturated_mfa.status, CCV_NNC_EXEC_SUCCESS, "sigmoid half-precision multi-token MFA routing should execute");
	REQUIRE(saturated_mfa.metadata_match, "sigmoid half-precision multi-token MFA metadata should match CPU with saturated scores");
	REQUIRE(saturated_mfa.gathered_match, "sigmoid half-precision multi-token MFA gathered activations should match CPU exactly");
	REQUIRE(saturated_mfa.max_weight_difference <= 5e-4f, "sigmoid half-precision multi-token MFA weights should match CPU with saturated scores");
	REQUIRE_EQ(saturated_graph.status, CCV_NNC_EXEC_SUCCESS, "sigmoid half-precision MPSGraph routing should execute");
	REQUIRE(saturated_graph.metadata_match, "sigmoid half-precision MPSGraph metadata should match CPU with saturated scores");
	REQUIRE(saturated_graph.gathered_match, "sigmoid half-precision MPSGraph gathered activations should match CPU exactly");
	REQUIRE(saturated_graph.max_weight_difference <= 5e-4f, "sigmoid half-precision MPSGraph weights should match CPU with saturated scores");
}

TEST_CASE("multi-token moe routing ignores the single input token flag and matches CPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const moe_routing_mps_result_t result = _moe_routing_mps_case(7, 16, 3, 65, 0, 0, CCV_16F, CCV_32F, CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN, 0, 0);
	REQUIRE_EQ(result.status, CCV_NNC_EXEC_SUCCESS, "multi-token routing should execute");
	REQUIRE(result.metadata_match, "multi-token grouping metadata should match CPU independent of equal-key sort order");
	REQUIRE(result.gathered_match, "multi-token gathered activations should match CPU exactly");
	REQUIRE(result.max_weight_difference <= 5e-4f, "multi-token normalized weights should match CPU");
}

TEST_CASE("single-token moe routing supports 384 experts and the 512-expert MFA boundary")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	// Alternate launch sizes while reusing the same kernel cache, including the
	// first size above the MFA limit to retain coverage of the graph fallback.
	const int expert_counts[] = {32, 255, 256, 257, 287, 288, 289, 383, 384, 385, 415, 416, 417, 511, 512, 256, 384, 513};
	int i, mode;
	for (i = 0; i < sizeof(expert_counts) / sizeof(expert_counts[0]); i++)
		for (mode = 0; mode < 5; mode++)
		{
			// Exercise top-32 selection as well as the preselected top-32 path.
			const moe_routing_mps_result_t result = _moe_routing_mps_case(1, expert_counts[i], mode == 2 || mode == 4 ? 32 : 6, 5120,
				mode == 2, mode == 3, mode == 0 ? CCV_16F : CCV_32F, mode == 0 ? CCV_16F : CCV_16BF,
				mode == 1 ? CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN : 0, 0, 0);
			REQUIRE_EQ(result.status, CCV_NNC_EXEC_SUCCESS, "routing should execute across the MFA expert-count boundary");
			REQUIRE(result.metadata_match, "selected experts and grouping metadata should match CPU");
			REQUIRE(result.gathered_match, "gathered activations should match CPU exactly");
			REQUIRE(result.max_weight_difference <= 5e-4f, "normalized weights should match CPU");
		}
}

TEST_CASE("few-token moe routing MFA kernel matches CPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	// A speculative verify routes a handful of tokens; they take the MFA kernel, not MPSGraph.
	// 16 tokens at top-8 fill the kernel's 128 (expert, token, slot) pairs.
	const int token_counts[] = {2, 3, 6, 16};
	const int kths[] = {6, 8};
	int i, j, preselected;
	for (i = 0; i < sizeof(token_counts) / sizeof(token_counts[0]); i++)
		for (j = 0; j < sizeof(kths) / sizeof(kths[0]); j++)
			for (preselected = 0; preselected < 2; preselected++)
			{
				const moe_routing_mps_result_t result = _moe_routing_mps_case(token_counts[i], 384, kths[j], 5120, preselected, 0, CCV_16F, CCV_32F, 0, 1e-20f, 0);
				REQUIRE_EQ(result.status, CCV_NNC_EXEC_SUCCESS, "few-token MFA routing should execute");
				REQUIRE(result.metadata_match, "few-token MFA grouping should match CPU for tokens=%d kth=%d preselected=%d", token_counts[i], kths[j], preselected);
				REQUIRE(result.gathered_match, "few-token MFA gathered rows should match CPU exactly");
				REQUIRE(result.max_weight_difference <= 5e-4f, "few-token MFA weights should match CPU");
			}
}

TEST_CASE("moe routing additive epsilon preserves legacy normalization and rejects invalid values on CPU")
{
	const int expert_count = 384;
	const int kth = 6;
	ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD(kth, 1.5f, 0);
	ccv_nnc_tensor_param_t input_params[] = {
		CPU_TENSOR_NHWC(32F, 1, expert_count),
		CPU_TENSOR_NHWC(32F, expert_count),
		CPU_TENSOR_NHWC(32F, 1, 65),
	};
	ccv_nnc_tensor_param_t output_params[5];
	ccv_nnc_hint_tensor_auto(cmd, input_params, 3, ccv_nnc_no_hint, output_params, 5);
	ccv_nnc_tensor_t* inputs[3];
	ccv_nnc_tensor_t* outputs[5];
	int i, step;
	for (i = 0; i < 3; i++)
		inputs[i] = ccv_nnc_tensor_new(0, input_params[i], 0);
	for (i = 0; i < 5; i++)
		outputs[i] = ccv_nnc_tensor_new(0, output_params[i], 0);
	for (i = 0; i < expert_count; i++)
	{
		inputs[0]->data.f32[i] = -30;
		inputs[1]->data.f32[i] = (float)i;
	}
	for (i = 0; i < 65; i++)
		inputs[2]->data.f32[i] = (float)i / 65;
	const float probability = _moe_routing_probability(-30);
	const float epsilons[] = {0, 1e-20f, kth * probability, 0};
	const float expected_weights[] = {probability * 1.5f / 6.103515625e-5f, 0.25f, 0.125f, probability * 1.5f / 6.103515625e-5f};
	for (step = 0; step < sizeof(epsilons) / sizeof(epsilons[0]); step++)
	{
		cmd.info.moe_routing.normalization_epsilon = epsilons[step];
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 3, outputs, 5, 0), CCV_NNC_EXEC_SUCCESS, "CPU normalization should execute");
		for (i = 0; i < kth; i++)
		{
			REQUIRE_EQ(outputs[3]->data.i32[i], expert_count - i - 1, "epsilon should preserve bias-based selection");
			REQUIRE_EQ_WITH_TOLERANCE(outputs[1]->data.f32[i], expected_weights[step], 1e-6, "equal unbiased scores should use the requested normalization");
		}
	}
	for (i = 0; i < expert_count; i++)
		inputs[0]->data.f32[i] = -INFINITY;
	cmd.info.moe_routing.normalization_epsilon = 1e-20f;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 3, outputs, 5, 0), CCV_NNC_EXEC_SUCCESS, "zero scores should execute with additive epsilon");
	for (i = 0; i < kth; i++)
		REQUIRE_EQ(outputs[1]->data.f32[i], 0, "zero scores should have finite zero weight");
	const float invalid_epsilons[] = {-1, NAN, INFINITY};
	for (step = 0; step < sizeof(invalid_epsilons) / sizeof(invalid_epsilons[0]); step++)
	{
		cmd.info.moe_routing.normalization_epsilon = invalid_epsilons[step];
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 3, outputs, 5, 0), CCV_NNC_EXEC_INVALID, "CPU should reject negative and nonfinite epsilon");
	}
	for (i = 0; i < 5; i++)
		ccv_nnc_tensor_free(outputs[i]);
	for (i = 0; i < 3; i++)
		ccv_nnc_tensor_free(inputs[i]);
}

TEST_CASE("moe routing MFA additive epsilon matches CPU across score magnitudes and cache changes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const float epsilons[] = {0, 1e-20f, 1e-5f, 0.25f, 0};
	const float shifts[] = {0, -30, -INFINITY};
	// One token, and a few tokens as a speculative verify routes them.
	const int token_counts[] = {1, 6};
	int tokens, dtype, preselected, magnitude, step;
	for (tokens = 0; tokens < sizeof(token_counts) / sizeof(token_counts[0]); tokens++)
		for (dtype = 0; dtype < 2; dtype++)
			for (preselected = 0; preselected < 2; preselected++)
				for (magnitude = 0; magnitude < sizeof(shifts) / sizeof(shifts[0]); magnitude++)
					for (step = 0; step < sizeof(epsilons) / sizeof(epsilons[0]); step++)
					{
						const moe_routing_mps_result_t result = _moe_routing_mps_case(token_counts[tokens], 384, 6, 5120, preselected, 0,
							dtype ? CCV_16F : CCV_32F, dtype ? CCV_16BF : CCV_32F,
							preselected ? CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN : 0, epsilons[step], shifts[magnitude]);
						REQUIRE_EQ(result.status, CCV_NNC_EXEC_SUCCESS, "MFA additive normalization should execute");
						REQUIRE(result.metadata_match, "epsilon should preserve selected IDs and routing metadata");
						REQUIRE(result.gathered_match, "gathered activations should match CPU exactly");
						REQUIRE(result.max_weight_difference <= 2e-6f, "MFA weights should match CPU for tokens=%d dtype=%d preselected=%d magnitude=%d epsilon=%g difference=%g", token_counts[tokens], dtype, preselected, magnitude, epsilons[step], result.max_weight_difference);
					}
}

TEST_CASE("moe routing MPSGraph additive epsilon matches CPU across fallback conditions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const float epsilons[] = {1e-20f, 1e-5f, 0.25f, 1e-20f};
	const float shifts[] = {0, -30, -INFINITY};
	const int activation_datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	int mode, dtype, preselected, magnitude, step;
	for (mode = 0; mode < 4; mode++)
		for (dtype = 0; dtype < 2; dtype++)
			for (preselected = 0; preselected < 2; preselected++)
				for (magnitude = 0; magnitude < sizeof(shifts) / sizeof(shifts[0]); magnitude++)
					for (step = 0; step < sizeof(epsilons) / sizeof(epsilons[0]); step++)
					{
						// Force the graph, or reach it through token-count (past the MFA kernel's 16) /
						// expert-count / top-k limits.
						const moe_routing_mps_result_t result = _moe_routing_mps_case(mode == 1 ? 17 : 1, mode == 2 ? 513 : 384,
							mode == 3 ? 33 : 6, 1024, preselected, mode == 0, dtype ? CCV_16F : CCV_32F,
							activation_datatypes[magnitude], CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN, epsilons[step], shifts[magnitude]);
						REQUIRE_EQ(result.status, CCV_NNC_EXEC_SUCCESS, "MPSGraph additive normalization should execute");
						REQUIRE(result.metadata_match, "selected IDs and grouping metadata should match CPU");
						REQUIRE(result.gathered_match, "gathered activations should match CPU exactly");
						REQUIRE(result.max_weight_difference <= 2e-6f, "MPSGraph weights should match CPU for mode=%d dtype=%d preselected=%d magnitude=%d epsilon=%g difference=%g", mode, dtype, preselected, magnitude, epsilons[step], result.max_weight_difference);
					}
}

TEST_CASE("moe routing MPS rejects invalid epsilon")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD(6, 1.5f, 0);
	ccv_nnc_tensor_param_t input_params[] = {
		GPU_TENSOR_NHWC(000, 32F, 1, 384),
		GPU_TENSOR_NHWC(000, 32F, 384),
		GPU_TENSOR_NHWC(000, 32F, 1, 65),
	};
	ccv_nnc_tensor_param_t output_params[5];
	ccv_nnc_hint_tensor_auto(cmd, input_params, 3, ccv_nnc_no_hint, output_params, 5);
	ccv_nnc_tensor_t* inputs[3];
	ccv_nnc_tensor_t* outputs[5];
	int i, step;
	for (i = 0; i < 3; i++)
		inputs[i] = ccv_nnc_tensor_new(0, input_params[i], 0);
	for (i = 0; i < 5; i++)
		outputs[i] = ccv_nnc_tensor_new(0, output_params[i], 0);
	const float invalid_epsilons[] = {-1, NAN, INFINITY};
	for (step = 0; step < sizeof(invalid_epsilons) / sizeof(invalid_epsilons[0]); step++)
	{
		cmd.info.moe_routing.normalization_epsilon = invalid_epsilons[step];
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 3, outputs, 5, 0), CCV_NNC_EXEC_INVALID, "MPS should reject negative and nonfinite epsilon before dispatch");
	}
	for (i = 0; i < 5; i++)
		ccv_nnc_tensor_free(outputs[i]);
	for (i = 0; i < 3; i++)
		ccv_nnc_tensor_free(inputs[i]);
}

TEST_CASE("multi-token moe routing gathers prefill rows and partial widths exactly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const struct {
		int tokens, hidden, datatype, preselected;
	} cases[] = {
		{4096, 5120, CCV_16F, 0},
		{129, 513, CCV_16BF, 1},
		{257, 31, CCV_32F, 0},
		{193, 5120, CCV_16F, 1},
	};
	int i;
	for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
	{
		const moe_routing_mps_result_t result = _moe_routing_mps_case(cases[i].tokens, 384, 6, cases[i].hidden,
			cases[i].preselected, 0, CCV_32F, cases[i].datatype, 0, 1e-6f, 0);
		REQUIRE_EQ(result.status, CCV_NNC_EXEC_SUCCESS, "prefill row gathering should execute");
		REQUIRE(result.metadata_match, "prefill expert and token grouping should match CPU");
		REQUIRE(result.gathered_match, "prefill gathered activation rows should match CPU exactly");
		REQUIRE(result.max_weight_difference <= 5e-4f, "prefill route weights should match CPU");
	}
}

static int _moe_routing_mixed_bias_case(const int token_count, const int disable_mfa)
{
	// FP16 logits with an FP32 bias must route exactly as the same logits widened to FP32.
	const int expert_count = 384, kth = 6, hidden = 256;
	const int pair_count = token_count * kth;
	const int group_count = ccv_min(expert_count, pair_count);
	ccv_nnc_tensor_t* const hlogits16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const hlogits32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, expert_count), 0);
	ccv_nnc_tensor_t* const hactivation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, hidden), 0);
	int i;
	for (i = 0; i < token_count * expert_count; i++)
		hlogits32->data.f32[i] = (float)(((i * 37) % expert_count) - expert_count / 2) * 0.03125f;
	ccv_float_to_half_precision(hlogits32->data.f32, (uint16_t*)hlogits16->data.f16, token_count * expert_count);
	ccv_half_precision_to_float((uint16_t*)hlogits16->data.f16, hlogits32->data.f32, token_count * expert_count);
	// Offsets finer than FP16 resolution near 0.3 decide ties, so a rounded bias would route differently.
	for (i = 0; i < expert_count; i++)
		hbias->data.f32[i] = 0.3f + (float)((i * 11) % expert_count) * 1e-6f;
	for (i = 0; i < token_count * hidden; i++)
		hactivation->data.f32[i] = (float)((i % 29) - 14) / 16.0f;
	ccv_nnc_tensor_t* const logits16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const logits32 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, token_count, expert_count), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, expert_count), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, token_count, hidden), 0);
	ccv_nnc_tensor_t* outputs[2][5];
	ccv_nnc_tensor_t* host[2][5];
	int j;
	for (j = 0; j < 2; j++)
	{
		// A single input token gathers its one activation row.
		const int gathered_rows = token_count == 1 ? 1 : pair_count;
		outputs[j][0] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, gathered_rows, hidden), 0);
		outputs[j][1] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, pair_count), 0);
		outputs[j][2] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, pair_count), 0);
		outputs[j][3] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, group_count), 0);
		outputs[j][4] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, group_count), 0);
		host[j][0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, gathered_rows, hidden), 0);
		host[j][1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, pair_count), 0);
		host[j][2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, pair_count), 0);
		host[j][3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
		host[j][4] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, group_count), 0);
	}
	const uint64_t old_flags = ccv_nnc_flags();
	if (disable_mfa)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hlogits16, hlogits32, hbias, hactivation), TENSOR_LIST(logits16, logits32, bias, activation), 0);
	ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 1.5f, 0, token_count == 1 ? CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN : 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	cmd.info.moe_routing.normalization_epsilon = 1e-20f;
	int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(logits16, bias, activation), outputs[0], 5, 0);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(logits32, bias, activation), outputs[1], 5, 0);
	if (!(old_flags & CCV_NNC_DISABLE_MFA))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	int match = status == CCV_NNC_EXEC_SUCCESS;
	for (j = 0; j < 2 && match; j++)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, outputs[j], 5, host[j], 5, 0);
	for (i = 0; i < 5 && match; i++)
		match = memcmp(host[0][i]->data.u8, host[1][i]->data.u8, ccv_nnc_tensor_data_size_without_padding(host[0][i]->info)) == 0;
	for (j = 0; j < 2; j++)
		for (i = 0; i < 5; i++)
		{
			ccv_nnc_tensor_free(outputs[j][i]);
			ccv_nnc_tensor_free(host[j][i]);
		}
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(logits32);
	ccv_nnc_tensor_free(logits16);
	ccv_nnc_tensor_free(hactivation);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hlogits32);
	ccv_nnc_tensor_free(hlogits16);
	return status != CCV_NNC_EXEC_SUCCESS ? -1 : match;
}

TEST_CASE("moe routing with FP16 logits and FP32 bias matches FP32 logits exactly on MPS")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	REQUIRE_EQ(_moe_routing_mixed_bias_case(1, 0), 1, "single-token MFA routing should match");
	REQUIRE_EQ(_moe_routing_mixed_bias_case(1, 1), 1, "single-token MPSGraph routing should match");
	REQUIRE_EQ(_moe_routing_mixed_bias_case(7, 0), 1, "multi-token MPSGraph routing should match");
}

TEST_CASE("moe routing infers outputs for FP16 logits with an FP32 bias")
{
	// Execution accepts a bias wider than the logits, so shape inference (as in model graphs) must too.
	const ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(2, 1.5f, 0, CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN);
	ccv_nnc_tensor_param_t input_params[] = {
		CPU_TENSOR_NHWC(16F, 1, 8),
		CPU_TENSOR_NHWC(32F, 8),
		CPU_TENSOR_NHWC(32F, 1, 5),
	};
	ccv_nnc_tensor_param_t output_params[5];
	ccv_nnc_hint_tensor_auto(cmd, input_params, 3, ccv_nnc_no_hint, output_params, 5);
	REQUIRE_EQ(output_params[0].datatype, CCV_32F, "gathered activations should preserve the activation datatype");
	REQUIRE_EQ(output_params[0].dim[0], 1, "a single input token should gather one row");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_params[1]), 2, "weights should contain one value per route");
	REQUIRE_EQ(output_params[3].dim[0], 2, "groups should be bounded by the routes");
}

static int _moe_routing_infinite_bias_case(const int expert_count, const int kth, const int finite_count, const int* const finite_experts)
{
	// Experts with a -INFINITY bias stay selectable: once the finite experts are taken, the
	// remaining slots take the lowest remaining indices, as on the CPU.
	const int hidden = 64;
	ccv_nnc_tensor_t* const hlogits = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, expert_count), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, expert_count), 0);
	ccv_nnc_tensor_t* const hactivation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, hidden), 0);
	int i, j;
	for (i = 0; i < expert_count; i++)
	{
		hlogits->data.f32[i] = (float)((i * 7) % 11 - 5) * 0.125f;
		hbias->data.f32[i] = -INFINITY;
	}
	for (i = 0; i < finite_count; i++)
		hbias->data.f32[finite_experts[i]] = (float)(finite_count - i);
	for (i = 0; i < hidden; i++)
		hactivation->data.f32[i] = (float)((i % 13) - 6) / 8.0f;
	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, expert_count), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, expert_count), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, hidden), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hlogits, hbias, hactivation), TENSOR_LIST(logits, bias, activation), 0);
	ccv_nnc_tensor_t* expected[5];
	ccv_nnc_tensor_t* outputs[5];
	ccv_nnc_tensor_t* actual[5];
	const ccv_nnc_tensor_param_t params[5] = {
		CPU_TENSOR_NHWC(32F, 1, hidden), CPU_TENSOR_NHWC(32F, kth), CPU_TENSOR_NHWC(32S, kth), CPU_TENSOR_NHWC(32S, kth), CPU_TENSOR_NHWC(32S, kth),
	};
	for (i = 0; i < 5; i++)
	{
		expected[i] = ccv_nnc_tensor_new(0, params[i], 0);
		actual[i] = ccv_nnc_tensor_new(0, params[i], 0);
		ccv_nnc_tensor_param_t gpu_params = params[i];
		gpu_params.type = CCV_TENSOR_GPU_MEMORY;
		outputs[i] = ccv_nnc_tensor_new(0, gpu_params, 0);
	}
	ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 1.5f, 0, CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN);
	cmd.info.moe_routing.normalization_epsilon = 1e-20f;
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hlogits, hbias, hactivation), expected, 5, 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(logits, bias, activation), outputs, 5, 0);
	int match = status == CCV_NNC_EXEC_SUCCESS;
	if (match)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, outputs, 5, actual, 5, 0);
		for (i = 0; i < kth && match; i++)
		{
			match = actual[3]->data.i32[i] == expected[3]->data.i32[i] && actual[4]->data.i32[i] == 1 &&
				fabsf(actual[1]->data.f32[i] - expected[1]->data.f32[i]) <= 1e-5f * ccv_max(1.0f, fabsf(expected[1]->data.f32[i]));
			for (j = 0; j < i && match; j++)
				match = actual[3]->data.i32[j] != actual[3]->data.i32[i];
		}
	}
	for (i = 0; i < 5; i++)
	{
		ccv_nnc_tensor_free(outputs[i]);
		ccv_nnc_tensor_free(actual[i]);
		ccv_nnc_tensor_free(expected[i]);
	}
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(logits);
	ccv_nnc_tensor_free(hactivation);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hlogits);
	return status != CCV_NNC_EXEC_SUCCESS ? -1 : match;
}

TEST_CASE("single-token moe routing selects distinct experts when biases are -INFINITY on MPS")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS));
	const int finite_experts[] = { 40, 3 };
	REQUIRE_EQ(_moe_routing_infinite_bias_case(32, 1, 0, 0), 1, "one slot over all -INFINITY biases should take expert 0");
	REQUIRE_EQ(_moe_routing_infinite_bias_case(32, 6, 0, 0), 1, "six slots over all -INFINITY biases should take experts 0 to 5");
	REQUIRE_EQ(_moe_routing_infinite_bias_case(64, 6, 2, finite_experts), 1, "slots past the finite experts should take the lowest remaining experts");
	REQUIRE_EQ(_moe_routing_infinite_bias_case(384, 6, 1, finite_experts + 1), 1, "-INFINITY experts across many lanes should stay selectable");
}

#include "case_main.h"
