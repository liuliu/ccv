#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <nnc/mps/ccv_nnc_mps.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

// The CPU reference against a direct loop, then the GPU against it, in FP32, FP16 or BF16.
static void _attention_residual_case(char* __case_name__, int* __case_result__, const int rows, const int channels, const int block_count, const int query_count, const int datatype)
{
	const float epsilon = 1e-6;
	const float scale = 1.0 / sqrt((double)channels);
	const int state_count = block_count + 1;
	ccv_nnc_tensor_t* const hblocks = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, block_count, rows, channels), 0);
	ccv_nnc_tensor_t* const hpartial = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const hqueries = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, query_count, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3 + rows + 5 * block_count + 7 * query_count);
	int i, s, t, c, q;
	// States of different magnitudes, which the normalization removes.
	for (s = 0; s < block_count; s++)
		for (i = 0; i < rows * channels; i++)
			hblocks->data.f32[s * rows * channels + i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * (0.5f + 2 * s);
	for (i = 0; i < rows * channels; i++)
		hpartial->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 3;
	// Queries large enough to spread the scores of the states by a few units.
	for (i = 0; i < query_count * channels; i++)
		hqueries->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 12;
	if (datatype != CCV_32F)
	{
		// The GPU reads FP16 or BF16; the references read the same rounded values.
		ccv_nnc_tensor_t* const tensors[3] = { hblocks, hpartial, hqueries };
		for (i = 0; i < 3; i++)
		{
			const size_t count = ccv_nnc_tensor_count(tensors[i]->info);
			uint16_t* const scratch = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
			if (datatype == CCV_16F)
			{
				ccv_float_to_half_precision(tensors[i]->data.f32, scratch, count);
				ccv_half_precision_to_float(scratch, tensors[i]->data.f32, count);
			} else {
				ccv_float_to_bfloat(tensors[i]->data.f32, scratch, count);
				ccv_bfloat_to_float(scratch, tensors[i]->data.f32, count);
			}
			ccfree(scratch);
		}
	}
	float* const expected = (float*)ccmalloc(sizeof(float) * query_count * rows * channels);
	double* const reciprocal = (double*)ccmalloc(sizeof(double) * state_count * 2);
	double* const weight = reciprocal + state_count;
	double spread = 0;
#define STATE(s, t) ((s) < block_count ? hblocks->data.f32 + ((size_t)(s) * rows + (t)) * channels : hpartial->data.f32 + (size_t)(t) * channels)
	for (t = 0; t < rows; t++)
	{
		for (s = 0; s < state_count; s++)
		{
			const float* const x = STATE(s, t);
			double sum = 0;
			for (c = 0; c < channels; c++)
				sum += (double)x[c] * x[c];
			reciprocal[s] = 1 / sqrt(sum / channels + epsilon);
		}
		for (q = 0; q < query_count; q++)
		{
			double maximum = -INFINITY, minimum = INFINITY;
			for (s = 0; s < state_count; s++)
			{
				const float* const x = STATE(s, t);
				double dot = 0;
				for (c = 0; c < channels; c++)
					dot += (double)hqueries->data.f32[q * channels + c] * x[c];
				weight[s] = dot * reciprocal[s] * scale;
				maximum = ccv_max(maximum, weight[s]);
				minimum = ccv_min(minimum, weight[s]);
			}
			spread += maximum - minimum;
			double sum = 0;
			for (s = 0; s < state_count; s++)
				sum += (weight[s] = exp(weight[s] - maximum));
			float* const y = expected + ((size_t)q * rows + t) * channels;
			for (c = 0; c < channels; c++)
			{
				double value = 0;
				for (s = 0; s < state_count; s++)
					value += weight[s] / sum * reciprocal[s] * STATE(s, t)[c];
				y[c] = (float)value;
			}
		}
	}
#undef STATE
	REQUIRE(spread > 0.5 * rows * query_count, "the queries should weigh the states differently (mean score spread %g)", spread / (rows * query_count));
	const ccv_nnc_cmd_t cmd = CMD_ATTENTION_RESIDUAL_FORWARD(epsilon, scale);
	ccv_nnc_tensor_t* cpu_outputs[4];
	for (q = 0; q < query_count; q++)
		cpu_outputs[q] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hblocks, hpartial, hqueries), cpu_outputs, query_count, 0), CCV_NNC_EXEC_SUCCESS, "CPU attention residual should execute");
	for (q = 0; q < query_count; q++)
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected + (size_t)q * rows * channels, cpu_outputs[q]->data.f32, rows * channels, 1e-5, "CPU attention residual output %d should match (rows %d, channels %d, blocks %d)", q, rows, channels, block_count);
	if (ccv_nnc_cmd_ok(CCV_NNC_ATTENTION_RESIDUAL_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_t* const hosts[3] = { hblocks, hpartial, hqueries };
		ccv_nnc_tensor_t* gpu[3];
		for (i = 0; i < 3; i++)
		{
			ccv_nnc_tensor_param_t params = hosts[i]->info;
			params.datatype = datatype;
			ccv_nnc_tensor_t* const converted = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hosts[i]), TENSOR_LIST(converted), 0);
			params.type = CCV_TENSOR_GPU_MEMORY | 000;
			gpu[i] = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(converted), TENSOR_LIST(gpu[i]), 0);
			ccv_nnc_tensor_free(converted);
		}
		ccv_nnc_tensor_param_t output_params = GPU_TENSOR_NHWC(000, 32F, rows, channels);
		output_params.datatype = datatype;
		ccv_nnc_tensor_t* gpu_outputs[4];
		for (q = 0; q < query_count; q++)
			gpu_outputs[q] = ccv_nnc_tensor_new(0, output_params, 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[0], gpu[1], gpu[2]), gpu_outputs, query_count, 0), CCV_NNC_EXEC_SUCCESS, "GPU attention residual should execute (rows %d, channels %d, blocks %d, queries %d, datatype %d)", rows, channels, block_count, query_count, datatype);
		output_params.type = CCV_TENSOR_CPU_MEMORY;
		ccv_nnc_tensor_t* const hyt = ccv_nnc_tensor_new(0, output_params, 0);
		ccv_nnc_tensor_t* const hy32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
		for (q = 0; q < query_count; q++)
		{
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_outputs[q]), TENSOR_LIST(hyt), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hyt), TENSOR_LIST(hy32), 0);
			for (i = 0; i < rows * channels; i++)
				{ REQUIRE(isfinite(hy32->data.f32[i]), "GPU attention residual output %d should be finite at %d", q, i); }
			// FP16 and BF16 round the output once.
			REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected + (size_t)q * rows * channels, hy32->data.f32, rows * channels, datatype == CCV_16BF ? 2e-2 : (datatype == CCV_16F ? 2e-3 : 1e-4), "GPU attention residual output %d should match (rows %d, channels %d, blocks %d, datatype %d)", q, rows, channels, block_count, datatype);
			ccv_nnc_tensor_free(gpu_outputs[q]);
		}
		for (i = 0; i < 3; i++)
			ccv_nnc_tensor_free(gpu[i]);
		ccv_nnc_tensor_free(hyt);
		ccv_nnc_tensor_free(hy32);
	}
	for (q = 0; q < query_count; q++)
		ccv_nnc_tensor_free(cpu_outputs[q]);
	ccfree(reciprocal);
	ccfree(expected);
	ccv_nnc_tensor_free(hblocks);
	ccv_nnc_tensor_free(hpartial);
	ccv_nnc_tensor_free(hqueries);
}

TEST_CASE("attention residual mixes one decoded token's states")
{
	// From one block state to eight, with one query and with the three of a full-attention layer.
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 1, 1, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 8, 1, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 5, 3, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 8, 3, CCV_16F);
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 8, 3, CCV_16BF);
}

TEST_CASE("attention residual mixes the states of a prefill chunk")
{
	_attention_residual_case(__case_name__, __case_result__, 261, 4608, 8, 1, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 261, 4608, 3, 3, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 261, 4608, 8, 3, CCV_16F);
	// Rows that do not fill a threadgroup, and channels that do not fill the lanes.
	_attention_residual_case(__case_name__, __case_result__, 7, 100, 2, 2, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 9, 33, 15, 4, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 2, 16, 1, 1, CCV_16F);
}

TEST_CASE("attention residual mixes eight block states with three FP32 queries")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ATTENTION_RESIDUAL_FORWARD, CCV_NNC_BACKEND_MPS));
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 8, 3, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 261, 4608, 8, 3, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 7, 4609, 8, 3, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 5, 3, CCV_32F);
	_attention_residual_case(__case_name__, __case_result__, 1, 4608, 8, 3, CCV_32F);
}

TEST_CASE("attention residual rejects what the GPU kernel cannot take")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ATTENTION_RESIDUAL_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 3;
	const int channels = 64;
	const ccv_nnc_cmd_t cmd = CMD_ATTENTION_RESIDUAL_FORWARD(1e-6, 0.125);
	ccv_nnc_tensor_t* const blocks = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, rows, channels), 0);
	ccv_nnc_tensor_t* const many_blocks = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 16, rows, channels), 0);
	ccv_nnc_tensor_t* const partial = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, channels), 0);
	ccv_nnc_tensor_t* const queries = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 5, channels), 0);
	ccv_nnc_tensor_t* outputs[5];
	int i;
	for (i = 0; i < 5; i++)
		outputs[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, channels), 0);
	ccv_nnc_tensor_t* const one_query = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, channels), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(many_blocks, partial, one_query), outputs, 1, 0), CCV_NNC_EXEC_INVALID, "17 states should be rejected");
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(blocks, partial, queries), outputs, 5, 0), CCV_NNC_EXEC_INVALID, "5 queries should be rejected");
	// The kernel reads the partial state again after it writes the first output.
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(blocks, partial, one_query), TENSOR_LIST(partial), 0), CCV_NNC_EXEC_INVALID, "an output over the partial state should be rejected");
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(blocks, partial, one_query), TENSOR_LIST(outputs[0]), 0), CCV_NNC_EXEC_SUCCESS, "the same call with its own output should execute");
	ccv_nnc_tensor_free(one_query);
	for (i = 0; i < 5; i++)
		ccv_nnc_tensor_free(outputs[i]);
	ccv_nnc_tensor_free(blocks);
	ccv_nnc_tensor_free(many_blocks);
	ccv_nnc_tensor_free(partial);
	ccv_nnc_tensor_free(queries);
}

TEST_CASE("attention residual model owns its queries and returns a mixture for each")
{
	const int rows = 5;
	const int channels = 96;
	const int block_count = 3;
	const int query_count = 3;
	const float scale = 1.0 / sqrt((double)channels);
	ccv_cnnp_model_t* const model = ccv_cnnp_attention_residual(query_count, 1e-6, scale, 0, "attention_residual");
	ccv_nnc_tensor_param_t input_params[2] = {
		CPU_TENSOR_NHWC(32F, block_count, rows, channels),
		CPU_TENSOR_NHWC(32F, rows, channels),
	};
	ccv_cnnp_model_compile(model, input_params, 2, CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const blocks = ccv_nnc_tensor_new(0, input_params[0], 0);
	ccv_nnc_tensor_t* const partial = ccv_nnc_tensor_new(0, input_params[1], 0);
	ccv_nnc_tensor_t* const queries = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, query_count, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 17);
	int i, q;
	for (i = 0; i < block_count * rows * channels; i++)
		blocks->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 4;
	for (i = 0; i < rows * channels; i++)
		partial->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	for (i = 0; i < query_count * channels; i++)
		queries->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 12;
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(model, ALL_PARAMETERS, ALL_PARAMETERS), queries);
	ccv_nnc_tensor_t* outputs[3];
	ccv_nnc_tensor_t* expected[3];
	for (q = 0; q < query_count; q++)
	{
		outputs[q] = ccv_nnc_tensor_new(0, input_params[1], 0);
		expected[q] = ccv_nnc_tensor_new(0, input_params[1], 0);
	}
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(blocks, partial), outputs, query_count, 0, 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_ATTENTION_RESIDUAL_FORWARD(1e-6, scale), ccv_nnc_no_hint, 0, TENSOR_LIST(blocks, partial, queries), expected, query_count, 0), CCV_NNC_EXEC_SUCCESS, "attention residual should execute");
	for (q = 0; q < query_count; q++)
	{
		REQUIRE_TENSOR_EQ(outputs[q], expected[q], "the model's output %d should be the command's", q);
		ccv_nnc_tensor_free(outputs[q]);
		ccv_nnc_tensor_free(expected[q]);
	}
	ccv_nnc_tensor_free(blocks);
	ccv_nnc_tensor_free(partial);
	ccv_nnc_tensor_free(queries);
	ccv_cnnp_model_free(model);
}

#include "case_main.h"
