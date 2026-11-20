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

enum {
	SINKS_NONE,
	SINKS_PER_HEAD,
	SINKS_PER_QUERY,
};

// Causal attention of B batches of R queries over C keys with a relative-position bias
// [B, R, Hq, extent], or none when extent is 0, and sinks: a direct loop, then the CPU reference
// against it, then the GPU with and without neural accelerators.
static void _relative_bias_attention_case(char* __case_name__, int* __case_result__, const int B, const int R, const int C, const int Hq, const int Hk, const int D, const int extent, const int sliding_window, const int sinks_mode, const int datatype)
{
	const float scale = 1.0 / sqrt((double)D);
	const int sink_count = sinks_mode == SINKS_PER_QUERY ? B * R * Hq : Hq;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, extent > 0 ? extent : 1), 0);
	ccv_nnc_tensor_t* const hsinks = sinks_mode == SINKS_PER_QUERY ? ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq), 0) : ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, Hq, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 11 + R + 3 * C + extent);
	int i;
	for (i = 0; i < B * R * Hq * D; i++)
		hq->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 4;
	for (i = 0; i < B * C * Hk * D; i++)
		hk->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 4;
	for (i = 0; i < B * C * Hk * D; i++)
		hv->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 2;
	// Biases as large as the scores, so a misplaced one moves the output.
	for (i = 0; i < B * R * Hq * (extent > 0 ? extent : 1); i++)
		hbias->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 6;
	for (i = 0; i < sink_count; i++)
		hsinks->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 4;
	if (datatype != CCV_32F)
	{
		// The GPU reads FP16 or BF16; the references read the same rounded values.
		ccv_nnc_tensor_t* const tensors[5] = { hq, hk, hv, hbias, hsinks };
		for (i = 0; i < 5; i++)
		{
			const size_t count = ccv_nnc_tensor_count(tensors[i]->info);
			uint16_t* const scratch = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
			if (datatype == CCV_16BF)
			{
				ccv_float_to_bfloat(tensors[i]->data.f32, scratch, count);
				ccv_bfloat_to_float(scratch, tensors[i]->data.f32, count);
			} else {
				ccv_float_to_half_precision(tensors[i]->data.f32, scratch, count);
				ccv_half_precision_to_float(scratch, tensors[i]->data.f32, count);
			}
			ccfree(scratch);
		}
	}
	float* const expected = (float*)ccmalloc(sizeof(float) * B * R * Hq * D);
	double* const scores = (double*)ccmalloc(sizeof(double) * C);
	int b, r, h, c, d;
	for (b = 0; b < B; b++)
	for (r = 0; r < R; r++)
		for (h = 0; h < Hq; h++)
		{
			const float* const keys = hk->data.f32 + (size_t)b * C * Hk * D;
			const float* const values = hv->data.f32 + (size_t)b * C * Hk * D;
			const float* const bias = hbias->data.f32 + ((size_t)(b * R + r) * Hq + h) * extent;
			const double sink = sinks_mode == SINKS_NONE ? 0 : (double)hsinks->data.f32[sinks_mode == SINKS_PER_QUERY ? (b * R + r) * Hq + h : h];
			const float* const q = hq->data.f32 + ((size_t)(b * R + r) * Hq + h) * D;
			// The query's own key.
			const int own = r + C - R;
			const int first = sliding_window > 0 ? ccv_max(own - sliding_window + 1, 0) : 0;
			double maximum = sinks_mode == SINKS_NONE ? -INFINITY : sink;
			for (c = first; c <= own; c++)
			{
				const float* const k = keys + ((size_t)c * Hk + h / (Hq / Hk)) * D;
				double score = 0;
				for (d = 0; d < D; d++)
					score += (double)q[d] * k[d];
				score *= scale;
				if (own - c < extent)
					score += bias[own - c];
				scores[c] = score;
				maximum = ccv_max(maximum, score);
			}
			double sum = sinks_mode == SINKS_NONE ? 0 : exp(sink - maximum);
			for (c = first; c <= own; c++)
				sum += (scores[c] = exp(scores[c] - maximum));
			float* const o = expected + ((size_t)(b * R + r) * Hq + h) * D;
			for (d = 0; d < D; d++)
			{
				double value = 0;
				for (c = first; c <= own; c++)
					value += scores[c] * values[((size_t)c * Hk + h / (Hq / Hk)) * D + d];
				// A row that sees no key and has no sink is zeros, as in the CPU reference.
				o[d] = sum > 0 ? (float)(value / sum) : 0;
			}
		}
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 1);
	cmd.info.scaled_dot_product_attention.sliding_window = sliding_window;
	cmd.info.scaled_dot_product_attention.attention_sinks = sinks_mode != SINKS_NONE;
	cmd.info.scaled_dot_product_attention.relative_bias = extent > 0;
	// The sinks are the ninth input, and the bias, if any, the tenth.
	const int input_size = extent > 0 ? 10 : 9;
	ccv_nnc_tensor_t* const ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
	ccv_nnc_tensor_t* const cpu_inputs[10] = { hq, hk, hv, 0, 0, 0, 0, 0, sinks_mode != SINKS_NONE ? hsinks : 0, hbias };
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, cpu_inputs, input_size, TENSOR_LIST(ho), 0), CCV_NNC_EXEC_SUCCESS, "CPU attention with a relative-position bias should execute");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected, ho->data.f32, B * R * Hq * D, 1e-4, "CPU attention with a relative-position bias should match the direct loop (R %d, C %d, extent %d, window %d, sinks %d)", R, C, extent, sliding_window, sinks_mode);
	if (ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_t* const hosts[5] = { hq, hk, hv, hbias, hsinks };
		ccv_nnc_tensor_t* gpu[5];
		for (i = 0; i < 5; i++)
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
		ccv_nnc_tensor_param_t output_params = GPU_TENSOR_NHWC(000, 32F, B, R, Hq, D);
		output_params.datatype = datatype;
		ccv_nnc_tensor_t* const o = ccv_nnc_tensor_new(0, output_params, 0);
		output_params.type = CCV_TENSOR_CPU_MEMORY;
		ccv_nnc_tensor_t* const hot = ccv_nnc_tensor_new(0, output_params, 0);
		ccv_nnc_tensor_t* const ho32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_inputs[10] = { gpu[0], gpu[1], gpu[2], 0, 0, 0, 0, 0, sinks_mode != SINKS_NONE ? gpu[4] : 0, gpu[3] };
		const uint64_t old_flags = ccv_nnc_flags();
		int pass;
		// With neural accelerators where there are any, whose products are half precision, then without.
		for (pass = 0; pass < 2; pass++)
		{
			if (pass == 0)
				ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
			else
				ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, gpu_inputs, input_size, TENSOR_LIST(o), 0), CCV_NNC_EXEC_SUCCESS, "GPU attention with a relative-position bias should execute (R %d, C %d, extent %d, window %d, sinks %d, datatype %d, pass %d)", R, C, extent, sliding_window, sinks_mode, datatype, pass);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o), TENSOR_LIST(hot), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hot), TENSOR_LIST(ho32), 0);
			for (i = 0; i < B * R * Hq * D; i++)
				{ REQUIRE(isfinite(ho32->data.f32[i]), "GPU attention output %d should be finite (R %d, C %d, extent %d, window %d, sinks %d, datatype %d, pass %d)", i, R, C, extent, sliding_window, sinks_mode, datatype, pass); }
			// BF16 keeps 8 bits of each output.
			REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected, ho32->data.f32, B * R * Hq * D, datatype == CCV_16BF ? 2e-2 : ((pass == 0 || datatype != CCV_32F) ? 5e-3 : 1e-4), "GPU attention with a relative-position bias should match the direct loop (R %d, C %d, extent %d, window %d, sinks %d, datatype %d, pass %d)", R, C, extent, sliding_window, sinks_mode, datatype, pass);
		}
		if (old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		else
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		for (i = 0; i < 5; i++)
			ccv_nnc_tensor_free(gpu[i]);
		ccv_nnc_tensor_free(o);
		ccv_nnc_tensor_free(hot);
		ccv_nnc_tensor_free(ho32);
	}
	ccfree(scores);
	ccfree(expected);
	ccv_nnc_tensor_free(ho);
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hsinks);
}

TEST_CASE("attention adds a relative-position bias to the keys before each query")
{
	// Prefill from an empty cache, then with cached keys, shorter and longer than the extent.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 37, 6, 2, 32, 16, 0, SINKS_NONE, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 37, 6, 2, 32, 64, 0, SINKS_PER_HEAD, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 90, 6, 2, 32, 16, 0, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 90, 6, 2, 32, 128, 0, SINKS_PER_QUERY, CCV_32F);
	// A sliding window as long as the extent, as wide as the keys, and shorter than the extent.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 90, 6, 2, 32, 24, 24, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 60, 6, 2, 32, 24, 60, SINKS_PER_HEAD, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 90, 6, 2, 32, 24, 9, SINKS_NONE, CCV_32F);
	// Two batches, each with its own biases and sinks.
	_relative_bias_attention_case(__case_name__, __case_result__, 2, 37, 90, 6, 2, 32, 24, 0, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 3, 21, 21, 6, 2, 32, 8, 12, SINKS_PER_HEAD, CCV_32F);
	// One query, as in decode.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 90, 6, 2, 32, 24, 0, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 1, 6, 2, 32, 24, 0, SINKS_PER_QUERY, CCV_32F);
	// One query over a window of keys as long as the sliding window, and over a longer one.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 64, 6, 2, 32, 64, 64, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 90, 6, 2, 32, 64, 64, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 1024, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_32F);
	// More queries than keys: the first rows see no key, with and without sinks.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 17, 6, 2, 128, 24, 0, SINKS_NONE, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 17, 6, 2, 32, 24, 0, SINKS_PER_HEAD, CCV_32F);
}

TEST_CASE("attention with a relative-position bias runs sliding-window and full layers in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	// 48 query heads over 8 key heads of 128, biased over the last 1,024 keys: a prefill chunk of a
	// sliding-window layer with its window of cached keys, of a full layer with every key, then
	// decode steps of one and two queries.
	// Smaller heads, and no sinks.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 90, 6, 2, 32, 16, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 90, 6, 2, 128, 16, 0, SINKS_NONE, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 64, 1087, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 64, 2500, 48, 8, 128, 1024, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 37, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 1024, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 2500, 48, 8, 128, 1024, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 700, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	// One decode row reads the last keys of a cache bound with more than its window; two rows, or
	// two batches, have windows that start at different keys.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 1500, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 5119, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 1025, 48, 8, 128, 1024, 1024, SINKS_NONE, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 1500, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 2, 1, 1500, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 64, 2500, 48, 8, 128, 1024, 0, SINKS_PER_QUERY, CCV_32F);
	// More queries than keys, so the first rows see no key.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 17, 6, 2, 128, 24, 0, SINKS_NONE, CCV_16F);
}

TEST_CASE("attention with a relative-position bias runs two decode rows and short caches through the decode kernel")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	// Two causal rows, with a window as wide as the keys and without one.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 700, 48, 8, 128, 1024, 1024, SINKS_PER_HEAD, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 2500, 48, 8, 128, 1024, 0, SINKS_NONE, CCV_16F);
	// Short caches, which one workgroup for each query and head reads together: one row, one row
	// over the last keys of a longer cache, and two rows.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 200, 48, 8, 128, 1024, 0, SINKS_PER_HEAD, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 1500, 48, 8, 128, 1024, 200, SINKS_PER_HEAD, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 60, 48, 8, 128, 1024, 0, SINKS_NONE, CCV_16F);
}

TEST_CASE("attention with a relative-position bias runs up to eight verify rows through the decode kernel")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	// Verify rows of a full layer with a sink for each query: split across workgroups over long
	// caches, a block of keys shorter than 16 at the end, and short caches split across workgroups
	// and read together by one workgroup for each query and head; only short caches give a sink
	// enough weight that reading another query's moves the output.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 3, 4099, 48, 8, 128, 1024, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 8, 9000, 48, 8, 128, 1024, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 5, 1500, 48, 8, 128, 1024, 0, SINKS_PER_HEAD, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 8, 100, 48, 8, 128, 1024, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 8, 60, 48, 8, 128, 1024, 0, SINKS_PER_QUERY, CCV_16F);
	// Sliding-window rows over the last keys of a cache bound with exactly and more than the last
	// row's window and the keys before it, so each row's window starts at a different key; then a
	// window shorter than the extent, and one short enough for the cooperative kernel.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 4, 1027, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 8, 3000, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 6, 1500, 48, 8, 128, 1024, 200, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 1500, 48, 8, 128, 1024, 40, SINKS_PER_QUERY, CCV_16F);
	// Two heads of a SIMD group (D = 128 with four query heads to a key head), and heads of 256,
	// the last with the largest workgroup: four heads of four groups of two rows, 512 threads.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 4, 5000, 32, 8, 128, 64, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 3, 5000, 16, 4, 256, 64, 0, SINKS_PER_HEAD, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 8, 5000, 16, 4, 256, 64, 0, SINKS_PER_QUERY, CCV_16F);
	// Sinks for each query without a bias, and bfloat.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 8, 2500, 48, 8, 128, 0, 0, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 8, 3000, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16BF);
}

TEST_CASE("attention with a relative-position bias runs in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	// A prefill chunk of a sliding-window layer, then decode rows.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 64, 1087, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16BF);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 1, 1500, 48, 8, 128, 1024, 1024, SINKS_PER_QUERY, CCV_16BF);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 700, 48, 8, 128, 1024, 1024, SINKS_PER_HEAD, CCV_16BF);
}

TEST_CASE("attention takes a sink for each query without a relative-position bias")
{
	// Without and with a sliding window, then in half precision.
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 37, 90, 6, 2, 32, 0, 0, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 2, 37, 90, 6, 2, 32, 0, 24, SINKS_PER_QUERY, CCV_32F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 64, 1087, 48, 8, 128, 0, 1024, SINKS_PER_QUERY, CCV_16F);
	_relative_bias_attention_case(__case_name__, __case_result__, 1, 2, 700, 48, 8, 128, 0, 0, SINKS_PER_QUERY, CCV_16F);
}

TEST_CASE("attention with a relative-position bias rejects what the GPU kernels cannot take")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 8, 6, 32), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 8, 2, 32), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 8, 2, 32), 0);
	ccv_nnc_tensor_t* const mask = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, 8, 8), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 8, 6, 4), 0);
	ccv_nnc_tensor_t* const o = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 8, 6, 32), 0);
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / sqrt(32), 0);
	cmd.info.scaled_dot_product_attention.relative_bias = 1;
	ccv_nnc_tensor_t* const inputs[10] = { q, k, v, 0, 0, 0, 0, 0, 0, bias };
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 10, TENSOR_LIST(o), 0), CCV_NNC_EXEC_INVALID, "a relative-position bias without causal attention should be rejected");
	cmd.info.scaled_dot_product_attention.is_causal = 1;
	ccv_nnc_tensor_t* const masked_inputs[10] = { q, k, v, mask, 0, 0, 0, 0, 0, bias };
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, masked_inputs, 10, TENSOR_LIST(o), 0), CCV_NNC_EXEC_INVALID, "a relative-position bias with an attention mask should be rejected");
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(mask);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(o);
}

TEST_CASE("attention model with a relative-position bias runs a decode row without the saved statistics")
{
	// One decode row of a sliding-window layer, bound with more keys than its window. Nothing reads
	// the model's saved softmax statistics, so its graph leaves them out.
	const int C = 1500, Hq = 48, Hk = 8, D = 128, extent = 1024, sliding_window = 1024;
	const float scale = 1.0 / sqrt((double)D);
	ccv_nnc_tensor_t* const hosts[5] = {
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, Hq, D), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, Hq), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, Hq, extent), 0),
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 29);
	int i, j;
	for (i = 0; i < 5; i++)
	{
		const size_t count = ccv_nnc_tensor_count(hosts[i]->info);
		uint16_t* const scratch = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
		for (j = 0; j < count; j++)
			hosts[i]->data.f32[j] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * 4;
		// The GPU reads FP16; the CPU reads the same rounded values.
		ccv_float_to_half_precision(hosts[i]->data.f32, scratch, count);
		ccv_half_precision_to_float(scratch, hosts[i]->data.f32, count);
		ccfree(scratch);
	}
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 1);
	cmd.info.scaled_dot_product_attention.sliding_window = sliding_window;
	cmd.info.scaled_dot_product_attention.attention_sinks = 1;
	cmd.info.scaled_dot_product_attention.relative_bias = 1;
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, Hq, D), 0);
	ccv_nnc_tensor_t* const cpu_inputs[10] = { hosts[0], hosts[1], hosts[2], 0, 0, 0, 0, 0, hosts[3], hosts[4] };
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, cpu_inputs, 10, TENSOR_LIST(expected), 0), CCV_NNC_EXEC_SUCCESS, "CPU attention with a relative-position bias should execute");
	ccv_nnc_tensor_param_t input_params[5];
	for (i = 0; i < 5; i++)
		input_params[i] = hosts[i]->info;
	ccv_cnnp_model_t* const cpu_model = ccv_cnnp_scaled_dot_product_attention(scale, 1, 0, 0, 0, 0, 0, 0, 1, sliding_window, 1, 0, 1, 0, "attention");
	ccv_cnnp_model_compile(cpu_model, input_params, 5, CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, Hq, D), 0);
	ccv_cnnp_model_evaluate(cpu_model, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, hosts, 5, TENSOR_LIST(ho), 0, 0);
	REQUIRE_TENSOR_EQ(ho, expected, "the CPU model's output should be the command's");
	ccv_cnnp_model_free(cpu_model);
	if (ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_t* gpu[5];
		for (i = 0; i < 5; i++)
		{
			ccv_nnc_tensor_param_t params = hosts[i]->info;
			params.datatype = CCV_16F;
			ccv_nnc_tensor_t* const converted = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hosts[i]), TENSOR_LIST(converted), 0);
			params.type = CCV_TENSOR_GPU_MEMORY | 000;
			input_params[i] = params;
			gpu[i] = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(converted), TENSOR_LIST(gpu[i]), 0);
			ccv_nnc_tensor_free(converted);
		}
		ccv_cnnp_model_t* const gpu_model = ccv_cnnp_scaled_dot_product_attention(scale, 1, 0, 0, 0, 0, 0, 0, 1, sliding_window, 1, 0, 1, 0, "attention");
		ccv_cnnp_model_compile(gpu_model, input_params, 5, CMD_NOOP(), CMD_NOOP());
		ccv_nnc_tensor_t* const o = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, Hq, D), 0);
		ccv_cnnp_model_evaluate(gpu_model, (ccv_cnnp_evaluate_param_t){
			.is_test = 1
		}, gpu, 5, TENSOR_LIST(o), 0, 0);
		ccv_nnc_tensor_t* const hot = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o), TENSOR_LIST(hot), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hot), TENSOR_LIST(ho), 0);
		for (i = 0; i < Hq * D; i++)
			{ REQUIRE(isfinite(ho->data.f32[i]), "the GPU model's output %d should be finite", i); }
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, ho->data.f32, Hq * D, 5e-3, "the GPU model's output should match the CPU command's");
		ccv_cnnp_model_free(gpu_model);
		ccv_nnc_tensor_free(o);
		ccv_nnc_tensor_free(hot);
		for (i = 0; i < 5; i++)
			ccv_nnc_tensor_free(gpu[i]);
	}
	for (i = 0; i < 5; i++)
		ccv_nnc_tensor_free(hosts[i]);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(ho);
}

#include "case_main.h"
