#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <math.h>
#include <string.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _segmented_swiglu_fill(float* const data, const size_t count, const int multiplier, const int modulus, const float scale)
{
	size_t i;
	for (i = 0; i < count; i++)
		data[i] = (float)((int)((i * multiplier) % modulus) - modulus / 2) * scale;
}

static void _segmented_swiglu_reference(const float* const a, const int* const indices, const int* const counts,
	const float* const gate_w, const float* const up_w, const float* const route_weight,
	const int rows, const int segments, const int n, const int k, const float clamp, float* const output)
{
	int row = 0;
	int segment;
	for (segment = 0; segment < segments; segment++)
	{
		const int expert = indices[segment];
		int segment_row;
		for (segment_row = 0; segment_row < counts[segment]; segment_row++, row++)
		{
			int col;
			for (col = 0; col < n; col++)
			{
				float gate = 0;
				float up = 0;
				int inner;
				for (inner = 0; inner < k; inner++)
				{
					const size_t weight_index = ((size_t)expert * n + col) * k + inner;
					gate += a[(size_t)row * k + inner] * gate_w[weight_index];
					up += a[(size_t)row * k + inner] * up_w[weight_index];
				}
				if (clamp > 0)
				{
					gate = ccv_min(gate, clamp);
					up = ccv_min(ccv_max(up, -clamp), clamp);
				}
				output[(size_t)row * n + col] = route_weight[row] * up * gate / (1 + expf(-gate));
			}
		}
	}
	assert(row == rows);
}

TEST_CASE("segmented SwiGLU composes routed gate and up projections on CPU")
{
	const int rows = 17;
	const int segments = 5;
	const int experts = 8;
	const int n = 29;
	const int k = 31;
	const float clamp = 2.5f;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const gate_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const up_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const route_weight = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	float* const expected = (float*)ccmalloc(sizeof(float) * rows * n);
	const int selected[] = { 6, 2, 1, 7, 4 };
	const int rows_per_segment[] = { 4, 0, 6, 3, 4 };
	memcpy(indices->data.i32, selected, sizeof(selected));
	memcpy(counts->data.i32, rows_per_segment, sizeof(rows_per_segment));
	_segmented_swiglu_fill(a->data.f32, (size_t)rows * k, 17, 101, 1.0f / 64);
	_segmented_swiglu_fill(gate_w->data.f32, (size_t)experts * n * k, 29, 113, 1.0f / 512);
	_segmented_swiglu_fill(up_w->data.f32, (size_t)experts * n * k, 37, 127, 1.0f / 512);
	int i;
	for (i = 0; i < rows; i++)
		route_weight->data.f32[i] = (float)(i + 1) / rows;
	_segmented_swiglu_reference(a->data.f32, indices->data.i32, counts->data.i32,
		gate_w->data.f32, up_w->data.f32, route_weight->data.f32,
		rows, segments, n, k, clamp, expected);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, gate_w, up_w, route_weight), TENSOR_LIST(output), 0),
		CCV_NNC_EXEC_SUCCESS, "segmented SwiGLU should execute on CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected, output->data.f32, rows * n, 1e-5,
		"segmented SwiGLU should match the high-level routed reference, including an empty segment");
	const float tiny_clamp = 1.0e-7f;
	_segmented_swiglu_reference(a->data.f32, indices->data.i32, counts->data.i32,
		gate_w->data.f32, up_w->data.f32, route_weight->data.f32,
		rows, segments, n, k, tiny_clamp, expected);
	command = CMD_SEGMENTED_SWIGLU_FORWARD(tiny_clamp);
	command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, gate_w, up_w, route_weight), TENSOR_LIST(output), 0),
		CCV_NNC_EXEC_SUCCESS, "segmented SwiGLU should apply every positive clamp");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected, output->data.f32, rows * n, 1e-9,
		"segmented SwiGLU should not use an epsilon threshold for clamp");
	ccv_nnc_tensor_param_t inferred = {};
	ccv_nnc_hint_tensor_auto(command, TENSOR_PARAM_LIST(a->info, indices->info, counts->info, gate_w->info, up_w->info, route_weight->info), ccv_nnc_no_hint, &inferred, 1);
	REQUIRE_EQ(inferred.dim[0], rows, "segmented SwiGLU should preserve the input row count");
	REQUIRE_EQ(inferred.dim[1], n, "segmented SwiGLU should infer the projection width");
	ccfree(expected);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(route_weight);
	ccv_nnc_tensor_free(up_w);
	ccv_nnc_tensor_free(gate_w);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("segmented SwiGLU broadcasts one activation row across routed experts on CPU")
{
	const int rows = 6;
	const int segments = 4;
	const int experts = 7;
	const int n = 23;
	const int k = 31;
	const float clamp = 3;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, k), 0);
	ccv_nnc_tensor_t* const grouped_a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const gate_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const up_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const route_weight = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const broadcast_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_tensor_t* const grouped_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	const int selected[] = { 6, 1, 4, 2 };
	const int rows_per_segment[] = { 1, 2, 0, 3 };
	memcpy(indices->data.i32, selected, sizeof(selected));
	memcpy(counts->data.i32, rows_per_segment, sizeof(rows_per_segment));
	_segmented_swiglu_fill(a->data.f32, k, 17, 101, 1.0f / 64);
	int i;
	for (i = 0; i < rows; i++)
		memcpy(grouped_a->data.f32 + (size_t)i * k, a->data.f32, sizeof(float) * k);
	_segmented_swiglu_fill(gate_w->data.f32, (size_t)experts * n * k, 29, 113, 1.0f / 512);
	_segmented_swiglu_fill(up_w->data.f32, (size_t)experts * n * k, 37, 127, 1.0f / 512);
	for (i = 0; i < rows; i++)
		route_weight->data.f32[i] = (float)(i + 1) / rows;
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(grouped_a, indices, counts, gate_w, up_w, route_weight), TENSOR_LIST(grouped_output), 0),
		CCV_NNC_EXEC_SUCCESS, "grouped segmented SwiGLU reference should execute");
	REQUIRE_EQ(ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, gate_w, up_w, route_weight), TENSOR_LIST(broadcast_output), 0),
		CCV_NNC_EXEC_SUCCESS, "broadcast segmented SwiGLU should execute");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, grouped_output->data.f32, broadcast_output->data.f32, rows * n, 1e-6,
		"a single activation row should produce the same result as explicitly grouped copies");
	ccv_nnc_tensor_param_t inferred = {};
	ccv_nnc_hint_tensor_auto(command, TENSOR_PARAM_LIST(a->info, indices->info, counts->info, gate_w->info, up_w->info, route_weight->info), ccv_nnc_no_hint, &inferred, 1);
	REQUIRE_EQ(inferred.dim[0], rows, "broadcast segmented SwiGLU should infer routed rows from route weights");
	REQUIRE_EQ(inferred.dim[1], n, "broadcast segmented SwiGLU should infer the projection width");
	ccv_nnc_tensor_free(grouped_output);
	ccv_nnc_tensor_free(broadcast_output);
	ccv_nnc_tensor_free(route_weight);
	ccv_nnc_tensor_free(up_w);
	ccv_nnc_tensor_free(gate_w);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(grouped_a);
	ccv_nnc_tensor_free(a);
}

static void _segmented_swiglu_compare_half(const ccv_float16_t* const actual_half, const float* const expected,
	const int count, const double relative_l2_tolerance, const char* const message, int* const __case_result__)
{
	float* const actual = (float*)ccmalloc(sizeof(float) * count);
	ccv_half_precision_to_float((const uint16_t*)actual_half, actual, count);
	double difference_norm = 0;
	double expected_norm = 0;
	double dot = 0;
	double actual_norm = 0;
	double max_abs = 0;
	int i;
	for (i = 0; i < count; i++)
	{
		const double difference = (double)actual[i] - expected[i];
		difference_norm += difference * difference;
		expected_norm += (double)expected[i] * expected[i];
		actual_norm += (double)actual[i] * actual[i];
		dot += (double)actual[i] * expected[i];
		max_abs = ccv_max(max_abs, fabs(difference));
	}
	const double relative_l2 = sqrt(difference_norm / ccv_max(expected_norm, 1e-20));
	const double cosine = dot / sqrt(ccv_max(actual_norm * expected_norm, 1e-20));
	REQUIRE(relative_l2 < relative_l2_tolerance,
		"%s (relative L2 %.8g, cosine %.8g, max abs %.8g)", message, relative_l2, cosine, max_abs);
	REQUIRE(cosine > 0.999, "%s should preserve direction (cosine %.8g)", message, cosine);
	ccfree(actual);
}

TEST_CASE("MPS segmented SwiGLU supports general dense routed groups")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 17;
	const int segments = 5;
	const int experts = 8;
	const int n = 29;
	const int k = 31;
	const float clamp = 2.5f;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hgate_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hup_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hroute_weight = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	const int selected[] = { 6, 2, 1, 7, 4 };
	const int rows_per_segment[] = { 4, 0, 6, 3, 4 };
	memcpy(hindices->data.i32, selected, sizeof(selected));
	memcpy(hcounts->data.i32, rows_per_segment, sizeof(rows_per_segment));
	_segmented_swiglu_fill(ha->data.f32, (size_t)rows * k, 17, 101, 1.0f / 64);
	_segmented_swiglu_fill(hgate_w->data.f32, (size_t)experts * n * k, 29, 113, 1.0f / 512);
	_segmented_swiglu_fill(hup_w->data.f32, (size_t)experts * n * k, 37, 127, 1.0f / 512);
	int i;
	for (i = 0; i < rows; i++)
		hroute_weight->data.f32[i] = (float)(i + 1) / rows;
	ccv_nnc_cmd_t cpu_command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	cpu_command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cpu_command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hindices, hcounts, hgate_w, hup_w, hroute_weight), TENSOR_LIST(expected), 0),
		CCV_NNC_EXEC_SUCCESS, "CPU segmented SwiGLU reference should execute");
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const gate_w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const up_w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const route_weight = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hindices, hcounts, hgate_w, hup_w, hroute_weight),
		TENSOR_LIST(a, indices, counts, gate_w, up_w, route_weight), stream), CCV_NNC_EXEC_SUCCESS,
		"dense segmented SwiGLU inputs should transfer");
	ccv_nnc_cmd_t mps_command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	mps_command.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, gate_w, up_w, route_weight), TENSOR_LIST(output), stream),
		CCV_NNC_EXEC_SUCCESS, "dense segmented SwiGLU should execute through its composed MPS path");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, rows * n, 1e-4,
		"dense MPS segmented SwiGLU should match CPU");
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(route_weight);
	ccv_nnc_tensor_free(up_w);
	ccv_nnc_tensor_free(gate_w);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hroute_weight);
	ccv_nnc_tensor_free(hup_w);
	ccv_nnc_tensor_free(hgate_w);
	ccv_nnc_tensor_free(ha);
}

static void _segmented_swiglu_mps_rowwise_case(const int format, const int broadcast_input, const int grouped_prefill, const float clamp, const char* const format_name, int* const __case_result__)
{
	const int rows = grouped_prefill ? 17 : 6;
	const int segments = 6;
	const int experts = 8;
	const int n = 256;
	const int k = 256;
	const int activation_rows = broadcast_input ? 1 : rows;
	const size_t activation_count = (size_t)activation_rows * k;
	const size_t weight_count = (size_t)experts * n * k;
	ccv_nnc_tensor_t* const ha_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, activation_rows, k), 0);
	ccv_nnc_tensor_t* const hgate_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hup_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hroute_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, activation_rows, k), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hup = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hroute = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	_segmented_swiglu_fill(ha_f32->data.f32, activation_count, 17, 101, 1.0f / 64);
	_segmented_swiglu_fill(hgate_f32->data.f32, weight_count, 29, 113, 1.0f / 1024);
	_segmented_swiglu_fill(hup_f32->data.f32, weight_count, 37, 127, 1.0f / 1024);
	const int selected[] = { 7, 1, 5, 2, 6, 3 };
	const int decode_rows_per_segment[] = { 0, 2, 1, 1, 1, 1 };
	const int prefill_rows_per_segment[] = { 0, 2, 4, 3, 5, 3 };
	memcpy(hindices->data.i32, selected, sizeof(selected));
	memcpy(hcounts->data.i32, grouped_prefill ? prefill_rows_per_segment : decode_rows_per_segment,
		sizeof(decode_rows_per_segment));
	int i;
	for (i = 0; i < rows; i++)
	{
		hroute_f32->data.f32[i] = (float)(i + 1) / (rows + 1);
	}
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha_f32, hgate_f32, hup_f32, hroute_f32), TENSOR_LIST(ha, hgate, hup, hroute), 0);
	const ccv_nnc_tensor_param_t q_params = format ?
		ccv_nnc_tensor_8i_rowwise_x(hgate->info, format) : ccv_nnc_tensor_8i_rowwise(hgate->info);
	ccv_nnc_tensor_t* const hgate_q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const hup_q = ccv_nnc_tensor_new(0, q_params, 0);
	const size_t gate_qsize = format ?
		ccv_nnc_quantize_8i_rowwise_x(hgate->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, weight_count, k, format, 0, 0,
			hgate_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hgate_q->info)) :
		ccv_nnc_quantize_8i_rowwise(hgate->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, weight_count, k, 0, 0,
			hgate_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hgate_q->info));
	const size_t up_qsize = format ?
		ccv_nnc_quantize_8i_rowwise_x(hup->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, weight_count, k, format, 0, 0,
			hup_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hup_q->info)) :
		ccv_nnc_quantize_8i_rowwise(hup->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, weight_count, k, 0, 0,
			hup_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hup_q->info));
	REQUIRE_EQ(gate_qsize, ccv_nnc_tensor_data_size_without_padding(hgate_q->info),
		"%s gate weights should quantize to the declared size", format_name);
	REQUIRE_EQ(up_qsize, ccv_nnc_tensor_data_size_without_padding(hup_q->info),
		"%s up weights should quantize to the declared size", format_name);
	ccv_nnc_tensor_t* const hgate_dequant = ccv_nnc_tensor_new(0, hgate->info, 0);
	ccv_nnc_tensor_t* const hup_dequant = ccv_nnc_tensor_new(0, hup->info, 0);
	if (format)
	{
		ccv_nnc_dequantize_8i_rowwise_x(hgate_q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, gate_qsize, k, format, hgate_dequant->data.u8, weight_count);
		ccv_nnc_dequantize_8i_rowwise_x(hup_q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, up_qsize, k, format, hup_dequant->data.u8, weight_count);
	} else {
		ccv_nnc_dequantize_8i_rowwise(hgate_q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, gate_qsize, k, hgate_dequant->data.u8, weight_count);
		ccv_nnc_dequantize_8i_rowwise(hup_q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, up_qsize, k, hup_dequant->data.u8, weight_count);
	}
	ccv_nnc_tensor_t* const ha_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, activation_rows, k), 0);
	ccv_nnc_tensor_t* const hgate_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hup_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hroute_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hgate_dequant, hup_dequant, hroute),
		TENSOR_LIST(ha_ref, hgate_ref, hup_ref, hroute_ref), 0);
	ccv_nnc_cmd_t cpu_command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	cpu_command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cpu_command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha_ref, hindices, hcounts, hgate_ref, hup_ref, hroute_ref), TENSOR_LIST(expected), 0),
		CCV_NNC_EXEC_SUCCESS, "%s CPU dequantized reference should execute", format_name);
	ccv_nnc_tensor_param_t gpu_q_params = q_params;
	gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, activation_rows, k), 0);
	ccv_nnc_tensor_t* const gate_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const up_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hindices, hcounts, hgate_q, hup_q, hroute),
		TENSOR_LIST(a, indices, counts, gate_q, up_q, route), stream);
	ccv_nnc_cmd_t mps_command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	mps_command.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	const int old_watermark = ccv_nnc_queue_watermark();
	ccv_nnc_set_queue_watermark(3);
	for (i = 0; i < 3; i++)
		REQUIRE_EQ(ccv_nnc_cmd_exec(mps_command, ccv_nnc_no_hint, 0,
			TENSOR_LIST(a, indices, counts, gate_q, up_q, route), TENSOR_LIST(output), stream),
			CCV_NNC_EXEC_SUCCESS, "%s path should queue", format_name);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_set_queue_watermark(old_watermark);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	_segmented_swiglu_compare_half(actual->data.f16, expected->data.f32, rows * n,
		grouped_prefill ? 0.04 : 0.02,
		"rowwise segmented SwiGLU should match its dequantized CPU reference", __case_result__);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(up_q);
	ccv_nnc_tensor_free(gate_q);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(hroute_ref);
	ccv_nnc_tensor_free(hup_ref);
	ccv_nnc_tensor_free(hgate_ref);
	ccv_nnc_tensor_free(ha_ref);
	ccv_nnc_tensor_free(hup_dequant);
	ccv_nnc_tensor_free(hgate_dequant);
	ccv_nnc_tensor_free(hup_q);
	ccv_nnc_tensor_free(hgate_q);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hroute);
	ccv_nnc_tensor_free(hup);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hroute_f32);
	ccv_nnc_tensor_free(hup_f32);
	ccv_nnc_tensor_free(hgate_f32);
	ccv_nnc_tensor_free(ha_f32);
}

TEST_CASE("MPS segmented SwiGLU fuses rowwise int8 decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_rowwise_case(0, 0, 0, 10, "Q8_0 decode", __case_result__);
}

TEST_CASE("MPS segmented SwiGLU fuses IQ2_XXS decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 0, 0, 10, "IQ2_XXS decode", __case_result__);
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 0, 0, 0.25f, "IQ2_XXS decode with specialized clamp", __case_result__);
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 0, 0, 0, "IQ2_XXS decode without clamp", __case_result__);
}

TEST_CASE("MPS segmented SwiGLU fuses IQ2_XS, IQ3_XXS, and Q2_K decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XS, 0, 0, 10, "IQ2_XS decode", __case_result__);
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ3_XXS, 0, 0, 10, "IQ3_XXS decode", __case_result__);
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_Q2_K, 0, 0, 10, "Q2_K decode", __case_result__);
}

TEST_CASE("MPS segmented SwiGLU broadcasts one activation row for rowwise int8 decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_rowwise_case(0, 1, 0, 10, "broadcast Q8_0 decode", __case_result__);
}

TEST_CASE("MPS segmented SwiGLU broadcasts one activation row for IQ2_XXS decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 1, 0, 10, "broadcast IQ2_XXS decode", __case_result__);
}

TEST_CASE("MPS segmented SwiGLU executes grouped rowwise int8 prefill")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_rowwise_case(0, 0, 1, 10, "Q8_0 grouped prefill", __case_result__);
}

static void _segmented_swiglu_mps_float_rowwise_case(const int format, const int n, const int k, const int large_groups, int* const __case_result__)
{
	const int rows = large_groups ? 389 : 97;
	const int segments = 6;
	const int experts = 8;
	const float clamp = 10;
	const size_t weight_count = (size_t)experts * n * k;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hup = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
	ccv_nnc_tensor_t* const hroute = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < rows * k; i++)
		ha->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	if (large_groups)
	{
		// Keep the expanded fixture off half-integer rounding boundaries: the
		// CPU and Metal reciprocal can otherwise choose adjacent int8 values.
		// Quarter-step offsets still require real activation quantization.
		for (i = 0; i < rows * k; i++)
			ha->data.f32[i] = (floorf(ha->data.f32[i] * 252) + 0.25f) / 128;
		for (i = 0; i < rows; i++)
			ha->data.f32[i * k] = 127.0f / 128;
	}
	// These values cannot be represented by FP16 directly. The fallback must first
	// quantize them to bounded q / 128 FP16 values and keep the row scale in FP32.
	ha->data.f32[0] = 1e8f;
	ha->data.f32[k] = -1e8f;
	for (i = 0; i < weight_count; i++)
	{
		hgate->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 8;
		hup->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 8;
	}
	const int selected[] = { 7, 1, 5, 2, 6, 3 };
	const int rows_per_segment[] = { 0, 2, large_groups ? 65 : 17, large_groups ? 111 : 31, large_groups ? 99 : 19, large_groups ? 112 : 28 };
	memcpy(hindices->data.i32, selected, sizeof(selected));
	memcpy(hcounts->data.i32, rows_per_segment, sizeof(rows_per_segment));
	for (i = 0; i < rows; i++)
		hroute->data.f32[i] = (float)(i + 1) / (rows + 1);
	const ccv_nnc_tensor_param_t q_params = format ?
		ccv_nnc_tensor_8i_rowwise_x(hgate->info, format) : ccv_nnc_tensor_8i_rowwise(hgate->info);
	ccv_nnc_tensor_t* const hgate_q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const hup_q = ccv_nnc_tensor_new(0, q_params, 0);
	const size_t gate_qsize = format ? ccv_nnc_quantize_8i_rowwise_x(
		hgate->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, weight_count, k, format, 0, 0,
		hgate_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hgate_q->info)) : ccv_nnc_quantize_8i_rowwise(
		hgate->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, weight_count, k, 0, 0,
		hgate_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hgate_q->info));
	const size_t up_qsize = format ? ccv_nnc_quantize_8i_rowwise_x(
		hup->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, weight_count, k, format, 0, 0,
		hup_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hup_q->info)) : ccv_nnc_quantize_8i_rowwise(
		hup->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, weight_count, k, 0, 0,
		hup_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hup_q->info));
	REQUIRE_EQ(gate_qsize, ccv_nnc_tensor_data_size_without_padding(hgate_q->info),
		"float gate weights should quantize to the declared rowwise size");
	REQUIRE_EQ(up_qsize, ccv_nnc_tensor_data_size_without_padding(hup_q->info),
		"float up weights should quantize to the declared rowwise size");
	ccv_nnc_tensor_t* const hgate_dequant = ccv_nnc_tensor_new(0, hgate->info, 0);
	ccv_nnc_tensor_t* const hup_dequant = ccv_nnc_tensor_new(0, hup->info, 0);
	if (format)
	{
		ccv_nnc_dequantize_8i_rowwise_x(
			hgate_q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, gate_qsize, k, format,
			hgate_dequant->data.u8, weight_count);
		ccv_nnc_dequantize_8i_rowwise_x(
			hup_q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, up_qsize, k, format,
			hup_dequant->data.u8, weight_count);
	} else {
		ccv_nnc_dequantize_8i_rowwise(
			hgate_q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, gate_qsize, k,
			hgate_dequant->data.u8, weight_count);
		ccv_nnc_dequantize_8i_rowwise(
			hup_q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, up_qsize, k,
			hup_dequant->data.u8, weight_count);
	}
	ccv_nnc_tensor_t* const ha_dequant = ccv_nnc_tensor_new(0, ha->info, 0);
	// Runtime activation quantization uses absmax, not the offline weight
	// quantizer's iteratively fitted scale.
	for (i = 0; i < rows; i++)
	{
		float max_abs = 0;
		int j;
		for (j = 0; j < k; j++)
			max_abs = ccv_max(max_abs, fabsf(ha->data.f32[i * k + j]));
		const float scale = max_abs > 0 ? max_abs / 127.0f : 1.0f / 127.0f;
		const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
		for (j = 0; j < k; j++)
			ha_dequant->data.f32[i * k + j] = scale *
				ccv_clamp((int)lrintf(ha->data.f32[i * k + j] * inv_scale), -127, 127);
	}
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	_segmented_swiglu_reference(
		ha_dequant->data.f32, hindices->data.i32, hcounts->data.i32,
		hgate_dequant->data.f32, hup_dequant->data.f32, hroute->data.f32,
		rows, segments, n, k, clamp, expected->data.f32);
	ccv_nnc_tensor_param_t gpu_q_params = q_params;
	gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const gate_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const up_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hindices, hcounts, hgate_q, hup_q, hroute),
		TENSOR_LIST(a, indices, counts, gate_q, up_q, route), stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	command.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	const int exec_status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, gate_q, up_q, route), TENSOR_LIST(output), stream);
	if (exec_status == CCV_NNC_EXEC_SUCCESS)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	REQUIRE_EQ(exec_status, CCV_NNC_EXEC_SUCCESS, "float rowwise segmented SwiGLU fallback should execute");
	double difference_norm = 0;
	double expected_norm = 0;
	double max_abs = 0;
	for (i = 0; i < rows * n; i++)
	{
		const double difference = (double)actual->data.f32[i] - expected->data.f32[i];
		difference_norm += difference * difference;
		expected_norm += (double)expected->data.f32[i] * expected->data.f32[i];
		max_abs = ccv_max(max_abs, fabs(difference));
	}
	const double relative_l2 = sqrt(difference_norm / ccv_max(expected_norm, 1e-20));
	REQUIRE(relative_l2 < 5e-3,
		"float rowwise segmented SwiGLU with Int8MatMul should match the quantized float reference (relative L2 %.8g, max abs %.8g)",
		relative_l2, max_abs);
	// Check each row too, so the large-scale rows cannot mask a broken tail or
	// silently selecting the unquantized-activation FP32 fallback.
	for (i = 0; i < rows; i++)
	{
		double row_difference_norm = 0;
		double row_expected_norm = 0;
		int j;
		for (j = 0; j < n; j++)
		{
			const double value = expected->data.f32[i * n + j];
			const double difference = (double)actual->data.f32[i * n + j] - value;
			row_difference_norm += difference * difference;
			row_expected_norm += value * value;
		}
		const double row_relative_l2 = sqrt(row_difference_norm / ccv_max(row_expected_norm, 1e-20));
		REQUIRE(row_relative_l2 < 1e-4,
			"Int8MatMul row should match quantized reference: format=%d N=%d K=%d row=%d relative L2=%g",
			format, n, k, i, row_relative_l2);
	}
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(up_q);
	ccv_nnc_tensor_free(gate_q);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(ha_dequant);
	ccv_nnc_tensor_free(hup_dequant);
	ccv_nnc_tensor_free(hgate_dequant);
	ccv_nnc_tensor_free(hup_q);
	ccv_nnc_tensor_free(hgate_q);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hroute);
	ccv_nnc_tensor_free(hup);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(ha);
}

TEST_CASE("MPS segmented SwiGLU emulates int8 with normalized half values and float scales")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_float_rowwise_case(0, 64, 128, 0, __case_result__);
}

TEST_CASE("MPS segmented SwiGLU uses Int8MatMul for packed weights and float scales")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q4_K, CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K, CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS, CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_Q5_K, CCV_NNC_QX_8I_ROWWISE_Q6_K,
	};
	int i;
	for (i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
		_segmented_swiglu_mps_float_rowwise_case(formats[i], 256, 256, 0, __case_result__);
	_segmented_swiglu_mps_float_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 512, 256, 0, __case_result__);
	_segmented_swiglu_mps_float_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 256, 512, 0, __case_result__);
	_segmented_swiglu_mps_float_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 256, 4096, 0, __case_result__);
	_segmented_swiglu_mps_float_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 256, 4096, 1, __case_result__);
}

TEST_CASE("MPS segmented SwiGLU executes grouped IQ2_XXS prefill")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_segmented_swiglu_mps_rowwise_case(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, 0, 1, 10, "IQ2_XXS grouped prefill", __case_result__);
}

TEST_CASE("MPS segmented SwiGLU handles small rowwise batches across formats")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		0, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS, CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K, CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q5_K, CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S, CCV_NNC_QX_8I_ROWWISE_IQ3_S,
	};
	const int row_counts[] = { 3, 17, 18, 19 };
	const int segments = 6, experts = 8, n = 256, k = 256;
	const size_t weight_count = (size_t)experts * n * k;
	const float clamp = 10;
	for (int f = 0; f < sizeof(formats) / sizeof(formats[0]); f++)
	{
		const int format = formats[f];
		ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, experts, n, k), 0);
		ccv_nnc_tensor_t* const hup = ccv_nnc_tensor_new(0, hgate->info, 0);
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 42);
		for (size_t i = 0; i < weight_count; i++)
		{
			hgate->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 8;
			hup->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 8;
		}
		const ccv_nnc_tensor_param_t packed_info = format ?
			ccv_nnc_tensor_8i_rowwise_x(hgate->info, format) : ccv_nnc_tensor_8i_rowwise(hgate->info);
		const size_t packed_size = ccv_nnc_tensor_data_size_without_padding(packed_info);
		ccv_nnc_tensor_t* const hgate_q = ccv_nnc_tensor_new(0, packed_info, 0);
		ccv_nnc_tensor_t* const hup_q = ccv_nnc_tensor_new(0, packed_info, 0);
		ccv_nnc_tensor_t* const weights[] = { hgate, hup };
		ccv_nnc_tensor_t* const packed[] = { hgate_q, hup_q };
		for (int projection = 0; projection < 2; projection++)
		{
			const size_t size = format ?
				ccv_nnc_quantize_8i_rowwise_x(weights[projection]->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
					weight_count, k, format, 0, 0, packed[projection]->data.u8, packed_size) :
				ccv_nnc_quantize_8i_rowwise(weights[projection]->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
					weight_count, k, 0, 0, packed[projection]->data.u8, packed_size);
			REQUIRE_EQ(size, packed_size, "weights should quantize to the declared size");
			if (format)
				ccv_nnc_dequantize_8i_rowwise_x(packed[projection]->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
					packed_size, k, format, weights[projection]->data.u8, weight_count);
			else
				ccv_nnc_dequantize_8i_rowwise(packed[projection]->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
					packed_size, k, weights[projection]->data.u8, weight_count);
		}
		ccv_nnc_tensor_param_t gpu_info = packed_info;
		gpu_info.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, gpu_info, 0);
		ccv_nnc_tensor_t* const up = ccv_nnc_tensor_new(0, gpu_info, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgate_q, hup_q), TENSOR_LIST(gate, up), 0);
		for (int step = 0; step < sizeof(row_counts) / sizeof(row_counts[0]); step++)
		{
			const int rows = row_counts[step];
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
			ccv_nnc_tensor_t* const ha_ref = ccv_nnc_tensor_new(0, ha->info, 0);
			ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
			ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, hi->info, 0);
			ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, expected->info, 0);
			const int selected[] = { 7, 1, 5, 2, 6, 3 };
			const int counts[] = { 0, 2, 0, rows - 2, 0, 0 };
			memcpy(hi->data.i32, selected, sizeof(selected));
			memcpy(hc->data.i32, counts, sizeof(counts));
			for (int i = 0; i < rows * k; i++)
				ha->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
			// Both direct projections and the matrix fallback must handle FP32 magnitudes.
			ha->data.f32[0] = 1e8f;
			ha->data.f32[k] = -1e8f;
			memcpy(ha_ref->data.f32, ha->data.f32, (size_t)rows * k * sizeof(float));
			for (int row = 0; row < rows; row++)
			{
				hr->data.f32[row] = (float)(row + 1) / (rows + 1);
				// At 19 rows, the existing matrix path still quantizes activations.
				if (rows == 19)
				{
					float max_abs = 0;
					for (int j = 0; j < k; j++)
						max_abs = ccv_max(max_abs, fabsf(ha->data.f32[row * k + j]));
					const float scale = max_abs > 0 ? max_abs / 127.f : 1.f / 127;
					const float inv_scale = max_abs > 0 ? 127.f / max_abs : 127;
					for (int j = 0; j < k; j++)
						ha_ref->data.f32[row * k + j] = scale * ccv_clamp((int)lrintf(ha->data.f32[row * k + j] * inv_scale), -127, 127);
				}
			}
			ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
			command.backend = CCV_NNC_BACKEND_CPU_REF;
			REQUIRE_EQ(ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
				TENSOR_LIST(ha_ref, hi, hc, hgate, hup, hr), TENSOR_LIST(expected), 0), CCV_NNC_EXEC_SUCCESS, "CPU reference should execute");
			ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
			ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
			ccv_nnc_tensor_t* const counts_gpu = ccv_nnc_tensor_new(0, indices->info, 0);
			ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows), 0);
			ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
				TENSOR_LIST(ha, hi, hc, hr), TENSOR_LIST(a, indices, counts_gpu, route), 0);
			command.backend = CCV_NNC_BACKEND_MPS;
			const uint64_t old_flags = ccv_nnc_flags();
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
			const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
				TENSOR_LIST(a, indices, counts_gpu, gate, up, route), TENSOR_LIST(output), 0);
			if (old_flags & CCV_NNC_DISABLE_MFA) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
			if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
			REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "small rowwise batch should execute: format=%d rows=%d", format, rows);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(actual), 0);
			for (int row = 0; row < rows; row++)
			{
				double squared_error = 0, squared_reference = 0;
				for (int j = 0; j < n; j++)
				{
					const float value = actual->data.f32[row * n + j];
					const float reference = expected->data.f32[row * n + j];
					REQUIRE(isfinite(value), "output should be finite: format=%d rows=%d row=%d", format, rows, row);
					const double difference = value - reference;
					squared_error += difference * difference;
					squared_reference += (double)reference * reference;
				}
				const double relative_l2 = sqrt(squared_error / ccv_max(squared_reference, 1e-20));
				// Affine packed decoders subtract large FP32 sums on the two outlier
				// rows. Keep the tighter bound for ordinary activations.
				const double tolerance = row < 2 ? 5e-3 : 1e-4;
				REQUIRE(relative_l2 < tolerance, "row should match CPU reference: format=%d rows=%d row=%d relative L2=%g", format, rows, row, relative_l2);
			}
			ccv_nnc_tensor_t* const tensors[] = { ha, ha_ref, hi, hc, hr, expected, actual, a, indices, counts_gpu, route, output };
			for (int i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++)
				ccv_nnc_tensor_free(tensors[i]);
		}
		ccv_nnc_tensor_free(gate);
		ccv_nnc_tensor_free(up);
		ccv_nnc_tensor_free(hgate_q);
		ccv_nnc_tensor_free(hup_q);
		ccv_nnc_tensor_free(hgate);
		ccv_nnc_tensor_free(hup);
	}
}

TEST_CASE("MPS segmented SwiGLU preserves route weights with 257 experts")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	// This shape exposed intermittent zero outputs with Metal shader validation.
	// 256 experts passed; 257 failed even though all selected IDs are below 228.
	const int rows = 228, experts = 257, n = 2304, k = 5120;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	for (int i = 0; i < rows * k; i++)
		ha->data.f32[i] = 1.f / k;
	for (int i = 0; i < rows; i++)
	{
		hi->data.i32[i] = i;
		hc->data.i32[i] = 1;
		hr->data.f32[i] = 0.25f;
	}
	ccv_nnc_tensor_param_t q = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, experts, n, k), CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, q, 0);
	ccv_nnc_tensor_t* const hu = ccv_nnc_tensor_new(0, q, 0);
	const size_t payload = (size_t)experts * n * (k / 32) * 8;
	// Construct valid packed blocks directly to avoid quantizing a dense bank.
	// Each expert repeats the same rows: one expert's payload is a multiple of
	// this 256-code pattern. CPU-dequantizing one expert therefore covers them all.
	for (size_t i = 0; i < payload; i += 8)
	{
		for (int j = 0; j < 4; j++)
		{
			hg->data.u8[i + j] = (i / 8 + j * 17) % 256;
			hu->data.u8[i + j] = (i / 8 + j * 23 + 9) % 256;
		}
		for (int j = 4; j < 7; j++)
			hg->data.u8[i + j] = hu->data.u8[i + j] = 0;
		hg->data.u8[i + 7] = 0x20;
		hu->data.u8[i + 7] = 0x30;
	}
	float* const gs = (float*)(hg->data.u8 + payload);
	float* const us = (float*)(hu->data.u8 + payload);
	for (int i = 0; i < experts * n; i++)
	{
		gs[i] = 1.f / 16;
		us[i] = 1.f / 32;
	}
	ccv_nnc_tensor_t* const gd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, n, k), 0);
	ccv_nnc_tensor_t* const ud = ccv_nnc_tensor_new(0, gd->info, 0);
	const ccv_nnc_tensor_param_t one_q = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, 1, n, k), CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
	ccv_nnc_tensor_t* const one = ccv_nnc_tensor_new(0, one_q, 0);
	const size_t one_payload = (size_t)n * (k / 32) * 8;
	const size_t one_size = ccv_nnc_tensor_data_size_without_padding(one_q);
	memcpy(one->data.u8, hg->data.u8, one_payload);
	memcpy(one->data.u8 + one_payload, gs, n * sizeof(float));
	ccv_nnc_dequantize_8i_rowwise_x(one->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, one_size, k, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, gd->data.u8, (size_t)n * k);
	memcpy(one->data.u8, hu->data.u8, one_payload);
	memcpy(one->data.u8 + one_payload, us, n * sizeof(float));
	ccv_nnc_dequantize_8i_rowwise_x(one->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, one_size, k, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, ud->data.u8, (size_t)n * k);
	float* const expected = (float*)ccmalloc(n * sizeof(float));
	for (int col = 0; col < n; col++)
	{
		double gate = 0, up = 0;
		for (int inner = 0; inner < k; inner++)
		{
			gate += gd->data.f32[col * k + inner] / k;
			up += ud->data.f32[col * k + inner] / k;
		}
		gate = ccv_min(gate, 10);
		up = ccv_min(ccv_max(up, -10), 10);
		expected[col] = 0.25 * up * gate / (1 + exp(-gate));
	}
	ccv_nnc_tensor_free(one);
	ccv_nnc_tensor_free(gd);
	ccv_nnc_tensor_free(ud);
	q.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, q, 0);
	ccv_nnc_tensor_t* const up = ccv_nnc_tensor_new(0, q, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hi, hc, hr, hg, hu), TENSOR_LIST(a, indices, counts, route, gate, up), stream), CCV_NNC_EXEC_SUCCESS, "copy test inputs");
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(10);
	command.backend = CCV_NNC_BACKEND_MPS;
	for (int run = 0; run < 16; run++)
	{
		// Distinguish zeroed results from skipped stores or stale output storage.
		for (int i = 0; i < rows * n; i++)
			actual->data.f32[i] = run % 2 ? 123.f : NAN;
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(actual), TENSOR_LIST(output), stream), CCV_NNC_EXEC_SUCCESS, "poison output");
		const uint64_t old_flags = ccv_nnc_flags();
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
			TENSOR_LIST(a, indices, counts, gate, up, route), TENSOR_LIST(output), stream);
		if (old_flags & CCV_NNC_DISABLE_MFA)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
		REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "execute packed SwiGLU");
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(output), TENSOR_LIST(actual), stream), CCV_NNC_EXEC_SUCCESS, "read completed output");
		ccv_nnc_stream_context_wait(stream);
		for (int i = 0; i < rows * n; i++)
		{
			REQUIRE(isfinite(actual->data.f32[i]), "finite result at run=%d index=%d", run, i);
			REQUIRE_EQ_WITH_TOLERANCE(actual->data.f32[i], expected[i % n], 1e-5, "route weight preserved at run=%d index=%d", run, i);
		}
	}
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_t* const tensors[] = { ha, hi, hc, hr, hg, hu, actual, a, indices, counts, gate, up, route, output };
	for (int i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++)
		ccv_nnc_tensor_free(tensors[i]);
	ccfree(expected);
}

#include "case_main.h"
