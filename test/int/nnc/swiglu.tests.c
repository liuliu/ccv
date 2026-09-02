#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _swiglu_fill(float* const data, const size_t count, const int multiplier, const int modulus, const float scale)
{
	size_t i;
	for (i = 0; i < count; i++)
		data[i] = (float)((int)((i * multiplier) % modulus) - modulus / 2) * scale;
}

static void _swiglu_reference(const float* const a, const float* const gate_w, const float* const up_w,
	const int rows, const int n, const int k, const float clamp, float* const output)
{
	int row;
	for (row = 0; row < rows; row++)
	{
		int column;
		for (column = 0; column < n; column++)
		{
			float gate = 0;
			float up = 0;
			int inner;
			for (inner = 0; inner < k; inner++)
			{
				gate += a[(size_t)row * k + inner] * gate_w[(size_t)column * k + inner];
				up += a[(size_t)row * k + inner] * up_w[(size_t)column * k + inner];
			}
			if (clamp > 0)
			{
				gate = ccv_min(gate, clamp);
				up = ccv_min(ccv_max(up, -clamp), clamp);
			}
			output[(size_t)row * n + column] = up * gate / (1 + expf(-gate));
		}
	}
}

static void _swiglu_compare_half(const ccv_float16_t* const actual_half, const float* const expected,
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

TEST_CASE("SwiGLU composes gate and up projections on CPU and infers metadata from the activation")
{
	const int rows = 6;
	const int n = 29;
	const int k = 31;
	const float clamp = 2.5f;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, k), 0);
	ccv_nnc_tensor_t* const gate_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const up_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, n), 0);
	float* const expected = (float*)ccmalloc(sizeof(float) * rows * n);
	_swiglu_fill(a->data.f32, (size_t)rows * k, 17, 101, 1.0f / 64);
	_swiglu_fill(gate_w->data.f32, (size_t)n * k, 29, 113, 1.0f / 512);
	_swiglu_fill(up_w->data.f32, (size_t)n * k, 37, 127, 1.0f / 512);
	_swiglu_reference(a->data.f32, gate_w->data.f32, up_w->data.f32,
		rows, n, k, clamp, expected);
	ccv_nnc_cmd_t command = CMD_SWIGLU_FORWARD(clamp);
	command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, gate_w, up_w), TENSOR_LIST(output), 0),
		CCV_NNC_EXEC_SUCCESS, "SwiGLU should execute on CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected, output->data.f32, rows * n, 1e-5,
		"SwiGLU should match the high-level reference");
	ccv_nnc_tensor_param_t inferred = {};
	ccv_nnc_hint_tensor_auto(command, TENSOR_PARAM_LIST(a->info, gate_w->info, up_w->info),
		ccv_nnc_no_hint, &inferred, 1);
	REQUIRE_EQ(inferred.type, a->info.type, "SwiGLU should inherit the activation memory type");
	REQUIRE_EQ(inferred.format, a->info.format, "SwiGLU should inherit the activation format");
	REQUIRE_EQ(inferred.datatype, a->info.datatype, "SwiGLU should inherit the activation datatype");
	REQUIRE_EQ(inferred.dim[0], 2, "SwiGLU should preserve the first activation dimension");
	REQUIRE_EQ(inferred.dim[1], 3, "SwiGLU should preserve the second activation dimension");
	REQUIRE_EQ(inferred.dim[2], n, "SwiGLU should replace the final dimension with the projection width");
	ccfree(expected);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(up_w);
	ccv_nnc_tensor_free(gate_w);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("SwiGLU model exposes one input and owns both projection weights")
{
	const ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 2, 3, 31);
	ccv_cnnp_model_t* const model = ccv_cnnp_swiglu(29, 2.5f, 0, "swiglu");
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_param_t output_params = {};
	ccv_cnnp_model_tensor_auto(model, &output_params, 1);
	REQUIRE_EQ(CCV_TENSOR_GET_MEMORY(output_params.type), CCV_TENSOR_GET_MEMORY(input_params.type),
		"SwiGLU model output should inherit input memory type");
	REQUIRE_EQ(output_params.format, input_params.format, "SwiGLU model output should inherit input format");
	REQUIRE_EQ(output_params.datatype, input_params.datatype, "SwiGLU model output should inherit input datatype");
	REQUIRE_EQ(output_params.dim[0], 2, "SwiGLU model should preserve the first input dimension");
	REQUIRE_EQ(output_params.dim[1], 3, "SwiGLU model should preserve the second input dimension");
	REQUIRE_EQ(output_params.dim[2], 29, "SwiGLU model should infer the projection width");
	ccv_cnnp_model_free(model);
}

TEST_CASE("MPS SwiGLU supports general dense projections")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 17;
	const int n = 29;
	const int k = 31;
	const float clamp = 2.5f;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hgate_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const hup_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	_swiglu_fill(ha->data.f32, (size_t)rows * k, 17, 101, 1.0f / 64);
	_swiglu_fill(hgate_w->data.f32, (size_t)n * k, 29, 113, 1.0f / 512);
	_swiglu_fill(hup_w->data.f32, (size_t)n * k, 37, 127, 1.0f / 512);
	ccv_nnc_cmd_t cpu_command = CMD_SWIGLU_FORWARD(clamp);
	cpu_command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cpu_command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hgate_w, hup_w), TENSOR_LIST(expected), 0),
		CCV_NNC_EXEC_SUCCESS, "CPU SwiGLU reference should execute");
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const gate_w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, n, k), 0);
	ccv_nnc_tensor_t* const up_w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, n, k), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hgate_w, hup_w), TENSOR_LIST(a, gate_w, up_w), stream),
		CCV_NNC_EXEC_SUCCESS, "dense SwiGLU inputs should transfer");
	ccv_nnc_cmd_t mps_command = CMD_SWIGLU_FORWARD(clamp);
	mps_command.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, gate_w, up_w), TENSOR_LIST(output), stream),
		CCV_NNC_EXEC_SUCCESS, "dense SwiGLU should execute through its composed MPS path");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, rows * n, 1e-4,
		"dense MPS SwiGLU should match CPU");
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(up_w);
	ccv_nnc_tensor_free(gate_w);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(hup_w);
	ccv_nnc_tensor_free(hgate_w);
	ccv_nnc_tensor_free(ha);
}

static void _swiglu_mps_rowwise_case(const int rows, const float clamp, const char* const case_name, int* const __case_result__)
{
	const int n = 256;
	const int k = 256;
	const size_t activation_count = (size_t)rows * k;
	const size_t weight_count = (size_t)n * k;
	ccv_nnc_tensor_t* const ha_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hgate_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const hup_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, k), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, n, k), 0);
	ccv_nnc_tensor_t* const hup = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, n, k), 0);
	_swiglu_fill(ha_f32->data.f32, activation_count, 17, 101, 1.0f / 64);
	_swiglu_fill(hgate_f32->data.f32, weight_count, 29, 113, 1.0f / 1024);
	_swiglu_fill(hup_f32->data.f32, weight_count, 37, 127, 1.0f / 1024);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha_f32, hgate_f32, hup_f32), TENSOR_LIST(ha, hgate, hup), 0);
	const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise(hgate->info);
	ccv_nnc_tensor_t* const hgate_q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const hup_q = ccv_nnc_tensor_new(0, q_params, 0);
	const size_t gate_qsize = ccv_nnc_quantize_8i_rowwise(hgate->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY,
		weight_count, k, 0, 0, hgate_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hgate_q->info));
	const size_t up_qsize = ccv_nnc_quantize_8i_rowwise(hup->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY,
		weight_count, k, 0, 0, hup_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hup_q->info));
	REQUIRE_EQ(gate_qsize, ccv_nnc_tensor_data_size_without_padding(hgate_q->info),
		"%s gate weights should quantize to the declared size", case_name);
	REQUIRE_EQ(up_qsize, ccv_nnc_tensor_data_size_without_padding(hup_q->info),
		"%s up weights should quantize to the declared size", case_name);
	ccv_nnc_tensor_t* const hgate_dequant = ccv_nnc_tensor_new(0, hgate->info, 0);
	ccv_nnc_tensor_t* const hup_dequant = ccv_nnc_tensor_new(0, hup->info, 0);
	ccv_nnc_dequantize_8i_rowwise(hgate_q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY,
		gate_qsize, k, hgate_dequant->data.u8, weight_count);
	ccv_nnc_dequantize_8i_rowwise(hup_q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY,
		up_qsize, k, hup_dequant->data.u8, weight_count);
	ccv_nnc_tensor_t* const ha_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hgate_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const hup_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hgate_dequant, hup_dequant), TENSOR_LIST(ha_ref, hgate_ref, hup_ref), 0);
	ccv_nnc_cmd_t cpu_command = CMD_SWIGLU_FORWARD(clamp);
	cpu_command.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cpu_command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha_ref, hgate_ref, hup_ref), TENSOR_LIST(expected), 0),
		CCV_NNC_EXEC_SUCCESS, "%s CPU dequantized reference should execute", case_name);
	ccv_nnc_tensor_param_t gpu_q_params = q_params;
	gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, k), 0);
	ccv_nnc_tensor_t* const gate_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const up_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hgate_q, hup_q), TENSOR_LIST(a, gate_q, up_q), stream),
		CCV_NNC_EXEC_SUCCESS, "%s inputs should transfer", case_name);
	ccv_nnc_cmd_t mps_command = CMD_SWIGLU_FORWARD(clamp);
	mps_command.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	const int old_watermark = ccv_nnc_queue_watermark();
	ccv_nnc_set_queue_watermark(3);
	int i;
	for (i = 0; i < 3; i++)
	{
		REQUIRE_EQ(ccv_nnc_cmd_exec(mps_command, ccv_nnc_no_hint, 0,
			TENSOR_LIST(a, gate_q, up_q), TENSOR_LIST(output), stream),
			CCV_NNC_EXEC_SUCCESS, "%s path should queue", case_name);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_set_queue_watermark(old_watermark);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	_swiglu_compare_half(actual->data.f16, expected->data.f32, rows * n,
		rows == 1 ? 0.02 : 0.04, case_name, __case_result__);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(up_q);
	ccv_nnc_tensor_free(gate_q);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(hup_ref);
	ccv_nnc_tensor_free(hgate_ref);
	ccv_nnc_tensor_free(ha_ref);
	ccv_nnc_tensor_free(hup_dequant);
	ccv_nnc_tensor_free(hgate_dequant);
	ccv_nnc_tensor_free(hup_q);
	ccv_nnc_tensor_free(hgate_q);
	ccv_nnc_tensor_free(hup);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hup_f32);
	ccv_nnc_tensor_free(hgate_f32);
	ccv_nnc_tensor_free(ha_f32);
}

TEST_CASE("MPS SwiGLU fuses rowwise int8 decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_swiglu_mps_rowwise_case(1, 10, "rowwise int8 decode SwiGLU should match its dequantized CPU reference", __case_result__);
	_swiglu_mps_rowwise_case(1, 0, "unclamped rowwise int8 decode SwiGLU should match its dequantized CPU reference", __case_result__);
}

TEST_CASE("MPS SwiGLU quantizes activation once for rowwise int8 prefill")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	_swiglu_mps_rowwise_case(17, 10, "rowwise int8 prefill SwiGLU should match its dequantized CPU reference", __case_result__);
}

TEST_CASE("MPS SwiGLU uses Int8MatMul for float rowwise int8 prefill without neural accelerators")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 17;
	const int n = 64;
	const int k = 72;
	const float clamp = 10;
	const size_t activation_count = (size_t)rows * k;
	const size_t weight_count = (size_t)n * k;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const hup = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	_swiglu_fill(ha->data.f32, activation_count, 17, 101, 1.0f / 64);
	_swiglu_fill(hgate->data.f32, weight_count, 29, 113, 1.0f / 1024);
	_swiglu_fill(hup->data.f32, weight_count, 37, 127, 1.0f / 1024);
	const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise(hgate->info);
	ccv_nnc_tensor_t* const hgate_q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const hup_q = ccv_nnc_tensor_new(0, q_params, 0);
	const size_t gate_qsize = ccv_nnc_quantize_8i_rowwise(
		hgate->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, weight_count, k, 0, 0,
		hgate_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hgate_q->info));
	const size_t up_qsize = ccv_nnc_quantize_8i_rowwise(
		hup->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, weight_count, k, 0, 0,
		hup_q->data.u8, ccv_nnc_tensor_data_size_without_padding(hup_q->info));
	REQUIRE_EQ(gate_qsize, ccv_nnc_tensor_data_size_without_padding(hgate_q->info),
		"float gate weights should quantize to the declared rowwise size");
	REQUIRE_EQ(up_qsize, ccv_nnc_tensor_data_size_without_padding(hup_q->info),
		"float up weights should quantize to the declared rowwise size");
	ccv_nnc_tensor_t* const hgate_dequant = ccv_nnc_tensor_new(0, hgate->info, 0);
	ccv_nnc_tensor_t* const hup_dequant = ccv_nnc_tensor_new(0, hup->info, 0);
	ccv_nnc_dequantize_8i_rowwise(
		hgate_q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, gate_qsize, k,
		hgate_dequant->data.u8, weight_count);
	ccv_nnc_dequantize_8i_rowwise(
		hup_q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, up_qsize, k,
		hup_dequant->data.u8, weight_count);
	const ccv_nnc_tensor_param_t a_q_params = ccv_nnc_tensor_8i_rowwise(ha->info);
	ccv_nnc_tensor_t* const ha_q = ccv_nnc_tensor_new(0, a_q_params, 0);
	ccv_nnc_tensor_t* const ha_dequant = ccv_nnc_tensor_new(0, ha->info, 0);
	const size_t a_qsize = ccv_nnc_quantize_8i_rowwise(
		ha->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, activation_count, k, 0, 0,
		ha_q->data.u8, ccv_nnc_tensor_data_size_without_padding(ha_q->info));
	REQUIRE_EQ(a_qsize, ccv_nnc_tensor_data_size_without_padding(ha_q->info),
		"float activations should quantize to the declared rowwise size");
	ccv_nnc_dequantize_8i_rowwise(
		ha_q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, a_qsize, k,
		ha_dequant->data.u8, activation_count);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(
		0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	_swiglu_reference(
		ha_dequant->data.f32, hgate_dequant->data.f32, hup_dequant->data.f32,
		rows, n, k, clamp, expected->data.f32);
	ccv_nnc_tensor_param_t gpu_q_params = q_params;
	gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(
		0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const gate_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const up_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(
		0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(
		0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hgate_q, hup_q), TENSOR_LIST(a, gate_q, up_q), stream);
	ccv_nnc_cmd_t command = CMD_SWIGLU_FORWARD(clamp);
	command.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	const int exec_status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, gate_q, up_q), TENSOR_LIST(output), stream);
	if (exec_status == CCV_NNC_EXEC_SUCCESS)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	REQUIRE_EQ(exec_status, CCV_NNC_EXEC_SUCCESS,
		"float rowwise SwiGLU fallback should execute");
	double difference_norm = 0;
	double expected_norm = 0;
	double max_abs = 0;
	int i;
	for (i = 0; i < rows * n; i++)
	{
		const double difference = (double)actual->data.f32[i] - expected->data.f32[i];
		difference_norm += difference * difference;
		expected_norm += (double)expected->data.f32[i] * expected->data.f32[i];
		max_abs = ccv_max(max_abs, fabs(difference));
	}
	const double relative_l2 = sqrt(difference_norm / ccv_max(expected_norm, 1e-20));
	REQUIRE(relative_l2 < 6e-3,
		"float rowwise SwiGLU with Int8MatMul should match the quantized float reference (relative L2 %.8g, max abs %.8g)",
		relative_l2, max_abs);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(up_q);
	ccv_nnc_tensor_free(gate_q);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(ha_dequant);
	ccv_nnc_tensor_free(ha_q);
	ccv_nnc_tensor_free(hup_dequant);
	ccv_nnc_tensor_free(hgate_dequant);
	ccv_nnc_tensor_free(hup_q);
	ccv_nnc_tensor_free(hgate_q);
	ccv_nnc_tensor_free(hup);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(ha);
}

#include "case_main.h"
