#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("swish mul in float")
{
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < 17 * 13; i++)
	{
		a_tensor->data.f32[i] = (float)(6.0 * dsfmt_genrand_open_close(&dsfmt) - 3.0);
		b_tensor->data.f32[i] = (float)(6.0 * dsfmt_genrand_open_close(&dsfmt) - 3.0);
	}
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_FORWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(a_tensor, b_tensor), TENSOR_LIST(c_tensor), 0);
	for (i = 0; i < 17 * 13; i++)
	{
		const float x = b_tensor->data.f32[i];
		const float sigmoid = 1.f / (1.f + expf(-1.7f * x));
		const float expected = 0.25f * a_tensor->data.f32[i] * x * sigmoid;
		REQUIRE_EQ_WITH_TOLERANCE(expected, c_tensor->data.f32[i], 1e-6, "swish mul should match reference");
	}
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_FORWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(a_tensor, b_tensor), TENSOR_LIST(a_tensor), 0);
	REQUIRE_TENSOR_EQ(a_tensor, c_tensor, "swish mul should support replacing the value input");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b_tensor);
	ccv_nnc_tensor_free(c_tensor);
}

TEST_CASE("weighted clamped swish mul applies one weight per row")
{
	const int rows = 7;
	const int cols = 13;
	const float limit = 10;
	ccv_nnc_tensor_t* const value = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_tensor_t* const weight = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i;
	for (i = 0; i < rows; i++)
	{
		weight->data.f32[i] = (float)(i + 1) / 8;
		int j;
		for (j = 0; j < cols; j++)
		{
			value->data.f32[i * cols + j] = (float)((i * 19 + j * 7) % 37 - 18);
			gate->data.f32[i * cols + j] = (float)((i * 11 + j * 13) % 41 - 20);
		}
	}
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_WEIGHTED_SWISH_MUL_FORWARD(1, 1, limit), ccv_nnc_no_hint, 0,
		TENSOR_LIST(value, gate, weight), TENSOR_LIST(output), 0), CCV_NNC_EXEC_SUCCESS,
		"weighted clamped swish mul should execute");
	for (i = 0; i < rows; i++)
	{
		int j;
		for (j = 0; j < cols; j++)
		{
			const float limited_value = ccv_min(ccv_max(value->data.f32[i * cols + j], -limit), limit);
			const float limited_gate = ccv_min(gate->data.f32[i * cols + j], limit);
			const float expected = weight->data.f32[i] * limited_value * limited_gate / (1 + expf(-limited_gate));
			REQUIRE_EQ_WITH_TOLERANCE(output->data.f32[i * cols + j], expected, 1e-5,
				"weighted clamped swish mul should match its row-wise reference");
		}
	}
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(weight);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(value);
}

TEST_CASE("MPSGraph weighted clamped swish mul accepts a rank-one row weight")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 7;
	const int cols = 13;
	const float limit = 10;
	ccv_nnc_tensor_t* const hvalue_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_tensor_t* const hgate_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_tensor_t* const hweight_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_tensor_t* const hvalue = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_nnc_tensor_t* const hweight = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows), 0);
	ccv_nnc_tensor_t* const value = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, cols), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, cols), 0);
	ccv_nnc_tensor_t* const weight = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, cols), 0);
	ccv_nnc_tensor_t* const houtput = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i;
	for (i = 0; i < rows; i++)
	{
		hweight_f32->data.f32[i] = (float)(i + 1) / 8;
		int j;
		for (j = 0; j < cols; j++)
		{
			hvalue_f32->data.f32[i * cols + j] = (float)((i * 19 + j * 7) % 37 - 18);
			hgate_f32->data.f32[i * cols + j] = (float)((i * 11 + j * 13) % 41 - 20);
		}
	}
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hvalue_f32, hgate_f32, hweight_f32), TENSOR_LIST(hvalue, hgate, hweight), 0);
	ccv_nnc_tensor_t* const value_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_tensor_t* const gate_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_tensor_t* const weight_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hvalue, hgate, hweight), TENSOR_LIST(value_f32, gate_f32, weight_f32), 0);
	ccv_nnc_cmd_exec(CMD_WEIGHTED_SWISH_MUL_FORWARD(1, 1, limit), ccv_nnc_no_hint, 0,
		TENSOR_LIST(value_f32, gate_f32, weight_f32), TENSOR_LIST(expected), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hvalue, hgate, hweight), TENSOR_LIST(value, gate, weight), 0);
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_t cmd = CMD_WEIGHTED_SWISH_MUL_FORWARD(1, 1, limit);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0,
		TENSOR_LIST(value, gate, weight), TENSOR_LIST(output), 0);
	if (!(old_flags & CCV_NNC_DISABLE_MFA))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "MPSGraph weighted clamped swish mul should execute");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(houtput), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(houtput), TENSOR_LIST(actual), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, actual->data.f32, expected->data.f32, rows * cols, 2e-2,
		"MPSGraph weighted clamped swish mul should match the quantized CPU reference");
	ccv_nnc_tensor_free(weight_f32);
	ccv_nnc_tensor_free(gate_f32);
	ccv_nnc_tensor_free(value_f32);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(houtput);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(weight);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(value);
	ccv_nnc_tensor_free(hweight);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(hvalue);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(hweight_f32);
	ccv_nnc_tensor_free(hgate_f32);
	ccv_nnc_tensor_free(hvalue_f32);
}

TEST_CASE("swish mul gradient in float")
{
	ccv_nnc_tensor_t* const value = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gradient = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const dvalue = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const dgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	int i;
	for (i = 0; i < 17 * 13; i++)
	{
		value->data.f32[i] = (float)(6.0 * dsfmt_genrand_open_close(&dsfmt) - 3.0);
		gate->data.f32[i] = (float)(6.0 * dsfmt_genrand_open_close(&dsfmt) - 3.0);
		gradient->data.f32[i] = (float)(2.0 * dsfmt_genrand_open_close(&dsfmt) - 1.0);
	}
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_BACKWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gradient, value, gate), TENSOR_LIST(dvalue, dgate), 0);
	for (i = 0; i < 17 * 13; i++)
	{
		const float x = gate->data.f32[i];
		const float sigmoid = 1.f / (1.f + expf(-1.7f * x));
		const float scale_gradient = 0.25f * gradient->data.f32[i];
		const float expected_dvalue = scale_gradient * x * sigmoid;
		const float expected_dgate = scale_gradient * value->data.f32[i] * (sigmoid + 1.7f * x * sigmoid * (1.f - sigmoid));
		REQUIRE_EQ_WITH_TOLERANCE(expected_dvalue, dvalue->data.f32[i], 1e-6, "swish mul value gradient should match reference");
		REQUIRE_EQ_WITH_TOLERANCE(expected_dgate, dgate->data.f32[i], 1e-6, "swish mul gate gradient should match reference");
	}
	ccv_nnc_tensor_t* const dvalue_only = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const dgate_only = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_BACKWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gradient, 0, gate), TENSOR_LIST(dvalue_only, 0), 0);
	REQUIRE_TENSOR_EQ(dvalue_only, dvalue, "swish mul backward should support only value gradient");
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_BACKWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gradient, value, gate), TENSOR_LIST(0, dgate_only), 0);
	REQUIRE_TENSOR_EQ(dgate_only, dgate, "swish mul backward should support only gate gradient");
	ccv_nnc_tensor_free(value);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(gradient);
	ccv_nnc_tensor_free(dvalue);
	ccv_nnc_tensor_free(dgate);
	ccv_nnc_tensor_free(dvalue_only);
	ccv_nnc_tensor_free(dgate_only);
}

TEST_CASE("gpu swish mul in mixed precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_MUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) ||
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const value = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	int i;
	for (i = 0; i < 17 * 13; i++)
	{
		value->data.f32[i] = (float)(4.0 * dsfmt_genrand_open_close(&dsfmt) - 2.0);
		gate->data.f32[i] = (float)(4.0 * dsfmt_genrand_open_close(&dsfmt) - 2.0);
	}
	ccv_nnc_tensor_t* const value_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 17, 13), 0);
	ccv_nnc_tensor_t* const value_f16_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gate_bf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16BF, 17, 13), 0);
	ccv_nnc_tensor_t* const gate_bf16_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const expected_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const expected_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_value_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_gate_bf16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, 17, 13), 0);
	ccv_nnc_tensor_t* const out_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 17, 13), 0);
	ccv_nnc_tensor_t* const out_f32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 17, 13), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(value), TENSOR_LIST(value_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(value_f16), TENSOR_LIST(value_f16_f32), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gate), TENSOR_LIST(gate_bf16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gate_bf16), TENSOR_LIST(gate_bf16_f32), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_FORWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(value_f16_f32, gate_bf16_f32), TENSOR_LIST(expected_f32), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected_f32), TENSOR_LIST(expected_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(value_f16, gate_bf16), TENSOR_LIST(gpu_value_f16, gpu_gate_bf16), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_FORWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_value_f16, gpu_gate_bf16), TENSOR_LIST(gpu_value_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_value_f16), TENSOR_LIST(out_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(out_f16), TENSOR_LIST(out_f32), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected_f16), TENSOR_LIST(expected_f32), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32->data.f32, out_f32->data.f32, 17 * 13, 1e-2, "gpu swish mul should support mixed fp16 / bf16 inputs with fp16 output");
	ccv_nnc_tensor_free(value);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(value_f16);
	ccv_nnc_tensor_free(value_f16_f32);
	ccv_nnc_tensor_free(gate_bf16);
	ccv_nnc_tensor_free(gate_bf16_f32);
	ccv_nnc_tensor_free(expected_f32);
	ccv_nnc_tensor_free(expected_f16);
	ccv_nnc_tensor_free(gpu_value_f16);
	ccv_nnc_tensor_free(gpu_gate_bf16);
	ccv_nnc_tensor_free(out_f16);
	ccv_nnc_tensor_free(out_f32);
}

TEST_CASE("gpu swish mul gradient in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_MUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF) ||
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const value = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gradient = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4);
	int i;
	for (i = 0; i < 17 * 13; i++)
	{
		value->data.f32[i] = (float)(4.0 * dsfmt_genrand_open_close(&dsfmt) - 2.0);
		gate->data.f32[i] = (float)(4.0 * dsfmt_genrand_open_close(&dsfmt) - 2.0);
		gradient->data.f32[i] = (float)(2.0 * dsfmt_genrand_open_close(&dsfmt) - 1.0);
	}
	ccv_nnc_tensor_t* const expected_dvalue = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const expected_dgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_value = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_gate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_gradient = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_dvalue = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_dgate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_dvalue_only = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 13), 0);
	ccv_nnc_tensor_t* const gpu_dgate_only = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 13), 0);
	ccv_nnc_tensor_t* const dvalue = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const dgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const dvalue_only = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_tensor_t* const dgate_only = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 13), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_BACKWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gradient, value, gate), TENSOR_LIST(expected_dvalue, expected_dgate), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(value, gate, gradient), TENSOR_LIST(gpu_value, gpu_gate, gpu_gradient), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_BACKWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_gradient, gpu_value, gpu_gate), TENSOR_LIST(gpu_dvalue, gpu_dgate), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dvalue, gpu_dgate), TENSOR_LIST(dvalue, dgate), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_dvalue->data.f32, dvalue->data.f32, 17 * 13, 1e-5, "gpu swish mul value gradient should match CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_dgate->data.f32, dgate->data.f32, 17 * 13, 1e-5, "gpu swish mul gate gradient should match CPU");
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_BACKWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_gradient, 0, gpu_gate), TENSOR_LIST(gpu_dvalue_only, 0), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dvalue_only), TENSOR_LIST(dvalue_only), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_dvalue->data.f32, dvalue_only->data.f32, 17 * 13, 1e-5, "gpu swish mul should support only value gradient");
	ccv_nnc_cmd_exec(CMD_SWISH_MUL_BACKWARD(1.7, 0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_gradient, gpu_value, gpu_gate), TENSOR_LIST(0, gpu_dgate_only), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dgate_only), TENSOR_LIST(dgate_only), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_dgate->data.f32, dgate_only->data.f32, 17 * 13, 1e-5, "gpu swish mul should support only gate gradient");
	ccv_nnc_tensor_free(value);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(gradient);
	ccv_nnc_tensor_free(expected_dvalue);
	ccv_nnc_tensor_free(expected_dgate);
	ccv_nnc_tensor_free(gpu_value);
	ccv_nnc_tensor_free(gpu_gate);
	ccv_nnc_tensor_free(gpu_gradient);
	ccv_nnc_tensor_free(gpu_dvalue);
	ccv_nnc_tensor_free(gpu_dgate);
	ccv_nnc_tensor_free(gpu_dvalue_only);
	ccv_nnc_tensor_free(gpu_dgate_only);
	ccv_nnc_tensor_free(dvalue);
	ccv_nnc_tensor_free(dgate);
	ccv_nnc_tensor_free(dvalue_only);
	ccv_nnc_tensor_free(dgate_only);
}

TEST_CASE("swish in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(1), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_TENSOR_EQ(ty, y_tensor, "swish from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(1), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty->data.f32, y_tensor->data.f32, 20 * 10, 1e-3, "swish from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish gradient in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(1), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_TENSOR_EQ(ty_tensor, y_tensor, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_TENSOR_EQ(tdx_tensor, dx_tensor, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("mps swish gradient in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(1), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_TENSOR_EQ(ty_tensor, y_tensor, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_TENSOR_EQ(tdx_tensor, dx_tensor, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish gradient in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(1), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dy16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dy16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy16_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dx16_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty_tensor->data.f32, y_tensor->data.f32, 10 * 100, 1e-3, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdx_tensor->data.f32, dx_tensor->data.f32, 10 * 100, 1e-3, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dx16_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dy16_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("mps swish gradient in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(1), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dy16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dy16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy16_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dx16_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty_tensor->data.f32, y_tensor->data.f32, 10 * 100, 1e-3, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdx_tensor->data.f32, dx_tensor->data.f32, 10 * 100, 1e-3, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dx16_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dy16_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish gradient with non-one beta in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	const float beta = 1.7;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(beta), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dy16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dy16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy16_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dx16_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(beta), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty_tensor->data.f32, y_tensor->data.f32, 10 * 100, 1e-3, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(beta), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdx_tensor->data.f32, dx_tensor->data.f32, 10 * 100, 1e-3, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dx16_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dy16_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("mps swish gradient with non-one beta in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_MPS));
	const float beta = 1.7;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(beta), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "swish");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dy16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dy16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy16_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dx16_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(beta), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty_tensor->data.f32, y_tensor->data.f32, 10 * 100, 1e-3, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(beta), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdx_tensor->data.f32, dx_tensor->data.f32, 10 * 100, 1e-3, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dx16_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dy16_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("mps swish with extreme beta stays finite")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_MPS));
	const float beta = 1e20f;
	static const float xv[] = {-1e20f, -1.0f, 0.0f, 1.0f, 1e20f};
	static const float expected_y[] = {0.0f, 0.0f, 0.0f, 1.0f, 1e20f};
	static const float expected_dx[] = {0.0f, 0.0f, 0.5f, 1.0f, 1.0f};
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	int i;
	for (i = 0; i < 5; i++)
	{
		x_tensor->data.f32[i] = xv[i];
		dy_tensor->data.f32[i] = 1.0f;
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(beta), ccv_nnc_no_hint, 0, TENSOR_LIST(xt), TENSOR_LIST(yt), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(beta), ccv_nnc_no_hint, 0, TENSOR_LIST(dyt, xt, 0), TENSOR_LIST(dxt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	for (i = 0; i < 5; i++)
	{
		REQUIRE(isfinite(y_tensor->data.f32[i]), "forward pass should stay finite at %d", i);
		REQUIRE(isfinite(dx_tensor->data.f32[i]), "backward pass should stay finite at %d", i);
		const float y_tolerance = fmaxf(fabsf(expected_y[i]) * 1e-6f, 1e-6f);
		REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[i], expected_y[i], y_tolerance, "forward pass should saturate cleanly at %d", i);
		REQUIRE_EQ_WITH_TOLERANCE(dx_tensor->data.f32[i], expected_dx[i], 1e-6f, "backward pass should saturate cleanly at %d", i);
	}
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(xt);
	ccv_nnc_tensor_free(yt);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_tensor_free(dxt);
}

#include "case_main.h"
