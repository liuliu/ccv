#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _conform_data_format_fill(dsfmt_t* const dsfmt, float* const values, const int count)
{
	int i;
	for (i = 0; i < count; i++)
		values[i] = (float)(dsfmt_genrand_open_close(dsfmt) * 2000.0 - 1000.0);
}

static void _conform_data_format_restore_mfa_flag(const uint64_t old_flags)
{
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
}

TEST_CASE("MFA conform data format matches CPU reference in place and out of place")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	const int formats[] = { CCV_TENSOR_FORMAT_NHWC, CCV_TENSOR_FORMAT_NCHW };
	const int rows = 5 * 3;
	const int head_dim = 320;
	const int count = rows * head_dim;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int f;
	for (f = 0; f < 2; f++)
	{
		ccv_nnc_tensor_param_t cpu_params = CPU_TENSOR_NHWC(32F, 5, 3, 320);
		cpu_params.format = formats[f];
		ccv_nnc_tensor_param_t gpu_params = GPU_TENSOR_NHWC(000, 32F, 5, 3, 320);
		gpu_params.format = formats[f];
		ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, cpu_params, 0);
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, cpu_params, 0);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, cpu_params, 0);
		ccv_nnc_tensor_t* const actual_in_place = ccv_nnc_tensor_new(0, cpu_params, 0);
		ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, gpu_params, 0);
		ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, gpu_params, 0);
		ccv_nnc_tensor_t* const gpu_in_place = ccv_nnc_tensor_new(0, gpu_params, 0);
		_conform_data_format_fill(&dsfmt, input->data.f32, count);
		memset(input->data.f32, 0, sizeof(float) * 64);
		input->data.f32[0] = 448.0f;
		input->data.f32[1] = 1.0625f;
		input->data.f32[2] = 1.1875f;
		input->data.f32[3] = -1.0625f;
		const ccv_nnc_cmd_t cpu_cmd = CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64);
		REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(input), TENSOR_LIST(expected), 0), "CPU E4M3 reference should execute");
		REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input, input), TENSOR_LIST(gpu_input, gpu_in_place), 0), "inputs should transfer to Metal");
		ccv_nnc_cmd_t mps_cmd = cpu_cmd;
		mps_cmd.backend = CCV_NNC_BACKEND_MPS;
		const int out_of_place_status = ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input), TENSOR_LIST(gpu_output), 0);
		const int in_place_status = ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_in_place), TENSOR_LIST(gpu_in_place), 0);
		REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output, gpu_in_place), TENSOR_LIST(actual, actual_in_place), 0), "Metal results should transfer to CPU");
		REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, out_of_place_status, "MFA E4M3 conformance should execute out of place");
		REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, in_place_status, "MFA E4M3 conformance should execute in place");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, count, 1e-6, "MFA E4M3 conformance should match CPU reference");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual_in_place->data.f32, count, 1e-6, "in-place MFA E4M3 conformance should match CPU reference");
		ccv_nnc_tensor_free(gpu_in_place);
		ccv_nnc_tensor_free(gpu_output);
		ccv_nnc_tensor_free(gpu_input);
		ccv_nnc_tensor_free(actual_in_place);
		ccv_nnc_tensor_free(actual);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(input);
	}
	_conform_data_format_restore_mfa_flag(old_flags);
}

TEST_CASE("MFA conform data format dispatch follows the current shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	const int count = 7 * 512;
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 512), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 512), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 512), 0);
	ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 512), 0);
	ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 512), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	_conform_data_format_fill(&dsfmt, input->data.f32, count);
	const ccv_nnc_cmd_t cpu_cmd = CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64);
	ccv_nnc_cmd_t mps_cmd = cpu_cmd;
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(input), TENSOR_LIST(expected), 0), "CPU E4M3 reference should execute for the larger shape");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input), TENSOR_LIST(gpu_input), 0);
	const int status = ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input), TENSOR_LIST(gpu_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(actual), 0);
	_conform_data_format_restore_mfa_flag(old_flags);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status, "MFA E4M3 conformance should execute for the larger shape");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, count, 1e-6, "MFA dispatch should cover every block of the current shape");
	ccv_nnc_tensor_free(gpu_output);
	ccv_nnc_tensor_free(gpu_input);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(input);
}

TEST_CASE("conform data format MPSGraph fallback matches CPU reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	const int count = 3 * 192;
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 192), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 192), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 192), 0);
	ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 192), 0);
	ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 192), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	_conform_data_format_fill(&dsfmt, input->data.f32, count);
	const ccv_nnc_cmd_t cpu_cmd = CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64);
	ccv_nnc_cmd_t mps_cmd = cpu_cmd;
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(input), TENSOR_LIST(expected), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input), TENSOR_LIST(gpu_input), 0);
	const int status = ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input), TENSOR_LIST(gpu_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(actual), 0);
	_conform_data_format_restore_mfa_flag(old_flags);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status, "MPSGraph E4M3 conformance should execute");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, count, 1e-4, "MPSGraph E4M3 conformance should match CPU reference");
	ccv_nnc_tensor_free(gpu_output);
	ccv_nnc_tensor_free(gpu_input);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(input);
}

TEST_CASE("conform data format backward and all-tail forward copy on Metal")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD, CCV_NNC_BACKEND_MPS));
	const int count = 2 * 128;
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* const actual_backward = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* const actual_forward = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 128), 0);
	ccv_nnc_tensor_t* const gpu_backward = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 128), 0);
	ccv_nnc_tensor_t* const gpu_forward = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4);
	_conform_data_format_fill(&dsfmt, input->data.f32, count);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input), TENSOR_LIST(gpu_input), 0);
	ccv_nnc_cmd_t backward = CMD_CONFORM_DATA_FORMAT_BACKWARD(CCV_NNC_FP8_E4M3, 64);
	backward.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_t all_tail = CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 128);
	all_tail.backend = CCV_NNC_BACKEND_MPS;
	const int backward_status = ccv_nnc_cmd_exec(backward, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input, 0, 0), TENSOR_LIST(gpu_backward), 0);
	const int forward_status = ccv_nnc_cmd_exec(all_tail, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input), TENSOR_LIST(gpu_forward), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_backward, gpu_forward), TENSOR_LIST(actual_backward, actual_forward), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, backward_status, "Metal E4M3 backward should execute");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, forward_status, "Metal all-tail E4M3 forward should execute");
	REQUIRE_ARRAY_EQ(float, input->data.f32, actual_backward->data.f32, count, "Metal E4M3 backward should pass gradients through unchanged");
	REQUIRE_ARRAY_EQ(float, input->data.f32, actual_forward->data.f32, count, "Metal E4M3 forward should preserve an all-tail tensor unchanged");
	ccv_nnc_tensor_free(gpu_forward);
	ccv_nnc_tensor_free(gpu_backward);
	ccv_nnc_tensor_free(gpu_input);
	ccv_nnc_tensor_free(actual_forward);
	ccv_nnc_tensor_free(actual_backward);
	ccv_nnc_tensor_free(input);
}

#include "case_main.h"
