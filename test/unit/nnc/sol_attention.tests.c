#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("Sol requires exactly Q K V and one output")
{
	ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(0.25, 0.5, 32, 0, 137);
	const uint64_t output_bitmask = 1;
	for (uint64_t input_bitmask = 0; input_bitmask < 16; input_bitmask++)
	{
		REQUIRE_EQ(ccv_nnc_cmd_bitmask(cmd, 3, 1, &input_bitmask, 1, &output_bitmask, 1), input_bitmask == 7, "require all three inputs");
		REQUIRE_EQ(ccv_nnc_cmd_bitmask(cmd, 4, 1, &input_bitmask, 1, &output_bitmask, 1), 0, "reject a fourth input slot");
	}
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 137, 2, 16), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, input->info, 0);
	ccv_nnc_tensor_t* const extra = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	for (int i = 0; i < 2; i++)
	{
		extra->data.i32[0] = i;
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(input, input, input, extra), TENSOR_LIST(output), 0), CCV_NNC_EXEC_INVALID, "reject removed dense override for either value");
	}
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(input, input), TENSOR_LIST(output), 0), CCV_NNC_EXEC_INVALID, "reject missing V");
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(input, 0, input), TENSOR_LIST(output), 0), CCV_NNC_EXEC_INVALID, "reject null K");
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(input, input, input), TENSOR_LIST(0), 0), CCV_NNC_EXEC_INVALID, "reject null output");
	ccv_nnc_tensor_free(extra);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(input);
}

TEST_CASE("CPU Sol all-exact matches dense attention and infers output shape")
{
	const int N = 2, T = 137, H = 2, D = 16, count = N * T * H * D;
	ccv_nnc_tensor_t* tensors[5];
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 42);
	for (int i = 0; i < 5; i++)
	{
		tensors[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, T, H, D), 0);
		if (i < 3)
			for (int j = 0; j < count; j++) tensors[i]->data.f32[j] = dsfmt_genrand_open_close(&rng) * 2 - 1;
	}
	ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(0.25, 0.5, 32, 0, 0);
	ccv_nnc_tensor_param_t output;
	ccv_nnc_tensor_param_t inputs[] = { tensors[0]->info, tensors[1]->info, tensors[2]->info };
	ccv_nnc_hint_tensor_auto(cmd, inputs, 3, ccv_nnc_no_hint, &output, 1);
	REQUIRE_ARRAY_EQ(int, output.dim, tensors[0]->info.dim, CCV_NNC_MAX_DIM_ALLOC, "Sol output shape matches Q");
	ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(0.25, 0), ccv_nnc_no_hint, 0, tensors, 3, &tensors[4], 1, 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, tensors, 3, &tensors[3], 1, 0), CCV_NNC_EXEC_SUCCESS, "CPU all-exact should run");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensors[3]->data.f32, tensors[4]->data.f32, count, 1e-5, "CPU all-exact equals dense");
	cmd.info.sol_attention.approximation_end = T + 1;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, tensors, 3, &tensors[3], 1, 0), CCV_NNC_EXEC_INVALID, "reject out-of-range interval");
	cmd.info.sol_attention.approximation_end = T;
	cmd.info.sol_attention.local_block_radius = 0;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, tensors, 3, &tensors[3], 1, 0), CCV_NNC_EXEC_INVALID, "reject zero radius");
	// The CPU oracle accepts FP32 only, including with an explicitly selected backend.
	ccv_nnc_tensor_t* const half_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, N, T, H, D), 0);
	ccv_nnc_tensor_t* const half_output = ccv_nnc_tensor_new(0, half_input->info, 0);
	ccv_nnc_tensor_zero(half_input);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	cmd.info.sol_attention.local_block_radius = 1;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(half_input, half_input, half_input), TENSOR_LIST(half_output), 0), CCV_NNC_EXEC_INVALID, "CPU Sol rejects FP16 tensors");
	ccv_nnc_tensor_free(half_input);
	ccv_nnc_tensor_free(half_output);
	for (int i = 0; i < 5; i++) ccv_nnc_tensor_free(tensors[i]);
}

TEST_CASE("CPU Sol summaries preserve uniform attention with protected boundaries and a partial block")
{
	const int T = 139, H = 2, D = 7;
	ccv_nnc_tensor_t* tensors[4];
	for (int i = 0; i < 4; i++) tensors[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, T, H, D), 0);
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 51);
	double means[14] = {};
	for (int j = 0; j < T * H * D; j++)
	{
		tensors[0]->data.f32[j] = dsfmt_genrand_open_close(&rng);
		tensors[1]->data.f32[j] = dsfmt_genrand_open_close(&rng);
		tensors[2]->data.f32[j] = (dsfmt_genrand_open_close(&rng) * 2 - 1) * 100;
		means[j % (H * D)] += tensors[2]->data.f32[j] / (double)T;
	}
	ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(0, 0.5, 16, 17, T - 3);
	cmd.info.sol_attention.query_block_size = 8;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, tensors, 3, &tensors[3], 1, 0), CCV_NNC_EXEC_SUCCESS, "CPU summary path should run");
	for (int j = 0; j < T * H * D; j++)
		REQUIRE(fabs(tensors[3]->data.f32[j] - means[j % (H * D)]) < 1e-5, "uniform attention preserves block multiplicity");
	for (int i = 0; i < 4; i++) ccv_nnc_tensor_free(tensors[i]);
}

#include "case_main.h"
