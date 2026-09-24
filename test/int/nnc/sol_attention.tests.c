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

TEST_CASE("MPS Sol rejects missing inputs and the removed dense override")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(1, 0.5, 64, 0, 257);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 257, 2, 128), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, input->info, 0);
	ccv_nnc_tensor_t* const extra = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
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

TEST_CASE("int8 Sol attention matches CPU for pooling, signed scales, tails and batches")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	extern uint8_t ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_mfa_context_t* context);
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	const int shapes[][5] = { {2, 257, 64, 64, 2}, {1, 197, 32, 16, 2}, {1, 133, 16, 16, 2}, {1, 389, 64, 32, 2}, {1, 129, 64, 16, 3} };
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 42);
	for (int c = 0; c < 5; c++)
	{
		const int N = shapes[c][0], T = shapes[c][1], H = shapes[c][4], count = N * T * H * 128;
		ccv_nnc_tensor_t* cpu[4];
		ccv_nnc_tensor_t* half[4];
		ccv_nnc_tensor_t* gpu[4];
		for (int i = 0; i < 4; i++)
		{
			cpu[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, T, H, 128), 0);
			half[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, N, T, H, 128), 0);
			gpu[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, N, T, H, 128), 0);
			if (i < 3)
			{
				for (int j = 0; j < count; j++)
					cpu[i]->data.f32[j] = (dsfmt_genrand_open_close(&rng) * 2 - 1) * (i == 0 ? 0.3 : 3);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu[i]), TENSOR_LIST(half[i]), 0);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(half[i]), TENSOR_LIST(cpu[i]), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(half[i]), TENSOR_LIST(gpu[i]), 0);
			}
		}
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, cpu[3]->info, 0);
		const float scales[] = { 1, -0.125, 0 };
		for (int k = 0; k < 3; k++)
		{
			ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(scales[k], 0.5, shapes[c][2], 17, T - 7);
			cmd.info.sol_attention.query_block_size = shapes[c][3];
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, cpu, 3, &cpu[3], 1, 0), CCV_NNC_EXEC_SUCCESS, "CPU Sol should run");
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, gpu, 3, &gpu[3], 1, 0), CCV_NNC_EXEC_SUCCESS, "INT8 Sol should run");
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[3]), TENSOR_LIST(half[3]), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(half[3]), TENSOR_LIST(actual), 0);
			double error = 0, norm = 0;
			for (int j = 0; j < count; j++)
			{
				REQUIRE(isfinite(actual->data.f32[j]), "Sol output must be finite");
				error += (double)(actual->data.f32[j] - cpu[3]->data.f32[j]) * (actual->data.f32[j] - cpu[3]->data.f32[j]);
				norm += (double)cpu[3]->data.f32[j] * cpu[3]->data.f32[j];
			}
			REQUIRE(sqrt(error / fmax(norm, 1e-30)) < 0.05, "INT8 Sol should match independent CPU reference (case %d scale %g error %g)", c, scales[k], sqrt(error / fmax(norm, 1e-30)));
		}
		ccv_nnc_tensor_free(actual);
		for (int i = 0; i < 4; i++)
		{
			ccv_nnc_tensor_free(cpu[i]);
			ccv_nnc_tensor_free(half[i]);
			ccv_nnc_tensor_free(gpu[i]);
		}
	}
}

TEST_CASE("int8 Sol all-exact matches native attention across cache shapes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	extern uint8_t ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_mfa_context_t* context);
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 17);
	// Cross both the mean-reduction and synchronization cutoffs, including one-row tails.
	const int lengths[] = { 513, 20501, 32769, 133 };
	for (int shape = 0; shape < 4; shape++)
	{
		const int T = lengths[shape], H = 2, count = T * H * 128;
		ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, T, H, 128), 0);
		ccv_nnc_tensor_t* const staging = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, T, H, 128), 0);
		ccv_nnc_tensor_t* gpu[5];
		for (int i = 0; i < 5; i++)
		{
			gpu[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, T, H, 128), 0);
			if (i < 3)
			{
				for (int j = 0; j < count; j++)
					host->data.f32[j] = (dsfmt_genrand_open_close(&rng) * 2 - 1) * (i == 0 ? 0.3 : 3) + (i == 2 ? 64 : 0);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(staging), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(gpu[i]), 0);
			}
		}
		ccv_nnc_cmd_t dense = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1, 0);
		dense.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
		ccv_nnc_cmd_exec(dense, ccv_nnc_no_hint, 0, gpu, 3, &gpu[4], 1, 0);
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, staging->info, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[4]), TENSOR_LIST(expected), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected), TENSOR_LIST(host), 0);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host->info, 0);
		ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(1, 0.5, 64, 0, T);
		cmd.info.sol_attention.local_block_radius = (T + 63) / 64;
		for (int trial = 0; trial < (shape == 0 ? 5 : (shape == 2 ? 10 : 1)); trial++)
		{
			// The final short-shape trial also exercises the empty-span all-exact control.
			if (shape == 0 && trial == 4) { cmd.info.sol_attention.approximation_end = 0; cmd.info.sol_attention.local_block_radius = 1; }
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[0], gpu[1], gpu[2]), TENSOR_LIST(gpu[3]), 0), CCV_NNC_EXEC_SUCCESS, "controlled Sol should run");
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[3]), TENSOR_LIST(staging), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(actual), 0);
			double error = 0, norm = 0;
			for (int j = 0; j < count; j++)
			{
				REQUIRE(isfinite(actual->data.f32[j]), "all-exact output must be finite (T=%d index=%d)", T, j);
				error += (double)(actual->data.f32[j] - host->data.f32[j]) * (actual->data.f32[j] - host->data.f32[j]);
				norm += (double)host->data.f32[j] * host->data.f32[j];
			}
			REQUIRE(sqrt(error / norm) < 1e-4, "all-exact Sol must match native INT8 (T=%d error=%g)", T, sqrt(error / norm));
		}
		ccv_nnc_tensor_free(actual);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(staging);
		ccv_nnc_tensor_free(host);
		for (int i = 0; i < 5; i++) ccv_nnc_tensor_free(gpu[i]);
	}
}

TEST_CASE("Sol zero logits preserve multiplicity with large centered values")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int T = 517, H = 2, count = T * H * 128;
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, T, H, 128), 0);
	ccv_nnc_tensor_t* const staging = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, T, H, 128), 0);
	ccv_nnc_tensor_t* gpu[4];
	for (int i = 0; i < 4; i++)
	{
		gpu[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, T, H, 128), 0);
		if (i < 3)
		{
			for (int j = 0; j < count; j++)
				host->data.f32[j] = i < 2 ? 0 : ((j / (H * 128) < 3 * T / 4 ? 60000 : -60000) * ((j / 128) % H ? -1 : 1));
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(staging), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(gpu[i]), 0);
		}
	}
	ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(0, 0.5, 64, 17, T - 9);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, gpu, 3, &gpu[3], 1, 0), CCV_NNC_EXEC_SUCCESS, "zero-logit Sol should run");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[3]), TENSOR_LIST(staging), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(host), 0);
	const float mean = 60000.0 * (2 * (3 * T / 4) - T) / T;
	for (int j = 0; j < count; j++)
	{
		const float expected = mean * ((j / 128) % H ? -1 : 1);
		REQUIRE(isfinite(host->data.f32[j]) && fabsf(host->data.f32[j] - expected) / fabsf(expected) < 0.015, "summary multiplicity and centered V should preserve uniform attention");
	}
	for (int i = 0; i < 4; i++) ccv_nnc_tensor_free(gpu[i]);
	ccv_nnc_tensor_free(host);
	ccv_nnc_tensor_free(staging);
}

TEST_CASE("Sol contiguous offset views preserve output sentinels")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int T = 513, H = 2, count = (T + 2) * H * 128;
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, T + 2, H, 128), 0);
	ccv_nnc_tensor_t* const staging = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, T + 2, H, 128), 0);
	ccv_nnc_tensor_t* backing[4];
	ccv_nnc_tensor_t* plain[4];
	ccv_nnc_tensor_view_t* views[4];
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 73);
	for (int i = 0; i < 4; i++)
	{
		backing[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, T + 2, H, 128), 0);
		plain[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, T, H, 128), 0);
		views[i] = ccv_nnc_tensor_view_new(backing[i], plain[i]->info, DIM_ALLOC(0, 1, 0, 0), DIM_ALLOC((T + 2) * H * 128, H * 128, 128, 1));
		for (int j = 0; j < count; j++) host->data.f32[j] = i == 3 ? 9 : (dsfmt_genrand_open_close(&rng) * 2 - 1) * (i == 0 ? 0.3 : 3);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(staging), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(backing[i]), 0);
		if (i < 3)
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)views[i]), TENSOR_LIST(plain[i]), 0);
	}
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, T, H, 128), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, actual->info, 0);
	ccv_nnc_cmd_t sol = CMD_SOL_ATTENTION_FORWARD(1, 0.5, 64, 17, T - 9);
	for (int trial = 0; trial < 3; trial++)
	{
		REQUIRE_EQ(ccv_nnc_cmd_exec(sol, ccv_nnc_no_hint, 0, plain, 3, &plain[3], 1, 0), CCV_NNC_EXEC_SUCCESS, "contiguous reference should run");
		REQUIRE_EQ(ccv_nnc_cmd_exec(sol, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)views[0], (ccv_nnc_tensor_t*)views[1], (ccv_nnc_tensor_t*)views[2]), TENSOR_LIST((ccv_nnc_tensor_t*)views[3]), 0), CCV_NNC_EXEC_SUCCESS, "offset Sol should run");
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)views[3], plain[3]), TENSOR_LIST(actual, expected), 0);
		REQUIRE(memcmp(actual->data.f16, expected->data.f16, T * H * 128 * 2) == 0, "offset and contiguous results should be identical");
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(backing[3]), TENSOR_LIST(staging), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(host), 0);
		for (int j = 0; j < H * 128; j++)
		{
			REQUIRE_EQ(host->data.f32[j], 9, "output prefix must be untouched");
			REQUIRE_EQ(host->data.f32[(T + 1) * H * 128 + j], 9, "output suffix must be untouched");
		}
	}
	// Backend validation must reject unsupported query tiles even if dimensions fit.
	sol.info.sol_attention.query_block_size = 48;
	REQUIRE_EQ(ccv_nnc_cmd_exec(sol, ccv_nnc_no_hint, 0, plain, 3, &plain[3], 1, 0), CCV_NNC_EXEC_INVALID, "reject unsupported query tile");
	for (int i = 0; i < 4; i++)
	{
		ccv_nnc_tensor_view_free(views[i]);
		ccv_nnc_tensor_free(backing[i]);
		ccv_nnc_tensor_free(plain[i]);
	}
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(host);
	ccv_nnc_tensor_free(staging);
}

TEST_CASE("non-NA Sol attention matches the FP32 CPU oracle across shapes, spans and scales")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	// Increasing and decreasing shapes share the default MFA context, exercising
	// pipeline constants and runtime dispatch independently of the library cache.
	const struct { int n, t, h, begin, end, b, qb, radius; float scale, v_shift; } cases[] = {
		{1, 257, 2, 0, 257, 64, 64, 1, 1, 0},
		{1, 1025, 3, 95, 1018, 64, 64, 1, 1, 0},
		{2, 513, 2, 71, 491, 64, 64, 1, 1, 0},
		{1, 193, 3, 0, 0, 64, 64, 1, 1, 0},
		{1, 513, 2, 0, 513, 64, 64, 1, -1, 0},
		{1, 321, 2, 0, 321, 64, 64, 1, 0, 0},
		{1, 257, 2, 0, 257, 64, 64, 1, 1, 1000},
		{1, 385, 2, 63, 350, 64, 16, 1, 1, 0},
		{1, 257, 2, 0, 257, 32, 32, 1, 1, 0},
		{1, 129, 2, 0, 129, 16, 16, 1, 1, 0},
		{1, 513, 2, 71, 491, 64, 64, 9, 1, 0},
		{1, 7, 2, 0, 7, 64, 64, 1, 1, 0},
	};
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 314);
	int c;
	for (c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
	{
		const int n = cases[c].n, t = cases[c].t, h = cases[c].h, count = n * t * h * 128;
		ccv_nnc_tensor_t* qkv[3];
		ccv_nnc_tensor_t* cpu[3];
		ccv_nnc_tensor_t* gpu[3];
		int i, j;
		for (i = 0; i < 3; i++)
		{
			qkv[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, n, t, h, 128), 0);
			gpu[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, n, t, h, 128), 0);
			cpu[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, t, h, 128), 0);
			for (j = 0; j < count; j++)
				((_Float16*)qkv[i]->data.f16)[j] = (dsfmt_genrand_open_close(&rng) * 2 - 1) * (i == 0 ? 0.25 : 1) + (i == 2 ? cases[c].v_shift : 0);
		}
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, qkv, 3, cpu, 3, 0);
		ccv_nnc_tensor_t* ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, t, h, 128), 0);
		ccv_nnc_tensor_t* got = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, n, t, h, 128), 0);
		ccv_nnc_tensor_t* out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, n, t, h, 128), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, qkv, 3, gpu, 3, 0);
		ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(cases[c].scale, 0.5, cases[c].b, cases[c].begin, cases[c].end);
		cmd.info.sol_attention.query_block_size = cases[c].qb;
		cmd.info.sol_attention.local_block_radius = cases[c].radius;
		cmd.backend = CCV_NNC_BACKEND_CPU_REF;
		REQUIRE(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, cpu, 3, &ref, 1, 0) == CCV_NNC_EXEC_SUCCESS, "CPU Sol should execute");
		cmd.backend = CCV_NNC_BACKEND_MPS;
		REQUIRE(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, gpu, 3, &out, 1, 0) == CCV_NNC_EXEC_SUCCESS, "non-NA Sol should execute");
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &out, 1, &got, 1, 0);
		double energy = 0, error = 0;
		for (j = 0; j < count; j++)
		{
			const float a = ref->data.f32[j], b = ((_Float16*)got->data.f16)[j];
			REQUIRE(isfinite(b), "Sol output should be finite");
			const float delta = a - b;
			energy += (double)a * a;
			error += (double)delta * delta;
			REQUIRE(fabsf(delta) / fmaxf(1, fmaxf(fabsf(a), fabsf(b))) < 0.005, "Sol should match CPU including tails and protected interactions");
		}
		REQUIRE(sqrt(error / fmax(energy, 1e-30)) < 0.01, "Sol relative L2 should match CPU");
		for (i = 0; i < 3; i++) { ccv_nnc_tensor_free(qkv[i]); ccv_nnc_tensor_free(cpu[i]); ccv_nnc_tensor_free(gpu[i]); }
		ccv_nnc_tensor_free(ref); ccv_nnc_tensor_free(got); ccv_nnc_tensor_free(out);
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

TEST_CASE("non-NA Sol long zero-scale traversal stays uniform across the routing threshold")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 529);
	const int lengths[] = {32767, 32769};
	for (int shape = 0; shape < 2; shape++)
	{
		const int T = lengths[shape], H = 2, count = T * H * 128;
		ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, T, H, 128), 0);
		ccv_nnc_tensor_t* const staging = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, T, H, 128), 0);
		ccv_nnc_tensor_t* const mean = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, H, 128), 0);
		ccv_nnc_tensor_t* gpu[4];
		for (int i = 0; i < 4; i++)
		{
			gpu[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, T, H, 128), 0);
			if (i < 3)
			{
				for (int j = 0; j < count; j++)
					host->data.f32[j] = (dsfmt_genrand_open_close(&rng) * 2 - 1) * (i == 0 ? 0.25 : 3);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(staging), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(gpu[i]), 0);
			}
		}
		// Zero scale makes attention the mean of V. Compute that independently
		// on CPU from the actual FP16 inputs, without quadratic CPU attention.
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(host), 0);
		ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(mean), 0);
		ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(0, 0.5, 64, 0, T);
		// Scale zero keeps all-exact queries in the mixed kernel. This traverses
		// every route bit, including the last partial word and partial query tile.
		cmd.info.sol_attention.local_block_radius = (T + 63) / 64;
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, gpu, 3, &gpu[3], 1, 0), CCV_NNC_EXEC_SUCCESS, "long zero-scale Sol should run");
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[3]), TENSOR_LIST(staging), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(host), 0);
		for (int t = 0; t < T; t++)
		{
			for (int j = 0; j < H * 128; j++)
				REQUIRE(isfinite(host->data.f32[t * H * 128 + j]), "long zero-scale Sol output must be finite");
			REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, host->data.f32 + t * H * 128, mean->data.f32, H * 128, 1e-4, "zero-scale Sol should broadcast the CPU V mean");
		}
		for (int i = 0; i < 4; i++) ccv_nnc_tensor_free(gpu[i]);
		ccv_nnc_tensor_free(host);
		ccv_nnc_tensor_free(staging);
		ccv_nnc_tensor_free(mean);
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

TEST_CASE("non-NA Sol all-exact matches native attention across long tail cache shapes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	dsfmt_t rng;
	dsfmt_init_gen_rand(&rng, 17);
	// Cross the bit-routing cutoff, including the real H3 length and one-row tails.
	const int lengths[] = { 513, 10423, 32769, 133 };
	for (int shape = 0; shape < 4; shape++)
	{
		const int T = lengths[shape], H = 2, count = T * H * 128;
		ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, T, H, 128), 0);
		ccv_nnc_tensor_t* const staging = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, T, H, 128), 0);
		ccv_nnc_tensor_t* gpu[5];
		for (int i = 0; i < 5; i++)
		{
			gpu[i] = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, T, H, 128), 0);
			if (i < 3)
			{
				for (int j = 0; j < count; j++)
					host->data.f32[j] = (dsfmt_genrand_open_close(&rng) * 2 - 1) * (i == 0 ? 0.3 : 3) + (i == 2 && shape == 0 ? 64 : 0);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(staging), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(gpu[i]), 0);
			}
		}
		ccv_nnc_cmd_t dense = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1, 0);
		dense.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;
		REQUIRE_EQ(ccv_nnc_cmd_exec(dense, ccv_nnc_no_hint, 0, gpu, 3, &gpu[4], 1, 0), CCV_NNC_EXEC_SUCCESS, "native FP16 reference should run");
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, staging->info, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[4]), TENSOR_LIST(expected), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected), TENSOR_LIST(host), 0);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host->info, 0);
		ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(1, 0.5, 64, 0, T);
		cmd.info.sol_attention.local_block_radius = (T + 63) / 64;
		for (int trial = 0; trial < (shape == 0 ? 5 : (shape == 2 ? 10 : 1)); trial++)
		{
			// The final short-shape trial also exercises the empty-span all-exact control.
			if (shape == 0 && trial == 4) { cmd.info.sol_attention.approximation_end = 0; cmd.info.sol_attention.local_block_radius = 1; }
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[0], gpu[1], gpu[2]), TENSOR_LIST(gpu[3]), 0), CCV_NNC_EXEC_SUCCESS, "controlled Sol should run");
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu[3]), TENSOR_LIST(staging), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(staging), TENSOR_LIST(actual), 0);
			double error = 0, norm = 0;
			for (int j = 0; j < count; j++)
			{
				REQUIRE(isfinite(actual->data.f32[j]), "all-exact output must be finite (T=%d index=%d)", T, j);
				error += (double)(actual->data.f32[j] - host->data.f32[j]) * (actual->data.f32[j] - host->data.f32[j]);
				norm += (double)host->data.f32[j] * host->data.f32[j];
			}
			REQUIRE(sqrt(error / fmax(norm, 1e-30)) < 1e-4, "all-exact Sol must match native FP16 (T=%d error=%g)", T, sqrt(error / fmax(norm, 1e-30)));
		}
		ccv_nnc_tensor_free(actual);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(staging);
		ccv_nnc_tensor_free(host);
		for (int i = 0; i < 5; i++) ccv_nnc_tensor_free(gpu[i]);
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

#include "case_main.h"
