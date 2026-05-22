#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <sys/time.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static double benchmark_sdpa(ccv_nnc_cmd_t cmd, ccv_nnc_tensor_t* const q_tensor, ccv_nnc_tensor_t* const k_tensor, ccv_nnc_tensor_t* const v_tensor, ccv_nnc_tensor_t* const mask_tensor, ccv_nnc_tensor_t* const sinks_tensor, ccv_nnc_tensor_t* const o_tensor, const int warmup, const int iterations)
{
	int i;
	if (sinks_tensor)
		for (i = 0; i < warmup; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, mask_tensor, NULL, NULL, NULL, NULL, sinks_tensor), TENSOR_LIST(o_tensor, NULL), 0);
	else if (mask_tensor)
		for (i = 0; i < warmup; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, mask_tensor), TENSOR_LIST(o_tensor, NULL), 0);
	else
		for (i = 0; i < warmup; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
	double elapsed_time = get_current_time();
	if (sinks_tensor)
		for (i = 0; i < iterations; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, mask_tensor, NULL, NULL, NULL, NULL, sinks_tensor), TENSOR_LIST(o_tensor, NULL), 0);
	else if (mask_tensor)
		for (i = 0; i < iterations; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, mask_tensor), TENSOR_LIST(o_tensor, NULL), 0);
	else
		for (i = 0; i < iterations; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
	return (get_current_time() - elapsed_time) / (double)iterations;
}

static void benchmark_sdpa_nomask_case(const char* const backend, const int flags, const int B, const int R, const int C, const int Hq, const int Hk, const int D, const int is_causal, const int attention_sinks, const int warmup, const int iterations)
{
	const float scale = 1.0 / sqrt((float)D);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const sinks_tensor = attention_sinks ? ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, Hq), 0) : 0;
	int i;
	for (i = 0; i < B * R * Hq * D; ++i)
		q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
	for (i = 0; i < B * C * Hk * D; ++i)
		k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
	for (i = 0; i < B * C * Hk * D; ++i)
		v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
	if (attention_sinks)
		for (i = 0; i < Hq; ++i)
			sinks_tensor->data.f32[i] = (float)(((i * 7 + D) % 13) - 6) / 32;
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const sinks_tensor_f16 = attention_sinks ? ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, Hq), 0) : 0;
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);
	if (attention_sinks)
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(sinks_tensor), TENSOR_LIST(sinks_tensor_f16), 0);
	ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
	ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
	ccv_nnc_tensor_t* const gpu_sinks_tensor = attention_sinks ? ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, Hq), 0) : 0;
	ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
	if (attention_sinks)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(sinks_tensor_f16), TENSOR_LIST(gpu_sinks_tensor), 0);
	ccv_nnc_cmd_t attention = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
	attention.info.scaled_dot_product_attention.attention_sinks = attention_sinks;
	attention.info.scaled_dot_product_attention.flags = flags;
	const double seconds = benchmark_sdpa(attention, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, gpu_sinks_tensor, gpu_o_tensor, warmup, iterations);
	printf("%s,%s,%d,%d,%d,%d,%d,%d,%2.6f,%2.3f\n", backend, attention_sinks ? (is_causal ? "causal_sink" : "plain_sink") : (is_causal ? "causal" : "plain"), B, R, C, Hq, Hk, D, seconds, 1.0);
	if (gpu_sinks_tensor)
		ccv_nnc_tensor_free(gpu_sinks_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	if (sinks_tensor)
		ccv_nnc_tensor_free(sinks_tensor);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	if (sinks_tensor_f16)
		ccv_nnc_tensor_free(sinks_tensor_f16);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	int force_generic = 1;
	int trial_limit = -1;
	int small_r_grid = 0;
	int small_r_causal_grid = 0;
	int small_r_Hq = 32;
	int small_r_Hk = 32;
	int c_sweep = 0;
	int c_sweep_from = 128;
	int c_sweep_to = 8192;
	int c_sweep_step = 128;
	int c_sweep_B = 1;
	int c_sweep_R = 1;
	int c_sweep_Hq = 40;
	int c_sweep_Hk = 4;
	int c_sweep_D = 256;
	int c_sweep_causal = 1;
	int nomask_warmup = 5;
	int nomask_iterations = 40;
	int custom_case = 0;
	int custom_B = 1;
	int custom_R = 1;
	int custom_C = 4096;
	int custom_Hq = 32;
	int custom_Hk = 32;
	int custom_D = 256;
	int attention_sinks = 0;
	int attention_flags = 0;
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--na") == 0)
			force_generic = 0;
		else if (strcmp(argv[i], "--generic") == 0)
			force_generic = 1;
		else if (strcmp(argv[i], "--quick") == 0)
			trial_limit = 8;
		else if (strcmp(argv[i], "--sinks") == 0)
			attention_sinks = 1;
		else if (strcmp(argv[i], "--int8") == 0)
			attention_flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
		else if (strcmp(argv[i], "--small-r-grid") == 0)
			small_r_grid = 1;
		else if (strcmp(argv[i], "--small-r-causal-grid") == 0)
			small_r_causal_grid = 1;
		else if (strncmp(argv[i], "--small-r-hq=", 13) == 0)
			small_r_Hq = atoi(argv[i] + 13);
		else if (strncmp(argv[i], "--small-r-hk=", 13) == 0)
			small_r_Hk = atoi(argv[i] + 13);
		else if (strcmp(argv[i], "--c-sweep") == 0)
			c_sweep = 1;
		else if (strcmp(argv[i], "--c-sweep-plain") == 0)
			c_sweep_causal = 0;
		else if (strncmp(argv[i], "--c-from=", 9) == 0)
			c_sweep_from = atoi(argv[i] + 9);
		else if (strncmp(argv[i], "--c-to=", 7) == 0)
			c_sweep_to = atoi(argv[i] + 7);
		else if (strncmp(argv[i], "--c-step=", 9) == 0)
			c_sweep_step = atoi(argv[i] + 9);
		else if (strncmp(argv[i], "--sweep-b=", 10) == 0)
			c_sweep_B = atoi(argv[i] + 10);
		else if (strncmp(argv[i], "--sweep-r=", 10) == 0)
			c_sweep_R = atoi(argv[i] + 10);
		else if (strncmp(argv[i], "--sweep-hq=", 11) == 0)
			c_sweep_Hq = atoi(argv[i] + 11);
		else if (strncmp(argv[i], "--sweep-hk=", 11) == 0)
			c_sweep_Hk = atoi(argv[i] + 11);
		else if (strncmp(argv[i], "--sweep-d=", 10) == 0)
			c_sweep_D = atoi(argv[i] + 10);
		else if (strncmp(argv[i], "--warmup=", 9) == 0)
			nomask_warmup = atoi(argv[i] + 9);
		else if (strncmp(argv[i], "--iterations=", 13) == 0)
			nomask_iterations = atoi(argv[i] + 13);
		else if (strncmp(argv[i], "--trials=", 9) == 0)
			trial_limit = atoi(argv[i] + 9);
		else if (strcmp(argv[i], "--case") == 0 && i + 6 < argc)
		{
			custom_case = 1;
			custom_B = atoi(argv[++i]);
			custom_R = atoi(argv[++i]);
			custom_C = atoi(argv[++i]);
			custom_Hq = atoi(argv[++i]);
			custom_Hk = atoi(argv[++i]);
			custom_D = atoi(argv[++i]);
		}
	}
	const uint64_t old_flags = ccv_nnc_flags();
	if (force_generic)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (c_sweep)
	{
		printf("backend,mode,B,R,C,Hq,Hk,D,seconds_per_iter,relative_to_plain\n");
		if (c_sweep_step <= 0)
			c_sweep_step = 128;
		for (int C = c_sweep_from; C <= c_sweep_to; C += c_sweep_step)
			benchmark_sdpa_nomask_case(force_generic ? "generic" : "default", attention_flags, c_sweep_B, c_sweep_R, C, c_sweep_Hq, c_sweep_Hk, c_sweep_D, c_sweep_causal, attention_sinks, nomask_warmup, nomask_iterations);
		if (force_generic && !(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		return 0;
	}
	if (small_r_grid || small_r_causal_grid)
	{
		printf("backend,mode,B,R,C,Hq,Hk,D,seconds_per_iter,relative_to_plain\n");
		const int R_values[] = { 1, 2, 3, 5, 7, 8, 15, 16, 32, 64 };
		const int C_values[] = { 2048, 4096, 8192, 16384 };
		const int D_values[] = { 64, 128, 256 };
		for (int d_idx = 0; d_idx < (int)(sizeof(D_values) / sizeof(D_values[0])); ++d_idx)
			for (int c_idx = 0; c_idx < (int)(sizeof(C_values) / sizeof(C_values[0])); ++c_idx)
				for (int r_idx = 0; r_idx < (int)(sizeof(R_values) / sizeof(R_values[0])); ++r_idx)
					benchmark_sdpa_nomask_case(force_generic ? "generic" : "default", attention_flags, 1, R_values[r_idx], C_values[c_idx], small_r_Hq, small_r_Hk, D_values[d_idx], small_r_causal_grid, attention_sinks, nomask_warmup, nomask_iterations);
		if (force_generic && !(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		return 0;
	}
	// Bypass error: variable-sized object may not be initialized
#define num_trials 18
	int B_candidates[num_trials] = {  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int R_candidates[num_trials] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 1024, 2048, 4096, 1024, 2048, 3072, 4096, 6144, 8192, 16384  };
	int C_candidates[num_trials] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 1024, 2048, 4096, 1024, 2048, 3072, 4096, 6144, 8192, 16384 };
	int Hq_candidates[num_trials] = {   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
	int Hk_candidates[num_trials] = {   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
	int D_candidates[num_trials] = {  64, 64, 64, 64, 64, 64, 64, 64, 80, 80, 80, 128, 128, 128, 128, 128, 128, 128 };
	if (custom_case)
		trial_limit = 1;
	else if (trial_limit < 0 || trial_limit > num_trials)
		trial_limit = num_trials;
	printf("backend,mode,B,R,C,Hq,Hk,D,seconds_per_iter,relative_to_plain\n");

	for (int trial = 0; trial < trial_limit; ++trial) {
		int B = custom_case ? custom_B : B_candidates[trial];
		int R = custom_case ? custom_R : R_candidates[trial];
		int C = custom_case ? custom_C : C_candidates[trial];
		int Hq = custom_case ? custom_Hq : Hq_candidates[trial];
		int Hk = custom_case ? custom_Hk : Hk_candidates[trial];
		int D = custom_case ? custom_D : D_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const sinks_tensor = attention_sinks ? ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, Hq), 0) : 0;

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		if (attention_sinks)
			for (i = 0; i < Hq; ++i)
				sinks_tensor->data.f32[i] = (float)(((i * 7 + D) % 13) - 6) / 32;

		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const sinks_tensor_f16 = attention_sinks ? ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, Hq), 0) : 0;
		ccv_nnc_tensor_t* const triangular_mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
		ccv_nnc_tensor_t* const triangular_mask_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
		for (i = 0; i < R; i++)
			for (int j = 0; j < C; j++)
				triangular_mask_tensor->data.f32[i * C + j] = (j <= i + C - R) ? 0 : -FLT_MAX;
		ccv_float_to_half_precision(triangular_mask_tensor->data.f32, (uint16_t*)triangular_mask_tensor_f16->data.f16, R * C);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);
		if (attention_sinks)
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(sinks_tensor), TENSOR_LIST(sinks_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_sinks_tensor = attention_sinks ? ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, Hq), 0) : 0;
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_triangular_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, triangular_mask_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_triangular_mask_tensor), 0);
		if (attention_sinks)
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(sinks_tensor_f16), TENSOR_LIST(gpu_sinks_tensor), 0);

		ccv_nnc_cmd_t plain_attention = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
		ccv_nnc_cmd_t causal_attention = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 1);
		plain_attention.info.scaled_dot_product_attention.attention_sinks = attention_sinks;
		causal_attention.info.scaled_dot_product_attention.attention_sinks = attention_sinks;
		plain_attention.info.scaled_dot_product_attention.flags = attention_flags;
		causal_attention.info.scaled_dot_product_attention.flags = attention_flags;
		const double plain_seconds = benchmark_sdpa(plain_attention, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, gpu_sinks_tensor, gpu_o_tensor, nomask_warmup, nomask_iterations);
		const double causal_seconds = benchmark_sdpa(causal_attention, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, gpu_sinks_tensor, gpu_o_tensor, nomask_warmup, nomask_iterations);
		const double triangular_mask_seconds = benchmark_sdpa(plain_attention, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_triangular_mask_tensor, gpu_sinks_tensor, gpu_o_tensor, nomask_warmup, nomask_iterations);
		printf("%s,%s,%d,%d,%d,%d,%d,%d,%2.6f,%2.3f\n", force_generic ? "generic" : "default", attention_sinks ? "plain_sink" : "plain", B, R, C, Hq, Hk, D, plain_seconds, 1.0);
		printf("%s,%s,%d,%d,%d,%d,%d,%d,%2.6f,%2.3f\n", force_generic ? "generic" : "default", attention_sinks ? "causal_sink" : "causal", B, R, C, Hq, Hk, D, causal_seconds, causal_seconds / plain_seconds);
		printf("%s,%s,%d,%d,%d,%d,%d,%d,%2.6f,%2.3f\n", force_generic ? "generic" : "default", attention_sinks ? "triangular_mask_sink" : "triangular_mask", B, R, C, Hq, Hk, D, triangular_mask_seconds, triangular_mask_seconds / plain_seconds);

		ccv_nnc_tensor_free(gpu_triangular_mask_tensor);
		if (gpu_sinks_tensor)
			ccv_nnc_tensor_free(gpu_sinks_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		if (sinks_tensor)
			ccv_nnc_tensor_free(sinks_tensor);
		ccv_nnc_tensor_free(triangular_mask_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		if (sinks_tensor_f16)
			ccv_nnc_tensor_free(sinks_tensor_f16);
		ccv_nnc_tensor_free(triangular_mask_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
	}
#undef num_trials
	if (force_generic && !(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	return 0;
}
