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

static double benchmark_sdpa(ccv_nnc_cmd_t cmd, ccv_nnc_tensor_t* const q_tensor, ccv_nnc_tensor_t* const k_tensor, ccv_nnc_tensor_t* const v_tensor, ccv_nnc_tensor_t* const mask_tensor, ccv_nnc_tensor_t* const o_tensor, const int warmup, const int iterations)
{
	int i;
	if (mask_tensor)
		for (i = 0; i < warmup; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, mask_tensor), TENSOR_LIST(o_tensor, NULL), 0);
	else
		for (i = 0; i < warmup; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
	double elapsed_time = get_current_time();
	if (mask_tensor)
		for (i = 0; i < iterations; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, mask_tensor), TENSOR_LIST(o_tensor, NULL), 0);
	else
		for (i = 0; i < iterations; i++)
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
	return (get_current_time() - elapsed_time) / (double)iterations;
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	int force_generic = 1;
	int trial_limit = -1;
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--na") == 0)
			force_generic = 0;
		else if (strcmp(argv[i], "--generic") == 0)
			force_generic = 1;
		else if (strcmp(argv[i], "--quick") == 0)
			trial_limit = 8;
		else if (strncmp(argv[i], "--trials=", 9) == 0)
			trial_limit = atoi(argv[i] + 9);
	}
	const uint64_t old_flags = ccv_nnc_flags();
	if (force_generic)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	// Bypass error: variable-sized object may not be initialized
#define num_trials 18
	int B_candidates[num_trials] = {  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int R_candidates[num_trials] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 1024, 2048, 4096, 1024, 2048, 3072, 4096, 6144, 8192, 16384  };
	int C_candidates[num_trials] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 1024, 2048, 4096, 1024, 2048, 3072, 4096, 6144, 8192, 16384 };
	int Hq_candidates[num_trials] = {   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
	int Hk_candidates[num_trials] = {   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
	int D_candidates[num_trials] = {  64, 64, 64, 64, 64, 64, 64, 64, 80, 80, 80, 128, 128, 128, 128, 128, 128, 128 };
	if (trial_limit < 0 || trial_limit > num_trials)
		trial_limit = num_trials;
	printf("backend,mode,B,R,C,Hq,Hk,D,seconds_per_iter,relative_to_plain\n");

	for (int trial = 0; trial < trial_limit; ++trial) {
		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}

		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const triangular_mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
		ccv_nnc_tensor_t* const triangular_mask_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
		for (i = 0; i < R; i++)
			for (int j = 0; j < C; j++)
				triangular_mask_tensor->data.f32[i * C + j] = (j <= i + C - R) ? 0 : -FLT_MAX;
		ccv_float_to_half_precision(triangular_mask_tensor->data.f32, (uint16_t*)triangular_mask_tensor_f16->data.f16, R * C);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_triangular_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, triangular_mask_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_triangular_mask_tensor), 0);

		const ccv_nnc_cmd_t plain_attention = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
		const ccv_nnc_cmd_t causal_attention = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 1);
		const double plain_seconds = benchmark_sdpa(plain_attention, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, gpu_o_tensor, 5, 40);
		const double causal_seconds = benchmark_sdpa(causal_attention, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, gpu_o_tensor, 5, 40);
		const double triangular_mask_seconds = benchmark_sdpa(plain_attention, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_triangular_mask_tensor, gpu_o_tensor, 5, 40);
		printf("%s,plain,%d,%d,%d,%d,%d,%d,%2.6f,%2.3f\n", force_generic ? "generic" : "default", B, R, C, Hq, Hk, D, plain_seconds, 1.0);
		printf("%s,causal,%d,%d,%d,%d,%d,%d,%2.6f,%2.3f\n", force_generic ? "generic" : "default", B, R, C, Hq, Hk, D, causal_seconds, causal_seconds / plain_seconds);
		printf("%s,triangular_mask,%d,%d,%d,%d,%d,%d,%2.6f,%2.3f\n", force_generic ? "generic" : "default", B, R, C, Hq, Hk, D, triangular_mask_seconds, triangular_mask_seconds / plain_seconds);

		ccv_nnc_tensor_free(gpu_triangular_mask_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(triangular_mask_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
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
