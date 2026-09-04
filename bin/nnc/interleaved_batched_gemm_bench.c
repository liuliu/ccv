#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
	TOKEN_DIM = 4096,
	GROUP_DIM = 8,
	N_DIM = 1024,
	K_DIM = 4096,
};

static double get_current_time(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int compare_double(const void* const a, const void* const b)
{
	const double va = *(const double*)a;
	const double vb = *(const double*)b;
	return (va > vb) - (va < vb);
}

static void benchmark_case(const char* const name, const int interleaved, const int rowwise, const int disable_nax, const int warmup, const int iterations)
{
	const ccv_nnc_tensor_param_t a_params = interleaved ?
		GPU_TENSOR_NHWC(000, 16F, TOKEN_DIM, GROUP_DIM, 1, K_DIM) :
		GPU_TENSOR_NHWC(000, 16F, GROUP_DIM, TOKEN_DIM, K_DIM);
	const ccv_nnc_tensor_param_t w_dense_params = GPU_TENSOR_NHWC(000, 16F, GROUP_DIM, N_DIM, K_DIM);
	const ccv_nnc_tensor_param_t w_params = rowwise ? ccv_nnc_tensor_8i_rowwise(w_dense_params) : w_dense_params;
	const ccv_nnc_tensor_param_t b_params = interleaved ?
		GPU_TENSOR_NHWC(000, 16F, TOKEN_DIM, GROUP_DIM, 1, N_DIM) :
		GPU_TENSOR_NHWC(000, 16F, GROUP_DIM, TOKEN_DIM, N_DIM);
	const ccv_nnc_tensor_param_t hb_params = interleaved ?
		CPU_TENSOR_NHWC(16F, TOKEN_DIM, GROUP_DIM, 1, N_DIM) :
		CPU_TENSOR_NHWC(16F, GROUP_DIM, TOKEN_DIM, N_DIM);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, hb_params, 0);
	ccv_nnc_tensor_t* const hw = rowwise ? ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16F, GROUP_DIM, N_DIM, K_DIM)), 0) : 0;
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	double* const samples = (double*)malloc(sizeof(double) * iterations);
	if (!a || !w || !b || !hb || !stream || !samples || (rowwise && !hw))
	{
		fprintf(stderr, "%s: allocation failed\n", name);
		exit(1);
	}
	if (disable_nax)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	int status = ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), stream);
	if (rowwise)
	{
		// Construct the resident int8 payload and FP16 scales directly, without IQ2 quantization.
		const size_t weight_count = (size_t)GROUP_DIM * N_DIM * K_DIM;
		const size_t scale_offset = (weight_count + 127) & ~(size_t)127;
		const size_t row_count = (size_t)GROUP_DIM * N_DIM;
		const float scale = 1.0f / 128.0f;
		uint16_t scale_f16;
		ccv_float_to_half_precision(&scale, &scale_f16, 1);
		memset(hw->data.u8, 1, weight_count);
		uint16_t* const scales = (uint16_t*)(hw->data.u8 + scale_offset);
		size_t i;
		for (i = 0; i < row_count; i++)
			scales[i] = scale_f16;
		if (status == CCV_NNC_EXEC_SUCCESS)
			status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(w), stream);
	} else if (status == CCV_NNC_EXEC_SUCCESS) {
		status = ccv_nnc_cmd_exec(CMD_SET_FORWARD(1.0f / 128.0f), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(w), stream);
	}
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS)
	{
		fprintf(stderr, "%s: setup failed with status %d\n", name, status);
		exit(1);
	}
	const ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	int i;
	for (i = 0; i < warmup; i++)
	{
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), stream);
		ccv_nnc_stream_context_wait(stream);
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "%s: warmup failed with status %d\n", name, status);
			exit(1);
		}
	}
	double total_ms = 0;
	for (i = 0; i < iterations; i++)
	{
		ccv_nnc_stream_context_wait(stream);
		const double start = get_current_time();
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), stream);
		ccv_nnc_stream_context_wait(stream);
		const double elapsed_ms = (get_current_time() - start) * 1000;
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "%s: iteration failed with status %d\n", name, status);
			exit(1);
		}
		samples[i] = elapsed_ms;
		total_ms += elapsed_ms;
	}
	qsort(samples, iterations, sizeof(double), compare_double);
	const double median_ms = (iterations & 1) ? samples[iterations / 2] : 0.5 * (samples[iterations / 2 - 1] + samples[iterations / 2]);
	const double tflops = (2.0 * TOKEN_DIM * GROUP_DIM * N_DIM * K_DIM) / (median_ms * 1e9);
	status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS)
	{
		fprintf(stderr, "%s: output transfer failed with status %d\n", name, status);
		exit(1);
	}
	// Smoke-check each group's output corners; nonuniform stride coverage lives in mpsblas.tests.
	float max_abs_error = 0;
	for (int t = 0; t < 2; t++)
		for (int g = 0; g < GROUP_DIM; g++)
			for (int n = 0; n < 2; n++)
			{
				const size_t token = t ? TOKEN_DIM - 1 : 0;
				const size_t column = n ? N_DIM - 1 : 0;
				const size_t index = interleaved ?
					(token * GROUP_DIM + g) * N_DIM + column :
					((size_t)g * TOKEN_DIM + token) * N_DIM + column;
				float value;
				ccv_half_precision_to_float((uint16_t*)hb->data.f16 + index, &value, 1);
				const float error = isfinite(value) ? fabsf(value - K_DIM / 128.0f) : INFINITY;
				if (error > max_abs_error)
					max_abs_error = error;
			}
	if (max_abs_error > 0.01f)
	{
		fprintf(stderr, "%s: output verification failed (max sample error %.6g)\n", name, max_abs_error);
		exit(1);
	}
	printf("%-31s median=%9.4f ms min=%9.4f ms avg=%9.4f ms %7.3f TFLOP/s sample_max_abs=%g\n",
		name, median_ms, samples[0], total_ms / iterations, tflops, max_abs_error);
	fflush(stdout);
	free(samples);
	ccv_nnc_stream_context_free(stream);
	if (hw)
		ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(a);
}

int main(int argc, char** argv)
{
	const int warmup = argc > 1 ? atoi(argv[1]) : 1;
	const int iterations = argc > 2 ? atoi(argv[2]) : 5;
	if (warmup < 0 || iterations < 1)
	{
		fprintf(stderr, "usage: interleaved_batched_gemm_bench [warmup>=0] [iterations>=1]\n");
		return 1;
	}
	ccv_nnc_init();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
	printf("DeepSeek v4 wo_a: A=[%d,%d,1,%d], W=[%d,%d,%d], C=[%d,%d,1,%d], FP16\n",
		TOKEN_DIM, GROUP_DIM, K_DIM, GROUP_DIM, N_DIM, K_DIM, TOKEN_DIM, GROUP_DIM, N_DIM);
	printf("ANE disabled; warmup=%d iterations=%d; quantized weights are prebuilt rowwise-int8 payloads\n", warmup, iterations);
	benchmark_case("interleaved rowwise auto", 1, 1, 0, warmup, iterations);
	benchmark_case("interleaved rowwise no NAX", 1, 1, 1, warmup, iterations);
	benchmark_case("interleaved dense generic", 1, 0, 1, warmup, iterations);
	benchmark_case("group-major rowwise auto", 0, 1, 0, warmup, iterations);
	return 0;
}
