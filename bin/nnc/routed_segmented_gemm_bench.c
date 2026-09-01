#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef enum {
	ROUTING_BALANCED,
	ROUTING_SPARSE6,
} routing_mode_t;

typedef enum {
	WEIGHT_ROWWISE,
	WEIGHT_Q5_K,
	WEIGHT_Q4_K,
	WEIGHT_Q3_K,
	WEIGHT_Q2_K,
	WEIGHT_IQ2_XXS,
	WEIGHT_IQ2_S,
	WEIGHT_IQ2_XS,
	WEIGHT_IQ3_S,
	WEIGHT_IQ3_XXS,
	WEIGHT_ALL,
} weight_format_t;

static double get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int compare_double(const void* const a, const void* const b)
{
	const double va = *(const double*)a;
	const double vb = *(const double*)b;
	return (va > vb) - (va < vb);
}

static void fill_tensor(ccv_nnc_tensor_t* const tensor, const int datatype)
{
	const int count = ccv_nnc_tensor_count(tensor->info);
	float* const f = (float*)malloc(sizeof(float) * count);
	int i;
	for (i = 0; i < count; i++)
		f[i] = (float)((i % 127) - 63) / 64.0f;
	if (datatype == CCV_32F)
		memcpy(tensor->data.f32, f, sizeof(float) * count);
	else
		ccv_float_to_half_precision(f, (uint16_t*)tensor->data.f16, count);
	free(f);
}

static void fill_routing(ccv_nnc_tensor_t* const indices, ccv_nnc_tensor_t* const counts, const int segments, const int token_count, const routing_mode_t routing_mode)
{
	int i;
	for (i = 0; i < segments; i++)
	{
		indices->data.i32[i] = i;
		counts->data.i32[i] = 0;
	}
	if (routing_mode == ROUTING_BALANCED)
	{
		const int total_rows = token_count * 6;
		for (i = 0; i < segments; i++)
			counts->data.i32[i] = total_rows / segments + (i < total_rows % segments);
		return;
	}
	const int selected_experts[6] = {0, 17, 42, 103, 199, 255};
	for (i = 0; i < 6; i++)
	{
		indices->data.i32[i] = selected_experts[i % 6];
		counts->data.i32[i] = token_count;
	}
}

static const char* routing_mode_name(const routing_mode_t routing_mode)
{
	return routing_mode == ROUTING_BALANCED ? "balanced256" : "sparse6";
}

static const char* weight_format_name(const weight_format_t weight_format)
{
	switch (weight_format)
	{
		case WEIGHT_Q5_K:
			return "q5_k";
		case WEIGHT_Q4_K:
			return "q4_k";
		case WEIGHT_Q3_K:
			return "q3_k";
		case WEIGHT_Q2_K:
			return "q2_k";
		case WEIGHT_IQ2_XXS:
			return "iq2_xxs";
		case WEIGHT_IQ2_S:
			return "iq2_s";
		case WEIGHT_IQ2_XS:
			return "iq2_xs";
		case WEIGHT_IQ3_S:
			return "iq3_s";
		case WEIGHT_IQ3_XXS:
			return "iq3_xxs";
		case WEIGHT_ALL:
			return "all";
		case WEIGHT_ROWWISE:
		default:
			return "rowwise";
	}
}

static ccv_nnc_tensor_param_t weight_params_for_format(const ccv_nnc_tensor_param_t dense_params, const weight_format_t weight_format)
{
	switch (weight_format)
	{
		case WEIGHT_Q5_K:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_Q5_K);
		case WEIGHT_Q4_K:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_Q4_K);
		case WEIGHT_Q3_K:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_Q3_K);
		case WEIGHT_Q2_K:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_Q2_K);
		case WEIGHT_IQ2_XXS:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
		case WEIGHT_IQ2_S:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ2_S);
		case WEIGHT_IQ2_XS:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ2_XS);
		case WEIGHT_IQ3_S:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ3_S);
		case WEIGHT_IQ3_XXS:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ3_XXS);
		case WEIGHT_ROWWISE:
		default:
			return ccv_nnc_tensor_8i_rowwise(dense_params);
	}
}

static double benchmark_case(const char* const name, const int n, const int k, const int expert_count, const int token_count, const routing_mode_t routing_mode, const weight_format_t weight_format, const int datatype, const int warmup, const int iterations)
{
	const int m = token_count * 6;
	const int segments = routing_mode == ROUTING_SPARSE6 ? 6 : ccv_min(m, expert_count);
	ccv_nnc_tensor_param_t ha_params = CPU_TENSOR_NHWC(16F, m, k);
	ccv_nnc_tensor_param_t a_params = GPU_TENSOR_NHWC(000, 16F, m, k);
	ccv_nnc_tensor_param_t w_dense_params = GPU_TENSOR_NHWC(000, 16F, expert_count, n, k);
	ccv_nnc_tensor_param_t b_params = GPU_TENSOR_NHWC(000, 16F, m, n);
	ha_params.datatype = datatype;
	a_params.datatype = datatype;
	w_dense_params.datatype = datatype;
	b_params.datatype = datatype;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ha_params, 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	const ccv_nnc_tensor_param_t w_params = weight_params_for_format(w_dense_params, weight_format);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
	if (!ha || !hindices || !hcounts || !a || !indices || !counts || !w || !b)
	{
		fprintf(stderr, "%s allocation failed\n", name);
		exit(1);
	}
	fill_tensor(ha, datatype);
	fill_routing(hindices, hcounts, segments, token_count, routing_mode);
	const double weight_gib = (double)ccv_nnc_tensor_data_size(w_params) / 1073741824.0;
	printf("%s/%s/%s/%s: T=%d M=%d groups=%d experts=%d N=%d K=%d weight=%.3f GiB\n",
		datatype == CCV_32F ? "fp32" : "fp16", routing_mode_name(routing_mode), weight_format_name(weight_format), name,
		token_count, m, segments, expert_count, n, k, weight_gib);
	fflush(stdout);

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts), TENSOR_LIST(a, indices, counts), 0);
	const ccv_nnc_cmd_t cmd = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	int i;
	for (i = 0; i < warmup; i++)
	{
		const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "%s/%s/%s warmup failed with status %d\n", routing_mode_name(routing_mode), weight_format_name(weight_format), name, status);
			exit(1);
		}
	}
	double* const samples = (double*)malloc(sizeof(double) * iterations);
	double total = 0;
	double min_ms = 1e100;
	for (i = 0; i < iterations; i++)
	{
		const double start = get_current_time();
		const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
		const double ms = (get_current_time() - start) * 1000;
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "%s/%s/%s iteration failed with status %d\n", routing_mode_name(routing_mode), weight_format_name(weight_format), name, status);
			exit(1);
		}
		samples[i] = ms;
		total += ms;
		if (ms < min_ms)
			min_ms = ms;
	}
	qsort(samples, iterations, sizeof(double), compare_double);
	const double median_ms = (iterations % 2) ? samples[iterations / 2] : 0.5 * (samples[iterations / 2 - 1] + samples[iterations / 2]);
	const double avg_ms = total / iterations;
	printf("%s/%s/%s/%s: median=%.4f ms min=%.4f ms avg=%.4f ms iters=%d warmup=%d\n",
		datatype == CCV_32F ? "fp32" : "fp16", routing_mode_name(routing_mode), weight_format_name(weight_format), name,
		median_ms, min_ms, avg_ms, iterations, warmup);
	fflush(stdout);

	free(samples);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(ha);
	return median_ms;
}

static double benchmark_gemm_case(const char* const name, const int n, const int k, const int token_count, const int datatype, const int warmup, const int iterations)
{
	ccv_nnc_tensor_param_t ha_params = CPU_TENSOR_NHWC(16F, token_count, k);
	ccv_nnc_tensor_param_t a_params = GPU_TENSOR_NHWC(000, 16F, token_count, k);
	ccv_nnc_tensor_param_t w_dense_params = GPU_TENSOR_NHWC(000, 16F, n, k);
	ccv_nnc_tensor_param_t b_params = GPU_TENSOR_NHWC(000, 16F, token_count, n);
	ha_params.datatype = datatype;
	a_params.datatype = datatype;
	w_dense_params.datatype = datatype;
	b_params.datatype = datatype;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ha_params, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_dense_params), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
	if (!ha || !a || !w || !b)
	{
		fprintf(stderr, "gemm/%s allocation failed\n", name);
		exit(1);
	}
	fill_tensor(ha, datatype);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	const ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
	int i;
	for (i = 0; i < warmup; i++)
	{
		const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "gemm/%s warmup failed with status %d\n", name, status);
			exit(1);
		}
	}
	double* const samples = (double*)malloc(sizeof(double) * iterations);
	double total = 0;
	double min_ms = 1e100;
	for (i = 0; i < iterations; i++)
	{
		const double start = get_current_time();
		const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
		const double ms = (get_current_time() - start) * 1000;
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "gemm/%s iteration failed with status %d\n", name, status);
			exit(1);
		}
		samples[i] = ms;
		total += ms;
		if (ms < min_ms)
			min_ms = ms;
	}
	qsort(samples, iterations, sizeof(double), compare_double);
	const double median_ms = samples[iterations / 2];
	printf("%s/gemm/%s: T=%d N=%d K=%d median=%.4f ms min=%.4f ms avg=%.4f ms\n",
		datatype == CCV_32F ? "fp32" : "fp16", name, token_count, n, k,
		median_ms, min_ms, total / iterations);
	fflush(stdout);
	free(samples);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(ha);
	return median_ms;
}

static routing_mode_t parse_routing_mode(const char* const arg)
{
	if (!arg || strcmp(arg, "balanced") == 0 || strcmp(arg, "balanced256") == 0)
		return ROUTING_BALANCED;
	if (strcmp(arg, "sparse") == 0 || strcmp(arg, "sparse6") == 0)
		return ROUTING_SPARSE6;
	fprintf(stderr, "usage: routed_segmented_gemm_bench [balanced|sparse6] [warmup] [iterations]\n");
	exit(1);
}

static int is_integer_arg(const char* const arg)
{
	if (!arg || !*arg)
		return 0;
	int i = arg[0] == '-' ? 1 : 0;
	for (; arg[i]; i++)
		if (!isdigit((unsigned char)arg[i]))
			return 0;
	return 1;
}

static weight_format_t parse_weight_format(const char* const arg)
{
	if (!arg || strcmp(arg, "rowwise") == 0 || strcmp(arg, "8i_rowwise") == 0)
		return WEIGHT_ROWWISE;
	if (strcmp(arg, "all") == 0)
		return WEIGHT_ALL;
	if (strcmp(arg, "q5_k") == 0 || strcmp(arg, "Q5_K") == 0)
		return WEIGHT_Q5_K;
	if (strcmp(arg, "q4_k") == 0 || strcmp(arg, "Q4_K") == 0)
		return WEIGHT_Q4_K;
	if (strcmp(arg, "q3_k") == 0 || strcmp(arg, "Q3_K") == 0)
		return WEIGHT_Q3_K;
	if (strcmp(arg, "q2_k") == 0 || strcmp(arg, "Q2_K") == 0)
		return WEIGHT_Q2_K;
	if (strcmp(arg, "iq2_xxs") == 0 || strcmp(arg, "IQ2_XXS") == 0)
		return WEIGHT_IQ2_XXS;
	if (strcmp(arg, "iq2_s") == 0 || strcmp(arg, "IQ2_S") == 0)
		return WEIGHT_IQ2_S;
	if (strcmp(arg, "iq2_xs") == 0 || strcmp(arg, "IQ2_XS") == 0)
		return WEIGHT_IQ2_XS;
	if (strcmp(arg, "iq3_s") == 0 || strcmp(arg, "IQ3_S") == 0)
		return WEIGHT_IQ3_S;
	if (strcmp(arg, "iq3_xxs") == 0 || strcmp(arg, "IQ3_XXS") == 0)
		return WEIGHT_IQ3_XXS;
	fprintf(stderr, "usage: routed_segmented_gemm_bench [balanced|sparse6] [rowwise|q5_k|q4_k|q3_k|q2_k|iq2_xxs|iq2_s|iq2_xs|iq3_s|iq3_xxs|all] [warmup] [iterations]\n");
	exit(1);
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	if (argc > 1 && strcmp(argv[1], "gemm") == 0)
	{
		const int warmup = argc > 2 ? atoi(argv[2]) : 3;
		const int iterations = argc > 3 ? atoi(argv[3]) : 12;
		const int token_count = argc > 4 ? atoi(argv[4]) : 2048;
		const int datatype = argc > 5 && strcmp(argv[5], "fp32") == 0 ? CCV_32F : CCV_16F;
		if (argc > 6 && strcmp(argv[6], "gpu") == 0)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
		const double gate_ms = benchmark_gemm_case("gate", 2048, 4096, token_count, datatype, warmup, iterations);
		const double up_ms = benchmark_gemm_case("up", 2048, 4096, token_count, datatype, warmup, iterations);
		const double down_ms = benchmark_gemm_case("down", 4096, 2048, token_count, datatype, warmup, iterations);
		printf("%s/gemm/total_ffn: T=%d median_sum=%.4f ms\n", datatype == CCV_32F ? "fp32" : "fp16", token_count, gate_ms + up_ms + down_ms);
		return 0;
	}
	const routing_mode_t routing_mode = parse_routing_mode(argc > 1 ? argv[1] : "balanced");
	int argi = 2;
	weight_format_t weight_format = WEIGHT_ROWWISE;
	if (argc > argi && !is_integer_arg(argv[argi]))
	{
		weight_format = parse_weight_format(argv[argi]);
		++argi;
	}
	const int warmup = argc > argi ? atoi(argv[argi]) : 3;
	const int iterations = argc > argi + 1 ? atoi(argv[argi + 1]) : 12;
	const int token_count = argc > argi + 2 ? atoi(argv[argi + 2]) : 2048;
	const int datatype = argc > argi + 3 && strcmp(argv[argi + 3], "fp32") == 0 ? CCV_32F : CCV_16F;
	const int expert_count = 256;
	const weight_format_t all_formats[] = {
		WEIGHT_ROWWISE,
		WEIGHT_Q5_K,
		WEIGHT_Q4_K,
		WEIGHT_Q3_K,
		WEIGHT_Q2_K,
		WEIGHT_IQ2_XXS,
		WEIGHT_IQ2_S,
		WEIGHT_IQ2_XS,
		WEIGHT_IQ3_S,
		WEIGHT_IQ3_XXS,
	};
	const int format_count = weight_format == WEIGHT_ALL ? sizeof(all_formats) / sizeof(all_formats[0]) : 1;
	int i;
	for (i = 0; i < format_count; i++)
	{
		const weight_format_t current_format = weight_format == WEIGHT_ALL ? all_formats[i] : weight_format;
		const double gate_ms = benchmark_case("gate", 2048, 4096, expert_count, token_count, routing_mode, current_format, datatype, warmup, iterations);
		const double up_ms = benchmark_case("up", 2048, 4096, expert_count, token_count, routing_mode, current_format, datatype, warmup, iterations);
		const double down_ms = benchmark_case("down", 4096, 2048, expert_count, token_count, routing_mode, current_format, datatype, warmup, iterations);
		printf("%s/%s/%s/total_ffn: T=%d median_sum=%.4f ms\n", datatype == CCV_32F ? "fp32" : "fp16", routing_mode_name(routing_mode), weight_format_name(current_format), token_count, gate_ms + up_ms + down_ms);
	}
	return 0;
}
