#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// CUDA throughput of the row-wise-x decodes that the GEMM command runs for TMCQ (and IQ2_XXS for comparison): the
// int8 row-wise conversion, the FP16 dequantization, the M = 1 FP16 GEMM (conversion + int8 GEMV), and the int8
// GEMV alone on already converted weights. Each measurement queues `iterations` runs on one stream and reports the
// mean time per run (the best of 5 batches). Usage:
//   tmcq_cuda_bench warmup iterations NxK[,NxK...]

extern void ccv_nnc_compat_dequantize_8i_rowwise_x(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context);
extern void ccv_nnc_compat_dequantize_8i_rowwise_x_fp(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context);

static double get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

typedef enum {
	RUN_CONVERT_8I,
	RUN_DEQUANTIZE_16F,
	RUN_GEMV,
	RUN_GEMV_8I,
} run_t;

typedef struct {
	ccv_nnc_tensor_t* q; // Row-wise-x payload.
	ccv_nnc_tensor_t* q8; // Converted row-wise int8 payload.
	ccv_nnc_tensor_t* dense; // FP16 weights.
	ccv_nnc_tensor_t* a;
	ccv_nnc_tensor_t* b;
	int format;
	int n;
	int k;
} bench_t;

static void run_once(const bench_t* const bench, const run_t run, ccv_nnc_stream_context_t* const stream)
{
	const size_t count = (size_t)bench->n * bench->k;
	switch (run)
	{
		case RUN_CONVERT_8I:
			ccv_nnc_compat_dequantize_8i_rowwise_x(bench->q->data.u8, CCV_16F, ccv_nnc_tensor_data_size_without_padding(bench->q->info), bench->k, bench->format, bench->q8->data.u8, count, stream);
			break;
		case RUN_DEQUANTIZE_16F:
			ccv_nnc_compat_dequantize_8i_rowwise_x_fp(bench->q->data.u8, CCV_16F, ccv_nnc_tensor_data_size_without_padding(bench->q->info), bench->k, bench->format, bench->dense->data.u8, count, stream);
			break;
		case RUN_GEMV:
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(bench->a, bench->q), TENSOR_LIST(bench->b), stream);
			break;
		case RUN_GEMV_8I:
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(bench->a, bench->q8), TENSOR_LIST(bench->b), stream);
			break;
	}
}

// The best of 5 batches, to be robust against other work on a shared GPU.
static double time_run(const bench_t* const bench, const run_t run, ccv_nnc_stream_context_t* const stream, const int warmup, const int iterations)
{
	int i, j;
	for (i = 0; i < warmup; i++)
		run_once(bench, run, stream);
	ccv_nnc_stream_context_wait(stream);
	double best = HUGE_VAL;
	for (j = 0; j < 5; j++)
	{
		const double start = get_current_time();
		for (i = 0; i < iterations; i++)
			run_once(bench, run, stream);
		ccv_nnc_stream_context_wait(stream);
		best = ccv_min(best, (get_current_time() - start) * 1e6 / iterations);
	}
	return best;
}

static void benchmark(const int format, const char* const name, const int n, const int k, const int warmup, const int iterations, ccv_nnc_stream_context_t* const stream)
{
	const ccv_nnc_tensor_param_t w_params = GPU_TENSOR_NHWC(000, 16F, n, k);
	bench_t bench = {
		.q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(w_params, format), 0),
		.q8 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_params), 0),
		.dense = ccv_nnc_tensor_new(0, w_params, 0),
		.a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, k), 0),
		.b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, n), 0),
		.format = format,
		.n = n,
		.k = k,
	};
	// Random payload bits are valid for both codecs; scales and activations are small and finite.
	const ccv_nnc_tensor_param_t hq_params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(16F, n, k), format);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, hq_params, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, k), 0);
	const size_t q_size = ccv_nnc_tensor_data_size_without_padding(hq_params);
	const size_t scale_offset = q_size - (size_t)n * 2;
	size_t i;
	uint32_t state = 0x12345678u;
	for (i = 0; i < scale_offset; i++)
	{
		state = state * 1664525u + 1013904223u;
		hq->data.u8[i] = (uint8_t)(state >> 24);
	}
	float* const f = (float*)malloc(sizeof(float) * ccv_max(n, k));
	for (i = 0; i < (size_t)n; i++)
		f[i] = (0.5f + (float)(i % 7) / 7.0f) / 4096.0f;
	ccv_float_to_half_precision(f, (uint16_t*)(hq->data.u8 + scale_offset), n);
	for (i = 0; i < (size_t)k; i++)
		f[i] = (float)(i * 2654435761u % 1021) / 510.0f - 1.0f;
	ccv_float_to_half_precision(f, (uint16_t*)ha->data.f16, k);
	free(f);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, ha), TENSOR_LIST(bench.q, bench.a), 0);
	run_once(&bench, RUN_CONVERT_8I, 0);
	const size_t count = (size_t)n * k;
	const size_t payload = scale_offset; // Packed bits plus the 128-byte padding.
	const size_t int8_bytes = ccv_nnc_tensor_data_size_without_padding(bench.q8->info);
	const double convert_us = time_run(&bench, RUN_CONVERT_8I, stream, warmup, iterations);
	const double dequantize_us = time_run(&bench, RUN_DEQUANTIZE_16F, stream, warmup, iterations);
	const double gemv_us = time_run(&bench, RUN_GEMV, stream, warmup, iterations);
	const double gemv_8i_us = time_run(&bench, RUN_GEMV_8I, stream, warmup, iterations);
	// Bandwidth counts the bytes each step must move: packed payload + scales in, plus the int8 / FP16 weights out.
	printf("%-9s %5dx%-5d bpw %.4f | int8 convert %8.2f us %7.1f GB/s | fp16 dequant %8.2f us %7.1f GB/s | gemv m=1 %8.2f us (%7.1f GB/s of packed weights) | int8 gemv alone %8.2f us\n",
		name, n, k, payload * 8.0 / count,
		convert_us, (q_size + int8_bytes) / (convert_us * 1e3),
		dequantize_us, (q_size + count * 2.0) / (dequantize_us * 1e3),
		gemv_us, q_size / (gemv_us * 1e3),
		gemv_8i_us);
	fflush(stdout);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(bench.b);
	ccv_nnc_tensor_free(bench.a);
	ccv_nnc_tensor_free(bench.dense);
	ccv_nnc_tensor_free(bench.q8);
	ccv_nnc_tensor_free(bench.q);
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "usage: %s warmup iterations NxK[,NxK...]\n", argv[0]);
		return 1;
	}
	ccv_nnc_init();
	const int warmup = atoi(argv[1]);
	const int iterations = atoi(argv[2]);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	static const struct {
		int format;
		const char* name;
	} formats[] = {
		{ CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, "IQ2_XXS" },
		{ CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, "TMCQ_N16" },
		{ CCV_NNC_QX_8I_ROWWISE_TMCQ_N20, "TMCQ_N20" },
		{ CCV_NNC_QX_8I_ROWWISE_TMCQ_N24, "TMCQ_N24" },
		{ CCV_NNC_QX_8I_ROWWISE_TMCQ_N32, "TMCQ_N32" },
	};
	char* shapes = strdup(argv[3]);
	char* saveptr = 0;
	char* shape;
	for (shape = strtok_r(shapes, ",", &saveptr); shape; shape = strtok_r(0, ",", &saveptr))
	{
		int n, k;
		if (sscanf(shape, "%dx%d", &n, &k) != 2)
			continue;
		int f;
		for (f = 0; f < sizeof(formats) / sizeof(formats[0]); f++)
			benchmark(formats[f].format, formats[f].name, n, k, warmup, iterations, stream);
	}
	free(shapes);
	ccv_nnc_stream_context_free(stream);
	return 0;
}
