#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// End-to-end steady-state latency: packed-weight decode, activation preparation,
// weight upload, CoreML / ANE evaluation, output epilogue and stream completion.
// Alternate AB / BA order to reduce drift. Compilation and offline packing are
// excluded. Both variants use Q6_K weights and the same inputs and GEMM shape.
static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

static double measure(ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* b, ccv_nnc_stream_context_t* stream, int iters)
{
	const double start = now();
	int i;
	for (i = 0; i < iters; i++)
	{
		const int status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), stream);
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "ANE GEMM failed: %d\n", status);
			exit(2);
		}
	}
	ccv_nnc_stream_context_wait(stream);
	return (now() - start) * 1000 / iters;
}

static int compare_double(const void* a, const void* b)
{
	return (*(const double*)a > *(const double*)b) - (*(const double*)a < *(const double*)b);
}

int main(int argc, char** argv)
{
	if (argc < 4 || argc > 7)
	{
		fprintf(stderr, "usage: %s M N K [pairs=31] [iters=10] [precision=0:fp16,1:fp32,2:bf16]\n", argv[0]);
		return 1;
	}
	const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
	const int pairs = argc > 4 ? atoi(argv[4]) : 31;
	const int iters = argc > 5 ? atoi(argv[5]) : 10;
	const int precision = argc > 6 ? atoi(argv[6]) : 0;
	if (m <= 0 || n <= 0 || k <= 0 || k % 256 || k > 65536 || precision < 0 || precision > 2 || pairs <= 0 || iters <= 0)
		return 1;
	ccv_nnc_init();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	const int datatype = precision == 2 ? CCV_16BF : precision == 1 ? CCV_32F : CCV_16F;
	ccv_nnc_tensor_param_t ap = CPU_TENSOR_NHWC(32F, m, k);
	ccv_nnc_tensor_param_t wp = CPU_TENSOR_NHWC(32F, n, k);
	ccv_nnc_tensor_param_t bp = GPU_TENSOR_NHWC(000, 32F, m, n);
	ap.datatype = wp.datatype = bp.datatype = datatype;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ap, 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, wp, 0);
	float* const row = (float*)ccmalloc(sizeof(float) * k);
	unsigned state = 42;
	int i, j, v;
	for (v = 0; v < 2; v++)
		for (i = 0; i < (v ? n : m); i++)
		{
			for (j = 0; j < k; j++)
			{
				state = state * 1664525u + 1013904223u;
				row[j] = ((int)(state >> 24) - 128) / 512.f;
			}
			ccv_nnc_tensor_t* const tensor = v ? hw : ha;
			if (precision == 2)
				ccv_float_to_bfloat(row, (uint16_t*)tensor->data.f16 + (size_t)i * k, k);
			else if (precision == 1)
				memcpy(tensor->data.f32 + (size_t)i * k, row, sizeof(float) * k);
			else
				ccv_float_to_half_precision(row, (uint16_t*)tensor->data.f16 + (size_t)i * k, k);
		}
	ccfree(row);
	ap.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ap, 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, bp, 0);
	ccv_nnc_tensor_t* w[2];
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), stream);
	for (v = 0; v < 2; v++)
	{
		const int format = CCV_NNC_QX_8I_ROWWISE_Q6_K | (v ? CCV_NNC_QX_8I_ROWWISE_HADAMARD_256 : 0);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(wp, format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
		const size_t written = ccv_nnc_quantize_8i_rowwise_x(hw->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, (size_t)n * k, k, format, 0, 0, hq->data.u8, size);
		assert(written == size);
		ccv_nnc_tensor_param_t qp = hq->info;
		qp.type = CCV_TENSOR_GPU_MEMORY;
		w[v] = ccv_nnc_tensor_new(0, qp, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(w[v]), stream);
		ccv_nnc_stream_context_wait(stream);
		ccv_nnc_tensor_free(hq);
		measure(a, w[v], b, stream, 5);
	}
	double* const base = (double*)ccmalloc(sizeof(double) * pairs);
	double* const h256 = (double*)ccmalloc(sizeof(double) * pairs);
	double* const overhead = (double*)ccmalloc(sizeof(double) * pairs);
	printf("# M=%d N=%d K=%d dtype=%s iters=%d\n", m, n, k, precision == 2 ? "bf16" : precision == 1 ? "fp32" : "fp16", iters);
	printf("pair,baseline_ms,h256_ms,overhead_pct\n");
	for (i = 0; i < pairs; i++)
	{
		for (j = 0; j < 2; j++)
		{
			v = j ^ (i & 1);
			(v ? h256 : base)[i] = measure(a, w[v], b, stream, iters);
		}
		overhead[i] = 100 * (h256[i] / base[i] - 1);
		printf("%d,%.6f,%.6f,%.4f\n", i, base[i], h256[i], overhead[i]);
		fflush(stdout);
	}
	qsort(base, pairs, sizeof(double), compare_double);
	qsort(h256, pairs, sizeof(double), compare_double);
	qsort(overhead, pairs, sizeof(double), compare_double);
	printf("# median baseline=%.6f ms H256=%.6f ms paired_overhead=%.4f%%\n", base[pairs / 2], h256[pairs / 2], overhead[pairs / 2]);
	ccfree(base); ccfree(h256); ccfree(overhead);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(ha); ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(a); ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(w[0]); ccv_nnc_tensor_free(w[1]);
	return 0;
}
