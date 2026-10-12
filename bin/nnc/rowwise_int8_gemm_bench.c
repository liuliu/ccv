#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Wall time of GEMM (optionally with bias) or SwiGLU with row-wise int8
// weights through the MPS backend, with ANE disabled. Usage:
//   rowwise_int8_gemm_bench fp16|fp32 gemm|bias|swiglu warmup iterations MxNxK[,MxNxK...]
// SwiGLU uses N x K gate and up projections.

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

static void fill_values(ccv_nnc_tensor_t* const tensor, const int datatype, const float range)
{
	const size_t count = ccv_nnc_tensor_count(tensor->info);
	float* const f = (float*)malloc(sizeof(float) * count);
	size_t i;
	for (i = 0; i < count; i++)
		f[i] = range * ((float)(i * 2654435761u % 1021) / 510.0f - 1.0f);
	if (datatype == CCV_32F)
		memcpy(tensor->data.f32, f, sizeof(float) * count);
	else
		ccv_float_to_half_precision(f, (uint16_t*)tensor->data.f16, count);
	free(f);
}

// Write the row-wise payload directly: int8 codes, then 128-byte aligned scales.
static void fill_rowwise(ccv_nnc_tensor_t* const tensor, const int datatype, const int n, const int k)
{
	int8_t* const codes = (int8_t*)tensor->data.u8;
	size_t i;
	for (i = 0; i < (size_t)n * k; i++)
		codes[i] = (int8_t)((int)(i * 2246822519u % 255) - 127);
	uint8_t* const scales = tensor->data.u8 + (((size_t)n * k + 127) & ~(size_t)127);
	float* const f = (float*)malloc(sizeof(float) * n);
	for (i = 0; i < (size_t)n; i++)
		f[i] = (0.5f + (float)(i % 7) / 7.0f) / (127.0f * 64.0f);
	if (datatype == CCV_32F)
		memcpy(scales, f, sizeof(float) * n);
	else
		ccv_float_to_half_precision(f, (uint16_t*)scales, n);
	free(f);
}

typedef enum {
	OPERATION_GEMM,
	OPERATION_GEMM_BIAS,
	OPERATION_SWIGLU,
} operation_t;

static void benchmark(const int datatype, const operation_t operation, const int m, const int n, const int k, const int warmup, const int iterations)
{
	const int use_bias = operation == OPERATION_GEMM_BIAS;
	ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, m, k);
	ccv_nnc_tensor_param_t w_params = CPU_TENSOR_NHWC(32F, n, k);
	ccv_nnc_tensor_param_t bias_params = CPU_TENSOR_NHWC(32F, n);
	ccv_nnc_tensor_param_t b_params = CPU_TENSOR_NHWC(32F, m, n);
	a_params.datatype = w_params.datatype = bias_params.datatype = b_params.datatype = datatype;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_params), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* const hup = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_params), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, b_params, 0);
	a_params.type = bias_params.type = b_params.type = w_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_params), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* const up = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_params), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
	if (!ha || !hw || !hbias || !hup || !hb || !a || !w || !bias || !up || !b)
	{
		fprintf(stderr, "allocation failed for %dx%dx%d\n", m, n, k);
		exit(1);
	}
	fill_values(ha, datatype, 1);
	fill_values(hbias, datatype, 0.25);
	fill_rowwise(hw, datatype, n, k);
	fill_rowwise(hup, datatype, n, k);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hup), TENSOR_LIST(a, w, bias, up), 0);
	ccv_nnc_cmd_t cmd = operation == OPERATION_SWIGLU ? CMD_SWIGLU_FORWARD(10) : CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
	cmd.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_tensor_t* const inputs[3] = { a, w, operation == OPERATION_SWIGLU ? up : (use_bias ? bias : 0) };
	int i;
	for (i = 0; i < warmup; i++)
		if (ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 3, TENSOR_LIST(b), 0) != CCV_NNC_EXEC_SUCCESS)
		{
			fprintf(stderr, "operation failed for %dx%dx%d\n", m, n, k);
			exit(1);
		}
	double* const samples = (double*)malloc(sizeof(double) * iterations);
	for (i = 0; i < iterations; i++)
	{
		const double start = get_current_time();
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 3, TENSOR_LIST(b), 0);
		samples[i] = (get_current_time() - start) * 1000;
	}
	qsort(samples, iterations, sizeof(double), compare_double);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	int finite = 1;
	const size_t count = (size_t)m * n;
	size_t j;
	for (j = 0; j < count && finite; j++)
		finite = datatype == CCV_32F ? isfinite(hb->data.f32[j]) : ((((uint16_t*)hb->data.f16)[j] & 0x7c00) != 0x7c00);
	static const char* const operation_names[] = { "gemm", "bias", "swiglu" };
	printf("%s,%s,%d,%d,%d,%.4f,%.4f,%s\n", datatype == CCV_32F ? "fp32" : "fp16", operation_names[operation], m, n, k,
		samples[iterations / 2], samples[0], finite ? "finite" : "NONFINITE");
	fflush(stdout);
	free(samples);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(up);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hup);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(ha);
}

int main(int argc, char** argv)
{
	if (argc < 6)
	{
		fprintf(stderr, "usage: %s fp16|fp32 gemm|bias|swiglu warmup iterations MxNxK[,MxNxK...]\n", argv[0]);
		return 1;
	}
	ccv_nnc_init();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
	const int datatype = strcmp(argv[1], "fp32") == 0 ? CCV_32F : CCV_16F;
	const operation_t operation = strcmp(argv[2], "swiglu") == 0 ? OPERATION_SWIGLU : (strcmp(argv[2], "bias") == 0 ? OPERATION_GEMM_BIAS : OPERATION_GEMM);
	const int warmup = atoi(argv[3]);
	const int iterations = atoi(argv[4]);
	char* shapes = strdup(argv[5]);
	char* saveptr = 0;
	char* shape;
	for (shape = strtok_r(shapes, ",", &saveptr); shape; shape = strtok_r(0, ",", &saveptr))
	{
		int m, n, k;
		if (sscanf(shape, "%dx%dx%d", &m, &n, &k) == 3)
			benchmark(datatype, operation, m, n, k, warmup, iterations);
	}
	free(shapes);
	return 0;
}
