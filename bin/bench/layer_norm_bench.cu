extern "C" {
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <nnc/gpu/3rdparty/flash_attn/layer_norm/ln_api.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

typedef struct {
	int rows;
	int cols;
	int warmup;
	int iterations;
	int datatype;
	int op;
	int affine;
	float epsilon;
} bench_config_t;

enum {
	BENCH_OP_BOTH,
	BENCH_OP_LAYER_NORM,
	BENCH_OP_RMSNORM,
};

static void usage(const char* const argv0)
{
	printf("Usage: %s [options]\n", argv0);
	printf("  --rows N\n");
	printf("  --cols N\n");
	printf("  --warmup N\n");
	printf("  --iterations N\n");
	printf("  --dtype {f16|bf16|f32}\n");
	printf("  --op {both|layernorm|rmsnorm}\n");
	printf("  --affine {0|1}\n");
	printf("  --epsilon N\n");
}

static int _parse_int_arg(const char* const value)
{
	char* endptr = 0;
	const long parsed = strtol(value, &endptr, 10);
	if (!endptr || endptr == value || *endptr != 0)
	{
		fprintf(stderr, "cannot parse integer argument: %s\n", value);
		exit(1);
	}
	return (int)parsed;
}

static float _parse_float_arg(const char* const value)
{
	char* endptr = 0;
	const float parsed = strtof(value, &endptr);
	if (!endptr || endptr == value || *endptr != 0)
	{
		fprintf(stderr, "cannot parse float argument: %s\n", value);
		exit(1);
	}
	return parsed;
}

static ccv_nnc_tensor_param_t _make_gpu_tensor_param(const int datatype, const int rows, const int cols)
{
	ccv_nnc_tensor_param_t params = {};
	params.type = CCV_COMPUTE_DEVICE_000 | CCV_TENSOR_GPU_MEMORY;
	params.format = CCV_TENSOR_FORMAT_NHWC;
	params.datatype = datatype;
	params.dim[0] = rows;
	params.dim[1] = cols;
	return params;
}

static const char* _dtype_name(const int datatype)
{
	switch (datatype)
	{
		case CCV_16F:
			return "f16";
		case CCV_16BF:
			return "bf16";
		case CCV_32F:
			return "f32";
		default:
			return "unknown";
	}
}

static const char* _op_name(const int is_rms_norm)
{
	return is_rms_norm ? "rmsnorm" : "layernorm";
}

static layer_norm::DataType _flash_dtype(const int datatype)
{
	switch (datatype)
	{
		case CCV_16F:
			return layer_norm::DATA_TYPE_FP16;
		case CCV_16BF:
			return layer_norm::DATA_TYPE_BF16;
		case CCV_32F:
			return layer_norm::DATA_TYPE_FP32;
		default:
			fprintf(stderr, "unsupported datatype: %d\n", datatype);
			exit(1);
	}
}

template<typename T>
__device__ T _from_float(const float v);

template<>
__device__ float _from_float<float>(const float v)
{
	return v;
}

template<>
__device__ __half _from_float<__half>(const float v)
{
	return __float2half(v);
}

template<>
__device__ __nv_bfloat16 _from_float<__nv_bfloat16>(const float v)
{
	return __float2bfloat16(v);
}

template<typename T>
__global__ void _fill_input_kernel(T* const x, const int count, const int cols)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const int row = i / cols;
		const int col = i - row * cols;
		const float v = 0.25f * sinf((float)(i % 4096) * 0.013f) + 0.5f * cosf((float)col * 0.017f) + (float)(row % 17) * 0.001f;
		x[i] = _from_float<T>(v);
	}
}

template<typename T>
__global__ void _fill_scale_kernel(T* const scale, const int cols, const int affine)
{
	CUDA_1D_KERNEL_LOOP(i, cols) {
		const float v = affine ? (0.75f + 0.25f * sinf((float)i * 0.019f)) : 1.f;
		scale[i] = _from_float<T>(v);
	}
}

template<typename T>
__global__ void _fill_bias_kernel(T* const bias, const int cols)
{
	CUDA_1D_KERNEL_LOOP(i, cols) {
		const float v = 0.1f * cosf((float)i * 0.023f);
		bias[i] = _from_float<T>(v);
	}
}

static void _fill_input(ccv_nnc_tensor_t* const x, const int count, const int cols, cudaStream_t stream)
{
	if (x->info.datatype == CCV_32F)
		_fill_input_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(x->data.f32, count, cols);
	else if (x->info.datatype == CCV_16F)
		_fill_input_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>((__half*)x->data.f16, count, cols);
	else if (x->info.datatype == CCV_16BF)
		_fill_input_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>((__nv_bfloat16*)x->data.f16, count, cols);
	CUDA_ENFORCE(cudaGetLastError());
}

static void _fill_scale(ccv_nnc_tensor_t* const scale, const int cols, const int affine, cudaStream_t stream)
{
	if (scale->info.datatype == CCV_32F)
		_fill_scale_kernel<<<CUDA_GET_BLOCKS(cols), CUDA_NUM_THREADS, 0, stream>>>(scale->data.f32, cols, affine);
	else if (scale->info.datatype == CCV_16F)
		_fill_scale_kernel<<<CUDA_GET_BLOCKS(cols), CUDA_NUM_THREADS, 0, stream>>>((__half*)scale->data.f16, cols, affine);
	else if (scale->info.datatype == CCV_16BF)
		_fill_scale_kernel<<<CUDA_GET_BLOCKS(cols), CUDA_NUM_THREADS, 0, stream>>>((__nv_bfloat16*)scale->data.f16, cols, affine);
	CUDA_ENFORCE(cudaGetLastError());
}

static void _fill_bias(ccv_nnc_tensor_t* const bias, const int cols, cudaStream_t stream)
{
	if (bias->info.datatype == CCV_32F)
		_fill_bias_kernel<<<CUDA_GET_BLOCKS(cols), CUDA_NUM_THREADS, 0, stream>>>(bias->data.f32, cols);
	else if (bias->info.datatype == CCV_16F)
		_fill_bias_kernel<<<CUDA_GET_BLOCKS(cols), CUDA_NUM_THREADS, 0, stream>>>((__half*)bias->data.f16, cols);
	else if (bias->info.datatype == CCV_16BF)
		_fill_bias_kernel<<<CUDA_GET_BLOCKS(cols), CUDA_NUM_THREADS, 0, stream>>>((__nv_bfloat16*)bias->data.f16, cols);
	CUDA_ENFORCE(cudaGetLastError());
}

static float _benchmark_command(const ccv_nnc_cmd_t cmd, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context, const int warmup, const int iterations)
{
	int i;
	for (i = 0; i < warmup; i++)
		assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, input_size, outputs, output_size, stream_context));
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_ENFORCE(cudaEventCreate(&start));
	CUDA_ENFORCE(cudaEventCreate(&stop));
	CUDA_ENFORCE(cudaEventRecord(start, stream));
	for (i = 0; i < iterations; i++)
		assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, input_size, outputs, output_size, stream_context));
	CUDA_ENFORCE(cudaEventRecord(stop, stream));
	CUDA_ENFORCE(cudaEventSynchronize(stop));
	float elapsed = 0;
	CUDA_ENFORCE(cudaEventElapsedTime(&elapsed, start, stop));
	CUDA_ENFORCE(cudaEventDestroy(start));
	CUDA_ENFORCE(cudaEventDestroy(stop));
	return elapsed / iterations;
}

static bool _launch_flash(layer_norm::LaunchParams<layer_norm::FwdParams>& launch_params, const int datatype, const int cols, const bool configure_params)
{
	const layer_norm::DataType dtype = _flash_dtype(datatype);
	return layer_norm::run_layer_norm_fwd(launch_params, dtype, dtype, dtype, dtype, cols, configure_params);
}

static float _benchmark_flash(layer_norm::LaunchParams<layer_norm::FwdParams>& launch_params, const int datatype, const int cols, const int warmup, const int iterations)
{
	int i;
	for (i = 0; i < warmup; i++)
	{
		if (!_launch_flash(launch_params, datatype, cols, false))
		{
			fprintf(stderr, "FlashAttention layer norm launcher is unavailable for dtype=%s cols=%d\n", _dtype_name(datatype), cols);
			exit(1);
		}
	}
	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_ENFORCE(cudaEventCreate(&start));
	CUDA_ENFORCE(cudaEventCreate(&stop));
	CUDA_ENFORCE(cudaEventRecord(start, launch_params.stream));
	for (i = 0; i < iterations; i++)
	{
		if (!_launch_flash(launch_params, datatype, cols, false))
		{
			fprintf(stderr, "FlashAttention layer norm launcher is unavailable for dtype=%s cols=%d\n", _dtype_name(datatype), cols);
			exit(1);
		}
	}
	CUDA_ENFORCE(cudaEventRecord(stop, launch_params.stream));
	CUDA_ENFORCE(cudaEventSynchronize(stop));
	float elapsed = 0;
	CUDA_ENFORCE(cudaEventElapsedTime(&elapsed, start, stop));
	CUDA_ENFORCE(cudaEventDestroy(start));
	CUDA_ENFORCE(cudaEventDestroy(stop));
	return elapsed / iterations;
}

static void _copy_to_float(const ccv_nnc_tensor_t* const tensor, const int count, std::vector<float>& out)
{
	out.resize(count);
	if (tensor->info.datatype == CCV_32F)
	{
		CUDA_ENFORCE(cudaMemcpy(out.data(), tensor->data.f32, sizeof(float) * count, cudaMemcpyDeviceToHost));
		return;
	}
	std::vector<uint16_t> half_data(count);
	CUDA_ENFORCE(cudaMemcpy(half_data.data(), tensor->data.f16, sizeof(uint16_t) * count, cudaMemcpyDeviceToHost));
	if (tensor->info.datatype == CCV_16F)
		ccv_half_precision_to_float(half_data.data(), out.data(), count);
	else if (tensor->info.datatype == CCV_16BF)
		ccv_bfloat_to_float(half_data.data(), out.data(), count);
}

static void _compare_outputs(const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, const int count, float* const max_abs, float* const mean_abs)
{
	std::vector<float> ahost;
	std::vector<float> bhost;
	_copy_to_float(a, count, ahost);
	_copy_to_float(b, count, bhost);
	double sum = 0;
	float maxv = 0;
	int i;
	for (i = 0; i < count; i++)
	{
		const float diff = fabsf(ahost[i] - bhost[i]);
		maxv = std::max(maxv, diff);
		sum += diff;
	}
	*max_abs = maxv;
	*mean_abs = (float)(sum / count);
}

static void _run_one(const bench_config_t* const config, const int is_rms_norm)
{
	if (is_rms_norm && !ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN))
	{
		fprintf(stderr, "CCV_NNC_RMSNORM_FORWARD on GPU_CUDNN is unavailable\n");
		exit(1);
	}
	if (!is_rms_norm && !ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN))
	{
		fprintf(stderr, "CCV_NNC_LAYER_NORM_FORWARD on GPU_CUDNN is unavailable\n");
		exit(1);
	}
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int count = config->rows * config->cols;
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(config->datatype, config->rows, config->cols), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(config->datatype, 1, config->cols), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(config->datatype, 1, config->cols), 0);
	ccv_nnc_tensor_t* const y_cudnn = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(config->datatype, config->rows, config->cols), 0);
	ccv_nnc_tensor_t* const y_flash = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(config->datatype, config->rows, config->cols), 0);
	const int cudnn_stats_datatype = config->datatype == CCV_16BF ? CCV_32F : config->datatype;
	ccv_nnc_tensor_t* const mean_cudnn = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(cudnn_stats_datatype, config->rows, 1), 0);
	ccv_nnc_tensor_t* const inv_cudnn = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(cudnn_stats_datatype, config->rows, 1), 0);
	ccv_nnc_tensor_t* const mean_flash = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(CCV_32F, config->rows, 1), 0);
	ccv_nnc_tensor_t* const inv_flash = ccv_nnc_tensor_new(0, _make_gpu_tensor_param(CCV_32F, config->rows, 1), 0);

	_fill_input(x, count, config->cols, stream);
	_fill_scale(scale, config->cols, config->affine, stream);
	_fill_bias(bias, config->cols, stream);

	ccv_nnc_cmd_param_t cmd_params = {};
	cmd_params.size.dim[0] = 1;
	cmd_params.size.dim[1] = 1;
	cmd_params.size.dim[2] = 1;
	ccv_nnc_cmd_t cmd;
	if (is_rms_norm)
	{
		cmd_params.rmsnorm.axis[0] = 1;
		cmd_params.rmsnorm.count = 1;
		cmd_params.rmsnorm.epsilon = config->epsilon;
		cmd_params.rmsnorm.elementwise_affine = config->affine;
		cmd = ccv_nnc_cmd(CCV_NNC_RMSNORM_FORWARD, 0, cmd_params, 0);
	} else {
		cmd_params.lnorm.axis[0] = 1;
		cmd_params.lnorm.count = 1;
		cmd_params.lnorm.epsilon = config->epsilon;
		cmd_params.lnorm.elementwise_affine = config->affine;
		cmd = ccv_nnc_cmd(CCV_NNC_LAYER_NORM_FORWARD, 0, cmd_params, 0);
	}
	cmd.backend = CCV_NNC_BACKEND_GPU_CUDNN;
	ccv_nnc_tensor_t* cudnn_inputs[3] = {};
	int cudnn_input_size = 0;
	cudnn_inputs[cudnn_input_size++] = x;
	if (config->affine)
	{
		cudnn_inputs[cudnn_input_size++] = scale;
		if (!is_rms_norm)
			cudnn_inputs[cudnn_input_size++] = bias;
	}
	ccv_nnc_tensor_t* cudnn_outputs[3] = {};
	int cudnn_output_size = 0;
	cudnn_outputs[cudnn_output_size++] = y_cudnn;
	if (is_rms_norm)
		cudnn_outputs[cudnn_output_size++] = inv_cudnn;
	else {
		cudnn_outputs[cudnn_output_size++] = mean_cudnn;
		cudnn_outputs[cudnn_output_size++] = inv_cudnn;
	}

	layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;
	int device = 0;
	cudaDeviceProp props = {};
	CUDA_ENFORCE(cudaGetDevice(&device));
	CUDA_ENFORCE(cudaGetDeviceProperties(&props, device));
	launch_params.props = &props;
	launch_params.stream = stream;
	launch_params.params.rows = config->rows;
	launch_params.params.cols = config->cols;
	launch_params.params.x0 = x->data.u8;
	launch_params.params.x1 = 0;
	launch_params.params.residual = 0;
	launch_params.params.x = 0;
	launch_params.params.dmask = 0;
	launch_params.params.dmask1 = 0;
	launch_params.params.mu = mean_flash->data.f32;
	launch_params.params.rs = inv_flash->data.f32;
	launch_params.params.gamma = scale->data.u8;
	launch_params.params.gamma1 = 0;
	launch_params.params.rowscale = 0;
	launch_params.params.colscale = 0;
	launch_params.params.x0_subset = 0;
	launch_params.params.z_subset = 0;
	launch_params.params.beta = (!is_rms_norm && config->affine) ? bias->data.u8 : 0;
	launch_params.params.beta1 = 0;
	launch_params.params.z = y_flash->data.u8;
	launch_params.params.z1 = 0;
	launch_params.params.epsilon = config->epsilon;
	launch_params.params.dropout_keep_p = 1.f;
	launch_params.params.dropout_scale = 1.f;
	launch_params.params.inverse_cols = 1.f / (float)config->cols;
	launch_params.params.rowscale_const = 1.f;
	launch_params.params.is_rms_norm = is_rms_norm;
	launch_params.params.workspace = 0;
	launch_params.params.barrier = 0;
	if (!_launch_flash(launch_params, config->datatype, config->cols, true))
	{
		fprintf(stderr, "FlashAttention layer norm launcher is unavailable for dtype=%s cols=%d\n", _dtype_name(config->datatype), config->cols);
		exit(1);
	}
	if (launch_params.workspace_bytes > 0)
		CUDA_ENFORCE(cudaMalloc(&launch_params.params.workspace, launch_params.workspace_bytes));
	if (launch_params.barrier_size > 0)
	{
		CUDA_ENFORCE(cudaMalloc(&launch_params.params.barrier, launch_params.barrier_size * sizeof(int)));
		CUDA_ENFORCE(cudaMemsetAsync(launch_params.params.barrier, 0, launch_params.barrier_size * sizeof(int), stream));
	}

	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, cudnn_inputs, cudnn_input_size, cudnn_outputs, cudnn_output_size, stream_context));
	if (!_launch_flash(launch_params, config->datatype, config->cols, false))
	{
		fprintf(stderr, "FlashAttention layer norm launcher is unavailable for dtype=%s cols=%d\n", _dtype_name(config->datatype), config->cols);
		exit(1);
	}
	CUDA_ENFORCE(cudaStreamSynchronize(stream));

	float max_abs = 0;
	float mean_abs = 0;
	_compare_outputs(y_cudnn, y_flash, count, &max_abs, &mean_abs);

	const float cudnn_ms = _benchmark_command(cmd, cudnn_inputs, cudnn_input_size, cudnn_outputs, cudnn_output_size, stream_context, config->warmup, config->iterations);
	const float flash_ms = _benchmark_flash(launch_params, config->datatype, config->cols, config->warmup, config->iterations);
	const float speedup = cudnn_ms / flash_ms;
	printf("%-9s dtype=%-4s rows=%d cols=%d affine=%d cudnn_ms=%.6f flash_ms=%.6f speedup=%.3fx max_abs=%.8g mean_abs=%.8g\n",
		_op_name(is_rms_norm), _dtype_name(config->datatype), config->rows, config->cols, config->affine, cudnn_ms, flash_ms, speedup, max_abs, mean_abs);

	if (launch_params.params.workspace)
		CUDA_ENFORCE(cudaFree(launch_params.params.workspace));
	if (launch_params.params.barrier)
		CUDA_ENFORCE(cudaFree(launch_params.params.barrier));
	ccv_nnc_tensor_free(inv_flash);
	ccv_nnc_tensor_free(mean_flash);
	ccv_nnc_tensor_free(inv_cudnn);
	ccv_nnc_tensor_free(mean_cudnn);
	ccv_nnc_tensor_free(y_flash);
	ccv_nnc_tensor_free(y_cudnn);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(x);
	ccv_nnc_stream_context_free(stream_context);
}

int main(int argc, char** argv)
{
	bench_config_t config = {
		.rows = 4096,
		.cols = 1024,
		.warmup = 20,
		.iterations = 100,
		.datatype = CCV_16F,
		.op = BENCH_OP_BOTH,
		.affine = 1,
		.epsilon = 1e-6f,
	};
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc)
			config.rows = _parse_int_arg(argv[++i]);
		else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc)
			config.cols = _parse_int_arg(argv[++i]);
		else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
			config.warmup = _parse_int_arg(argv[++i]);
		else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
			config.iterations = _parse_int_arg(argv[++i]);
		else if (strcmp(argv[i], "--affine") == 0 && i + 1 < argc)
			config.affine = _parse_int_arg(argv[++i]);
		else if (strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc)
			config.epsilon = _parse_float_arg(argv[++i]);
		else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc)
		{
			const char* const dtype = argv[++i];
			if (strcmp(dtype, "f16") == 0)
				config.datatype = CCV_16F;
			else if (strcmp(dtype, "bf16") == 0)
				config.datatype = CCV_16BF;
			else if (strcmp(dtype, "f32") == 0)
				config.datatype = CCV_32F;
			else {
				fprintf(stderr, "unsupported dtype: %s\n", dtype);
				return 1;
			}
		} else if (strcmp(argv[i], "--op") == 0 && i + 1 < argc) {
			const char* const op = argv[++i];
			if (strcmp(op, "both") == 0)
				config.op = BENCH_OP_BOTH;
			else if (strcmp(op, "layernorm") == 0)
				config.op = BENCH_OP_LAYER_NORM;
			else if (strcmp(op, "rmsnorm") == 0)
				config.op = BENCH_OP_RMSNORM;
			else {
				fprintf(stderr, "unsupported op: %s\n", op);
				return 1;
			}
		} else if (strcmp(argv[i], "--help") == 0) {
			usage(argv[0]);
			return 0;
		} else {
			fprintf(stderr, "unknown argument: %s\n", argv[i]);
			usage(argv[0]);
			return 1;
		}
	}
	if (config.rows <= 0 || config.cols <= 0 || config.warmup < 0 || config.iterations <= 0)
	{
		fprintf(stderr, "invalid benchmark dimensions or iteration counts\n");
		return 1;
	}
	if (config.cols % 8 != 0)
	{
		fprintf(stderr, "cols must be divisible by 8 for the imported FlashAttention kernels\n");
		return 1;
	}
	if (layer_norm::round_hidden_size(config.cols) > 8192)
	{
		fprintf(stderr, "cols=%d is larger than the imported FlashAttention layer norm buckets\n", config.cols);
		return 1;
	}
	ccv_nnc_init();
	if (config.op == BENCH_OP_BOTH || config.op == BENCH_OP_LAYER_NORM)
		_run_one(&config, 0);
	if (config.op == BENCH_OP_BOTH || config.op == BENCH_OP_RMSNORM)
		_run_one(&config, 1);
	return 0;
}
