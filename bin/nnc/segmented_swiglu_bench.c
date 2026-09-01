#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef enum {
	SEGMENTED_SWIGLU_ROWWISE,
	SEGMENTED_SWIGLU_IQ2_XXS,
	SEGMENTED_SWIGLU_IQ2_XS,
	SEGMENTED_SWIGLU_IQ3_XXS,
	SEGMENTED_SWIGLU_Q2_K,
} segmented_swiglu_weight_format_t;

typedef enum {
	SEGMENTED_SWIGLU_COMPOSED_GROUPED,
	SEGMENTED_SWIGLU_FUSED_GROUPED,
	SEGMENTED_SWIGLU_FUSED_BROADCAST,
	SEGMENTED_SWIGLU_PATH_COUNT,
} segmented_swiglu_path_t;

typedef struct {
	double median_us;
	double mean_us;
	double min_us;
	int status;
} segmented_swiglu_benchmark_t;

static double _segmented_swiglu_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int _segmented_swiglu_double_compare(const void* const a, const void* const b)
{
	const double av = *(const double*)a;
	const double bv = *(const double*)b;
	return (av > bv) - (av < bv);
}

static const char* _segmented_swiglu_format_name(const segmented_swiglu_weight_format_t format)
{
	switch (format)
	{
		case SEGMENTED_SWIGLU_IQ2_XXS:
			return "iq2_xxs";
		case SEGMENTED_SWIGLU_IQ2_XS:
			return "iq2_xs";
		case SEGMENTED_SWIGLU_IQ3_XXS:
			return "iq3_xxs";
		case SEGMENTED_SWIGLU_Q2_K:
			return "q2_k";
		default:
			return "rowwise";
	}
}

static segmented_swiglu_weight_format_t _segmented_swiglu_parse_format(const char* const format)
{
	if (strcmp(format, "rowwise") == 0 || strcmp(format, "8i_rowwise") == 0)
		return SEGMENTED_SWIGLU_ROWWISE;
	if (strcmp(format, "iq2_xxs") == 0 || strcmp(format, "IQ2_XXS") == 0)
		return SEGMENTED_SWIGLU_IQ2_XXS;
	if (strcmp(format, "iq2_xs") == 0 || strcmp(format, "IQ2_XS") == 0)
		return SEGMENTED_SWIGLU_IQ2_XS;
	if (strcmp(format, "iq3_xxs") == 0 || strcmp(format, "IQ3_XXS") == 0)
		return SEGMENTED_SWIGLU_IQ3_XXS;
	if (strcmp(format, "q2_k") == 0 || strcmp(format, "Q2_K") == 0)
		return SEGMENTED_SWIGLU_Q2_K;
	fprintf(stderr, "weight format must be rowwise, iq2_xxs, iq2_xs, iq3_xxs, or q2_k\n");
	exit(1);
}

static ccv_nnc_tensor_param_t _segmented_swiglu_quantized_params(const ccv_nnc_tensor_param_t dense_params, const segmented_swiglu_weight_format_t format)
{
	switch (format)
	{
		case SEGMENTED_SWIGLU_IQ2_XXS:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
		case SEGMENTED_SWIGLU_IQ2_XS:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ2_XS);
		case SEGMENTED_SWIGLU_IQ3_XXS:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_IQ3_XXS);
		case SEGMENTED_SWIGLU_Q2_K:
			return ccv_nnc_tensor_8i_rowwise_x(dense_params, CCV_NNC_QX_8I_ROWWISE_Q2_K);
		default:
			return ccv_nnc_tensor_8i_rowwise(dense_params);
	}
}

static void _segmented_swiglu_fill_half(ccv_nnc_tensor_t* const tensor, const int row_length, const int multiplier, const int modulus, const float scale)
{
	const size_t count = ccv_nnc_tensor_count(tensor->info);
	assert(row_length > 0 && count % row_length == 0);
	float* const row = (float*)malloc(sizeof(float) * row_length);
	if (!row)
	{
		fprintf(stderr, "failed to allocate the deterministic fill row\n");
		exit(1);
	}
	const size_t row_count = count / row_length;
	size_t i;
	for (i = 0; i < row_count; i++)
	{
		int j;
		for (j = 0; j < row_length; j++)
		{
			const size_t index = i * row_length + j;
			row[j] = (float)((int)((index * multiplier) % modulus) - modulus / 2) * scale;
		}
		ccv_float_to_half_precision(row, (uint16_t*)tensor->data.f16 + i * row_length, row_length);
	}
	free(row);
}

static void _segmented_swiglu_fill_quantized(ccv_nnc_tensor_t* const tensor, const int seed, const float scale)
{
	const int tensor_nd = ccv_nnc_tensor_nd(tensor->info.dim);
	const size_t row_length = tensor->info.dim[tensor_nd - 1];
	const size_t row_count = ccv_nnc_tensor_count(tensor->info) / row_length;
	const size_t tensor_size = ccv_nnc_tensor_data_size_without_padding(tensor->info);
	const size_t scale_size = row_count * sizeof(ccv_float16_t);
	assert(tensor_size >= scale_size);
	const size_t scale_offset = tensor_size - scale_size;
	size_t i;
	for (i = 0; i < scale_offset; i++)
		tensor->data.u8[i] = (uint8_t)(i * seed + (i >> 7) * 17 + 31);
	float* const scales = (float*)malloc(sizeof(float) * row_count);
	if (!scales)
	{
		fprintf(stderr, "failed to allocate quantized weight scales\n");
		exit(1);
	}
	for (i = 0; i < row_count; i++)
		scales[i] = scale * (0.875f + (float)(i % 5) * 0.0625f);
	ccv_float_to_half_precision(scales, (uint16_t*)(tensor->data.u8 + scale_offset), row_count);
	free(scales);
}

static int _segmented_swiglu_composed(const ccv_nnc_cmd_t segmented, const ccv_nnc_cmd_t weighted_swish,
	ccv_nnc_tensor_t* const a, ccv_nnc_tensor_t* const indices, ccv_nnc_tensor_t* const counts,
	ccv_nnc_tensor_t* const gate_w, ccv_nnc_tensor_t* const up_w, ccv_nnc_tensor_t* const route_weight,
	ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const output,
	ccv_nnc_stream_context_t* const stream)
{
	int status = ccv_nnc_cmd_exec(segmented, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, gate_w), TENSOR_LIST(gate), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(segmented, ccv_nnc_no_hint, 0,
			TENSOR_LIST(a, indices, counts, up_w), TENSOR_LIST(up), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(weighted_swish, ccv_nnc_no_hint, 0,
			TENSOR_LIST(up, gate, route_weight), TENSOR_LIST(output), stream);
	return status;
}

static int _segmented_swiglu_fused(const ccv_nnc_cmd_t fused,
	ccv_nnc_tensor_t* const a, ccv_nnc_tensor_t* const indices, ccv_nnc_tensor_t* const counts,
	ccv_nnc_tensor_t* const gate_w, ccv_nnc_tensor_t* const up_w, ccv_nnc_tensor_t* const route_weight,
	ccv_nnc_tensor_t* const output, ccv_nnc_stream_context_t* const stream)
{
	return ccv_nnc_cmd_exec(fused, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, gate_w, up_w, route_weight), TENSOR_LIST(output), stream);
}

static int _segmented_swiglu_run(const segmented_swiglu_path_t path,
	const ccv_nnc_cmd_t segmented, const ccv_nnc_cmd_t weighted_swish, const ccv_nnc_cmd_t fused,
	ccv_nnc_tensor_t* const* const activations, ccv_nnc_tensor_t* const indices, ccv_nnc_tensor_t* const counts,
	ccv_nnc_tensor_t* const gate_w, ccv_nnc_tensor_t* const up_w, ccv_nnc_tensor_t* const route_weight,
	ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const* const outputs,
	ccv_nnc_stream_context_t* const stream)
{
	if (path == SEGMENTED_SWIGLU_COMPOSED_GROUPED)
		return _segmented_swiglu_composed(segmented, weighted_swish, activations[path], indices, counts,
			gate_w, up_w, route_weight, gate, up, outputs[path], stream);
	return _segmented_swiglu_fused(fused, activations[path], indices, counts,
		gate_w, up_w, route_weight, outputs[path], stream);
}

static int _segmented_swiglu_benchmark(
	const ccv_nnc_cmd_t segmented, const ccv_nnc_cmd_t weighted_swish, const ccv_nnc_cmd_t fused,
	ccv_nnc_tensor_t* const* const activations, ccv_nnc_tensor_t* const indices, ccv_nnc_tensor_t* const counts,
	ccv_nnc_tensor_t* const gate_w, ccv_nnc_tensor_t* const up_w, ccv_nnc_tensor_t* const route_weight,
	ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const* const outputs,
	ccv_nnc_stream_context_t* const stream, const int warmup, const int iterations, const int batch_size,
	const int path_count, segmented_swiglu_benchmark_t results[SEGMENTED_SWIGLU_PATH_COUNT])
{
	int path;
	for (path = 0; path < SEGMENTED_SWIGLU_PATH_COUNT; path++)
		results[path] = (segmented_swiglu_benchmark_t){
			.median_us = 0,
			.mean_us = 0,
			.min_us = DBL_MAX,
			.status = CCV_NNC_EXEC_SUCCESS,
		};
	int status = CCV_NNC_EXEC_SUCCESS;
	int iteration;
	for (iteration = 0; iteration < warmup && status == CCV_NNC_EXEC_SUCCESS; iteration++)
	{
		int order;
		for (order = 0; order < path_count && status == CCV_NNC_EXEC_SUCCESS; order++)
		{
			const segmented_swiglu_path_t scheduled_path = (iteration + order) % path_count;
			int batch;
			for (batch = 0; batch < batch_size && status == CCV_NNC_EXEC_SUCCESS; batch++)
				status = _segmented_swiglu_run(scheduled_path, segmented, weighted_swish, fused,
					activations, indices, counts, gate_w, up_w, route_weight, gate, up, outputs, stream);
			ccv_nnc_stream_context_wait(stream);
		}
	}
	double* const samples = (double*)malloc(sizeof(double) * iterations * path_count);
	if (!samples)
		return CCV_NNC_EXEC_OOM;
	for (iteration = 0; iteration < iterations && status == CCV_NNC_EXEC_SUCCESS; iteration++)
	{
		int order;
		for (order = 0; order < path_count && status == CCV_NNC_EXEC_SUCCESS; order++)
		{
			const segmented_swiglu_path_t scheduled_path = (iteration + order) % path_count;
			const double start = _segmented_swiglu_current_time();
			int batch;
			for (batch = 0; batch < batch_size && status == CCV_NNC_EXEC_SUCCESS; batch++)
				status = _segmented_swiglu_run(scheduled_path, segmented, weighted_swish, fused,
					activations, indices, counts, gate_w, up_w, route_weight, gate, up, outputs, stream);
			ccv_nnc_stream_context_wait(stream);
			const double sample_us = (_segmented_swiglu_current_time() - start) * 1000000.0 / batch_size;
			samples[scheduled_path * iterations + iteration] = sample_us;
			results[scheduled_path].mean_us += sample_us;
			results[scheduled_path].min_us = ccv_min(results[scheduled_path].min_us, sample_us);
		}
	}
	for (path = 0; path < path_count; path++)
	{
		results[path].status = status;
		if (status == CCV_NNC_EXEC_SUCCESS)
		{
			double* const path_samples = samples + path * iterations;
			qsort(path_samples, iterations, sizeof(double), _segmented_swiglu_double_compare);
			results[path].median_us = iterations & 1 ? path_samples[iterations / 2] :
				(path_samples[iterations / 2 - 1] + path_samples[iterations / 2]) * 0.5;
			results[path].mean_us /= iterations;
		}
	}
	free(samples);
	return status;
}

static int _segmented_swiglu_compare(const char* const path, const ccv_nnc_tensor_t* const composed, const ccv_nnc_tensor_t* const fused)
{
	const int count = ccv_nnc_tensor_count(composed->info);
	float* const composed_float = (float*)malloc(sizeof(float) * count);
	float* const fused_float = (float*)malloc(sizeof(float) * count);
	if (!composed_float || !fused_float)
	{
		free(fused_float);
		free(composed_float);
		return 0;
	}
	ccv_half_precision_to_float((const uint16_t*)composed->data.f16, composed_float, count);
	ccv_half_precision_to_float((const uint16_t*)fused->data.f16, fused_float, count);
	double difference_norm = 0;
	double composed_norm = 0;
	double fused_norm = 0;
	double dot = 0;
	double max_abs = 0;
	int i;
	for (i = 0; i < count; i++)
	{
		const double difference = (double)fused_float[i] - composed_float[i];
		difference_norm += difference * difference;
		composed_norm += (double)composed_float[i] * composed_float[i];
		fused_norm += (double)fused_float[i] * fused_float[i];
		dot += (double)fused_float[i] * composed_float[i];
		max_abs = ccv_max(max_abs, fabs(difference));
	}
	const double relative_l2 = sqrt(difference_norm / ccv_max(composed_norm, 1e-20));
	const double cosine = dot / sqrt(ccv_max(composed_norm * fused_norm, 1e-20));
	printf("validation,path=%s,relative_l2=%.8g,cosine=%.8g,max_abs=%.8g\n", path, relative_l2, cosine, max_abs);
	free(fused_float);
	free(composed_float);
	return relative_l2 < 0.02 && cosine > 0.999;
}

static int _segmented_swiglu_fp32_prefill_benchmark(const int warmup, const int iterations, const int token_count)
{
	const int experts = 256;
	const int n = 2048;
	const int k = 4096;
	const int rows = token_count * 6;
	const int segments = ccv_min(rows, experts);
	ccv_nnc_init();
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hroute = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows), 0);
	const ccv_nnc_tensor_param_t dense_weight_params = GPU_TENSOR_NHWC(000, 32F, experts, n, k);
	const ccv_nnc_tensor_param_t quantized_weight_params = ccv_nnc_tensor_8i_rowwise(dense_weight_params);
	ccv_nnc_tensor_t* const gate_w = ccv_nnc_tensor_new(0, quantized_weight_params, 0);
	ccv_nnc_tensor_t* const up_w = ccv_nnc_tensor_new(0, quantized_weight_params, 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	if (!ha || !hindices || !hcounts || !hroute || !a || !indices || !counts || !route || !gate_w || !up_w || !output)
	{
		fprintf(stderr, "fp32 prefill allocation failed\n");
		return 1;
	}
	const int activation_count = ccv_nnc_tensor_count(ha->info);
	int i;
	for (i = 0; i < activation_count; i++)
		ha->data.f32[i] = (float)((i % 127) - 63) / 64;
	for (i = 0; i < segments; i++)
	{
		hindices->data.i32[i] = i;
		hcounts->data.i32[i] = rows / segments + (i < rows % segments);
	}
	for (i = 0; i < rows; i++)
		hroute->data.f32[i] = 1;
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hindices, hcounts, hroute), TENSOR_LIST(a, indices, counts, route), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(10);
	command.backend = CCV_NNC_BACKEND_MPS;
	for (i = 0; i < warmup && status == CCV_NNC_EXEC_SUCCESS; i++)
	{
		status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
			TENSOR_LIST(a, indices, counts, gate_w, up_w, route), TENSOR_LIST(output), stream);
		ccv_nnc_stream_context_wait(stream);
	}
	double* const samples = (double*)malloc(sizeof(double) * iterations);
	for (i = 0; i < iterations && status == CCV_NNC_EXEC_SUCCESS; i++)
	{
		const double start = _segmented_swiglu_current_time();
		status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
			TENSOR_LIST(a, indices, counts, gate_w, up_w, route), TENSOR_LIST(output), stream);
		ccv_nnc_stream_context_wait(stream);
		samples[i] = (_segmented_swiglu_current_time() - start) * 1000;
	}
	if (status == CCV_NNC_EXEC_SUCCESS)
	{
		qsort(samples, iterations, sizeof(double), _segmented_swiglu_double_compare);
		printf("fp32,rowwise,prefill,T=%d,M=%d,S=%d,E=%d,N=%d,K=%d,median_ms=%.4f,min_ms=%.4f\n",
			token_count, rows, segments, experts, n, k, samples[iterations / 2], samples[0]);
	}
	free(samples);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(up_w);
	ccv_nnc_tensor_free(gate_w);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hroute);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(ha);
	return status != CCV_NNC_EXEC_SUCCESS;
}

int main(int argc, char** argv)
{
	if (argc > 1 && strcmp(argv[1], "prefill_fp32") == 0)
		return _segmented_swiglu_fp32_prefill_benchmark(
			argc > 2 ? atoi(argv[2]) : 2,
			argc > 3 ? atoi(argv[3]) : 7,
			argc > 4 ? atoi(argv[4]) : 2048);
	const segmented_swiglu_weight_format_t format = _segmented_swiglu_parse_format(argc > 1 ? argv[1] : "iq2_xxs");
	const int warmup = argc > 2 ? atoi(argv[2]) : 5;
	const int iterations = argc > 3 ? atoi(argv[3]) : 30;
	const int batch_size = argc > 4 ? atoi(argv[4]) : 8;
	const int experts = argc > 5 ? atoi(argv[5]) : 6;
	const int rows = argc > 6 ? atoi(argv[6]) : 6;
	const int segments = argc > 7 ? atoi(argv[7]) : 6;
	if (warmup < 0 || iterations <= 0 || batch_size <= 0 || experts <= 0 || rows <= 0 || segments <= 0 || rows < segments)
	{
		fprintf(stderr, "usage: %s [rowwise|iq2_xxs|iq2_xs|iq3_xxs|q2_k] [warmup>=0] [iterations>0] [batch_size>0] [experts>0] [rows>=segments] [segments>0]\n", argv[0]);
		return 1;
	}
	ccv_nnc_init();
	enum {
		n = 2048,
		k = 4096,
	};
	const float clamp = 10;
	ccv_nnc_tensor_t* const ha_grouped = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, k), 0);
	ccv_nnc_tensor_t* const ha_broadcast = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, k), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hroute = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows), 0);
	const ccv_nnc_tensor_param_t dense_params = CPU_TENSOR_NHWC(16F, experts, n, k);
	const ccv_nnc_tensor_param_t q_params = _segmented_swiglu_quantized_params(dense_params, format);
	ccv_nnc_tensor_t* const hgate_q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const hup_q = ccv_nnc_tensor_new(0, q_params, 0);
	if (!ha_grouped || !ha_broadcast || !hindices || !hcounts || !hroute || !hgate_q || !hup_q)
	{
		fprintf(stderr, "host tensor allocation failed\n");
		return 1;
	}
	int i;
	_segmented_swiglu_fill_half(ha_broadcast, k, 17, 251, 1.0f / 128);
	for (i = 0; i < rows; i++)
		memcpy(ha_grouped->data.f16 + i * k, ha_broadcast->data.f16, sizeof(ccv_float16_t) * k);
	for (i = 0; i < segments; i++)
	{
		hindices->data.i32[i] = (i * 5 + 3) % experts;
		hcounts->data.i32[i] = rows / segments + (i < rows % segments);
	}
	float route_float[rows];
	for (i = 0; i < rows; i++)
		route_float[i] = (float)(i + 2) / 8;
	ccv_float_to_half_precision(route_float, (uint16_t*)hroute->data.f16, rows);
	// Packed values do not affect kernel timing. Build valid deterministic payloads directly so
	// benchmark setup does not measure the much slower offline IQ2_XXS quantizer.
	fprintf(stderr, "synthesizing two %s weight tables (%.3f GiB each)...\n",
		_segmented_swiglu_format_name(format),
		(double)ccv_nnc_tensor_data_size_without_padding(q_params) / 1073741824.0);
	_segmented_swiglu_fill_quantized(hgate_q, 29, format != SEGMENTED_SWIGLU_ROWWISE ? 1.0f / 1024 : 1.0f / 8192);
	_segmented_swiglu_fill_quantized(hup_q, 37, format != SEGMENTED_SWIGLU_ROWWISE ? 1.0f / 1024 : 1.0f / 8192);

	ccv_nnc_tensor_param_t gpu_q_params = q_params;
	gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const a_grouped = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, k), 0);
	ccv_nnc_tensor_t* const a_broadcast = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows), 0);
	ccv_nnc_tensor_t* const gate_w = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const up_w = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, n), 0);
	ccv_nnc_tensor_t* const up = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, n), 0);
	ccv_nnc_tensor_t* const composed_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, n), 0);
	ccv_nnc_tensor_t* const fused_grouped_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, n), 0);
	ccv_nnc_tensor_t* const fused_broadcast_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, n), 0);
	ccv_nnc_tensor_t* const hcomposed_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, n), 0);
	ccv_nnc_tensor_t* const hfused_grouped_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, n), 0);
	ccv_nnc_tensor_t* const hfused_broadcast_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha_grouped, ha_broadcast, hindices, hcounts, hgate_q, hup_q, hroute),
		TENSOR_LIST(a_grouped, a_broadcast, indices, counts, gate_w, up_w, route), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS)
	{
		fprintf(stderr, "input transfer failed with status %d\n", status);
		return 1;
	}

	ccv_nnc_cmd_t segmented = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	ccv_nnc_cmd_t weighted_swish = CMD_WEIGHTED_SWISH_MUL_FORWARD(1, 1, clamp);
	ccv_nnc_cmd_t fused = CMD_SEGMENTED_SWIGLU_FORWARD(clamp);
	segmented.backend = CCV_NNC_BACKEND_MPS;
	weighted_swish.backend = CCV_NNC_BACKEND_MPS;
	fused.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	const int old_watermark = ccv_nnc_queue_watermark();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_set_queue_watermark(batch_size * 3 + 1);
	status = _segmented_swiglu_composed(segmented, weighted_swish, a_grouped, indices, counts,
		gate_w, up_w, route, gate, up, composed_output, stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _segmented_swiglu_fused(fused, a_grouped, indices, counts, gate_w, up_w, route, fused_grouped_output, stream);
	const int path_count = rows == segments ? SEGMENTED_SWIGLU_PATH_COUNT : 2;
	if (status == CCV_NNC_EXEC_SUCCESS && path_count == SEGMENTED_SWIGLU_PATH_COUNT)
		status = _segmented_swiglu_fused(fused, a_broadcast, indices, counts, gate_w, up_w, route, fused_broadcast_output, stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(composed_output, fused_grouped_output, fused_broadcast_output),
			TENSOR_LIST(hcomposed_output, hfused_grouped_output, hfused_broadcast_output), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS ||
		!_segmented_swiglu_compare("fused_grouped", hcomposed_output, hfused_grouped_output) ||
		(path_count == SEGMENTED_SWIGLU_PATH_COUNT &&
			!_segmented_swiglu_compare("fused_broadcast", hcomposed_output, hfused_broadcast_output)))
	{
		fprintf(stderr, "fused-vs-composed validation failed (status %d)\n", status);
		return 1;
	}

	ccv_nnc_tensor_t* const activations[SEGMENTED_SWIGLU_PATH_COUNT] = {
		a_grouped, a_grouped, a_broadcast,
	};
	ccv_nnc_tensor_t* const outputs[SEGMENTED_SWIGLU_PATH_COUNT] = {
		composed_output, fused_grouped_output, fused_broadcast_output,
	};
	segmented_swiglu_benchmark_t results[SEGMENTED_SWIGLU_PATH_COUNT];
	status = _segmented_swiglu_benchmark(segmented, weighted_swish, fused,
		activations, indices, counts, gate_w, up_w, route, gate, up, outputs, stream,
		warmup, iterations, batch_size, path_count, results);
	printf("format,M,S,E,N,K,clamp,path,median_us,mean_us,min_us,speedup_vs_composed,speedup_vs_fused_grouped\n");
	printf("%s,%d,%d,%d,%d,%d,%.1f,composed_grouped,%.3f,%.3f,%.3f,1.000,%.3f\n",
		_segmented_swiglu_format_name(format), rows, segments, experts, n, k, clamp,
		results[SEGMENTED_SWIGLU_COMPOSED_GROUPED].median_us,
		results[SEGMENTED_SWIGLU_COMPOSED_GROUPED].mean_us,
		results[SEGMENTED_SWIGLU_COMPOSED_GROUPED].min_us,
		results[SEGMENTED_SWIGLU_FUSED_GROUPED].median_us /
			results[SEGMENTED_SWIGLU_COMPOSED_GROUPED].median_us);
	printf("%s,%d,%d,%d,%d,%d,%.1f,fused_grouped,%.3f,%.3f,%.3f,%.3f,1.000\n",
		_segmented_swiglu_format_name(format), rows, segments, experts, n, k, clamp,
		results[SEGMENTED_SWIGLU_FUSED_GROUPED].median_us,
		results[SEGMENTED_SWIGLU_FUSED_GROUPED].mean_us,
		results[SEGMENTED_SWIGLU_FUSED_GROUPED].min_us,
		results[SEGMENTED_SWIGLU_COMPOSED_GROUPED].median_us /
			results[SEGMENTED_SWIGLU_FUSED_GROUPED].median_us);
	if (path_count == SEGMENTED_SWIGLU_PATH_COUNT)
		printf("%s,%d,%d,%d,%d,%d,%.1f,fused_broadcast,%.3f,%.3f,%.3f,%.3f,%.3f\n",
			_segmented_swiglu_format_name(format), rows, segments, experts, n, k, clamp,
			results[SEGMENTED_SWIGLU_FUSED_BROADCAST].median_us,
			results[SEGMENTED_SWIGLU_FUSED_BROADCAST].mean_us,
			results[SEGMENTED_SWIGLU_FUSED_BROADCAST].min_us,
			results[SEGMENTED_SWIGLU_COMPOSED_GROUPED].median_us /
				results[SEGMENTED_SWIGLU_FUSED_BROADCAST].median_us,
			results[SEGMENTED_SWIGLU_FUSED_GROUPED].median_us /
				results[SEGMENTED_SWIGLU_FUSED_BROADCAST].median_us);
	if (status != CCV_NNC_EXEC_SUCCESS)
		fprintf(stderr, "benchmark failed with status %d\n", status);

	ccv_nnc_set_queue_watermark(old_watermark);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(hfused_broadcast_output);
	ccv_nnc_tensor_free(hfused_grouped_output);
	ccv_nnc_tensor_free(hcomposed_output);
	ccv_nnc_tensor_free(fused_broadcast_output);
	ccv_nnc_tensor_free(fused_grouped_output);
	ccv_nnc_tensor_free(composed_output);
	ccv_nnc_tensor_free(up);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(up_w);
	ccv_nnc_tensor_free(gate_w);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a_broadcast);
	ccv_nnc_tensor_free(a_grouped);
	ccv_nnc_tensor_free(hup_q);
	ccv_nnc_tensor_free(hgate_q);
	ccv_nnc_tensor_free(hroute);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(ha_broadcast);
	ccv_nnc_tensor_free(ha_grouped);
	return status != CCV_NNC_EXEC_SUCCESS;
}
