#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct {
	double median_us;
	double mean_us;
	double min_us;
	int status;
} moe_routing_benchmark_t;

static double _moe_routing_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int _moe_routing_double_compare(const void* const a, const void* const b)
{
	const double av = *(const double*)a;
	const double bv = *(const double*)b;
	return (av > bv) - (av < bv);
}

static moe_routing_benchmark_t _moe_routing_benchmark(const int preselected, const int disable_mfa, const int single_input_token, const int warmup, const int iterations, const int batch_size, const int hidden)
{
	const int expert_count = 256;
	const int kth = 6;
	moe_routing_benchmark_t result = { .median_us = 0, .mean_us = 0, .min_us = DBL_MAX, .status = CCV_NNC_EXEC_SUCCESS };
	if (disable_mfa)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_set_queue_watermark(batch_size);
	ccv_nnc_tensor_t* const hlogits = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, expert_count), 0);
	ccv_nnc_tensor_t* const hroute = ccv_nnc_tensor_new(0, preselected ? CPU_TENSOR_NHWC(32S, 1, kth) : CPU_TENSOR_NHWC(32F, expert_count), 0);
	ccv_nnc_tensor_t* const hactivation_float = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, hidden), 0);
	ccv_nnc_tensor_t* const hactivation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, hidden), 0);
	ccv_nnc_tensor_t* const logits = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, expert_count), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, preselected ? GPU_TENSOR_NHWC(000, 32S, 1, kth) : GPU_TENSOR_NHWC(000, 32F, expert_count), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, hidden), 0);
	ccv_nnc_tensor_t* const gathered = ccv_nnc_tensor_new(0, single_input_token ? GPU_TENSOR_NHWC(000, 16F, 1, hidden) : GPU_TENSOR_NHWC(000, 16F, kth, hidden), 0);
	ccv_nnc_tensor_t* const route_weights = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, kth), 0);
	ccv_nnc_tensor_t* const token_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, kth), 0);
	ccv_nnc_tensor_t* const expert_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, kth), 0);
	ccv_nnc_tensor_t* const expert_counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, kth), 0);
	int i;
	for (i = 0; i < expert_count; i++)
		hlogits->data.f32[i] = (float)((i * 37) % 257 - 128) * 0.03125f;
	if (preselected)
	{
		const int selected[] = { 83, 122, 142, 141, 0, 121 };
		memcpy(hroute->data.i32, selected, sizeof(selected));
	} else {
		for (i = 0; i < expert_count; i++)
			hroute->data.f32[i] = (float)i * 0.00001f;
	}
	for (i = 0; i < hidden; i++)
		hactivation_float->data.f32[i] = (float)((i * 13) % 127 - 63) / 64.0f;
	ccv_float_to_half_precision(hactivation_float->data.f32, (uint16_t*)hactivation->data.f16, hidden);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	result.status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hlogits, hroute, hactivation), TENSOR_LIST(logits, route, activation), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_cmd_t cmd = CMD_MOE_ROUTING_FORWARD_FLAGS(kth, 1.5f, preselected,
		single_input_token ? CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN : 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	int iteration;
	for (iteration = 0; iteration < warmup && result.status == CCV_NNC_EXEC_SUCCESS; iteration++)
	{
		int batch;
		for (batch = 0; batch < batch_size; batch++)
			result.status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0,
				TENSOR_LIST(logits, route, activation),
				TENSOR_LIST(gathered, route_weights, token_indices, expert_indices, expert_counts), stream);
		ccv_nnc_stream_context_wait(stream);
	}
	double* const samples = (double*)malloc(sizeof(double) * iterations);
	double total_us = 0;
	for (iteration = 0; iteration < iterations && result.status == CCV_NNC_EXEC_SUCCESS; iteration++)
	{
		const double start = _moe_routing_current_time();
		int batch;
		for (batch = 0; batch < batch_size; batch++)
			result.status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0,
				TENSOR_LIST(logits, route, activation),
				TENSOR_LIST(gathered, route_weights, token_indices, expert_indices, expert_counts), stream);
		ccv_nnc_stream_context_wait(stream);
		const double sample_us = (_moe_routing_current_time() - start) * 1000000.0 / batch_size;
		samples[iteration] = sample_us;
		total_us += sample_us;
		result.min_us = ccv_min(result.min_us, sample_us);
	}
	if (result.status == CCV_NNC_EXEC_SUCCESS)
	{
		qsort(samples, iterations, sizeof(double), _moe_routing_double_compare);
		result.median_us = iterations & 1 ? samples[iterations / 2] : (samples[iterations / 2 - 1] + samples[iterations / 2]) * 0.5;
		result.mean_us = total_us / iterations;
	}
	free(samples);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(expert_counts);
	ccv_nnc_tensor_free(expert_indices);
	ccv_nnc_tensor_free(token_indices);
	ccv_nnc_tensor_free(route_weights);
	ccv_nnc_tensor_free(gathered);
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(logits);
	ccv_nnc_tensor_free(hactivation);
	ccv_nnc_tensor_free(hactivation_float);
	ccv_nnc_tensor_free(hroute);
	ccv_nnc_tensor_free(hlogits);
	return result;
}

static int _moe_routing_report(const int preselected, const int warmup, const int iterations, const int batch_size, const int hidden)
{
	const moe_routing_benchmark_t graph_expanded = _moe_routing_benchmark(preselected, 1, 0, warmup, iterations, batch_size, hidden);
	const moe_routing_benchmark_t graph_single_input_token = _moe_routing_benchmark(preselected, 1, 1, warmup, iterations, batch_size, hidden);
	const moe_routing_benchmark_t mfa_expanded = _moe_routing_benchmark(preselected, 0, 0, warmup, iterations, batch_size, hidden);
	const moe_routing_benchmark_t mfa_single_input_token = _moe_routing_benchmark(preselected, 0, 1, warmup, iterations, batch_size, hidden);
	if (graph_expanded.status != CCV_NNC_EXEC_SUCCESS || graph_single_input_token.status != CCV_NNC_EXEC_SUCCESS ||
		mfa_expanded.status != CCV_NNC_EXEC_SUCCESS || mfa_single_input_token.status != CCV_NNC_EXEC_SUCCESS)
	{
		fprintf(stderr, "moe routing benchmark failed: graph-expanded=%d graph-single-input-token=%d mfa-expanded=%d mfa-single-input-token=%d\n",
			graph_expanded.status, graph_single_input_token.status, mfa_expanded.status, mfa_single_input_token.status);
		return 1;
	}
	printf("%s,mpsgraph,expanded,1,256,6,%d,fp16,%.3f,%.3f,%.3f,1.000\n",
		preselected ? "preselected" : "standard", hidden, graph_expanded.median_us, graph_expanded.mean_us, graph_expanded.min_us);
	printf("%s,mpsgraph,single_input_token,1,256,6,%d,fp16,%.3f,%.3f,%.3f,%.3f\n",
		preselected ? "preselected" : "standard", hidden, graph_single_input_token.median_us, graph_single_input_token.mean_us, graph_single_input_token.min_us, graph_expanded.median_us / graph_single_input_token.median_us);
	printf("%s,mfa,expanded,1,256,6,%d,fp16,%.3f,%.3f,%.3f,1.000\n",
		preselected ? "preselected" : "standard", hidden, mfa_expanded.median_us, mfa_expanded.mean_us, mfa_expanded.min_us);
	printf("%s,mfa,single_input_token,1,256,6,%d,fp16,%.3f,%.3f,%.3f,%.3f\n",
		preselected ? "preselected" : "standard", hidden, mfa_single_input_token.median_us, mfa_single_input_token.mean_us, mfa_single_input_token.min_us, mfa_expanded.median_us / mfa_single_input_token.median_us);
	return 0;
}

int main(int argc, char** argv)
{
	const int warmup = argc > 1 ? atoi(argv[1]) : 10;
	const int iterations = argc > 2 ? atoi(argv[2]) : 100;
	const int batch_size = argc > 3 ? atoi(argv[3]) : 32;
	const int hidden = argc > 4 ? atoi(argv[4]) : 4096;
	if (warmup < 0 || iterations <= 0 || batch_size <= 0 || hidden <= 0)
	{
		fprintf(stderr, "usage: %s [warmup>=0] [iterations>0] [batch_size>0] [hidden>0]\n", argv[0]);
		return 1;
	}
	ccv_nnc_init();
	const uint64_t old_flags = ccv_nnc_flags();
	const int old_watermark = ccv_nnc_queue_watermark();
	printf("mode,backend,activation,T,E,K,H,dtype,median_us,mean_us,min_us,speedup_vs_expanded\n");
	const int standard_status = _moe_routing_report(0, warmup, iterations, batch_size, hidden);
	const int preselected_status = _moe_routing_report(1, warmup, iterations, batch_size, hidden);
	ccv_nnc_set_queue_watermark(old_watermark);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	return standard_status || preselected_status;
}
