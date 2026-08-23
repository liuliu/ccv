#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

extern "C" {
#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
}

namespace {

struct Config {
	int warmup_samples = 5;
	int timed_samples = 20;
	int dispatches_per_sample = 50;
};

struct Shape {
	const char* name;
	int batch_size;
	int rows;
	int cols;
	int source_row_stride;
	int source_offset;
};

struct Stats {
	double average_ms;
	double median_ms;
	double p10_ms;
	double p90_ms;
	double min_ms;
	double max_ms;
};

enum class Path {
	MFA,
	MPSGraph,
};

bool parse_positive(const char* const text, int* const value)
{
	const long parsed = std::strtol(text, 0, 10);
	if (parsed <= 0 || parsed > INT32_MAX)
		return false;
	*value = (int)parsed;
	return true;
}

void print_usage(const char* const argv0)
{
	std::cerr << "usage: " << argv0 << " [warmup_samples timed_samples dispatches_per_sample]\n";
}

void set_path(const Path path)
{
	if (path == Path::MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
}

void restore_flags(const uint64_t saved_flags)
{
	if (saved_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
}

Stats compute_stats(std::vector<double> values)
{
	double sum = 0;
	for (const double value : values)
		sum += value;
	std::sort(values.begin(), values.end());
	return {
		.average_ms = sum / values.size(),
		.median_ms = values[values.size() / 2],
		.p10_ms = values[(values.size() - 1) / 10],
		.p90_ms = values[((values.size() - 1) * 9) / 10],
		.min_ms = values.front(),
		.max_ms = values.back(),
	};
}

bool run_sample(const ccv_nnc_cmd_t cmd, ccv_nnc_tensor_t* const source, ccv_nnc_tensor_t* const output, ccv_nnc_stream_context_t* const stream, const int dispatches, double* const elapsed_ms)
{
	const auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < dispatches; i++)
		if (ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(source), TENSOR_LIST(output), stream) != CCV_NNC_EXEC_SUCCESS)
			return false;
	ccv_nnc_stream_context_wait(stream);
	const auto end = std::chrono::steady_clock::now();
	if (elapsed_ms)
		*elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / dispatches;
	return true;
}

bool output_matches(const ccv_nnc_tensor_t* const expected, const ccv_nnc_tensor_t* const actual, const uint64_t count, const char* const path, const Shape& shape)
{
	if (std::memcmp(expected->data.f16, actual->data.f16, count * sizeof(uint16_t)) == 0)
		return true;
	uint64_t i;
	for (i = 0; i < count; i++)
		if (((const uint16_t*)expected->data.f16)[i] != ((const uint16_t*)actual->data.f16)[i])
			break;
	std::cerr << shape.name << " " << path << " differs at element " << i
		<< ": expected=" << ((const uint16_t*)expected->data.f16)[i]
		<< " actual=" << ((const uint16_t*)actual->data.f16)[i] << "\n";
	return false;
}

void print_stats(const char* const name, const Stats& stats, const uint64_t effective_bytes)
{
	const double effective_gbps = effective_bytes / (stats.median_ms * 1e6);
	std::cout << name
		<< " average_ms=" << stats.average_ms
		<< " median_ms=" << stats.median_ms
		<< " p10_ms=" << stats.p10_ms
		<< " p90_ms=" << stats.p90_ms
		<< " min_ms=" << stats.min_ms
		<< " max_ms=" << stats.max_ms
		<< " effective_GBps=" << effective_gbps
		<< "\n";
}

bool benchmark_shape(const Shape& shape, const Config& config, ccv_nnc_stream_context_t* const stream)
{
	const int source_batch_stride = shape.rows * shape.source_row_stride;
	const uint64_t source_count = (uint64_t)shape.batch_size * source_batch_stride;
	const uint64_t output_count = (uint64_t)shape.batch_size * shape.rows * shape.cols;
	ccv_nnc_tensor_param_t host_source_params = {};
	host_source_params.type = CCV_TENSOR_CPU_MEMORY;
	host_source_params.format = CCV_TENSOR_FORMAT_NHWC;
	host_source_params.datatype = CCV_16F;
	host_source_params.dim[0] = shape.batch_size;
	host_source_params.dim[1] = shape.rows;
	host_source_params.dim[2] = shape.source_row_stride;
	ccv_nnc_tensor_param_t source_params = host_source_params;
	source_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_param_t source_view_params = source_params;
	source_view_params.dim[2] = shape.cols;
	ccv_nnc_tensor_param_t host_output_params = {};
	host_output_params.type = CCV_TENSOR_CPU_MEMORY;
	host_output_params.format = CCV_TENSOR_FORMAT_NHWC;
	host_output_params.datatype = CCV_16F;
	host_output_params.dim[0] = shape.batch_size;
	host_output_params.dim[1] = shape.cols;
	host_output_params.dim[2] = shape.rows;
	ccv_nnc_tensor_param_t output_params = host_output_params;
	output_params.type = CCV_TENSOR_GPU_MEMORY | 000;

	ccv_nnc_tensor_t* const host_source = ccv_nnc_tensor_new(0, host_source_params, 0);
	ccv_nnc_tensor_t* const host_expected = ccv_nnc_tensor_new(0, host_output_params, 0);
	ccv_nnc_tensor_t* const host_actual = ccv_nnc_tensor_new(0, host_output_params, 0);
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, source_params, 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, output_params, 0);
	uint16_t* const source_values = (uint16_t*)host_source->data.f16;
	uint16_t* const expected_values = (uint16_t*)host_expected->data.f16;
	for (uint64_t i = 0; i < source_count; i++)
		source_values[i] = (uint16_t)(0x3c00 + ((i * 73 + 19) & 0x3ff));
	for (int batch = 0; batch < shape.batch_size; batch++)
		for (int row = 0; row < shape.rows; row++)
			for (int col = 0; col < shape.cols; col++)
				expected_values[((uint64_t)batch * shape.cols + col) * shape.rows + row] =
					source_values[(uint64_t)batch * source_batch_stride + row * shape.source_row_stride + shape.source_offset + col];
	if (ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host_source), TENSOR_LIST(source), stream) != CCV_NNC_EXEC_SUCCESS)
		return false;
	ccv_nnc_stream_context_wait(stream);
	const int source_offset[CCV_NNC_MAX_DIM_ALLOC] = { 0, 0, shape.source_offset };
	const int source_stride[CCV_NNC_MAX_DIM_ALLOC] = { source_batch_stride, shape.source_row_stride, 1 };
	ccv_nnc_tensor_view_t* const source_view = ccv_nnc_tensor_view_new(source, source_view_params, source_offset, source_stride);
	ccv_nnc_cmd_t transpose = CMD_TRANSPOSE_FORWARD(1, 2);
	transpose.backend = CCV_NNC_BACKEND_MPS;

	const Path paths[] = { Path::MFA, Path::MPSGraph };
	const char* const path_names[] = { "MFA", "MPSGraph" };
	for (int path_index = 0; path_index < 2; path_index++)
	{
		set_path(paths[path_index]);
		if (!run_sample(transpose, (ccv_nnc_tensor_t*)source_view, output, stream, 1, 0))
			return false;
		if (ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(host_actual), stream) != CCV_NNC_EXEC_SUCCESS)
			return false;
		ccv_nnc_stream_context_wait(stream);
		if (!output_matches(host_expected, host_actual, output_count, path_names[path_index], shape))
			return false;
	}

	for (int i = 0; i < config.warmup_samples; i++)
		for (int j = 0; j < 2; j++)
		{
			const Path path = paths[(i + j) % 2];
			set_path(path);
			if (!run_sample(transpose, (ccv_nnc_tensor_t*)source_view, output, stream, config.dispatches_per_sample, 0))
				return false;
		}
	std::vector<double> mfa_samples;
	std::vector<double> mps_graph_samples;
	mfa_samples.reserve(config.timed_samples);
	mps_graph_samples.reserve(config.timed_samples);
	for (int i = 0; i < config.timed_samples; i++)
		for (int j = 0; j < 2; j++)
		{
			const Path path = paths[(i + j) % 2];
			double elapsed_ms;
			set_path(path);
			if (!run_sample(transpose, (ccv_nnc_tensor_t*)source_view, output, stream, config.dispatches_per_sample, &elapsed_ms))
				return false;
			if (path == Path::MFA)
				mfa_samples.push_back(elapsed_ms);
			else
				mps_graph_samples.push_back(elapsed_ms);
		}
	const Stats mfa_stats = compute_stats(mfa_samples);
	const Stats mps_graph_stats = compute_stats(mps_graph_samples);
	const uint64_t effective_bytes = output_count * sizeof(uint16_t) * 2;
	std::cout << "shape name=" << shape.name
		<< " batch=" << shape.batch_size
		<< " rows=" << shape.rows
		<< " cols=" << shape.cols
		<< " source_row_stride=" << shape.source_row_stride
		<< " source_offset=" << shape.source_offset
		<< " validation=exact\n";
	print_stats("mfa_wall", mfa_stats, effective_bytes);
	print_stats("mps_graph_wall", mps_graph_stats, effective_bytes);
	std::cout << "comparison mfa_vs_mps_graph_speedup_median=" << mps_graph_stats.median_ms / mfa_stats.median_ms << "\n";

	ccv_nnc_tensor_view_free(source_view);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(source);
	ccv_nnc_tensor_free(host_actual);
	ccv_nnc_tensor_free(host_expected);
	ccv_nnc_tensor_free(host_source);
	return true;
}

} // namespace

int main(int argc, char** argv)
{
	Config config;
	if (argc > 1 && argc != 4)
	{
		print_usage(argv[0]);
		return 1;
	}
	if (argc == 4 && (!parse_positive(argv[1], &config.warmup_samples) ||
		!parse_positive(argv[2], &config.timed_samples) ||
		!parse_positive(argv[3], &config.dispatches_per_sample)))
	{
		print_usage(argv[0]);
		return 1;
	}
	ccv_nnc_init();
	if (ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) <= 0 || !ccv_nnc_cmd_ok(CCV_NNC_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		std::cerr << "MPS transpose is not available\n";
		return 1;
	}
	const Shape shapes[] = {
		{ "deepseek4_ratio4_b1", 1, 4, 512, 1024, 512 },
		{ "deepseek4_ratio4_b16", 16, 4, 512, 1024, 512 },
		{ "deepseek4_ratio4_b256", 256, 4, 512, 1024, 512 },
		{ "deepseek4_ratio128_b1", 1, 128, 512, 512, 0 },
		{ "deepseek4_ratio128_b4", 4, 128, 512, 512, 0 },
		{ "deepseek4_ratio128_b16", 16, 128, 512, 512, 0 },
	};
	const uint64_t saved_flags = ccv_nnc_flags();
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	std::cout << std::fixed << std::setprecision(6)
		<< "transpose benchmark dtype=fp16 warmup_samples=" << config.warmup_samples
		<< " timed_samples=" << config.timed_samples
		<< " dispatches_per_sample=" << config.dispatches_per_sample
		<< " timing=wall_per_dispatch_interleaved\n";
	for (const Shape& shape : shapes)
		if (!benchmark_shape(shape, config, stream))
		{
			restore_flags(saved_flags);
			return 1;
		}
	restore_flags(saved_flags);
	ccv_nnc_stream_context_free(stream);
	return 0;
}
