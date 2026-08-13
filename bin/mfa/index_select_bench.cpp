#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
}

namespace {

struct Config {
	int rows = 32768;
	int cols = 512;
	int selected = 512;
	int datatype = CCV_16F;
	int warmup = 10;
	int timed = 50;
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
	MPSGraph,
	MFASpecialized,
	MFADynamic,
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
	std::cerr << "usage: " << argv0 << " [rows cols selected int32|fp16|bf16 warmup timed]\n";
}

uint32_t element_size(const int datatype)
{
	return datatype == CCV_32S ? sizeof(int32_t) : sizeof(uint16_t);
}

const char* datatype_name(const int datatype)
{
	switch (datatype)
	{
		case CCV_32S:
			return "int32";
		case CCV_16F:
			return "fp16";
		case CCV_16BF:
			return "bf16";
	}
	return "unknown";
}

bool outputs_equal(const ccv_nnc_tensor_t* const expected, const ccv_nnc_tensor_t* const actual, const uint64_t count, const uint32_t value_size, const char* const path)
{
	if (std::memcmp(expected->data.u8, actual->data.u8, count * value_size) == 0)
		return true;
	uint64_t i;
	for (i = 0; i < count; i++)
		if (std::memcmp(expected->data.u8 + i * value_size, actual->data.u8 + i * value_size, value_size) != 0)
			break;
	std::cerr << path << " and CPU reference results differ at element " << i;
	if (value_size == sizeof(uint32_t))
		std::cerr << ": cpu=" << ((const uint32_t*)expected->data.i32)[i] << " " << path << "=" << ((const uint32_t*)actual->data.i32)[i];
	else
		std::cerr << ": cpu=" << ((const uint16_t*)expected->data.f16)[i] << " " << path << "=" << ((const uint16_t*)actual->data.f16)[i];
	std::cerr << "\n";
	return false;
}

void set_path(const Path path)
{
	if (path == Path::MPSGraph)
	{
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	} else {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		if (path == Path::MFADynamic)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
		else
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	}
}

void restore_flags(const uint64_t saved_flags)
{
	if (saved_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	if (saved_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
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

bool run_once(const ccv_nnc_cmd_t cmd, ccv_nnc_tensor_t* const source, ccv_nnc_tensor_t* const indices, ccv_nnc_tensor_t* const output, ccv_nnc_stream_context_t* const stream, double* const elapsed_ms)
{
	const auto start = std::chrono::steady_clock::now();
	const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(source, indices), TENSOR_LIST(output), stream);
	ccv_nnc_stream_context_wait(stream);
	const auto end = std::chrono::steady_clock::now();
	if (status != CCV_NNC_EXEC_SUCCESS)
		return false;
	if (elapsed_ms)
		*elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	return true;
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
		<< " effective_gbps=" << effective_gbps
		<< "\n";
}

} // namespace

int main(int argc, char** argv)
{
	Config config;
	if (argc > 1 && argc != 7)
	{
		print_usage(argv[0]);
		return 1;
	}
	if (argc == 7)
	{
		if (!parse_positive(argv[1], &config.rows) ||
			!parse_positive(argv[2], &config.cols) ||
			!parse_positive(argv[3], &config.selected) ||
			!parse_positive(argv[5], &config.warmup) ||
			!parse_positive(argv[6], &config.timed))
		{
			print_usage(argv[0]);
			return 1;
		}
		const std::string datatype = argv[4];
		if (datatype == "int32")
			config.datatype = CCV_32S;
		else if (datatype == "fp16")
			config.datatype = CCV_16F;
		else if (datatype == "bf16")
			config.datatype = CCV_16BF;
		else {
			print_usage(argv[0]);
			return 1;
		}
	}
	ccv_nnc_init();
	if (ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) <= 0 ||
		!ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		std::cerr << "MPS index select is not available\n";
		return 1;
	}
	const uint64_t source_count = (uint64_t)config.rows * config.cols;
	const uint64_t output_count = (uint64_t)config.selected * config.cols;
	const uint32_t value_size = element_size(config.datatype);
	if (source_count > SIZE_MAX / value_size || output_count > SIZE_MAX / value_size)
	{
		std::cerr << "shape is too large\n";
		return 1;
	}

	ccv_nnc_tensor_param_t host_source_params = {};
	host_source_params.type = CCV_TENSOR_CPU_MEMORY;
	host_source_params.format = CCV_TENSOR_FORMAT_NHWC;
	host_source_params.datatype = config.datatype;
	host_source_params.dim[0] = config.rows;
	host_source_params.dim[1] = config.cols;
	ccv_nnc_tensor_param_t host_output_params = host_source_params;
	host_output_params.dim[0] = config.selected;
	ccv_nnc_tensor_param_t gpu_source_params = host_source_params;
	gpu_source_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_param_t gpu_output_params = host_output_params;
	gpu_output_params.type = CCV_TENSOR_GPU_MEMORY | 000;

	ccv_nnc_tensor_t* const host_source = ccv_nnc_tensor_new(0, host_source_params, 0);
	ccv_nnc_tensor_t* const host_indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, config.selected), 0);
	ccv_nnc_tensor_t* const host_expected_output = ccv_nnc_tensor_new(0, host_output_params, 0);
	ccv_nnc_tensor_t* const host_mpsgraph_output = ccv_nnc_tensor_new(0, host_output_params, 0);
	ccv_nnc_tensor_t* const host_mfa_specialized_output = ccv_nnc_tensor_new(0, host_output_params, 0);
	ccv_nnc_tensor_t* const host_mfa_dynamic_output = ccv_nnc_tensor_new(0, host_output_params, 0);
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, gpu_source_params, 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, config.selected), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, gpu_output_params, 0);
	{
		if (config.datatype == CCV_32S)
		{
			uint32_t* const values = (uint32_t*)host_source->data.i32;
			for (uint64_t i = 0; i < source_count; i++)
				values[i] = (uint32_t)(int32_t)((int64_t)((i * 104729 + 1543) % 240001) - 120000);
		} else if (config.datatype == CCV_16F)
		{
			uint16_t* const values = (uint16_t*)host_source->data.f16;
			for (uint64_t i = 0; i < source_count; i++)
				values[i] = (uint16_t)(0x3c00 + ((i * 73 + 19) & 0x3ff));
		}
		else
		{
			uint16_t* const values = (uint16_t*)host_source->data.f16;
			for (uint64_t i = 0; i < source_count; i++)
				values[i] = (uint16_t)(0x3f80 + ((i * 73 + 19) & 0x7f));
		}
	}
	for (int i = 0; i < config.selected; i++)
	{
		host_indices->data.i32[i] = (int)(((uint64_t)i * 104729 + 1543) % config.rows);
		std::memcpy(host_expected_output->data.u8 + (uint64_t)i * config.cols * value_size, host_source->data.u8 + (uint64_t)host_indices->data.i32[i] * config.cols * value_size, (uint64_t)config.cols * value_size);
	}
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host_source, host_indices), TENSOR_LIST(source, indices), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS)
	{
		std::cerr << "input transfer failed\n";
		return 1;
	}
	ccv_nnc_tensor_free(host_source);

	const uint64_t saved_flags = ccv_nnc_flags();
	const ccv_nnc_cmd_t cmd = CMD_INDEX_SELECT_FORWARD();
	set_path(Path::MPSGraph);
	if (!run_once(cmd, source, indices, output, stream, 0))
	{
		std::cerr << "MPSGraph index select failed\n";
		return 1;
	}
	status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(host_mpsgraph_output), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS || !outputs_equal(host_expected_output, host_mpsgraph_output, output_count, value_size, "MPSGraph"))
	{
		return 1;
	}
	set_path(Path::MFASpecialized);
	if (!run_once(cmd, source, indices, output, stream, 0))
	{
		std::cerr << "specialized MFA index select failed\n";
		return 1;
	}
	status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(host_mfa_specialized_output), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS || !outputs_equal(host_expected_output, host_mfa_specialized_output, output_count, value_size, "specialized MFA"))
	{
		return 1;
	}
	set_path(Path::MFADynamic);
	if (!run_once(cmd, source, indices, output, stream, 0))
	{
		std::cerr << "dynamic MFA index select failed\n";
		return 1;
	}
	status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(host_mfa_dynamic_output), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status != CCV_NNC_EXEC_SUCCESS || !outputs_equal(host_expected_output, host_mfa_dynamic_output, output_count, value_size, "dynamic MFA"))
	{
		return 1;
	}

	const Path paths[] = { Path::MFASpecialized, Path::MFADynamic, Path::MPSGraph };
	for (int i = 0; i < config.warmup; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			set_path(paths[(i + j) % 3]);
			if (!run_once(cmd, source, indices, output, stream, 0))
				return 1;
		}
	}
	std::vector<double> mfa_specialized_samples;
	std::vector<double> mfa_dynamic_samples;
	std::vector<double> mpsgraph_samples;
	mfa_specialized_samples.reserve(config.timed);
	mfa_dynamic_samples.reserve(config.timed);
	mpsgraph_samples.reserve(config.timed);
	for (int i = 0; i < config.timed; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			const Path path = paths[(i + j) % 3];
			double elapsed_ms;
			set_path(path);
			if (!run_once(cmd, source, indices, output, stream, &elapsed_ms))
				return 1;
			switch (path)
			{
				case Path::MFASpecialized:
					mfa_specialized_samples.push_back(elapsed_ms);
					break;
				case Path::MFADynamic:
					mfa_dynamic_samples.push_back(elapsed_ms);
					break;
				case Path::MPSGraph:
					mpsgraph_samples.push_back(elapsed_ms);
					break;
			}
		}
	}
	restore_flags(saved_flags);
	const Stats mfa_specialized_stats = compute_stats(mfa_specialized_samples);
	const Stats mfa_dynamic_stats = compute_stats(mfa_dynamic_samples);
	const Stats mpsgraph_stats = compute_stats(mpsgraph_samples);
	const uint64_t effective_bytes = output_count * value_size * 2 + (uint64_t)config.selected * sizeof(int32_t);
	std::cout << std::fixed << std::setprecision(6)
		<< "shape rows=" << config.rows
		<< " cols=" << config.cols
		<< " selected=" << config.selected
		<< " dtype=" << datatype_name(config.datatype)
		<< " warmup_per_path=" << config.warmup
		<< " timed_per_path=" << config.timed
		<< " validation=exact\n";
	print_stats("mfa_specialized_wall", mfa_specialized_stats, effective_bytes);
	print_stats("mfa_dynamic_wall", mfa_dynamic_stats, effective_bytes);
	print_stats("mpsgraph_wall", mpsgraph_stats, effective_bytes);
	std::cout << "comparison specialized_speedup_median=" << mpsgraph_stats.median_ms / mfa_specialized_stats.median_ms
		<< " dynamic_speedup_median=" << mpsgraph_stats.median_ms / mfa_dynamic_stats.median_ms
		<< " dynamic_over_specialized_median=" << mfa_dynamic_stats.median_ms / mfa_specialized_stats.median_ms
		<< "\n";

	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(source);
	ccv_nnc_tensor_free(host_mfa_dynamic_output);
	ccv_nnc_tensor_free(host_mfa_specialized_output);
	ccv_nnc_tensor_free(host_mpsgraph_output);
	ccv_nnc_tensor_free(host_expected_output);
	ccv_nnc_tensor_free(host_indices);
	return 0;
}
