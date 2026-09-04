#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifdef HAVE_MPS

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _moe_weights_streaming_fill_half(ccv_float16_t* const data, const size_t count, const int seed)
{
	float* const values = (float*)ccmalloc(sizeof(float) * count);
	size_t i;
	for (i = 0; i < count; i++)
		values[i] = (float)((int)((i * (size_t)(seed * 2 + 1) + seed * 11) % 127) - 63) / 512;
	ccv_float_to_half_precision(values, (uint16_t*)data, count);
	ccfree(values);
}

static int _moe_weights_streaming_write_file(const ccv_nnc_tensor_t* const tensor, char path[64])
{
	const int fd = mkstemp(path);
	if (fd < 0)
		return 0;
	const size_t data_size = ccv_nnc_tensor_data_size_without_padding(tensor->info);
	size_t written = 0;
	while (written < data_size)
	{
		const ssize_t result = write(fd, tensor->data.u8 + written, data_size - written);
		if (result <= 0)
		{
			close(fd);
			return 0;
		}
		written += (size_t)result;
	}
	const size_t mapped_size = ccv_nnc_tensor_data_size((ccv_nnc_tensor_param_t){
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = tensor->info.format,
		.datatype = tensor->info.datatype,
		.reserved = tensor->info.reserved,
		.dim = {
			tensor->info.dim[0], tensor->info.dim[1], tensor->info.dim[2],
		},
	});
	const int success = ftruncate(fd, (off_t)mapped_size) == 0;
	close(fd);
	return success;
}

static int _moe_weights_streaming_run_case(
	ccv_nnc_tensor_t* const gate_source, ccv_nnc_tensor_t* const up_source,
	ccv_nnc_tensor_t* const down_source, ccv_nnc_tensor_t* const gate_reference,
	ccv_nnc_tensor_t* const up_reference, ccv_nnc_tensor_t* const down_reference,
	const int* const selected, const int* const segment_counts, const int segment_count,
	const int row_count, const int case_seed, double* const max_difference)
{
	const int n = gate_source->info.dim[1];
	const int k = gate_source->info.dim[2];
	int status = CCV_NNC_EXEC_SUCCESS;
	ccv_nnc_tensor_t* const host_activation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, row_count, k), 0);
	ccv_nnc_tensor_t* const host_indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segment_count), 0);
	ccv_nnc_tensor_t* const host_counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segment_count), 0);
	ccv_nnc_tensor_t* const host_route_weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, row_count), 0);
	ccv_nnc_tensor_t* const activation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, row_count, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segment_count), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segment_count), 0);
	ccv_nnc_tensor_t* const route_weights = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, row_count), 0);
	ccv_nnc_tensor_t* const reference_hidden = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, row_count, n), 0);
	ccv_nnc_tensor_t* const reference_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, row_count, k), 0);
	ccv_nnc_tensor_t* const streamed_hidden = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, row_count, n), 0);
	ccv_nnc_tensor_t* const streamed_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, row_count, k), 0);
	ccv_nnc_tensor_t* const host_reference = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, row_count, k), 0);
	ccv_nnc_tensor_t* const host_streamed = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, row_count, k), 0);
	_moe_weights_streaming_fill_half(host_activation->data.f16, (size_t)row_count * k, case_seed);
	memcpy(host_indices->data.i32, selected, sizeof(int) * segment_count);
	memcpy(host_counts->data.i32, segment_counts, sizeof(int) * segment_count);
	float* const route_values = (float*)ccmalloc(sizeof(float) * row_count);
	int i;
	for (i = 0; i < row_count; i++)
		route_values[i] = (float)(i + 1) / (row_count + 1);
	ccv_float_to_half_precision(route_values, (uint16_t*)host_route_weights->data.f16, row_count);
	ccfree(route_values);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(host_activation, host_indices, host_counts, host_route_weights),
		TENSOR_LIST(activation, indices, counts, route_weights), stream);
	ccv_nnc_cmd_t swiglu = CMD_SEGMENTED_SWIGLU_FORWARD(10);
	swiglu.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_t gemm = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	gemm.backend = CCV_NNC_BACKEND_MPS;
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(swiglu, ccv_nnc_no_hint, 0,
			TENSOR_LIST(activation, indices, counts, gate_reference, up_reference, route_weights),
			TENSOR_LIST(reference_hidden), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0,
			TENSOR_LIST(reference_hidden, indices, counts, down_reference),
			TENSOR_LIST(reference_output), stream);
	ccv_nnc_cmd_t streaming = CMD_MOE_WEIGHTS_STREAMING_FORWARD(2, 1);
	streaming.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_tensor_param_t streamed_params[6] = {};
	ccv_nnc_hint_tensor_auto(streaming,
		TENSOR_PARAM_LIST(indices->info, counts->info, route_weights->info,
			gate_source->info, up_source->info, down_source->info),
		ccv_nnc_no_hint, streamed_params, 6);
	ccv_nnc_tensor_t* streamed[6];
	for (i = 0; i < 6; i++)
	{
		// Model graphs infer this handle before the checkpoint decoder replaces a
		// dense parameter with its quantized file-backed source.
		if (i >= 3)
			streamed_params[i].reserved = 0;
		streamed[i] = ccv_nnc_tensor_new(0, streamed_params[i], 0);
	}
	for (i = 3; i < 6 && status == CCV_NNC_EXEC_SUCCESS; i++)
		if ((streamed_params[i].datatype & 0xf00) != CCV_NNC_QX_EPHERMAL_STAGING ||
			ccv_nnc_tensor_data_size_without_padding(streamed_params[i]) != 1)
			status = CCV_NNC_EXEC_INVALID;
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(streaming, ccv_nnc_no_hint, 0,
			TENSOR_LIST(indices, counts, route_weights, gate_source, up_source, down_source),
			TENSOR_LIST(streamed[0], streamed[1], streamed[2], streamed[3], streamed[4], streamed[5]), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(swiglu, ccv_nnc_no_hint, 0,
			TENSOR_LIST(activation, streamed[0], streamed[1], streamed[3], streamed[4], streamed[2]),
			TENSOR_LIST(streamed_hidden), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0,
			TENSOR_LIST(streamed_hidden, streamed[0], streamed[1], streamed[5]),
			TENSOR_LIST(streamed_output), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
			TENSOR_LIST(reference_output, streamed_output),
			TENSOR_LIST(host_reference, host_streamed), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
	{
		float* const reference_values = (float*)ccmalloc(sizeof(float) * row_count * k);
		float* const streamed_values = (float*)ccmalloc(sizeof(float) * row_count * k);
		ccv_half_precision_to_float((const uint16_t*)host_reference->data.f16, reference_values, row_count * k);
		ccv_half_precision_to_float((const uint16_t*)host_streamed->data.f16, streamed_values, row_count * k);
		*max_difference = 0;
		for (i = 0; i < row_count * k; i++)
			*max_difference = ccv_max(*max_difference, fabs((double)reference_values[i] - streamed_values[i]));
		ccfree(streamed_values);
		ccfree(reference_values);
	}
	for (i = 0; i < 6; i++)
		ccv_nnc_tensor_free(streamed[i]);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(host_streamed);
	ccv_nnc_tensor_free(host_reference);
	ccv_nnc_tensor_free(streamed_output);
	ccv_nnc_tensor_free(streamed_hidden);
	ccv_nnc_tensor_free(reference_output);
	ccv_nnc_tensor_free(reference_hidden);
	ccv_nnc_tensor_free(route_weights);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(activation);
	ccv_nnc_tensor_free(host_route_weights);
	ccv_nnc_tensor_free(host_counts);
	ccv_nnc_tensor_free(host_indices);
	ccv_nnc_tensor_free(host_activation);
	return status;
}

TEST_CASE("MoE weights streaming infers lightweight shape-compatible handles")
{
	const ccv_nnc_tensor_param_t indices = GPU_TENSOR_NHWC(000, 32S, 4);
	const ccv_nnc_tensor_param_t counts = GPU_TENSOR_NHWC(000, 32S, 4);
	const ccv_nnc_tensor_param_t route_weights = GPU_TENSOR_NHWC(000, 16F, 1);
	const ccv_nnc_tensor_param_t gate = ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16F, 4, 256, 256));
	const ccv_nnc_tensor_param_t up = gate;
	const ccv_nnc_tensor_param_t down = gate;
	ccv_nnc_tensor_param_t outputs[6] = {};
	ccv_nnc_hint_tensor_auto(CMD_MOE_WEIGHTS_STREAMING_FORWARD(2, 1),
		TENSOR_PARAM_LIST(indices, counts, route_weights, gate, up, down),
		ccv_nnc_no_hint, outputs, 6);
	REQUIRE(memcmp(&outputs[0], &indices, sizeof(indices)) == 0,
		"streaming should preserve the indices metadata");
	REQUIRE(memcmp(&outputs[1], &counts, sizeof(counts)) == 0,
		"streaming should preserve the counts metadata");
	REQUIRE(memcmp(&outputs[2], &route_weights, sizeof(route_weights)) == 0,
		"streaming should preserve the route-weight metadata");
	int i;
	const ccv_nnc_tensor_param_t weights[] = { gate, up, down };
	for (i = 0; i < 3; i++)
	{
		REQUIRE_EQ(outputs[i + 3].datatype & 0xf00, CCV_NNC_QX_EPHERMAL_STAGING,
			"streaming handles should carry the ephemeral-staging subtype");
		REQUIRE_EQ(outputs[i + 3].datatype & 0xff, weights[i].datatype & 0xff,
			"streaming handles should preserve the source base datatype");
		REQUIRE(memcmp(outputs[i + 3].dim, weights[i].dim, sizeof(weights[i].dim)) == 0,
			"streaming handles should preserve the source logical shape");
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(outputs[i + 3]), 1,
			"an ephemeral staging handle should occupy one physical byte before padding");
	}
}

TEST_CASE("MPS MoE weights streaming covers resident decode, eviction, and prefill")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD, CCV_NNC_BACKEND_MPS));
	const int experts = 4;
	const int n = 256;
	const int k = 512;
	const size_t weight_count = (size_t)experts * n * k;
	ccv_nnc_tensor_t* host_weights[3];
	ccv_nnc_tensor_t* quantized_weights[3];
	ccv_nnc_tensor_t* reference_weights[3];
	ccv_nnc_tensor_t* streamed_weights[3];
	char paths[3][64] = {
		"/tmp/ccv-moe-gate-XXXXXX",
		"/tmp/ccv-moe-up-XXXXXX",
		"/tmp/ccv-moe-down-XXXXXX",
	};
	const int formats[3] = {
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
	};
	int i;
	for (i = 0; i < 3; i++)
	{
		const ccv_nnc_tensor_param_t host_params = i < 2 ?
			CPU_TENSOR_NHWC(16F, experts, n, k) : CPU_TENSOR_NHWC(16F, experts, k, n);
		host_weights[i] = ccv_nnc_tensor_new(0, host_params, 0);
		_moe_weights_streaming_fill_half(host_weights[i]->data.f16, weight_count, 17 + i * 7);
		const int format = formats[i];
		const ccv_nnc_tensor_param_t quantized_params = ccv_nnc_tensor_8i_rowwise_x(host_weights[i]->info, format);
		quantized_weights[i] = ccv_nnc_tensor_new(0, quantized_params, 0);
		const size_t quantized_size = ccv_nnc_quantize_8i_rowwise_x(
			host_weights[i]->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY,
			weight_count, host_weights[i]->info.dim[2], format, 0, 0, quantized_weights[i]->data.u8,
			ccv_nnc_tensor_data_size_without_padding(quantized_params));
		REQUIRE_EQ(quantized_size, ccv_nnc_tensor_data_size_without_padding(quantized_params),
			"each source projection should quantize to its declared size");
		REQUIRE(_moe_weights_streaming_write_file(quantized_weights[i], paths[i]),
			"each source projection should be written to a page-sized temporary file");
		ccv_nnc_tensor_param_t gpu_params = quantized_params;
		gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		reference_weights[i] = ccv_nnc_tensor_new(0, gpu_params, 0);
		streamed_weights[i] = ccv_nnc_tensor_new_from_file(
			gpu_params, paths[i], 0, CCV_NNC_TENSOR_MEMORY_MAP_ON_DEMAND);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(quantized_weights[0], quantized_weights[1], quantized_weights[2]),
		TENSOR_LIST(reference_weights[0], reference_weights[1], reference_weights[2]), 0);
	ccv_nnc_tensor_t* const disabled_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 4), 0);
	ccv_nnc_tensor_t* const disabled_counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 4), 0);
	ccv_nnc_tensor_t* const disabled_route_weights = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1), 0);
	ccv_nnc_cmd_t disabled_command = CMD_MOE_WEIGHTS_STREAMING_FORWARD(2, 1);
	disabled_command.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_tensor_param_t disabled_output_params[6] = {};
	ccv_nnc_hint_tensor_auto(disabled_command,
		TENSOR_PARAM_LIST(disabled_indices->info, disabled_counts->info, disabled_route_weights->info,
			reference_weights[0]->info, reference_weights[1]->info, reference_weights[2]->info),
		ccv_nnc_no_hint, disabled_output_params, 6);
	ccv_nnc_tensor_t* disabled_outputs[6];
	for (i = 0; i < 6; i++)
		disabled_outputs[i] = ccv_nnc_tensor_new(0, disabled_output_params[i], 0);
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	const int disabled_status = ccv_nnc_cmd_exec(disabled_command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(disabled_indices, disabled_counts, disabled_route_weights,
			reference_weights[0], reference_weights[1], reference_weights[2]),
		TENSOR_LIST(disabled_outputs[0], disabled_outputs[1], disabled_outputs[2],
			disabled_outputs[3], disabled_outputs[4], disabled_outputs[5]), 0);
	if (!(old_flags & CCV_NNC_DISABLE_MFA))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(disabled_status, CCV_NNC_EXEC_INVALID,
		"MoE weights streaming should fail when MFA is disabled");
	for (i = 0; i < 6; i++)
		ccv_nnc_tensor_free(disabled_outputs[i]);
	ccv_nnc_tensor_free(disabled_route_weights);
	ccv_nnc_tensor_free(disabled_counts);
	ccv_nnc_tensor_free(disabled_indices);
	const int decode_indices[] = { 0, 1, 2, 3 };
	const int decode_experts[] = { 0, 1, 0, 2, 1 };
	double max_difference = 0;
	for (i = 0; i < 5; i++)
	{
		int decode_counts[] = { 0, 0, 0, 0 };
		decode_counts[decode_experts[i]] = 1;
		REQUIRE_EQ(_moe_weights_streaming_run_case(
			streamed_weights[0], streamed_weights[1], streamed_weights[2],
			reference_weights[0], reference_weights[1], reference_weights[2],
			decode_indices, decode_counts, 4, 1, 31 + i, &max_difference),
			CCV_NNC_EXEC_SUCCESS,
			"decode should execute across cold misses, a resident hit, and eviction");
		REQUIRE(max_difference < 1e-3,
			"streamed decode should match the same rowwise weights without streaming (max difference %.8g)",
			max_difference);
	}
	const int prefill_indices[] = { 1, 3, 0, 2 };
	const int prefill_counts[] = { 2, 2, 0, 0 };
	REQUIRE_EQ(_moe_weights_streaming_run_case(
		streamed_weights[0], streamed_weights[1], streamed_weights[2],
		reference_weights[0], reference_weights[1], reference_weights[2],
		prefill_indices, prefill_counts, 4, 4, 47, &max_difference),
		CCV_NNC_EXEC_SUCCESS,
		"prefill should combine a resident hit with a file-backed miss");
	REQUIRE(max_difference < 1e-3,
		"partially resident prefill should match the same rowwise weights without streaming (max difference %.8g)",
		max_difference);
	for (i = 0; i < 3; i++)
	{
		ccv_nnc_tensor_free(streamed_weights[i]);
		ccv_nnc_tensor_free(reference_weights[i]);
		ccv_nnc_tensor_free(quantized_weights[i]);
		ccv_nnc_tensor_free(host_weights[i]);
		unlink(paths[i]);
	}
}

#endif

#include "case_main.h"
