#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include "../ccv_nnc_hyper_connection_internal.h"

static int _ccv_nnc_hyper_connection_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (!((input_size == 3 && output_size == 3) || (input_size == 4 && (output_size == 1 || output_size == 3))))
		return CCV_NNC_EXEC_INVALID;
	const int hc = cmd.info.hyper_connection.count;
	if (hc <= 0 || hc > 16)
		return CCV_NNC_EXEC_INVALID;
	if (output_size == 1)
	{
		const ccv_nnc_tensor_view_t* const block = (const ccv_nnc_tensor_view_t*)inputs[0];
		const ccv_nnc_tensor_view_t* const residual = (const ccv_nnc_tensor_view_t*)inputs[1];
		const ccv_nnc_tensor_view_t* const post = (const ccv_nnc_tensor_view_t*)inputs[2];
		const ccv_nnc_tensor_view_t* const comb = (const ccv_nnc_tensor_view_t*)inputs[3];
		ccv_nnc_tensor_view_t* const expanded = (ccv_nnc_tensor_view_t*)outputs[0];
		if (block->info.datatype != CCV_32F || residual->info.datatype != CCV_32F || post->info.datatype != CCV_32F || comb->info.datatype != CCV_32F || expanded->info.datatype != CCV_32F)
			return CCV_NNC_EXEC_INVALID;
		if (!CCV_IS_TENSOR_CONTIGUOUS(block) || !CCV_IS_TENSOR_CONTIGUOUS(residual) || !CCV_IS_TENSOR_CONTIGUOUS(post) || !CCV_IS_TENSOR_CONTIGUOUS(comb) || !CCV_IS_TENSOR_CONTIGUOUS(expanded))
			return CCV_NNC_EXEC_INVALID;
		if (!_ccv_nnc_hyper_connection_expand_shapes_are_valid(hc, block->info, residual->info, post->info, comb->info, expanded->info))
			return CCV_NNC_EXEC_INVALID;
		const int nd = ccv_nnc_tensor_nd(residual->info.dim);
		const uint32_t hidden = residual->info.dim[nd - 1];
		const size_t residual_count = ccv_nnc_tensor_count(residual->info);
		const size_t row_count = residual_count / ((size_t)hc * hidden);
		if (row_count > UINT32_MAX)
			return CCV_NNC_EXEC_INVALID;
		const uint32_t rows = (uint32_t)row_count;
		@autoreleasepool {
			ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
			if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
				return CCV_NNC_EXEC_INVALID;
			const ccv_nnc_mfa_hyper_connection_params_t params = {
				.row_count = rows,
				.count = (uint32_t)hc,
				.hidden = hidden,
				.operation = 2,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_hyper_connection(context, params);
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[9] = {
				mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(inputs[2]), mpgetbuffer(inputs[3]),
				mpgetbuffer(outputs[0]), mpgetbuffer(outputs[0]), mpgetbuffer(outputs[0]), mpgetbuffer(outputs[0]), NULL,
			};
			size_t tensor_offsets[8] = {
				block->dataof, residual->dataof, post->dataof, comb->dataof,
				expanded->dataof, expanded->dataof, expanded->dataof, expanded->dataof,
			};
			ccv_nnc_mfa_encode_hyper_connection(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	const ccv_nnc_tensor_view_t* const mix = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const scale = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const base = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const pre = input_size == 3 ? (ccv_nnc_tensor_view_t*)outputs[0] : 0;
	ccv_nnc_tensor_view_t* const post = (ccv_nnc_tensor_view_t*)outputs[input_size == 3 ? 1 : 0];
	ccv_nnc_tensor_view_t* const comb = (ccv_nnc_tensor_view_t*)outputs[input_size == 3 ? 2 : 1];
	if (mix->info.datatype != CCV_32F || scale->info.datatype != CCV_32F || base->info.datatype != CCV_32F || (pre && pre->info.datatype != CCV_32F) || post->info.datatype != CCV_32F || comb->info.datatype != CCV_32F)
		return CCV_NNC_EXEC_INVALID;
	if (!CCV_IS_TENSOR_CONTIGUOUS(mix) || !CCV_IS_TENSOR_CONTIGUOUS(scale) || !CCV_IS_TENSOR_CONTIGUOUS(base) || (pre && !CCV_IS_TENSOR_CONTIGUOUS(pre)) || !CCV_IS_TENSOR_CONTIGUOUS(post) || !CCV_IS_TENSOR_CONTIGUOUS(comb))
		return CCV_NNC_EXEC_INVALID;
	const int iterations = cmd.info.hyper_connection.sinkhorn_iterations;
	const int mix_dim = 2 * hc + hc * hc;
	const size_t mix_count = ccv_nnc_tensor_count(mix->info);
	if (iterations <= 0 || !(cmd.info.hyper_connection.epsilon >= 0))
		return CCV_NNC_EXEC_INVALID;
	const size_t row_count = mix_count / mix_dim;
	if (row_count > UINT32_MAX)
		return CCV_NNC_EXEC_INVALID;
	const uint32_t rows = (uint32_t)row_count;
	uint32_t hidden = 0;
	const ccv_nnc_tensor_view_t* residual = 0;
	ccv_nnc_tensor_view_t* weighted = 0;
	if (input_size == 4)
	{
		residual = (const ccv_nnc_tensor_view_t*)inputs[3];
		weighted = (ccv_nnc_tensor_view_t*)outputs[2];
		if (residual->info.datatype != CCV_32F || weighted->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(residual) || !CCV_IS_TENSOR_CONTIGUOUS(weighted))
			return CCV_NNC_EXEC_INVALID;
		const int nd = ccv_nnc_tensor_nd(residual->info.dim);
		hidden = residual->info.dim[nd - 1];
	}
	if (!_ccv_nnc_hyper_connection_split_shapes_are_valid(hc, mix->info, scale->info, base->info, residual ? &residual->info : 0, pre ? &pre->info : 0, post->info, comb->info, weighted ? &weighted->info : 0))
		return CCV_NNC_EXEC_INVALID;
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			return CCV_NNC_EXEC_INVALID;
		const ccv_nnc_mfa_hyper_connection_params_t params = {
			.row_count = rows,
			.count = (uint32_t)hc,
			.hidden = hidden,
			.sinkhorn_iterations = (uint32_t)iterations,
			.epsilon = cmd.info.hyper_connection.epsilon,
			.operation = input_size == 4,
			.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
		};
		ccv_nnc_mfa_prepare_hyper_connection(context, params);
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		mtl_buffer_t* tensors[9] = {
			mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(inputs[2]),
			mpgetbuffer(input_size == 4 ? inputs[3] : inputs[0]),
			mpgetbuffer(input_size == 3 ? outputs[0] : inputs[0]),
			mpgetbuffer(outputs[input_size == 3 ? 1 : 0]),
			mpgetbuffer(outputs[input_size == 3 ? 2 : 1]),
			mpgetbuffer(input_size == 4 ? outputs[2] : outputs[0]), NULL,
		};
		size_t tensor_offsets[8] = {
			mix->dataof, scale->dataof, base->dataof,
			input_size == 4 ? inputs[3]->dataof : mix->dataof,
			pre ? pre->dataof : mix->dataof, post->dataof, comb->dataof,
			input_size == 4 ? outputs[2]->dataof : outputs[0]->dataof,
		};
		ccv_nnc_mfa_encode_hyper_connection(context, params, command_batch, tensors, tensor_offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_hyper_connection_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_hyper_connection_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_HYPER_CONNECTION_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_hyper_connection_back;
}
