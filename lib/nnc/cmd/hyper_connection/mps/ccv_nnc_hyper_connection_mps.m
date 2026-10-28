#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include "../ccv_nnc_hyper_connection_internal.h"

static size_t _ccv_nnc_hyper_connection_align(const size_t value)
{
	return (value + 255) & ~(size_t)255;
}

// Mix in two dispatches for decode-sized row counts: the partials, then the split alongside the
// collapse normalization. Longer inputs collapse and normalize a row per threadgroup, project the
// raw residual with the GEMM kernel, and fold the RMS into the split.
static int _ccv_nnc_hyper_connection_mix_forw(const ccv_nnc_cmd_t cmd, ccv_nnc_tensor_t* const* const inputs, ccv_nnc_tensor_t* const* const outputs, ccv_nnc_stream_context_t* const stream_context)
{
	const int hc = cmd.info.hyper_connection.count;
	int i;
	for (i = 0; i < 6; i++)
		if (inputs[i]->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(inputs[i]))
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < 4; i++)
		if (outputs[i]->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(outputs[i]))
			return CCV_NNC_EXEC_INVALID;
	ccv_nnc_tensor_t* const collapsed = outputs[3];
	const int iterations = cmd.info.hyper_connection.sinkhorn_iterations;
	const float norm_epsilon = cmd.info.hyper_connection.norm_epsilon;
	if (iterations <= 0 || !(cmd.info.hyper_connection.epsilon >= 0) || !(norm_epsilon > 0))
		return CCV_NNC_EXEC_INVALID;
	if (!_ccv_nnc_hyper_connection_mix_shapes_are_valid(hc, inputs[0]->info, inputs[1]->info, inputs[2]->info, inputs[3]->info, inputs[4]->info, inputs[5]->info, outputs[0]->info, outputs[1]->info, outputs[2]->info, collapsed->info))
		return CCV_NNC_EXEC_INVALID;
	const int nd = ccv_nnc_tensor_nd(inputs[0]->info.dim);
	const uint32_t hidden = inputs[0]->info.dim[nd - 1];
	const uint32_t flat = hc * hidden;
	const uint32_t mix_dim = 2 * hc + hc * hc;
	const size_t row_count = ccv_nnc_tensor_count(inputs[0]->info) / flat;
	// The split keeps a mix value per SIMD lane, and the row kernel caches 16 values per thread.
	if (mix_dim > 32 || (hidden % 4) != 0 || hidden > 16 * 1024 || row_count > UINT32_MAX)
		return CCV_NNC_EXEC_INVALID;
	const uint32_t rows = (uint32_t)row_count;
	const int decode = rows <= 16;
	const uint32_t dot_splits = decode ? (flat + CCV_NNC_MFA_HYPER_CONNECTION_MIX_CHUNK - 1) / CCV_NNC_MFA_HYPER_CONNECTION_MIX_CHUNK : 1;
	const uint32_t collapse_splits = decode ? (hidden + CCV_NNC_MFA_HYPER_CONNECTION_MIX_THREADS - 1) / CCV_NNC_MFA_HYPER_CONNECTION_MIX_THREADS : 0;
	const size_t dots_offset = 0;
	const size_t squares_offset = _ccv_nnc_hyper_connection_align(dots_offset + (size_t)rows * dot_splits * mix_dim * sizeof(float));
	const size_t collapsed_offset = _ccv_nnc_hyper_connection_align(squares_offset + (size_t)rows * dot_splits * sizeof(float));
	const size_t collapsed_squares_offset = _ccv_nnc_hyper_connection_align(collapsed_offset + (decode ? (size_t)rows * hidden * sizeof(float) : 0));
	const size_t scratch_size = collapsed_squares_offset + (size_t)rows * collapse_splits * sizeof(float);
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			return CCV_NNC_EXEC_INVALID;
		// The GEMM may use the shared scratch, so the mix keeps its partials in a durable lease.
		ccv_nnc_mfa_durable_scratch_t* const lease = ccv_nnc_mfa_request_durable_scratch(context, scratch_size);
		if (!lease)
			return CCV_NNC_EXEC_INVALID;
		mtl_buffer_t* const scratch = ccv_nnc_mfa_durable_scratch_buffer(lease);
		ccv_nnc_mfa_hyper_connection_params_t params = {
			.row_count = rows,
			.count = (uint32_t)hc,
			.hidden = hidden,
			.sinkhorn_iterations = (uint32_t)iterations,
			.epsilon = cmd.info.hyper_connection.epsilon,
			.norm_epsilon = norm_epsilon,
			.dot_splits = dot_splits,
			.collapse_splits = collapse_splits,
		};
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		if (decode)
		{
			params.operation = 3;
			mtl_buffer_t* partial_tensors[8] = {
				mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(inputs[2]),
				scratch, scratch, scratch, scratch, NULL,
			};
			size_t partial_offsets[7] = {
				inputs[0]->dataof, inputs[1]->dataof, inputs[2]->dataof,
				dots_offset, squares_offset, collapsed_offset, collapsed_squares_offset,
			};
			ccv_nnc_mfa_encode_hyper_connection(context, params, command_batch, partial_tensors, partial_offsets);
		} else {
			params.operation = 6;
			mtl_buffer_t* row_tensors[6] = {
				mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(inputs[5]),
				mpgetbuffer(collapsed), scratch, NULL,
			};
			size_t row_offsets[5] = {
				inputs[0]->dataof, inputs[1]->dataof, inputs[5]->dataof, collapsed->dataof, squares_offset,
			};
			ccv_nnc_mfa_encode_hyper_connection(context, params, command_batch, row_tensors, row_offsets);
			// Project the raw residual; the split scales the result by the residual's RMS reciprocal.
			const ccv_nnc_mfa_gemm_params_t gemm_params = {
				.data_type = 3,
				.M = rows,
				.N = mix_dim,
				.K = flat,
				.A_trans = 0,
				.B_trans = 1,
				.D_trans = 0,
				.fused_bias = 0,
				.register_float = 1,
				.use_neural_accelerators = !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) && ccv_nnc_mfa_has_neural_accelerators(context),
				.batch_dimension = 1,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_gemm(context, gemm_params);
			mtl_buffer_t* gemm_tensors[5] = {
				mpgetbuffer(inputs[0]), mpgetbuffer(inputs[2]), scratch, NULL, NULL,
			};
			size_t gemm_offsets[4] = {
				inputs[0]->dataof, inputs[2]->dataof, dots_offset, 0,
			};
			ccv_nnc_mfa_encode_gemm(context, gemm_params, command_batch, gemm_tensors, gemm_offsets);
		}
		if (decode)
		{
			params.operation = 5;
			mtl_buffer_t* finish_tensors[12] = {
				scratch, scratch, mpgetbuffer(inputs[3]), mpgetbuffer(inputs[4]),
				mpgetbuffer(outputs[0]), mpgetbuffer(outputs[1]), mpgetbuffer(outputs[2]),
				scratch, scratch, mpgetbuffer(inputs[5]), mpgetbuffer(collapsed), NULL,
			};
			size_t finish_offsets[11] = {
				dots_offset, squares_offset, inputs[3]->dataof, inputs[4]->dataof,
				outputs[0]->dataof, outputs[1]->dataof, outputs[2]->dataof,
				collapsed_offset, collapsed_squares_offset, inputs[5]->dataof, collapsed->dataof,
			};
			ccv_nnc_mfa_encode_hyper_connection(context, params, command_batch, finish_tensors, finish_offsets);
		} else {
			params.operation = 4;
			mtl_buffer_t* split_tensors[8] = {
				scratch, scratch, mpgetbuffer(inputs[3]), mpgetbuffer(inputs[4]),
				mpgetbuffer(outputs[0]), mpgetbuffer(outputs[1]), mpgetbuffer(outputs[2]), NULL,
			};
			size_t split_offsets[7] = {
				dots_offset, squares_offset, inputs[3]->dataof, inputs[4]->dataof,
				outputs[0]->dataof, outputs[1]->dataof, outputs[2]->dataof,
			};
			ccv_nnc_mfa_encode_hyper_connection(context, params, command_batch, split_tensors, split_offsets);
		}
		const int retired = ccv_nnc_mfa_retire_durable_scratch(lease, command_batch);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		if (!retired)
		{
			// Without a completion handler, wait before returning the lease to the pool.
			ccv_nnc_stream_context_wait(stream_context);
			ccv_nnc_mfa_release_durable_scratch(lease);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_hyper_connection_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (!((input_size == 3 && output_size == 3) || (input_size == 4 && (output_size == 1 || output_size == 3)) || (input_size == 5 && output_size == 1) || (input_size == 6 && output_size == 4)))
		return CCV_NNC_EXEC_INVALID;
	if (outputs[0]->info.dim[0] == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const int hc = cmd.info.hyper_connection.count;
	if (hc <= 0 || hc > 16)
		return CCV_NNC_EXEC_INVALID;
	if (input_size == 6)
		return _ccv_nnc_hyper_connection_mix_forw(cmd, inputs, outputs, stream_context);
	if (output_size == 1)
	{
		const ccv_nnc_tensor_view_t* const block = (const ccv_nnc_tensor_view_t*)inputs[0];
		const ccv_nnc_tensor_view_t* const residual = (const ccv_nnc_tensor_view_t*)inputs[1];
		const ccv_nnc_tensor_view_t* const post = (const ccv_nnc_tensor_view_t*)inputs[2];
		const ccv_nnc_tensor_view_t* const comb = (const ccv_nnc_tensor_view_t*)inputs[3];
		const ccv_nnc_tensor_view_t* const addend = input_size == 5 ? (const ccv_nnc_tensor_view_t*)inputs[4] : 0;
		ccv_nnc_tensor_view_t* const expanded = (ccv_nnc_tensor_view_t*)outputs[0];
		if ((block->info.datatype != CCV_16F && block->info.datatype != CCV_32F) || residual->info.datatype != CCV_32F || post->info.datatype != CCV_32F || comb->info.datatype != CCV_32F || expanded->info.datatype != CCV_32F)
			return CCV_NNC_EXEC_INVALID;
		if (addend && (block->info.datatype != CCV_32F || !_ccv_nnc_hyper_connection_addend_shape_is_valid(block->info, addend->info) || !CCV_IS_TENSOR_CONTIGUOUS(addend)))
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
				.block_fp16 = block->info.datatype == CCV_16F,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
				.expand_addend = addend != 0,
			};
			ccv_nnc_mfa_prepare_hyper_connection(context, params);
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			// The addend takes the second output's slot, which expansion does not write.
			mtl_buffer_t* tensors[9] = {
				mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(inputs[2]), mpgetbuffer(inputs[3]),
				mpgetbuffer(outputs[0]), addend ? mpgetbuffer(inputs[4]) : mpgetbuffer(outputs[0]), mpgetbuffer(outputs[0]), mpgetbuffer(outputs[0]), NULL,
			};
			size_t tensor_offsets[8] = {
				block->dataof, residual->dataof, post->dataof, comb->dataof,
				expanded->dataof, addend ? addend->dataof : expanded->dataof, expanded->dataof, expanded->dataof,
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
	registry->tensor_datatypes = CCV_32F | CCV_16F;
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
