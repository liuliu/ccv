#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_attention_residual_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 3 || output_size < 1 || output_size > CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_QUERIES)
		return CCV_NNC_EXEC_INVALID;
	int i, j;
	const int datatype = inputs[1]->info.datatype;
	if (datatype != CCV_32F && datatype != CCV_16F && datatype != CCV_16BF)
		return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < 3; i++)
		if (!inputs[i] || inputs[i]->info.datatype != datatype || !CCV_IS_TENSOR_CONTIGUOUS(inputs[i]))
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < output_size; i++)
		if (!outputs[i] || outputs[i]->info.datatype != datatype || !CCV_IS_TENSOR_CONTIGUOUS(outputs[i]))
			return CCV_NNC_EXEC_INVALID;
	// Shapes only matter through their counts: C is the partial state's last axis, every block
	// state has as many values as the partial state, and every query has C.
	const int channels = inputs[1]->info.dim[ccv_nnc_tensor_nd(inputs[1]->info.dim) - 1];
	const size_t state_size = ccv_nnc_tensor_count(inputs[1]->info);
	if (channels < 1 || state_size == 0)
		return CCV_NNC_EXEC_INVALID;
	const size_t rows = state_size / channels;
	const size_t block_count = ccv_nnc_tensor_count(inputs[0]->info) / state_size;
	if (rows > UINT32_MAX || block_count + 1 > CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_STATES ||
		ccv_nnc_tensor_count(inputs[0]->info) != block_count * state_size ||
		ccv_nnc_tensor_count(inputs[2]->info) != (size_t)output_size * channels)
		return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < output_size; i++)
		if (ccv_nnc_tensor_count(outputs[i]->info) != state_size)
			return CCV_NNC_EXEC_INVALID;
	// The kernel reads the states again while it writes, so no output may overlap them or another
	// output. A graph places tensors at offsets of shared buffers, so compare the byte ranges.
	const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatype);
	for (i = 0; i < output_size; i++)
	{
		const off_t output_begin = outputs[i]->dataof;
		const off_t output_end = output_begin + (off_t)(state_size * element_size);
		for (j = 0; j < 2 + i; j++)
		{
			const ccv_nnc_tensor_t* const other = j < 2 ? inputs[j] : outputs[j - 2];
			// On MPS, data.u8 is the buffer and dataof the offset into it.
			if (outputs[i]->data.u8 != other->data.u8)
				continue;
			const off_t other_begin = other->dataof;
			const off_t other_end = other_begin + (off_t)(ccv_nnc_tensor_count(other->info) * element_size);
			if (output_begin < other_end && other_begin < output_end)
				return CCV_NNC_EXEC_INVALID;
		}
	}
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			return CCV_NNC_EXEC_INVALID;
		const ccv_nnc_mfa_attention_residual_params_t params = {
			.data_type = datatype == CCV_32F ? 3 : (datatype == CCV_16BF ? 121 : 16),
			.rows = (uint32_t)rows,
			.channels = (uint32_t)channels,
			.block_count = (uint32_t)block_count,
			.query_count = (uint32_t)output_size,
			.epsilon = cmd.info.attention_residual.epsilon,
			.scale = cmd.info.attention_residual.scale,
		};
		// The block states, the partial state, the queries, then the outputs.
		mtl_buffer_t* tensors[3 + CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_QUERIES + 1] = {};
		size_t offsets[3 + CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_QUERIES] = {};
		for (i = 0; i < 3; i++)
		{
			tensors[i] = mpgetbuffer(inputs[i]);
			offsets[i] = inputs[i]->dataof;
		}
		for (i = 0; i < output_size; i++)
		{
			tensors[3 + i] = mpgetbuffer(outputs[i]);
			offsets[3 + i] = outputs[i]->dataof;
		}
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		ccv_nnc_mfa_encode_attention_residual(context, params, command_batch, tensors, offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ATTENTION_RESIDUAL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_attention_residual_forw;
}
