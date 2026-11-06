#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_causal_convolution_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 3 || output_size != 2)
		return CCV_NNC_EXEC_INVALID;
	int i;
	const int datatype = inputs[0]->info.datatype;
	if (datatype != CCV_32F && datatype != CCV_16F && datatype != CCV_16BF)
		return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < 3; i++)
		if (!inputs[i] || inputs[i]->info.datatype != datatype || !CCV_IS_TENSOR_CONTIGUOUS(inputs[i]))
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < 2; i++)
		if (!outputs[i] || outputs[i]->info.datatype != datatype || !CCV_IS_TENSOR_CONTIGUOUS(outputs[i]))
			return CCV_NNC_EXEC_INVALID;
	// Shapes only matter through their counts: C is x's last axis and the previous rows hold B batches.
	const int channels = inputs[0]->info.dim[ccv_nnc_tensor_nd(inputs[0]->info.dim) - 1];
	const size_t weight_count = ccv_nnc_tensor_count(inputs[2]->info);
	const int kernel_size = (int)(weight_count / channels);
	// One tap has no previous rows to carry.
	if (kernel_size < 2 || weight_count != (size_t)kernel_size * channels)
		return CCV_NNC_EXEC_INVALID;
	const size_t history = kernel_size - 1;
	const int B = (int)(ccv_nnc_tensor_count(inputs[1]->info) / (history * channels));
	if (B < 1)
		return CCV_NNC_EXEC_INVALID;
	const int T = (int)(ccv_nnc_tensor_count(inputs[0]->info) / ((size_t)B * channels));
	const int rows = (int)(ccv_nnc_tensor_count(outputs[1]->info) / ((size_t)B * channels));
	if (ccv_nnc_tensor_count(outputs[0]->info) != (size_t)B * T * channels || rows > kernel_size - 1 + T)
		return CCV_NNC_EXEC_INVALID;
	// The kernel reads x and the previous rows while it writes, so neither output may overlap them.
	// A graph places tensors at offsets of shared buffers, so compare the byte ranges.
	const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatype);
	int j;
	for (i = 0; i < 2; i++)
		for (j = 0; j < 2; j++)
		{
			// On MPS, data.u8 is the buffer and dataof the offset into it.
			if (outputs[i]->data.u8 != inputs[j]->data.u8)
				continue;
			const off_t output_begin = outputs[i]->dataof;
			const off_t output_end = output_begin + (off_t)(ccv_nnc_tensor_count(outputs[i]->info) * element_size);
			const off_t input_begin = inputs[j]->dataof;
			const off_t input_end = input_begin + (off_t)(ccv_nnc_tensor_count(inputs[j]->info) * element_size);
			if (output_begin < input_end && input_begin < output_end)
				return CCV_NNC_EXEC_INVALID;
		}
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			return CCV_NNC_EXEC_INVALID;
		const ccv_nnc_mfa_causal_convolution_params_t params = {
			.data_type = datatype == CCV_32F ? 3 : (datatype == CCV_16BF ? 121 : 16),
			.batch_size = B,
			.sequence_length = T,
			.channels = channels,
			.kernel_size = kernel_size,
			.next_rows = rows,
			.silu = cmd.info.causal_convolution.silu != 0,
		};
		ccv_nnc_mfa_prepare_causal_convolution(context, params);
		mtl_buffer_t* tensors[6] = {};
		size_t offsets[5];
		for (i = 0; i < 3; i++)
		{
			tensors[i] = mpgetbuffer(inputs[i]);
			offsets[i] = inputs[i]->dataof;
		}
		for (i = 0; i < 2; i++)
		{
			tensors[3 + i] = mpgetbuffer(outputs[i]);
			offsets[3 + i] = outputs[i]->dataof;
		}
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		ccv_nnc_mfa_encode_causal_convolution(context, params, command_batch, tensors, offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_causal_convolution_forw;
}
