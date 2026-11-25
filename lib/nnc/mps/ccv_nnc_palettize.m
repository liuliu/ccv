#include "ccv_nnc_mps.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_codebook.h"

void ccv_nnc_mps_depalettize(const void* input, const int datatype, const size_t input_length, const int qbits, const int number_in_blocks, void* output, const size_t output_length, void* const command_buffer)
{
	uint32_t mtl_data_type = UINT32_MAX;
	switch (datatype) {
		case CCV_16F: {
			mtl_data_type = 16;
			break;
		}
		case CCV_32F: {
			mtl_data_type = 3;
			break;
		}
		default: {
			break;
		}
	}
	ccv_nnc_mfa_depalettize_params_t params = {
		.data_type = mtl_data_type,
		.qbits = (uint32_t)qbits,
		.number_in_blocks = (uint32_t)number_in_blocks,
		.length = (uint64_t)output_length,
	};
	ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

	ccv_nnc_mfa_prepare_depalettize(context, params);

	mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(0);
	mtl_buffer_t* tensors[3] = {
		(mtl_buffer_t*)input, // A
		(mtl_buffer_t*)output, // B
		NULL,
	};
	size_t tensor_offsets[2] = {
		0, // A offset
		0, // B offset
	};
	ccv_nnc_mfa_encode_depalettize(context, params, command_batch, tensors, tensor_offsets);
	ccv_nnc_stream_context_finish_command_batch(0, command_batch);
}

void ccv_nnc_mps_dequantize_8i_rowwise(const void* input, const int datatype, const size_t input_length, const size_t row_length, void* output, const size_t output_length, void* const command_buffer)
{
	uint32_t mtl_data_type = UINT32_MAX;
	switch (datatype) {
		case CCV_16F: {
			mtl_data_type = 16;
			break;
		}
		case CCV_16BF: {
			mtl_data_type = 121;
			break;
		}
		case CCV_32F: {
			mtl_data_type = 3;
			break;
		}
		default: {
			break;
		}
	}
	assert(mtl_data_type != UINT32_MAX);
	ccv_nnc_mfa_dequantize_8i_rowwise_params_t params = {
		.data_type = mtl_data_type,
		.row_length = row_length,
		.length = output_length,
	};
	ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

	ccv_nnc_mfa_prepare_dequantize_8i_rowwise(context, params);

	mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(0);
	mtl_buffer_t* tensors[3] = {
		(mtl_buffer_t*)input,
		(mtl_buffer_t*)output,
		NULL,
	};
	size_t tensor_offsets[2] = {
		0,
		0,
	};
	ccv_nnc_mfa_encode_dequantize_8i_rowwise(context, params, command_batch, tensors, tensor_offsets);
	ccv_nnc_stream_context_finish_command_batch(0, command_batch);
}

// The MFA format of a TMCQ tensor format whose bits 16 - 23 name a codebook slot (ccv_nnc_codebook.h): the
// slot is replaced by log2 of the table's entry count, and *codebook is the table's buffer (NULL for slot 0).
static uint32_t _ccv_nnc_mps_codebook_format(ccv_nnc_mfa_context_t* const context, const int format, mtl_buffer_t** const codebook)
{
	const int slot = ccv_nnc_codebook_slot_of(format);
	// Kernels see the codec without the slot (ccv_nnc_8i_rowwise_x_kernel_format).
	const uint32_t kernel = (uint32_t)ccv_nnc_8i_rowwise_x_kernel_format(format);
	*codebook = NULL;
	if (!slot)
		return kernel;
	assert(ccv_nnc_8i_rowwise_x_is_tmcq(format) && "codebook slots exist for TMCQ only");
	int entries = 0;
	*codebook = ccv_nnc_mfa_codebook_buffer(context, slot, &entries);
	assert(*codebook && "the tensor names a codebook slot that is not registered");
	return (kernel & ~(uint32_t)CCV_NNC_CODEBOOK_SLOT_MASK) | ((uint32_t)ccv_nnc_codebook_log2_entries(entries) << 16);
}

void ccv_nnc_mps_dequantize_8i_rowwise_x(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, void* const command_buffer)
{
	uint32_t mtl_data_type = UINT32_MAX;
	switch (datatype) {
		case CCV_16F: {
			mtl_data_type = 16;
			break;
		}
		case CCV_16BF: {
			mtl_data_type = 121;
			break;
		}
		case CCV_32F: {
			mtl_data_type = 3;
			break;
		}
		default: {
			break;
		}
	}
	assert(mtl_data_type != UINT32_MAX);
	ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
	// An inline codebook is the start of the input: decode the data after it with the table's slot.
	const size_t inline_size = ccv_nnc_8i_rowwise_x_inline_codebook_size(format);
	const int payload_format = ccv_nnc_mps_8i_rowwise_x_resolve_inline((id<MTLBuffer>)input, 0, format);
	assert(payload_format >= 0 && "the inline codebook is invalid or its buffer is not CPU-visible");
	// An external codebook is supported for TMCQ.
	mtl_buffer_t* codebook = NULL;
	const uint32_t mfa_format = _ccv_nnc_mps_codebook_format(context, payload_format, &codebook);
	ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_params_t params = {
		.data_type = mtl_data_type,
		.format = mfa_format,
		.row_length = row_length,
		.length = output_length,
		.codebook = codebook,
	};

	ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x_fp(context, params);

	mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(0);
	mtl_buffer_t* tensors[3] = {
		(mtl_buffer_t*)input,
		(mtl_buffer_t*)output,
		NULL,
	};
	size_t tensor_offsets[2] = {
		inline_size,
		0,
	};
	ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_fp(context, params, command_batch, tensors, tensor_offsets);
	ccv_nnc_stream_context_finish_command_batch(0, command_batch);
}

void ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, void* const command_buffer)
{
	(void)input_length;
	(void)command_buffer;
	uint32_t mtl_data_type = UINT32_MAX;
	switch (datatype) {
		case CCV_16F: {
			mtl_data_type = 16;
			break;
		}
		case CCV_16BF: {
			mtl_data_type = 121;
			break;
		}
		case CCV_32F: {
			mtl_data_type = 3;
			break;
		}
		default: {
			break;
		}
	}
	assert(mtl_data_type != UINT32_MAX);
	assert(!(format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256));
	ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
	const size_t inline_size = ccv_nnc_8i_rowwise_x_inline_codebook_size(format);
	const int payload_format = ccv_nnc_mps_8i_rowwise_x_resolve_inline((id<MTLBuffer>)input, 0, format);
	assert(payload_format >= 0 && "the inline codebook is invalid or its buffer is not CPU-visible");
	mtl_buffer_t* codebook = NULL;
	const uint32_t mfa_format = _ccv_nnc_mps_codebook_format(context, payload_format & (CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK | CCV_NNC_CODEBOOK_SLOT_MASK), &codebook);
	ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params = {
		.data_type = mtl_data_type,
		.format = mfa_format,
		.row_length = row_length,
		.length = output_length,
		.codebook = codebook,
	};

	ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x(context, params);

	mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(0);
	mtl_buffer_t* tensors[3] = {
		(mtl_buffer_t*)input,
		(mtl_buffer_t*)output,
		NULL,
	};
	size_t tensor_offsets[2] = {
		inline_size,
		0,
	};
	ccv_nnc_mfa_encode_dequantize_8i_rowwise_x(context, params, command_batch, tensors, tensor_offsets);
	ccv_nnc_stream_context_finish_command_batch(0, command_batch);
}
