#include "ccv_nnc_mps.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"

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
	ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_params_t params = {
		.data_type = mtl_data_type,
		.format = (uint32_t)format,
		.row_length = row_length,
		.length = output_length,
	};
	ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

	ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x_fp(context, params);

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
	ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_fp(context, params, command_batch, tensors, tensor_offsets);
	ccv_nnc_stream_context_finish_command_batch(0, command_batch);
}
