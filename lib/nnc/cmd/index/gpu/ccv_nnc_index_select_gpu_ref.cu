extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM>
__global__ void _ccv_nnc_index_select_forw_kernel(const int n, const int d, const NUM* const a, const int a_inc, const int* const indices, NUM* const b, const int b_inc)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		const int j = i % d;
		const int dest = i / d;
		const int src = indices[dest];
		b[dest * b_inc + j] = a[src * a_inc + j];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_index_select_forw_kernel(const int n, const int d, const NUM* const a, const int a_inc, const int a_rows, const float* const indices, NUM* const b, const int b_inc)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		const int j = i % d;
		const int dest = i / d;
		const float src = indices[dest];
		const int src0 = (int)src;
		const int src1 = min(src0 + 1, a_rows - 1);
		const float w1 = src - src0;
		const float w0 = 1 - w1;
		b[dest * b_inc + j] = (NUM)((float)a[src0 * a_inc + j] * w0 + (float)a[src1 * a_inc + j] * w1);
	}
}

// Gather packed rows and their scales before decoding. Only the selected rows need FP storage,
// and the existing row-wise decoders retain support for each codec and the H256 transform.
static __global__ void _ccv_nnc_index_select_packed_kernel(const size_t count, const size_t row_bits, const size_t payload_size, const size_t input_scale_offset, const size_t output_scale_offset, const size_t scale_size, const uint8_t* const input, const int* const indices, uint8_t* const output)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		if (i >= payload_size)
		{
			const size_t scale_byte = i - payload_size;
			const size_t row = scale_byte / scale_size;
			output[output_scale_offset + scale_byte] = input[input_scale_offset + (size_t)indices[row] * scale_size + scale_byte % scale_size];
		} else if (row_bits % 8 == 0) {
			const size_t row_bytes = row_bits / 8;
			output[i] = input[(size_t)indices[i / row_bytes] * row_bytes + i % row_bytes];
		} else {
			uint8_t value = 0;
			const size_t rows = (count - payload_size) / scale_size;
			for (int bit = 0; bit < 8; bit++)
			{
				const size_t dest_bit = i * 8 + bit;
				const size_t row = dest_bit / row_bits;
				if (row < rows)
				{
					const size_t source_bit = (size_t)indices[row] * row_bits + dest_bit % row_bits;
					value |= ((input[source_bit / 8] >> (source_bit % 8)) & 1) << bit;
				}
			}
			output[i] = value;
		}
	}
}

static int _ccv_nnc_index_select_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	if (outputs[0]->info.dim[0] == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd <= 2);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == 1);
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd <= 2);
	const int a_cols = a_nd < 2 ? 1 : a->info.dim[1];
	const int a_cols_inc = CCV_IS_TENSOR_VIEW(a) ? (a_nd < 2 ? 1 : a->stride[0]) : a_cols;
	const int a_rows = a->info.dim[0];
	const int b_cols = b_nd < 2 ? 1 : b->info.dim[1];
	const int b_cols_inc = CCV_IS_TENSOR_VIEW(b) ? (b_nd < 2 ? 1 : b->stride[0]) : b_cols;
	const int b_rows = b->info.dim[0];
	assert(b_rows == indices->info.dim[0]);
	assert(a_cols == b_cols);
	if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
	{
		const int subtype = a->info.datatype & 0xf00;
		const int datatype = (a->info.datatype & 0xff) << 12;
		if ((subtype != CCV_NNC_QX_8I_ROWWISE && subtype != CCV_NNC_QX_8I_ROWWISE_X) ||
			(datatype != CCV_32F && datatype != CCV_16F && datatype != CCV_16BF) ||
			b->info.datatype != datatype || indices->info.datatype != CCV_32S || a_nd != 2 || b_nd != 2 ||
			!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(indices) || !CCV_IS_TENSOR_CONTIGUOUS(b))
			return CCV_NNC_EXEC_INVALID;
		const int format = a->info.reserved & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
		if (subtype == CCV_NNC_QX_8I_ROWWISE_X && (format == CCV_NNC_QX_8I_ROWWISE_IQ1_S || format == CCV_NNC_QX_8I_ROWWISE_IQ1_M))
			return CCV_NNC_EXEC_INVALID;
		const size_t group_size = subtype == CCV_NNC_QX_8I_ROWWISE ? 1 : ccv_nnc_8i_rowwise_x_group_size(a->info.reserved);
		const size_t group_bits = subtype == CCV_NNC_QX_8I_ROWWISE ? 8 : ccv_nnc_8i_rowwise_x_group_bits(a->info.reserved);
		if (group_size == 0 || group_bits == 0)
			return CCV_NNC_EXEC_INVALID;
		const size_t row_bits = ((a_cols + group_size - 1) / group_size) * group_bits;
		const size_t payload_size = ((size_t)b_rows * row_bits + 7) / 8;
		// The selected rows form a packed tensor of the same format, so both scale offsets come from the packed size.
		const size_t scale_size = CCV_GET_DATA_TYPE_SIZE(datatype);
		const size_t input_scale_offset = ccv_nnc_tensor_data_size_without_padding(a->info) - (size_t)a_rows * scale_size;
		ccv_nnc_tensor_param_t packed_params = a->info;
		packed_params.dim[0] = b_rows;
		const size_t packed_size = ccv_nnc_tensor_data_size_without_padding(packed_params);
		const size_t output_scale_offset = packed_size - (size_t)b_rows * scale_size;
		assert(output_scale_offset >= payload_size);
		uint8_t* const packed = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, packed_size, CCV_TENSOR_GPU_MEMORY);
		// No prefetch of a: the gather touches only the selected rows, so mapped tables page in just those.
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		const size_t count = payload_size + (size_t)b_rows * scale_size;
		_ccv_nnc_index_select_packed_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, row_bits, payload_size, input_scale_offset, output_scale_offset, scale_size, a->data.u8, indices->data.i32, packed);
		if (subtype == CCV_NNC_QX_8I_ROWWISE)
			ccv_nnc_compat_dequantize_8i_rowwise(packed, datatype, packed_size, a_cols, b->data.u8, (size_t)b_rows * b_cols, stream_context);
		else
			ccv_nnc_compat_dequantize_8i_rowwise_x_fp(packed, datatype, packed_size, a_cols, a->info.reserved, b->data.u8, (size_t)b_rows * b_cols, stream_context);
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(a->info.datatype == b->info.datatype);
	if (indices->info.datatype == CCV_32S)
	{
		assert(a->info.datatype == CCV_32F || a->info.datatype == CCV_16F || a->info.datatype == CCV_16BF || a->info.datatype == CCV_32S);
	} else {
		assert(indices->info.datatype == CCV_32F);
		assert(a->info.datatype == CCV_32F || a->info.datatype == CCV_16F || a->info.datatype == CCV_16BF);
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int count = b_rows * b_cols;
	if (indices->info.datatype == CCV_32S)
	{
		if (a->info.datatype == CCV_32S)
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, a->data.i32, a_cols_inc, indices->data.i32, b->data.i32, b_cols_inc);
		else if (a->info.datatype == CCV_16F || a->info.datatype == CCV_16BF)
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, (__half*)a->data.f16, a_cols_inc, indices->data.i32, (__half*)b->data.f16, b_cols_inc);
		else
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, a->data.f32, a_cols_inc, indices->data.i32, b->data.f32, b_cols_inc);
	} else {
		assert(indices->info.datatype == CCV_32F);
		if (a->info.datatype == CCV_16F)
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, (__half*)a->data.f16, a_cols_inc, a_rows, indices->data.f32, (__half*)b->data.f16, b_cols_inc);
		else if (a->info.datatype == CCV_16BF)
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, (__nv_bfloat16*)a->data.f16, a_cols_inc, a_rows, indices->data.f32, (__nv_bfloat16*)b->data.f16, b_cols_inc);
		else
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, a->data.f32, a_cols_inc, a_rows, indices->data.f32, b->data.f32, b_cols_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_index_select_zero_kernel(const int n, const int d, NUM* const a, const int a_inc)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		const int j = i % d;
		const int k = i / d;
		a[k * a_inc + j] = 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_index_select_back_kernel(const int n, const int d, const NUM* const a, const int a_inc, const int* const indices, NUM* const b, const int b_inc)
{
	CUDA_1D_KERNEL_LOOP(i, d) {
		for (int dest = 0; dest < n; dest++)
		{
			const int src = indices[dest];
			b[src * b_inc + i] += a[dest * a_inc + i];
		}
	}
}

static int _ccv_nnc_index_select_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size <= 2);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd <= 2);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)inputs[2];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == 1);
	const ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
	assert(h_nd <= 2);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (output_size >= 2 && outputs[1])
	{
		const ccv_nnc_tensor_view_t* const output = (ccv_nnc_tensor_view_t*)outputs[1];
		const int output_nd = ccv_nnc_tensor_nd(output->info.dim);
		const int output_cols = output_nd < 2 ? 1 : output->info.dim[1];
		const int output_cols_inc = CCV_IS_TENSOR_VIEW(output) ? (output_nd < 2 ? 1 : output->stride[0]) : output_cols;
		const int output_rows = output->info.dim[0];
		const int output_count = output_rows * output_cols;
		_ccv_nnc_index_select_zero_kernel<<<CUDA_GET_BLOCKS(output_count), CUDA_NUM_THREADS, 0, stream>>>(output_count, output_cols, output->data.i32, output_cols_inc);
	}
	const int g_cols = g_nd < 2 ? 1 : g->info.dim[1];
	const int g_cols_inc = CCV_IS_TENSOR_VIEW(g) ? (g_nd < 2 ? 1 : g->stride[0]) : g_cols;
	const int g_rows = g->info.dim[0];
	const int h_cols = h_nd < 2 ? 1 : h->info.dim[1];
	const int h_cols_inc = CCV_IS_TENSOR_VIEW(h) ? (h_nd < 2 ? 1 : h->stride[0]) : h_cols;
	const int h_rows = h->info.dim[0];
	assert(g_rows == indices->info.dim[0]);
	assert(g_cols == h_cols);
	assert(indices->info.datatype == CCV_32S);
	assert(g->info.datatype == h->info.datatype);
	assert(g->info.datatype == CCV_32F || g->info.datatype == CCV_16F || g->info.datatype == CCV_16BF);
	const int h_count = h_rows * h_cols;
	if (g->info.datatype == CCV_16F)
	{
		_ccv_nnc_index_select_zero_kernel<<<CUDA_GET_BLOCKS(h_count), CUDA_NUM_THREADS, 0, stream>>>(h_count, h_cols, (__half*)h->data.f16, h_cols_inc);
		_ccv_nnc_index_select_back_kernel<<<CUDA_GET_BLOCKS(g_cols), CUDA_NUM_THREADS, 0, stream>>>(g_rows, g_cols, (__half*)g->data.f16, g_cols_inc, indices->data.i32, (__half*)h->data.f16, h_cols_inc);
	} else if (g->info.datatype == CCV_16BF) {
		_ccv_nnc_index_select_zero_kernel<<<CUDA_GET_BLOCKS(h_count), CUDA_NUM_THREADS, 0, stream>>>(h_count, h_cols, (__nv_bfloat16*)h->data.f16, h_cols_inc);
		_ccv_nnc_index_select_back_kernel<<<CUDA_GET_BLOCKS(g_cols), CUDA_NUM_THREADS, 0, stream>>>(g_rows, g_cols, (__nv_bfloat16*)g->data.f16, g_cols_inc, indices->data.i32, (__nv_bfloat16*)h->data.f16, h_cols_inc);
	} else {
		_ccv_nnc_index_select_zero_kernel<<<CUDA_GET_BLOCKS(h_count), CUDA_NUM_THREADS, 0, stream>>>(h_count, h_cols, h->data.f32, h_cols_inc);
		_ccv_nnc_index_select_back_kernel<<<CUDA_GET_BLOCKS(g_cols), CUDA_NUM_THREADS, 0, stream>>>(g_rows, g_cols, g->data.f32, g_cols_inc, indices->data.i32, h->data.f32, h_cols_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F | CCV_16BF | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_index_select_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_index_select_back;
}
