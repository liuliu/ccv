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
		const float w0 = 1 - src1;
		b[dest * b_inc + j] = (NUM)((float)a[src0 * a_inc + j] * w0 + (float)a[src1 * a_inc + j] * w1);
	}
}

static int _ccv_nnc_index_select_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
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
	assert(a->info.datatype == b->info.datatype);
	assert(a->info.datatype == CCV_32F || a->info.datatype == CCV_16F);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int count = b_rows * b_cols;
	if (indices->info.datatype == CCV_32S)
	{
		if (a->info.datatype == CCV_16F)
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, (__half*)a->data.f16, a_cols_inc, indices->data.i32, (__half*)b->data.f16, b_cols_inc);
		else
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, a->data.f32, a_cols_inc, indices->data.i32, b->data.f32, b_cols_inc);
	} else {
		assert(indices->info.datatype == CCV_32F);
		if (a->info.datatype == CCV_16F)
			_ccv_nnc_index_select_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b_cols, (__half*)a->data.f16, a_cols_inc, a_rows, indices->data.f32, (__half*)b->data.f16, b_cols_inc);
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
	assert(g->info.datatype == CCV_32F || g->info.datatype == CCV_16F);
	const int h_count = h_rows * h_cols;
	if (g->info.datatype == CCV_16F)
	{
		_ccv_nnc_index_select_zero_kernel<<<CUDA_GET_BLOCKS(h_count), CUDA_NUM_THREADS, 0, stream>>>(h_count, h_cols, (__half*)h->data.f16, h_cols_inc);
		_ccv_nnc_index_select_back_kernel<<<CUDA_GET_BLOCKS(g_cols), CUDA_NUM_THREADS, 0, stream>>>(g_rows, g_cols, (__half*)g->data.f16, g_cols_inc, indices->data.i32, (__half*)h->data.f16, h_cols_inc);
	} else {
		_ccv_nnc_index_select_zero_kernel<<<CUDA_GET_BLOCKS(h_count), CUDA_NUM_THREADS, 0, stream>>>(h_count, h_cols, h->data.f32, h_cols_inc);
		_ccv_nnc_index_select_back_kernel<<<CUDA_GET_BLOCKS(g_cols), CUDA_NUM_THREADS, 0, stream>>>(g_rows, g_cols, g->data.f32, g_cols_inc, indices->data.i32, h->data.f32, h_cols_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_index_select_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_index_select_back;
}
