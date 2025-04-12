extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM>
__global__ void _ccv_nnc_scatter_add_zero_kernel(const int n, const int d, NUM* const a, const int a_inc)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		const int j = i % d;
		const int k = i / d;
		a[k * a_inc + j] = 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_scatter_add_forw_kernel(const int n, const int d, const NUM* const a, const int a_inc, const int* const indices, NUM* const b, const int b_inc)
{
	CUDA_1D_KERNEL_LOOP(i, d) {
		for (int dest = 0; dest < n; dest++)
		{
			const int src = indices[dest];
			b[src * b_inc + i] += a[dest * a_inc + i];
		}
	}
}

static int _ccv_nnc_scatter_add_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
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
	assert(a_rows == indices->info.dim[0]);
	assert(a_cols == b_cols);
	assert(a->info.datatype == b->info.datatype);
	assert(a->info.datatype == CCV_32F || a->info.datatype == CCV_16F);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int b_count = b_rows * b_cols;
	if (a->info.datatype == CCV_16F)
	{
		_ccv_nnc_scatter_add_zero_kernel<<<CUDA_GET_BLOCKS(b_count), CUDA_NUM_THREADS, 0, stream>>>(b_count, b_cols, (__half*)b->data.f16, b_cols_inc);
		_ccv_nnc_scatter_add_forw_kernel<<<CUDA_GET_BLOCKS(a_cols), CUDA_NUM_THREADS, 0, stream>>>(a_rows, a_cols, (__half*)a->data.f16, a_cols_inc, indices->data.i32, (__half*)b->data.f16, b_cols_inc);
	} else {
		_ccv_nnc_scatter_add_zero_kernel<<<CUDA_GET_BLOCKS(b_count), CUDA_NUM_THREADS, 0, stream>>>(b_count, b_cols, b->data.f32, b_cols_inc);
		_ccv_nnc_scatter_add_forw_kernel<<<CUDA_GET_BLOCKS(a_cols), CUDA_NUM_THREADS, 0, stream>>>(a_rows, a_cols, a->data.f32, a_cols_inc, indices->data.i32, b->data.f32, b_cols_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_scatter_add_back_kernel(const int n, const int d, const NUM* const a, const int a_inc, const int* const indices, NUM* const b, const int b_inc)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		const int j = i % d;
		const int dest = i / d;
		const int src = indices[dest];
		b[dest * b_inc + j] = a[src * a_inc + j];
	}
}

static int _ccv_nnc_scatter_add_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
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
		_ccv_nnc_scatter_add_zero_kernel<<<CUDA_GET_BLOCKS(output_count), CUDA_NUM_THREADS, 0, stream>>>(output_count, output_cols, output->data.i32, output_cols_inc);
	}
	const int g_cols = g_nd < 2 ? 1 : g->info.dim[1];
	const int g_cols_inc = CCV_IS_TENSOR_VIEW(g) ? (g_nd < 2 ? 1 : g->stride[0]) : g_cols;
	const int h_cols = h_nd < 2 ? 1 : h->info.dim[1];
	const int h_cols_inc = CCV_IS_TENSOR_VIEW(h) ? (h_nd < 2 ? 1 : h->stride[0]) : h_cols;
	const int h_rows = h->info.dim[0];
	assert(h_rows == indices->info.dim[0]);
	assert(g_cols == h_cols);
	assert(indices->info.datatype == CCV_32S);
	assert(g->info.datatype == h->info.datatype);
	assert(g->info.datatype == CCV_32F || g->info.datatype == CCV_16F);
	const int count = h_rows * h_cols;
	if (g->info.datatype == CCV_16F)
		_ccv_nnc_scatter_add_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, h_cols, (__half*)g->data.f16, g_cols_inc, indices->data.i32, (__half*)h->data.f16, h_cols_inc);
	else
		_ccv_nnc_scatter_add_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, h_cols, g->data.f32, g_cols_inc, indices->data.i32, h->data.f32, h_cols_inc);
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scatter_add_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scatter_add_back;
}
