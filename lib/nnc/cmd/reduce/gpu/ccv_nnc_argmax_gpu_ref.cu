extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_argmax_forw_kernel(const int dim_without_axis, const int axis_dim, const int dim_after_axis, const NUM1* const a, NUM2* const b)
{
	CUDA_1D_KERNEL_LOOP(i, dim_without_axis) {
		const int x = i / dim_after_axis;
		const int y = i % dim_after_axis;
		const NUM1* const ap = a + x * dim_after_axis * axis_dim + y;
		NUM1 max = ap[0];
		int idx = 0;
		for (int j = 1; j < axis_dim; j++)
			if (ap[j * dim_after_axis] > max)
				max = ap[j * dim_after_axis], idx = j;
		b[x * dim_after_axis + y] = (NUM2)idx;
	}
}

static int _ccv_nnc_argmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	ccv_nnc_tensor_t* const a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(b));
	const int axis = cmd.info.reduce.axis[0];
	assert(cmd.info.reduce.count == 1);
	assert(axis >= 0);
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(axis < a_nd);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	int i;
	for (i = axis; i < a_nd; i++)
		{ assert(a->info.dim[i] == (b_nd - (a_nd - i) >= 0) ? b->info.dim[b_nd - (a_nd - i)] : 1); }
	assert(1 == (b_nd - (a_nd - axis) >= 0) ? b->info.dim[b_nd - (a_nd - axis)] : 1);
	for (i = 0; i < axis; i++)
		{ assert(a->info.dim[i] == (b_nd - (a_nd - i) >= 0) ? b->info.dim[b_nd - (a_nd - i)] : 1); }
	const int tensor_count = ccv_nnc_tensor_count(a->info);
	const int axis_dim = a->info.dim[axis];
	int dim_after_axis = 1;
	for (i = axis + 1; i < a_nd; i++)
		dim_after_axis *= a->info.dim[i];
	assert(ccv_nnc_tensor_count(b->info) == tensor_count / axis_dim);
	assert(a->info.datatype == CCV_32F || a->info.datatype == CCV_16F);
	const int dim_without_axis = tensor_count / axis_dim;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (b->info.datatype == CCV_32F)
	{
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_argmax_forw_kernel<<<CUDA_GET_BLOCKS(dim_without_axis), CUDA_NUM_THREADS, 0, stream>>>(dim_without_axis, axis_dim, dim_after_axis, a->data.f32, b->data.f32);
		else
			_ccv_nnc_argmax_forw_kernel<<<CUDA_GET_BLOCKS(dim_without_axis), CUDA_NUM_THREADS, 0, stream>>>(dim_without_axis, axis_dim, dim_after_axis, (__half*)a->data.f16, b->data.f32);
	} else if (b->info.datatype == CCV_32S) {
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_argmax_forw_kernel<<<CUDA_GET_BLOCKS(dim_without_axis), CUDA_NUM_THREADS, 0, stream>>>(dim_without_axis, axis_dim, dim_after_axis, a->data.f32, b->data.i32);
		else
			_ccv_nnc_argmax_forw_kernel<<<CUDA_GET_BLOCKS(dim_without_axis), CUDA_NUM_THREADS, 0, stream>>>(dim_without_axis, axis_dim, dim_after_axis, (__half*)a->data.f16, b->data.i32);
	} else {
		assert(b->info.datatype == CCV_16F);
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_argmax_forw_kernel<<<CUDA_GET_BLOCKS(dim_without_axis), CUDA_NUM_THREADS, 0, stream>>>(dim_without_axis, axis_dim, dim_after_axis, a->data.f32, (__half*)b->data.f16);
		else
			_ccv_nnc_argmax_forw_kernel<<<CUDA_GET_BLOCKS(dim_without_axis), CUDA_NUM_THREADS, 0, stream>>>(dim_without_axis, axis_dim, dim_after_axis, (__half*)a->data.f16, (__half*)b->data.f16);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_argmax_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_argmax_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ARGMAX_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_argmax_back;
}
