extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM>
__global__ void _ccv_nnc_rotate_half_kernel(const size_t count, const int half, const int dim, const NUM* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const size_t row = i / dim;
		const int x = i - row * dim;
		b[i] = a[row * dim + ((x < half) ? (x + half) : (x - half))];
	}
}

static int _ccv_nnc_rotate_half_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd > 0);
	assert(a_nd == ccv_nnc_tensor_nd(b->info.dim));
	int i;
	for (i = 0; i < a_nd; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	assert(a->info.datatype == b->info.datatype);
	const int dim = a->info.dim[a_nd - 1];
	const int half = dim / 2;
	assert(half > 0);
	assert(dim == half * 2);
	const size_t tensor_count = ccv_nnc_tensor_count(a->info);
	assert(tensor_count == ccv_nnc_tensor_count(b->info));
	assert(tensor_count % dim == 0);
	const size_t count = tensor_count;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	int handled = 1;
	if (a->data.u8 == b->data.u8)
		return CCV_NNC_EXEC_INVALID;
	if (a->info.datatype == CCV_32F)
		_ccv_nnc_rotate_half_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, half, dim, a->data.f32, b->data.f32);
	else if (a->info.datatype == CCV_16F)
		_ccv_nnc_rotate_half_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, half, dim, (__half*)a->data.f16, (__half*)b->data.f16);
	else if (a->info.datatype == CCV_16BF)
		_ccv_nnc_rotate_half_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, half, dim, (__nv_bfloat16*)a->data.f16, (__nv_bfloat16*)b->data.f16);
	else
		handled = 0;
	if (!handled)
		return CCV_NNC_EXEC_INVALID;
	CUDA_ENFORCE(cudaGetLastError());
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_rotate_half_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size == 1);
	return _ccv_nnc_rotate_half_forw(cmd, hint, flags, inputs, 1, outputs, output_size, stream_context);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROTATE_HALF_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rotate_half_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROTATE_HALF_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rotate_half_back;
}
