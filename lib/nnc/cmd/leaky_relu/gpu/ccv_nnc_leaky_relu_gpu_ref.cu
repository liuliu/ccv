extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_leaky_relu_forw_kernel(const size_t count, const float negative_slope, const NUM1* const a, NUM2* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		float x = (float)a[i];
		b[i] = (NUM2)(x >= 0 ? x : x * negative_slope);
	}
}

static int _ccv_nnc_leaky_relu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const size_t count = ccv_nnc_tensor_count(a->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const float negative_slope = cmd.info.leaky_relu.negative_slope;
	if (a->info.datatype == CCV_32F && b->info.datatype == CCV_32F)
	{
		_ccv_nnc_leaky_relu_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, a->data.f32, b->data.f32);
	} else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_16F) {
		_ccv_nnc_leaky_relu_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, a->data.f32, (__half*)b->data.f16);
	} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32F) {
		_ccv_nnc_leaky_relu_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, (__half*)a->data.f16, b->data.f32);
	} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_16F) {
		_ccv_nnc_leaky_relu_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, (__half*)a->data.f16, (__half*)b->data.f16);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_leaky_relu_back_kernel(const size_t count, const float negative_slope, const NUM1* const b, const NUM2* const g, NUM1* const h)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float x = (float)b[i];
		h[i] = x >= 0 ? (NUM1)g[i] : (NUM1)((float)g[i] * negative_slope);
	}
}

static int _ccv_nnc_leaky_relu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_t* const g = inputs[0]; // gradient
	assert(CCV_IS_TENSOR_CONTIGUOUS(g));
	const ccv_nnc_tensor_t* const b = inputs[2];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const h = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(h));
	const size_t count = ccv_nnc_tensor_count(g->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
	{
		assert(b->info.dim[i] == g->info.dim[i]);
		assert(g->info.dim[i] == h->info.dim[i]);
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(b->info.datatype == h->info.datatype);
	const float negative_slope = cmd.info.leaky_relu.negative_slope;
	if (b->info.datatype == CCV_32F && g->info.datatype == CCV_32F)
	{
		_ccv_nnc_leaky_relu_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, b->data.f32, g->data.f32, h->data.f32);
	} else if (b->info.datatype == CCV_32F && g->info.datatype == CCV_16F) {
		_ccv_nnc_leaky_relu_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, b->data.f32, (__half*)g->data.f16, h->data.f32);
	} else if (b->info.datatype == CCV_16F && g->info.datatype == CCV_32F) {
		_ccv_nnc_leaky_relu_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, (__half*)b->data.f16, g->data.f32, (__half*)h->data.f16);
	} else if (b->info.datatype == CCV_16F && g->info.datatype == CCV_16F) {
		_ccv_nnc_leaky_relu_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, negative_slope, (__half*)b->data.f16, (__half*)g->data.f16, (__half*)h->data.f16);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LEAKY_RELU_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_leaky_relu_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LEAKY_RELU_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_leaky_relu_back;
}
