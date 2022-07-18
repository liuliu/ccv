extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <curand_kernel.h>

#ifdef HAVE_CUDNN

__global__ void _ccv_nnc_drop_entirety_select_kernel(const uint32_t seed, const float p, int* const mask)
{
	curandStatePhilox4_32_10_t state;
	curand_init(seed, 0, 0, &state);
	const float r = curand_uniform(&state); // This is from 0 to 1 open close.
	mask[0] = (r <= p);
}

template<typename NUM>
__global__ void _ccv_nnc_drop_entirety_kernel(const int count, const NUM* const a, const int* const mask, const float inv_p, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = mask[0] ? (NUM)0 : a[i] * (NUM)inv_p;
	}
}

static int _ccv_nnc_dropout_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 2);
	ccv_nnc_tensor_t* const mask = outputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(mask));
	const float p = cmd.info.dropout.p;
	const uint32_t seed = ccv_nnc_stream_context_genrand_uint32(stream_context);
	if (cmd.info.dropout.entirety)
	{
		const ccv_nnc_tensor_t* const a = inputs[0];
		ccv_nnc_tensor_t* const b = outputs[0];
		assert(CCV_IS_TENSOR_CONTIGUOUS(a));
		assert(CCV_IS_TENSOR_CONTIGUOUS(b));
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		_ccv_nnc_drop_entirety_select_kernel<<<1, 1, 0, stream>>>(seed, p, mask->data.i32);
		assert(a->info.datatype == b->info.datatype);
		const int count = ccv_nnc_tensor_count(a->info);
		assert(count == ccv_nnc_tensor_count(b->info));
		const float inv_p = 1. / (1. - p);
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_drop_entirety_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, mask->data.i32, inv_p, b->data.f32);
		else if (a->info.datatype == CCV_16F)
			_ccv_nnc_drop_entirety_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, mask->data.i32, inv_p, (__half*)b->data.f16);
	} else {
		cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
		cudnnDropoutDescriptor_t dropout = ccv_nnc_stream_context_get_dropout_descriptor(stream_context, p);
		const int tensor_count = ccv_nnc_tensor_count(mask->info);
		const size_t reserved_size = CCV_GET_DATA_TYPE_SIZE(mask->info.datatype) * tensor_count;
		CUDNN_ENFORCE(cudnnDropoutForward(cudnn, dropout, a.descriptor, a.data.u8, b.descriptor, b.data.u8, mask->data.u8, reserved_size));
		ccv_nnc_stream_context_return_dropout_descriptor(stream_context, dropout);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_dropout_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 5);
	assert(output_size >= 1);
	ccv_nnc_tensor_t* const mask = inputs[4];
	assert(CCV_IS_TENSOR_CONTIGUOUS(mask));
	if (cmd.info.dropout.entirety)
	{
		const ccv_nnc_tensor_t* const g = inputs[0];
		ccv_nnc_tensor_t* const h = outputs[0];
		assert(CCV_IS_TENSOR_CONTIGUOUS(g));
		assert(CCV_IS_TENSOR_CONTIGUOUS(h));
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		assert(g->info.datatype == h->info.datatype);
		const int count = ccv_nnc_tensor_count(g->info);
		assert(count == ccv_nnc_tensor_count(h->info));
		const float inv_p = 1. / (1. - cmd.info.dropout.p);
		if (g->info.datatype == CCV_32F)
			_ccv_nnc_drop_entirety_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, mask->data.i32, inv_p, h->data.f32);
		else if (g->info.datatype == CCV_16F)
			_ccv_nnc_drop_entirety_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, mask->data.i32, inv_p, (__half*)h->data.f16);
	} else {
		cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
		const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
		cudnnDropoutDescriptor_t dropout = ccv_nnc_stream_context_get_dropout_descriptor(stream_context, cmd.info.dropout.p);
		const int tensor_count = ccv_nnc_tensor_count(mask->info);
		const size_t reserved_size = CCV_GET_DATA_TYPE_SIZE(mask->info.datatype) * tensor_count;
		CUDNN_ENFORCE(cudnnDropoutBackward(cudnn, dropout, g.descriptor, g.data.u8, h.descriptor, h.data.u8, mask->data.u8, reserved_size));
		ccv_nnc_stream_context_return_dropout_descriptor(stream_context, dropout);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_DROPOUT_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_dropout_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DROPOUT_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_dropout_back;
#endif
}
