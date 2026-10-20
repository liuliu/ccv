extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_tanh_soft_cap_forw_kernel(const size_t count, const NUM1* const a, NUM2* const b, const float soft_cap, const float inv_soft_cap)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		// With --use_fast_math, tanhf lowers to tanh.approx (~5e-4 relative error), which the soft cap scales up.
		// Go through expf instead, on the negative side so it never overflows.
		const float x = (float)a[i] * inv_soft_cap;
		const float t = expf(-2 * fabsf(x));
		b[i] = (NUM2)(soft_cap * copysignf((1 - t) / (1 + t), x));
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_tanh_soft_cap_back_kernel(const size_t count, const NUM1* const g, const NUM2* const b, NUM2* const h, const float inv_soft_cap)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float y = (float)b[i] * inv_soft_cap;
		h[i] = (NUM2)((float)g[i] * (1 - y * y));
	}
}

template<typename NUM>
__global__ void _ccv_nnc_tanh_soft_cap_back_kernel(const size_t count, const NUM* const b, NUM* const h, const float inv_soft_cap)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float y = (float)b[i] * inv_soft_cap;
		h[i] = (NUM)(1 - y * y);
	}
}

// cuDNN activation cannot scale its input, soft cap goes through its own elementwise kernel.
static int _ccv_nnc_tanh_soft_cap_forw(const float soft_cap, const ccv_nnc_tensor_t* const a, ccv_nnc_tensor_t* const b, ccv_nnc_stream_context_t* const stream_context)
{
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const size_t count = ccv_nnc_tensor_count(a->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const float inv_soft_cap = 1. / soft_cap;
	if (a->info.datatype == CCV_32F && b->info.datatype == CCV_32F)
	{
		_ccv_nnc_tanh_soft_cap_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, soft_cap, inv_soft_cap);
	} else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_16F) {
		_ccv_nnc_tanh_soft_cap_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, (__half*)b->data.f16, soft_cap, inv_soft_cap);
	} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32F) {
		_ccv_nnc_tanh_soft_cap_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, b->data.f32, soft_cap, inv_soft_cap);
	} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_16F) {
		_ccv_nnc_tanh_soft_cap_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, soft_cap, inv_soft_cap);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

// d(soft_cap * tanh(x / soft_cap)) / dx = 1 - (y / soft_cap)^2.
static int _ccv_nnc_tanh_soft_cap_back(const float soft_cap, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const h, ccv_nnc_stream_context_t* const stream_context)
{
	assert(!g || CCV_IS_TENSOR_CONTIGUOUS(g));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(h));
	const size_t count = ccv_nnc_tensor_count(b->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
	{
		assert(b->info.dim[i] == h->info.dim[i]);
	}
	assert(b->info.datatype == h->info.datatype);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const float inv_soft_cap = 1. / soft_cap;
	if (g)
	{
		if (g->info.datatype == CCV_32F && b->info.datatype == CCV_32F)
		{
			_ccv_nnc_tanh_soft_cap_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, b->data.f32, h->data.f32, inv_soft_cap);
		} else if (g->info.datatype == CCV_32F && b->info.datatype == CCV_16F) {
			_ccv_nnc_tanh_soft_cap_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, (__half*)b->data.f16, (__half*)h->data.f16, inv_soft_cap);
		} else if (g->info.datatype == CCV_16F && b->info.datatype == CCV_32F) {
			_ccv_nnc_tanh_soft_cap_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, b->data.f32, h->data.f32, inv_soft_cap);
		} else if (g->info.datatype == CCV_16F && b->info.datatype == CCV_16F) {
			_ccv_nnc_tanh_soft_cap_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, (__half*)b->data.f16, (__half*)h->data.f16, inv_soft_cap);
		}
	} else {
		if (b->info.datatype == CCV_32F)
		{
			_ccv_nnc_tanh_soft_cap_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, h->data.f32, inv_soft_cap);
		} else if (b->info.datatype == CCV_16F) {
			_ccv_nnc_tanh_soft_cap_back_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)h->data.f16, inv_soft_cap);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_tanh_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	if (cmd.info.tanh.soft_cap > 0 && cmd.info.tanh.soft_cap != 1)
		return _ccv_nnc_tanh_soft_cap_forw(cmd.info.tanh.soft_cap, inputs[0], outputs[0], stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudnnActivationDescriptor_t tanh = ccv_nnc_stream_context_get_activation_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetActivationDescriptor(tanh, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	static const float one = 1, zero = 0;
	CUDNN_ENFORCE(cudnnActivationForward(cudnn, tanh, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
	ccv_nnc_stream_context_return_activation_descriptor(stream_context, tanh);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_tanh_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	if (cmd.info.tanh.soft_cap > 0 && cmd.info.tanh.soft_cap != 1)
		return _ccv_nnc_tanh_soft_cap_back(cmd.info.tanh.soft_cap, inputs[0], inputs[2], outputs[0], stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudnnActivationDescriptor_t tanh = ccv_nnc_stream_context_get_activation_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetActivationDescriptor(tanh, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	static const float one = 1, zero = 0;
	CUDNN_ENFORCE(cudnnActivationBackward(cudnn, tanh, &one, b.descriptor, b.data.u8, g.descriptor, g.data.u8, b.descriptor, b.data.u8, &zero, h.descriptor, h.data.u8));
	ccv_nnc_stream_context_return_activation_descriptor(stream_context, tanh);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_TANH_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_tanh_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_TANH_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_tanh_back;
#endif
}
