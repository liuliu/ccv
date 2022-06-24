extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_sgd_nesterov_kernel(const size_t tensor_count, const float rate, const float decay, const float scale, const float momentum, const NUM1* const g, const NUM2* const a, const NUM2* const mom, NUM2* const b, NUM2* const new_mom)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		float grad = scale * (float)g[i];
		const float m = momentum * (float)mom[i] + grad + decay * (float)a[i];
		grad += momentum * m;
		b[i] = (NUM2)((float)a[i] - rate * grad);
		new_mom[i] = (NUM2)m;
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_sgd_kernel(const size_t tensor_count, const float rate, const float decay, const float scale, const float momentum, const float inv_dampening, const NUM1* const g, const NUM2* const a, const NUM2* const mom, NUM2* const b, NUM2* const new_mom)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		const float m = momentum * (float)mom[i] + inv_dampening * (scale * (float)g[i] + decay * (float)a[i]);
		b[i] = (NUM2)((float)a[i] - rate * m);
		new_mom[i] = (NUM2)m;
	}
}

static int _ccv_nnc_sgd_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 2);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int nesterov = cmd.info.sgd.nesterov;
	const float rate = cmd.info.sgd.rate;
	const float scale = cmd.info.sgd.scale;
	const float decay = cmd.info.sgd.decay;
	const float momentum = cmd.info.sgd.momentum;
	const float dampening = cmd.info.sgd.dampening;
	const float inv_dampening = 1 - dampening;
	if (nesterov)
		{ assert(dampening == 0); }
	assert(inputs[1]->info.datatype == inputs[2]->info.datatype &&
		inputs[2]->info.datatype == outputs[0]->info.datatype &&
		outputs[0]->info.datatype == outputs[1]->info.datatype);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[0]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[1]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[2]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[0]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[1]));
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const n = (ccv_nnc_tensor_view_t*)outputs[1];
	const size_t tensor_count = ccv_nnc_tensor_count(g->info);
	assert(tensor_count ==  ccv_nnc_tensor_count(a->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(m->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(b->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(n->info));
	if (nesterov)
	{
		if (g->info.datatype == CCV_16F)
		{
			if (b->info.datatype == CCV_16F)
				_ccv_nnc_sgd_nesterov_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, (__half*)g->data.f16, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)b->data.f16, (__half*)n->data.f16);
			else if (b->info.datatype == CCV_32F)
				_ccv_nnc_sgd_nesterov_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, (__half*)g->data.f16, a->data.f32, m->data.f32, b->data.f32, n->data.f32);
		} else if (g->info.datatype == CCV_32F) {
			if (b->info.datatype == CCV_16F)
				_ccv_nnc_sgd_nesterov_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, g->data.f32, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)b->data.f16, (__half*)n->data.f16);
			else if (b->info.datatype == CCV_32F)
				_ccv_nnc_sgd_nesterov_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, g->data.f32, a->data.f32, m->data.f32, b->data.f32, n->data.f32);
		}
	} else {
		if (g->info.datatype == CCV_16F)
		{
			if (b->info.datatype == CCV_16F)
				_ccv_nnc_sgd_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, inv_dampening, (__half*)g->data.f16, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)b->data.f16, (__half*)n->data.f16);
			else if (b->info.datatype == CCV_32F)
				_ccv_nnc_sgd_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, inv_dampening, (__half*)g->data.f16, a->data.f32, m->data.f32, b->data.f32, n->data.f32);
		} else if (g->info.datatype == CCV_32F) {
			if (b->info.datatype == CCV_16F)
				_ccv_nnc_sgd_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, inv_dampening, g->data.f32, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)b->data.f16, (__half*)n->data.f16);
			else if (b->info.datatype == CCV_32F)
				_ccv_nnc_sgd_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, scale, momentum, inv_dampening, g->data.f32, a->data.f32, m->data.f32, b->data.f32, n->data.f32);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sgd_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sgd_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SGD_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sgd_back;
#endif
}
