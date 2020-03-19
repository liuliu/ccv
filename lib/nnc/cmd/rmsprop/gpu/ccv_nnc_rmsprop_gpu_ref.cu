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
__global__ void _ccv_nnc_rmsprop_kernel(const size_t tensor_count, const float rate, const float decay, const float alpha, const float momentum, const float epsilon, const NUM1* const g, const NUM2* const a, const NUM2* const mom, const NUM2* const vel, NUM2* const b, NUM2* const new_mom, NUM2* const new_vel)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		float grad = (float)g[i];
		grad += decay * (float)a[i];
		const float v = alpha * (float)vel[i] + (1 - alpha) * grad * grad;
		const float m = momentum * (float)mom[i] + grad / (sqrtf(v) + epsilon);
		b[i] = (NUM2)((float)a[i] - rate * m);
		new_mom[i] = (NUM2)m;
		new_vel[i] = (NUM2)v;
	}
}

static int _ccv_nnc_rmsprop_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 4);
	assert(output_size == 3);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const float rate = cmd.info.rmsprop.rate;
	const float decay = cmd.info.rmsprop.decay;
	const float alpha = cmd.info.rmsprop.alpha;
	const float momentum = cmd.info.rmsprop.momentum;
	const float epsilon = cmd.info.rmsprop.epsilon;
	assert(inputs[1]->info.datatype == inputs[2]->info.datatype &&
		inputs[2]->info.datatype == inputs[3]->info.datatype &&
		inputs[3]->info.datatype == outputs[0]->info.datatype &&
		outputs[0]->info.datatype == outputs[1]->info.datatype &&
		outputs[1]->info.datatype == outputs[2]->info.datatype);
	assert(!CCV_IS_TENSOR_VIEW(outputs[1]));
	assert(!CCV_IS_TENSOR_VIEW(outputs[2]));
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const n = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const u = (ccv_nnc_tensor_view_t*)outputs[2];
	const size_t tensor_count = ccv_nnc_tensor_count(g->info);
	assert(tensor_count ==  ccv_nnc_tensor_count(a->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(m->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(v->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(b->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(n->info));
	assert(tensor_count ==  ccv_nnc_tensor_count(u->info));
	if (g->info.datatype == CCV_16F)
	{
		if (b->info.datatype == CCV_16F)
			_ccv_nnc_rmsprop_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, alpha, momentum, epsilon, (__half*)g->data.f16, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)v->data.f16, (__half*)b->data.f16, (__half*)n->data.f16, (__half*)u->data.f16);
		else if (b->info.datatype == CCV_32F)
			_ccv_nnc_rmsprop_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, alpha, momentum, epsilon, (__half*)g->data.f16, a->data.f32, m->data.f32, v->data.f32, outputs[0]->data.f32, n->data.f32, u->data.f32);
	} else if (g->info.datatype == CCV_32F) {
		if (b->info.datatype == CCV_16F)
			_ccv_nnc_rmsprop_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, alpha, momentum, epsilon, g->data.f32, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)v->data.f16, (__half*)b->data.f16, (__half*)n->data.f16, (__half*)u->data.f16);
		else if (b->info.datatype == CCV_32F)
			_ccv_nnc_rmsprop_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate, decay, alpha, momentum, epsilon, g->data.f32, a->data.f32, m->data.f32, v->data.f32, outputs[0]->data.f32, n->data.f32, u->data.f32);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_rmsprop_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSPROP_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsprop_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSPROP_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsprop_back;
#endif
}
