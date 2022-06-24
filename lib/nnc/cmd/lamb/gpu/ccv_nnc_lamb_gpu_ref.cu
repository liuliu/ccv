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
__global__ void _ccv_nnc_adam_kernel(const size_t tensor_count, const float beta1, const float beta2, const float decay, const float inv_bias_correction1, const float inv_bias_correction2, const float epsilon, const NUM1* const g, const NUM2* const a, const NUM2* const mom, const NUM2* const vel, float* const b, NUM2* const new_mom, NUM2* const new_vel)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		float grad = (float)g[i];
		const float m = beta1 * (float)mom[i] + (1 - beta1) * grad;
		const float v = beta2 * (float)vel[i] + (1 - beta2) * grad * grad;
		b[i] = (m * inv_bias_correction1) / (sqrtf(v * inv_bias_correction2) + epsilon) + (float)a[i] * decay;
		new_mom[i] = (NUM2)m;
		new_vel[i] = (NUM2)v;
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_adam_kernel_to_float(const size_t tensor_count, const float beta1, const float beta2, const float decay, const float inv_bias_correction1, const float inv_bias_correction2, const float epsilon, const NUM1* const g, const NUM2* const a, const NUM2* const mom, const NUM2* const vel, float* const b, float* const c, NUM2* const new_mom, NUM2* const new_vel)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		float grad = (float)g[i];
		const float m = beta1 * (float)mom[i] + (1 - beta1) * grad;
		const float v = beta2 * (float)vel[i] + (1 - beta2) * grad * grad;
		b[i] = (m * inv_bias_correction1) / (sqrtf(v * inv_bias_correction2) + epsilon) + (float)a[i] * decay;
		c[i] = (float)a[i];
		new_mom[i] = (NUM2)m;
		new_vel[i] = (NUM2)v;
	}
}

__global__ void _ccv_nnc_rate_trust_ratio(const float rate, const float* const w_norm, const float* const update_norm, float* const rate_trust_ratio)
{
	const float trust_ratio = w_norm[0] > 0 && update_norm[0] > 0 ? w_norm[0] / update_norm[0] : 1;
	rate_trust_ratio[0] = rate * trust_ratio;
}

template<typename NUM>
__global__ void _ccv_nnc_lamb_kernel(const size_t tensor_count, const float* const rate_trust_ratio, const float* const g, const float* const a, NUM* const b)
{
	const float r = rate_trust_ratio[0];
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		b[i] = (NUM)(a[i] - r * g[i]);
	}
}

static int _ccv_nnc_lamb_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 4);
	assert(output_size == 3);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	const int step = cmd.info.lamb.step;
	const float rate = cmd.info.lamb.rate;
	const float beta1 = cmd.info.lamb.beta1;
	const float beta2 = cmd.info.lamb.beta2;
	const float decay = cmd.info.lamb.decay;
	const float epsilon = cmd.info.lamb.epsilon;
	assert(step >= 1);
	const float inv_bias_correction1 = 1. / (1 - powf(beta1, step));
	const float inv_bias_correction2 = 1. / (1 - powf(beta2, step));
	assert(inputs[1]->info.datatype == inputs[2]->info.datatype &&
		inputs[2]->info.datatype == inputs[3]->info.datatype &&
		inputs[3]->info.datatype == outputs[0]->info.datatype &&
		outputs[0]->info.datatype == outputs[1]->info.datatype &&
		outputs[1]->info.datatype == outputs[2]->info.datatype);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[0]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[1]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[2]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[3]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[0]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[1]));
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[2]));
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[3];
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
	if (b->info.datatype == CCV_16F)
	{
		float* const update = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * (tensor_count * 2 + 2), CCV_TENSOR_GPU_MEMORY);
		float* const af = update + tensor_count;
		if (g->info.datatype == CCV_16F)
		{
			_ccv_nnc_adam_kernel_to_float<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, beta1, beta2, decay, inv_bias_correction1, inv_bias_correction2, epsilon, (__half*)g->data.f16, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)v->data.f16, update, af, (__half*)n->data.f16, (__half*)u->data.f16);
		} else if (g->info.datatype == CCV_32F) {
			_ccv_nnc_adam_kernel_to_float<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, beta1, beta2, decay, inv_bias_correction1, inv_bias_correction2, epsilon, g->data.f32, (__half*)a->data.f16, (__half*)m->data.f16, (__half*)v->data.f16, update, af, (__half*)n->data.f16, (__half*)u->data.f16);
		}
		float* const w_norm = af + tensor_count;
		float* const update_norm = w_norm + 1;
		cublasPointerMode_t oldMode;
		cublasGetPointerMode(cublas, &oldMode);
		cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
		cublasSnrm2(cublas, tensor_count, af, 1, w_norm);
		cublasSnrm2(cublas, tensor_count, update, 1, update_norm);
		// Set it back.
		cublasSetPointerMode(cublas, oldMode);
		float* const rate_trust_ratio = update_norm; // Reuse the memory.
		_ccv_nnc_rate_trust_ratio<<<1, 1, 0, stream>>>(rate, w_norm, update_norm, rate_trust_ratio);
		_ccv_nnc_lamb_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate_trust_ratio, update, af, (__half*)b->data.f16);
	} else if (b->info.datatype == CCV_32F) {
		float* const update = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * (tensor_count + 2), CCV_TENSOR_GPU_MEMORY);
		if (g->info.datatype == CCV_16F)
		{
			_ccv_nnc_adam_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, beta1, beta2, decay, inv_bias_correction1, inv_bias_correction2, epsilon, (__half*)g->data.f16, a->data.f32, m->data.f32, v->data.f32, update, n->data.f32, u->data.f32);
		} else if (g->info.datatype == CCV_32F) {
			_ccv_nnc_adam_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, beta1, beta2, decay, inv_bias_correction1, inv_bias_correction2, epsilon, g->data.f32, a->data.f32, m->data.f32, v->data.f32, update, n->data.f32, u->data.f32);
		}
		float* const w_norm = update + tensor_count;
		float* const update_norm = w_norm + 1;
		cublasPointerMode_t oldMode;
		cublasGetPointerMode(cublas, &oldMode);
		cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
		cublasSnrm2(cublas, tensor_count, a->data.f32, 1, w_norm);
		cublasSnrm2(cublas, tensor_count, update, 1, update_norm);
		// Set it back.
		cublasSetPointerMode(cublas, oldMode);
		float* const rate_trust_ratio = update_norm; // Reuse the memory.
		_ccv_nnc_rate_trust_ratio<<<1, 1, 0, stream>>>(rate, w_norm, update_norm, rate_trust_ratio);
		_ccv_nnc_lamb_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate_trust_ratio, update, a->data.f32, b->data.f32);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_lamb_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_LAMB_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lamb_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAMB_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lamb_back;
#endif
}
