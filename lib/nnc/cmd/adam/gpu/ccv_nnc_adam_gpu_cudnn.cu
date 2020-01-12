extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

template<typename NUM>
__global__ void _ccv_nnc_adam_kernel(const size_t tensor_count, const float rate_inv_bias_correction1, const float inv_bias_correction2, const float epsilon, const NUM* const a, const NUM* const mom, const NUM* const vel, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		const float inv_std = rate_inv_bias_correction1 / (sqrtf((float)vel[i] * inv_bias_correction2) + epsilon);
		b[i] = (NUM)((float)a[i] - (float)mom[i] * inv_std);
	}
}

static int _ccv_nnc_adam_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 4);
	assert(output_size == 3);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int step = cmd.info.adam.step;
	const float rate = cmd.info.adam.rate;
	const float beta1 = cmd.info.adam.beta1;
	const float beta2 = cmd.info.adam.beta2;
	const float neg_beta1 = 1. - beta1;
	const float neg_beta2 = 1. - beta2;
	const float decay = cmd.info.adam.decay;
	const float epsilon = cmd.info.adam.epsilon;
	assert(step >= 1);
	const float rate_inv_bias_correction1 = rate / (1 - powf(beta1, step));
	const float inv_bias_correction2 = 1. / (1 - powf(beta2, step));
	assert(inputs[1]->info.datatype == inputs[2]->info.datatype &&
		inputs[2]->info.datatype == inputs[3]->info.datatype &&
		inputs[3]->info.datatype == outputs[0]->info.datatype &&
		outputs[0]->info.datatype == outputs[1]->info.datatype &&
		outputs[1]->info.datatype == outputs[2]->info.datatype);
	assert(!CCV_IS_TENSOR_VIEW(outputs[1]));
	assert(!CCV_IS_TENSOR_VIEW(outputs[2]));
	static const float one = 1;
	static const float zero = 0;
	ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const size_t tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
	void* buf = ccv_nnc_stream_context_get_workspace(stream_context, tensor_count * CCV_GET_DATA_TYPE_SIZE(inputs[1]->info.datatype), CCV_TENSOR_GPU_MEMORY);
	assert(buf);
	ccv_nnc_tensor_param_t params = inputs[0]->info;
	params.datatype = inputs[1]->info.datatype;
	ccv_nnc_tensor_t t = ccv_nnc_tensor(buf, params, 0);
	ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)&t);
	ccv_nnc_cudnn_tensor_view_descriptor_t* grad = &g;
	// Cast from fp16 to fp32 if needed.
	if (inputs[0]->info.datatype != inputs[1]->info.datatype)
	{
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, g.descriptor, g.data.u8, &zero, h.descriptor, h.data.u8));
		grad = &h;
	}
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t m = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t v = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[3]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t n = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t u = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[2]);
	cudnnOpTensorDescriptor_t op = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	// This is done with float no matter what the datatype inputs / outputs are.
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	if (decay != 0)
	{
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, grad->descriptor, grad->data.u8, &decay, a.descriptor, a.data.u8, &zero, h.descriptor, h.data.u8));
		grad = &h;
	}
	if (m.data.f32 == n.data.f32)
		CUDNN_ENFORCE(cudnnAddTensor(cudnn, &neg_beta1, grad->descriptor, grad->data.u8, &beta1, n.descriptor, n.data.u8));
	else
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &beta1, m.descriptor, m.data.u8, &neg_beta1, grad->descriptor, grad->data.u8, &zero, n.descriptor, n.data.u8));
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, grad->descriptor, grad->data.u8, &one, grad->descriptor, grad->data.u8, &zero, h.descriptor, h.data.u8));
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	if (v.data.f32 == u.data.f32)
		CUDNN_ENFORCE(cudnnAddTensor(cudnn, &neg_beta2, h.descriptor, h.data.u8, &beta2, u.descriptor, u.data.u8));
	else
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &beta2, v.descriptor, v.data.u8, &neg_beta2, h.descriptor, h.data.u8, &zero, u.descriptor, u.data.u8));
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
	if (outputs[0]->info.datatype == CCV_16F)
		_ccv_nnc_adam_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate_inv_bias_correction1, inv_bias_correction2, epsilon, (__half*)a.data.f16, (__half*)n.data.f16, (__half*)u.data.f16, (__half*)outputs[0]->data.f16);
	else
		_ccv_nnc_adam_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rate_inv_bias_correction1, inv_bias_correction2, epsilon, a.data.f32, n.data.f32, u.data.f32, outputs[0]->data.f32);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(m);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(v);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(n);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(u);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_adam_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_ADAM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_adam_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADAM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_adam_back;
#endif
}
