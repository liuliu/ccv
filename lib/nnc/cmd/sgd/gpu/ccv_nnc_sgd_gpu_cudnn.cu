extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_sgd_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 2);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	const float neg_rate = -cmd.info.minimize.rate;
	const float decay = cmd.info.minimize.decay;
	const float momentum = cmd.info.minimize.momentum;
	const float dampening = cmd.info.minimize.dampening;
	const float inv_dampening = 1 - dampening;
	const float inv_dampening_decay = inv_dampening * decay;
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t m = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t n = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[1]);
	cudnnOpTensorDescriptor_t add = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	cudnnSetOpTensorDescriptor(add, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	static const float one = 1;
	static const float zero = 0;
	if (m.data.f32 == n.data.f32)
	{
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, add, &inv_dampening, g.descriptor, g.data.u8, &inv_dampening_decay, a.descriptor, a.data.u8, &momentum, n.descriptor, n.data.u8));
		if (a.data.f32 == b.data.f32)
		{
			CUDNN_ENFORCE(cudnnAddTensor(cudnn, &neg_rate, n.descriptor, n.data.u8, &one, b.descriptor, b.data.u8));
		} else {
			CUDNN_ENFORCE(cudnnOpTensor(cudnn, add, &neg_rate, n.descriptor, n.data.u8, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		}
	} else {
		CUDNN_ENFORCE(cudnnAddTensor(cudnn, &momentum, m.descriptor, m.data.u8, &zero, n.descriptor, n.data.u8));
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, add, &inv_dampening, g.descriptor, g.data.u8, &inv_dampening_decay, a.descriptor, a.data.u8, &one, n.descriptor, n.data.u8));
		if (a.data.f32 == b.data.f32)
		{
			CUDNN_ENFORCE(cudnnAddTensor(cudnn, &neg_rate, n.descriptor, n.data.u8, &one, b.descriptor, b.data.u8));
		} else {
			CUDNN_ENFORCE(cudnnOpTensor(cudnn, add, &neg_rate, n.descriptor, n.data.u8, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		}
	}
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, add);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(m);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(n);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sgd_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sgd_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SGD_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sgd_back;
#endif
}
