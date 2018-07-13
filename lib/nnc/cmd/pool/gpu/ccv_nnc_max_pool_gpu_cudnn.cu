extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_max_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	cudnnPoolingDescriptor_t max_pool = ccv_nnc_stream_context_get_pooling_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(max_pool, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
		cmd.info.size.dim[0], cmd.info.size.dim[1],
		hint.border.begin[0], hint.border.begin[1],
		hint.stride.dim[0], hint.stride.dim[1]
	));
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	static const float one = 1, zero = 0;
	CUDNN_ENFORCE(cudnnPoolingForward(cudnn, max_pool, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
	ccv_nnc_stream_context_return_pooling_descriptor(stream_context, max_pool);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_max_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	cudnnPoolingDescriptor_t max_pool = ccv_nnc_stream_context_get_pooling_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(max_pool, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
		cmd.info.size.dim[0], cmd.info.size.dim[1],
		hint.border.begin[0], hint.border.begin[1],
		hint.stride.dim[0], hint.stride.dim[1]
	));
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	static const float one = 1, zero = 0;
	CUDNN_ENFORCE(cudnnPoolingBackward(cudnn, max_pool, &one, b.descriptor, b.data.u8, g.descriptor, g.data.u8, a.descriptor, a.data.u8, &zero, h.descriptor, h.data.u8));
	ccv_nnc_stream_context_return_pooling_descriptor(stream_context, max_pool);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_MAX_POOL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_max_pool_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MAX_POOL_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_max_pool_back;
#endif
}
