extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_format_transform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[i]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[i]);
		static const float one = 1, zero = 0;
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FORMAT_TRANSFORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
#endif
}

#ifdef HAVE_CUDNN

static int _ccv_nnc_set_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 0);
	int i;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[i]);
		CUDNN_ENFORCE(cudnnSetTensor(cudnn, a.descriptor, a.data.u8, &cmd.info.blas.a[0]));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_set_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[i]);
		static const float zero = 0;
		CUDNN_ENFORCE(cudnnSetTensor(cudnn, a.descriptor, a.data.u8, &zero));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_back;
#endif
}
