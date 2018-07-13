extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_dropout_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	assert(output_size == 2);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	cudnnDropoutDescriptor_t dropout = ccv_nnc_stream_context_get_dropout_descriptor(stream_context, cmd.info.dropout.p);
	ccv_nnc_tensor_t* const mask = outputs[1];
	assert(!CCV_IS_TENSOR_VIEW(mask));
	const int tensor_count = ccv_nnc_tensor_count(mask->info);
	const size_t reserved_size = CCV_GET_DATA_TYPE_SIZE(mask->info.datatype) * tensor_count;
	CUDNN_ENFORCE(cudnnDropoutForward(cudnn, dropout, a.descriptor, a.data.u8, b.descriptor, b.data.u8, mask->data.u8, reserved_size));
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_stream_context_return_dropout_descriptor(stream_context, dropout);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_dropout_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 5);
	assert(output_size >= 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	cudnnDropoutDescriptor_t dropout = ccv_nnc_stream_context_get_dropout_descriptor(stream_context, cmd.info.dropout.p);
	ccv_nnc_tensor_t* const mask = inputs[4];
	assert(!CCV_IS_TENSOR_VIEW(mask));
	const int tensor_count = ccv_nnc_tensor_count(mask->info);
	const size_t reserved_size = CCV_GET_DATA_TYPE_SIZE(mask->info.datatype) * tensor_count;
	CUDNN_ENFORCE(cudnnDropoutBackward(cudnn, dropout, g.descriptor, g.data.u8, h.descriptor, h.data.u8, mask->data.u8, reserved_size));
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	ccv_nnc_stream_context_return_dropout_descriptor(stream_context, dropout);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_DROPOUT_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_dropout_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DROPOUT_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_dropout_back;
#endif
}
