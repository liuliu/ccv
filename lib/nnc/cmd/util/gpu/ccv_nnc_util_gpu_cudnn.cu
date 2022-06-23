extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_format_transform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[i]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[i]);
		assert(inputs[i]->info.datatype == outputs[i]->info.datatype);
		if (inputs[i]->info.datatype == CCV_64F)
		{
			static const double one = 1, zero = 0;
			CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		} else {
			static const float one = 1, zero = 0;
			CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		}
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
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FORMAT_TRANSFORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
#endif
}

#ifdef HAVE_CUDNN

static int _ccv_nnc_transpose(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_view_t* const tensor = (const ccv_nnc_tensor_view_t*)inputs[i];
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, tensor);
		const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[i]);
		// Reset the descriptor for a such that the dimension matches b, the strides different.
		int dim[CCV_NNC_MAX_DIM_ALLOC] = {};
		int stride[CCV_NNC_MAX_DIM_ALLOC] = {};
		const int axis_count = ccv_nnc_tensor_nd(tensor->info.dim);
		assert(ccv_nnc_tensor_nd(outputs[i]->info.dim) == axis_count);
		const int* const inc = CCV_IS_TENSOR_VIEW(tensor) ? tensor->inc : tensor->info.dim;
		int j;
		for (j = axis_count; j < CCV_NNC_MAX_DIM + 2; j++)
			dim[j] = stride[j] = 1;
		dim[axis_count - 1] = tensor->info.dim[axis_count - 1];
		stride[axis_count - 1] = 1;
		for (j = axis_count - 2; j >= 0; j--)
		{
			dim[j] = tensor->info.dim[j];
			stride[j] = stride[j + 1] * inc[j + 1];
		}
		int k;
		CCV_SWAP(dim[cmd.info.transpose.axis[0]], dim[cmd.info.transpose.axis[1]], k);
		CCV_SWAP(stride[cmd.info.transpose.axis[0]], stride[cmd.info.transpose.axis[1]], k);
		for (j = 0; j < axis_count; j++)
			{ assert(dim[j] == outputs[i]->info.dim[j]); }
		if (axis_count <= 4)
		{
			CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(a.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), dim[0], dim[1], dim[2], dim[3], stride[0], stride[1], stride[2], stride[3]));
		} else {
			CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(a.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), axis_count, dim, stride));
		}
		static const float one = 1, zero = 0;
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_transpose;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_TRANSPOSE_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_transpose;
#endif
}

#ifdef HAVE_CUDNN

static int _ccv_nnc_set_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[i]);
		if (outputs[i]->info.datatype == CCV_64F)
		{
			double v = cmd.info.blas.a[0];
			CUDNN_ENFORCE(cudnnSetTensor(cudnn, a.descriptor, a.data.u8, &v));
		} else {
			CUDNN_ENFORCE(cudnnSetTensor(cudnn, a.descriptor, a.data.u8, &cmd.info.blas.a[0]));
		}
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_set_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[i]);
		if (outputs[i]->info.datatype == CCV_64F)
		{
			static const double zero = 0;
			CUDNN_ENFORCE(cudnnSetTensor(cudnn, a.descriptor, a.data.u8, &zero));
		} else {
			static const float zero = 0;
			CUDNN_ENFORCE(cudnnSetTensor(cudnn, a.descriptor, a.data.u8, &zero));
		}
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_back;
#endif
}
