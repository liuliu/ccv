extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_add_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	const float p = cmd.info.blas.a[0];
	static const float zero = 0;
	if (inputs[1] == 0)
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t c = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &p, a.descriptor, a.data.u8,  &zero, c.descriptor, c.data.u8));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(c);
		return CCV_NNC_EXEC_SUCCESS;
	}
	const float q = cmd.info.blas.a[1];
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t c = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	cudnnOpTensorDescriptor_t add = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	cudnnSetOpTensorDescriptor(add, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, add, &p, a.descriptor, a.data.u8, &q, b.descriptor, b.data.u8, &zero, c.descriptor, c.data.u8));
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, add);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(c);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_add_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	const float p = cmd.info.blas.a[0];
	const float q = cmd.info.blas.a[1];
	if (inputs[0] == 0)
	{
		if (outputs[0])
		{
			const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
			CUDNN_ENFORCE(cudnnSetTensor(cudnn, a.descriptor, a.data.u8, &p));
			ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
		}
		if (output_size > 1 && outputs[1])
		{
			const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[1]);
			CUDNN_ENFORCE(cudnnSetTensor(cudnn, b.descriptor, b.data.u8, &q));
			ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	int gdim[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_get_dim(g, gdim);
	static const float zero = 0;
	const ccv_nnc_cudnn_tensor_view_descriptor_t gcu = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, g);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_cudnn_tensor_view_descriptor_t acu;
	if (a)
		acu = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, a);
	const int reduce_a_dim = a ? !ccv_nnc_tensor_view_check_dim(a, gdim) : 0;
	ccv_nnc_tensor_view_t* const b = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	ccv_nnc_cudnn_tensor_view_descriptor_t bcu;
	if (b)
		 bcu = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, b);
	const int reduce_b_dim = b ? !ccv_nnc_tensor_view_check_dim(b, gdim) : 0;
	cudnnReduceTensorDescriptor_t reduce_sum;
	if ((a && reduce_a_dim) || (b && reduce_b_dim))
	{
		reduce_sum = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
		cudnnSetReduceTensorDescriptor(reduce_sum, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	}
	size_t workspace_size = 0;
	void* workspace = 0;
	if (a && reduce_a_dim)
	{
		size_t a_workspace_size = 0;
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce_sum, gcu.descriptor, acu.descriptor, &a_workspace_size));
		if (a_workspace_size > workspace_size)
			workspace_size = a_workspace_size;
	}
	if (b && reduce_b_dim)
	{
		size_t b_workspace_size = 0;
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce_sum, gcu.descriptor, bcu.descriptor, &b_workspace_size));
		if (b_workspace_size > workspace_size)
			workspace_size = b_workspace_size;
	}
	if (workspace_size)
		cudaMalloc(&workspace, workspace_size);
	if (a)
	{
		if (reduce_a_dim)
		{
			CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce_sum, 0, 0, workspace, workspace_size, &p, gcu.descriptor, gcu.data.u8, &zero, acu.descriptor, acu.data.u8));
		} else {
			CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &p, gcu.descriptor, gcu.data.u8,  &zero, acu.descriptor, acu.data.u8));
		}
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(acu);
	}
	if (b)
	{
		if (reduce_b_dim)
		{
			CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce_sum, 0, 0, workspace, workspace_size, &q, gcu.descriptor, gcu.data.u8, &zero, bcu.descriptor, bcu.data.u8));
		} else {
			CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &q, gcu.descriptor, gcu.data.u8,  &zero, bcu.descriptor, bcu.data.u8));
		}
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(bcu);
	}
	if ((a && reduce_a_dim) || (b && reduce_b_dim))
		ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce_sum);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(gcu);
	if (workspace_size)
		cudaFreeAsync(workspace, stream);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_back;
#endif
}

