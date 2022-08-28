extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_reduce_norm2_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	ccv_nnc_tensor_view_t atv = ccv_nnc_get_tensor_view(inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	ccv_nnc_tensor_view_t btv = ccv_nnc_get_tensor_view(outputs[0]);
	ccv_nnc_tensor_view_t* tvs[] = {
		&atv, &btv
	};
	ccv_nnc_tensor_view_alignment(tvs, 2);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, &btv);
	int can_reduce = 0;
	int i;
	for (i = 0; !can_reduce && i < cmd.info.reduce.count; i++)
		can_reduce = (inputs[0]->info.dim[cmd.info.reduce.axis[i]] > 1);
	static const float one = 1, zero = 0;
	if (can_reduce)
	{
		cudnnReduceTensorDescriptor_t reduce_norm2 = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
		cudnnSetReduceTensorDescriptor(reduce_norm2, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
		void* workspace = 0;
		size_t workspace_size = 0;
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce_norm2, a.descriptor, b.descriptor, &workspace_size));
		if (workspace_size)
		{
			workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_GPU_MEMORY);
			assert(workspace);
		}
		CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce_norm2, 0, 0, workspace, workspace_size, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce_norm2);
	} else if (a.data.u8 != b.data.u8) {
		// Don't need to reduce, just transfer to b, if the pointer doesn't match.
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8,  &zero, b.descriptor, b.data.u8));
	}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2>
__global__ static void _ccv_nnc_reciprocal_kernel(const int count, const NUM1* const a, NUM2* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = (NUM2)1 / (NUM2)a[i];
	}
}

static int _ccv_nnc_reduce_norm2_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	static const float zero = 0;
	static const float one = 1;
	ccv_nnc_tensor_view_t* const htv = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const atv = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_t* const b = inputs[2];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const int tensor_count = ccv_nnc_tensor_count(b->info);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, htv);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	void* workspace = ccv_nnc_stream_context_get_workspace(stream_context, tensor_count * CCV_GET_DATA_TYPE_SIZE(b->info.datatype), CCV_TENSOR_GPU_MEMORY);
	ccv_nnc_tensor_t const rbt = ccv_nnc_tensor(workspace, b->info, 0);
	if (b->info.datatype == CCV_32F)
		_ccv_nnc_reciprocal_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, b->data.f32, rbt.data.f32);
	else
		_ccv_nnc_reciprocal_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, (__half*)b->data.f16, (__half*)rbt.data.f16);
	ccv_nnc_tensor_view_t rbtv = ccv_nnc_get_tensor_view(&rbt);
	ccv_nnc_tensor_view_t* tvs[] = {
		htv, &rbtv
	};
	ccv_nnc_tensor_view_alignment(tvs, 2);
	const ccv_nnc_cudnn_tensor_view_descriptor_t rb = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, &rbtv);
	cudnnOpTensorDescriptor_t mul = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	cudnnSetOpTensorDescriptor(mul, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, atv);
	if (inputs[0] == 0)
	{
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, mul, &one, a.descriptor, a.data.u8, &one, rb.descriptor, rb.data.u8, &zero, h.descriptor, h.data.u8));
	} else {
		ccv_nnc_tensor_view_t gtv = ccv_nnc_get_tensor_view(inputs[0]);
		ccv_nnc_tensor_view_t* tvs[] = {
			htv, &gtv
		};
		ccv_nnc_tensor_view_alignment(tvs, 2);
		const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, &gtv);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, mul, &one, rb.descriptor, rb.data.u8, &one, g.descriptor, g.data.u8, &zero, rb.descriptor, rb.data.u8));
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, mul, &one, a.descriptor, a.data.u8, &one, rb.descriptor, rb.data.u8, &zero, h.descriptor, h.data.u8));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(rb);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_norm2_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_REDUCE_NORM2_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_norm2_back;
#endif
}

