extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

__global__ void _ccv_nnc_isnan_kernel(const size_t tensor_count, const float* const a, int* const b)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		b[i] = isnan(a[i]) ? 1 : 0;
	}
}

static int _ccv_nnc_reduce_isnan_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	ccv_nnc_tensor_view_t atv = ccv_nnc_get_tensor_view(inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	ccv_nnc_tensor_view_t btv = ccv_nnc_get_tensor_view(outputs[0]);
	btv.info.datatype = CCV_32F; // First set it to 32F so it can be properly computed. Then we will turn it into 32S.
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
	const size_t tensor_count = ccv_nnc_tensor_count(outputs[0]->info);
	if (can_reduce)
	{
		cudnnReduceTensorDescriptor_t reduce_max = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
		cudnnSetReduceTensorDescriptor(reduce_max, CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
		void* workspace = 0;
		size_t workspace_size = 0;
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce_max, a.descriptor, b.descriptor, &workspace_size));
		if (workspace_size)
		{
			workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_GPU_MEMORY);
			assert(workspace);
		}
		CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce_max, 0, 0, workspace, workspace_size, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
		ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce_max);
	} else if (a.data.u8 != b.data.u8) {
		// Don't need to reduce, just transfer to b, if the pointer doesn't match.
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8,  &zero, b.descriptor, b.data.u8));
	}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	_ccv_nnc_isnan_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, outputs[0]->data.f32, outputs[0]->data.i32);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_reduce_isnan_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_isnan_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_REDUCE_ISNAN_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_isnan_back;
#endif
}

