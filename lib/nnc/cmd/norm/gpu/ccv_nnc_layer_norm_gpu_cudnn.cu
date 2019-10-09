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
__global__ void _ccv_nnc_inv_std_kernel(const int count, const float epsilon, const NUM* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = (NUM)(1. / ((float)a[i] + epsilon));
	}
}

static int _ccv_nnc_layer_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	static const float one = 1, zero = 0, neg_one = -1;
	assert(output_size == 3);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	assert(!CCV_IS_TENSOR_VIEW(inputs[1]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	assert(!CCV_IS_TENSOR_VIEW(inputs[2]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t bias = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	assert(!CCV_IS_TENSOR_VIEW(outputs[1]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_mean = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[1]);
	assert(!CCV_IS_TENSOR_VIEW(outputs[2]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_inv_std = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[2]);
	int adim[CCV_NNC_MAX_DIM + 2];
	int rdim[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[0], adim);
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[1], rdim);
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)inputs[2], rdim));
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)outputs[1], rdim));
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)outputs[2], rdim));
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)outputs[0], adim));
	int x;
	int n = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n *= adim[x];
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n /= rdim[x];
	int rcount = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		rcount *= rdim[x];
	const float inv_n = 1. / n;
	cudnnReduceTensorDescriptor_t reduce = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
	size_t saved_mean_workspace_size = 0;
	size_t saved_inv_std_workspace_size = 0;
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, a.descriptor, saved_mean.descriptor, &saved_mean_workspace_size));
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, b.descriptor, saved_inv_std.descriptor, &saved_inv_std_workspace_size));
	const size_t workspace_size = ccv_max(saved_mean_workspace_size, saved_inv_std_workspace_size);
	void* workspace = 0;
	if (workspace_size)
		workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_GPU_MEMORY);
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &inv_n, a.descriptor, a.data.u8, &zero, saved_mean.descriptor, saved_mean.data.u8));
	cudnnOpTensorDescriptor_t op = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, a.descriptor, a.data.u8, &neg_one, saved_mean.descriptor, saved_mean.data.u8, &zero, b.descriptor, b.data.u8));
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	const float inv_n_sqrt = sqrt(inv_n);
	CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &inv_n_sqrt, b.descriptor, b.data.u8, &zero, saved_inv_std.descriptor, saved_inv_std.data.u8));
	// The epsilon is used a little bit differently from batch norm, it is outside of the sqrt in this case.
	const float epsilon = cmd.info.lnorm.epsilon;
	assert(outputs[2]->info.datatype == CCV_16F || outputs[2]->info.datatype == CCV_32F);
	if (outputs[2]->info.datatype == CCV_16F)
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, (__half*)saved_inv_std.data.f16, (__half*)saved_inv_std.data.f16);
	else
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, saved_inv_std.data.f32, saved_inv_std.data.f32);
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b.descriptor, b.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, b.descriptor, b.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b.descriptor, b.data.u8, &one, scale.descriptor, scale.data.u8, &zero, b.descriptor, b.data.u8));
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b.descriptor, b.data.u8, &one, bias.descriptor, bias.data.u8, &zero, b.descriptor, b.data.u8));
	ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce);
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_layer_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 9);
	assert(output_size == 3);
	// cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	// static const float one = 1, zero = 0;
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[5]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[6]);
	// ccv_nnc_tensor_t* const saved_mean = inputs[13];
	// ccv_nnc_tensor_t* const saved_inv_std = inputs[14];
	// ccv_nnc_tensor_t* const dscale = outputs[1];
	// ccv_nnc_tensor_t* const dbias = outputs[2];
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_back;
#endif
}
