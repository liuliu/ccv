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
		b[i] = (NUM)(1. / sqrt((float)a[i] * (float)a[i] + epsilon));
	}
}

static int _ccv_nnc_rmsnorm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	static const float one = 1, zero = 0;
	assert(output_size == 2);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[1]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[1]));
	const int saved_datatype = outputs[1]->info.datatype;
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_inv_std = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[1]);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[0], adim);
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)outputs[1], rdim);
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
	size_t saved_inv_std_workspace_size = 0;
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, a.descriptor, saved_inv_std.descriptor, &saved_inv_std_workspace_size));
	const size_t workspace_size = saved_inv_std_workspace_size + sizeof(float) * rcount;
	uint8_t* const workspace = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_GPU_MEMORY);
	const float inv_n_sqrt = sqrt(inv_n);
	CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &inv_n_sqrt, a.descriptor, a.data.u8, &zero, saved_inv_std.descriptor, saved_inv_std.data.u8));
	// The epsilon is used a little bit differently from batch norm, it is inside the sqrt.
	const float epsilon = cmd.info.lnorm.epsilon;
	if (saved_datatype == CCV_32F)
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, saved_inv_std.data.f32, saved_inv_std.data.f32);
	else if (saved_datatype == CCV_16F)
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, (__half*)saved_inv_std.data.f16, (__half*)saved_inv_std.data.f16);
	cudnnOpTensorDescriptor_t op = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, a.descriptor, a.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, b.descriptor, b.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b.descriptor, b.data.u8, &one, scale.descriptor, scale.data.u8, &zero, b.descriptor, b.data.u8));
	ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce);
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_rmsnorm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 6);
	assert(output_size >= 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[3]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[3]);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[5]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_inv_std = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[5]);
	if (output_size > 1 && outputs[1])
		{ assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[1])); }
	const ccv_nnc_cudnn_tensor_view_descriptor_t dscale = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, output_size > 1 ? (const ccv_nnc_tensor_view_t*)outputs[1] : 0);
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[0], gdim);
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[5], rdim);
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)inputs[2], gdim));
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)outputs[0], gdim));
	static const float one = 1, zero = 0;
	int x;
	int n = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n *= gdim[x];
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n /= rdim[x];
	int gcount = 1, rcount = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		gcount *= gdim[x], rcount *= rdim[x];
	const float neg_inv_n = -1. / n;
	cudnnReduceTensorDescriptor_t reduce = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	size_t scale_workspace_size = 0;
	if (dscale.descriptor)
		{ CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, g.descriptor, dscale.descriptor, &scale_workspace_size)); }
	size_t inv_std_workspace_size = 0;
	CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, g.descriptor, saved_inv_std.descriptor, &inv_std_workspace_size));
	const size_t workspace_size = ccv_max(scale_workspace_size, inv_std_workspace_size);
	uint8_t* const workspace = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, workspace_size + sizeof(float) * gcount * 3 + sizeof(float) * rcount, CCV_TENSOR_GPU_MEMORY);
	float* const ahp = (float*)(workspace + workspace_size);
	const ccv_nnc_tensor_t aht = ccv_nnc_tensor(ahp, inputs[0]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t ah = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&aht);
	float* const gssp = ahp + gcount;
	const ccv_nnc_tensor_t gsst = ccv_nnc_tensor(gssp, inputs[0]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t gss = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&gsst);
	float* const ahgssp = gssp + gcount;
	const ccv_nnc_tensor_t ahgsst = ccv_nnc_tensor(ahgssp, inputs[0]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t ahgss = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&ahgsst);
	float* const ahgssrp = ahgssp + gcount;
	const ccv_nnc_tensor_t ahgssrt = ccv_nnc_tensor(ahgssrp, inputs[5]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t ahgssr = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&ahgssrt);
	cudnnOpTensorDescriptor_t op = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, a.descriptor, a.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, ah.descriptor, ah.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, g.descriptor, g.data.u8, &zero, ahgss.descriptor, ahgss.data.u8));
	if (dscale.descriptor)
		{ CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &one, ahgss.descriptor, ahgss.data.u8, &zero, dscale.descriptor, dscale.data.u8)); }
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, g.descriptor, g.data.u8, &one, scale.descriptor, scale.data.u8, &zero, gss.descriptor, gss.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, gss.descriptor, gss.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, gss.descriptor, gss.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, gss.descriptor, gss.data.u8, &zero, ahgss.descriptor, ahgss.data.u8));
	CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &one, ahgss.descriptor, ahgss.data.u8, &zero, ahgssr.descriptor, ahgssr.data.u8));
	ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, ahgssr.descriptor, ahgssr.data.u8, &zero, ah.descriptor, ah.data.u8));
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, gss.descriptor, gss.data.u8, &neg_inv_n, ah.descriptor, ah.data.u8, &zero, h.descriptor, h.data.u8));
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(dscale);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(ah);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(gss);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(ahgss);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(ahgssr);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_back;
#endif
}
