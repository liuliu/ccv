extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include "ccv_nnc_flash_norm_gpu.h"

#ifdef HAVE_CUDNN

template<typename NUM>
__global__ void _ccv_nnc_inv_std_kernel(const int count, const float epsilon, const NUM* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = (NUM)(1. / sqrt((float)a[i] * (float)a[i] + epsilon));
	}
}

static int _ccv_nnc_layer_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3 || input_size == 1);
	assert(output_size == 3);
	if (cmd.info.lnorm.scale != 1)
		return CCV_NNC_EXEC_INVALID;
	const int elementwise_affine = cmd.info.lnorm.elementwise_affine;
#ifdef HAVE_CUDA_SM80
	ccv_nnc_flash_norm_info_t flash_norm = {};
	if (_ccv_nnc_flash_norm_check(cmd.info.lnorm.axis, cmd.info.lnorm.count, inputs[0], outputs[0], outputs[1], outputs[2], elementwise_affine ? inputs[1] : 0, elementwise_affine ? inputs[2] : 0, elementwise_affine, &flash_norm) &&
		_ccv_nnc_flash_norm_forw(stream_context, inputs[0], elementwise_affine ? inputs[1] : 0, elementwise_affine ? inputs[2] : 0, outputs[0], outputs[1], outputs[2], flash_norm, cmd.info.lnorm.epsilon, 0))
		return CCV_NNC_EXEC_SUCCESS;
#endif
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	static const float one = 1, zero = 0, neg_one = -1;
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	if (elementwise_affine)
	{
		assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[1]));
		assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[2]));
	}
	const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, elementwise_affine ? (const ccv_nnc_tensor_view_t*)inputs[1] : 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t bias = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, elementwise_affine ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[1]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_mean = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[1]);
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[2]));
	assert(outputs[1]->info.datatype == outputs[2]->info.datatype);
	const int saved_datatype = outputs[1]->info.datatype;
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_inv_std = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[2]);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[0], adim);
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)outputs[1], rdim);
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
	const int use_bfloat = inputs[0]->info.datatype == CCV_16BF || outputs[0]->info.datatype == CCV_16BF || outputs[1]->info.datatype == CCV_16BF || outputs[2]->info.datatype == CCV_16BF || (elementwise_affine && (inputs[1]->info.datatype == CCV_16BF || inputs[2]->info.datatype == CCV_16BF));
	if (use_bfloat)
	{
		const int acount = n * rcount;
		ccv_nnc_tensor_param_t a32_info = inputs[0]->info;
		a32_info.datatype = CCV_32F;
		ccv_nnc_tensor_param_t b32_info = outputs[0]->info;
		b32_info.datatype = CCV_32F;
		ccv_nnc_tensor_param_t saved_mean32_info = outputs[1]->info;
		saved_mean32_info.datatype = CCV_32F;
		ccv_nnc_tensor_param_t saved_inv_std32_info = outputs[2]->info;
		saved_inv_std32_info.datatype = CCV_32F;
		size_t placeholder = 0;
		ccv_nnc_tensor_t a32t = ccv_nnc_tensor(&placeholder, a32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t a32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&a32t);
		ccv_nnc_tensor_t b32t = ccv_nnc_tensor(&placeholder, b32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t b32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&b32t);
		ccv_nnc_tensor_t saved_mean32t = ccv_nnc_tensor(&placeholder, saved_mean32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t saved_mean32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&saved_mean32t);
		ccv_nnc_tensor_t saved_inv_std32t = ccv_nnc_tensor(&placeholder, saved_inv_std32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t saved_inv_std32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&saved_inv_std32t);
		ccv_nnc_cudnn_tensor_view_descriptor_t scale32 = {};
		ccv_nnc_cudnn_tensor_view_descriptor_t bias32 = {};
		int scale_count = 0;
		int bias_count = 0;
		if (elementwise_affine)
		{
			ccv_nnc_tensor_param_t scale32_info = inputs[1]->info;
			scale32_info.datatype = CCV_32F;
			scale_count = ccv_nnc_tensor_count(scale32_info);
			ccv_nnc_tensor_t scale32t = ccv_nnc_tensor(&placeholder, scale32_info, 0);
			scale32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&scale32t);
			ccv_nnc_tensor_param_t bias32_info = inputs[2]->info;
			bias32_info.datatype = CCV_32F;
			bias_count = ccv_nnc_tensor_count(bias32_info);
			ccv_nnc_tensor_t bias32t = ccv_nnc_tensor(&placeholder, bias32_info, 0);
			bias32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&bias32t);
		}
		cudnnReduceTensorDescriptor_t reduce = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
		size_t saved_mean_workspace_size = 0;
		size_t saved_inv_std_workspace_size = 0;
		cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, a32.descriptor, saved_mean32.descriptor, &saved_mean_workspace_size));
		cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, b32.descriptor, saved_inv_std32.descriptor, &saved_inv_std_workspace_size));
		const size_t workspace_size = ccv_max(saved_mean_workspace_size, saved_inv_std_workspace_size);
		uint8_t* const workspace = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, workspace_size + sizeof(float) * (acount * 2 + rcount * 2 + scale_count + bias_count), CCV_TENSOR_GPU_MEMORY);
		float* const a32p = (float*)(workspace + workspace_size);
		a32.data.u8 = (uint8_t*)a32p;
		float* const b32p = a32p + acount;
		b32.data.u8 = (uint8_t*)b32p;
		float* const saved_mean32p = b32p + acount;
		saved_mean32.data.u8 = (uint8_t*)saved_mean32p;
		float* const saved_inv_std32p = saved_mean32p + rcount;
		saved_inv_std32.data.u8 = (uint8_t*)saved_inv_std32p;
		if (elementwise_affine)
		{
			float* const scale32p = saved_inv_std32p + rcount;
			scale32.data.u8 = (uint8_t*)scale32p;
			float* const bias32p = scale32p + scale_count;
			bias32.data.u8 = (uint8_t*)bias32p;
		}
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8, &zero, a32.descriptor, a32.data.u8));
		if (elementwise_affine)
		{
			CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, scale.descriptor, scale.data.u8, &zero, scale32.descriptor, scale32.data.u8));
			CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, bias.descriptor, bias.data.u8, &zero, bias32.descriptor, bias32.data.u8));
		}
		cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
		CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &inv_n, a32.descriptor, a32.data.u8, &zero, saved_mean32.descriptor, saved_mean32.data.u8));
		cudnnOpTensorDescriptor_t op = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
		cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, a32.descriptor, a32.data.u8, &neg_one, saved_mean32.descriptor, saved_mean32.data.u8, &zero, b32.descriptor, b32.data.u8));
		cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
		const float inv_n_sqrt = sqrt(inv_n);
		CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &inv_n_sqrt, b32.descriptor, b32.data.u8, &zero, saved_inv_std32.descriptor, saved_inv_std32.data.u8));
		const float epsilon = cmd.info.lnorm.epsilon;
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, saved_inv_std32.data.f32, saved_inv_std32.data.f32);
		cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b32.descriptor, b32.data.u8, &one, saved_inv_std32.descriptor, saved_inv_std32.data.u8, &zero, b32.descriptor, b32.data.u8));
		if (elementwise_affine)
		{
			CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b32.descriptor, b32.data.u8, &one, scale32.descriptor, scale32.data.u8, &zero, b32.descriptor, b32.data.u8));
			cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
			CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b32.descriptor, b32.data.u8, &one, bias32.descriptor, bias32.data.u8, &zero, b32.descriptor, b32.data.u8));
		}
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, b32.descriptor, b32.data.u8, &zero, b.descriptor, b.data.u8));
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, saved_mean32.descriptor, saved_mean32.data.u8, &zero, saved_mean.descriptor, saved_mean.data.u8));
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, saved_inv_std32.descriptor, saved_inv_std32.data.u8, &zero, saved_inv_std.descriptor, saved_inv_std.data.u8));
		ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce);
		ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(bias);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_mean);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(b32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(bias32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_mean32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std32);
		return CCV_NNC_EXEC_SUCCESS;
	}
	cudnnReduceTensorDescriptor_t reduce = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
	size_t saved_mean_workspace_size = 0;
	size_t saved_inv_std_workspace_size = 0;
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, a.descriptor, saved_mean.descriptor, &saved_mean_workspace_size));
	cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
	CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, b.descriptor, saved_inv_std.descriptor, &saved_inv_std_workspace_size));
	const size_t workspace_size = ccv_max(saved_mean_workspace_size, saved_inv_std_workspace_size) + sizeof(float) * rcount;
	uint8_t* const workspace = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_GPU_MEMORY);
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
	if (saved_datatype == CCV_32F)
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, saved_inv_std.data.f32, saved_inv_std.data.f32);
	else if (saved_datatype == CCV_16F)
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, (__half*)saved_inv_std.data.f16, (__half*)saved_inv_std.data.f16);
	else if (saved_datatype == CCV_16BF)
		_ccv_nnc_inv_std_kernel<<<CUDA_GET_BLOCKS(rcount), CUDA_NUM_THREADS, 0, stream>>>(rcount, epsilon, (__nv_bfloat16*)saved_inv_std.data.f16, (__nv_bfloat16*)saved_inv_std.data.f16);
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b.descriptor, b.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, b.descriptor, b.data.u8));
	if (elementwise_affine)
	{
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b.descriptor, b.data.u8, &one, scale.descriptor, scale.data.u8, &zero, b.descriptor, b.data.u8));
		cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, b.descriptor, b.data.u8, &one, bias.descriptor, bias.data.u8, &zero, b.descriptor, b.data.u8));
	}
	ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce);
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(bias);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_mean);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_layer_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 9 || input_size == 7);
	assert(output_size >= 1);
	if (cmd.info.lnorm.scale != 1)
		return CCV_NNC_EXEC_INVALID;
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[3]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const int elementwise_affine = cmd.info.lnorm.elementwise_affine;
	if (elementwise_affine)
		{ assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[4])); }
	const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, elementwise_affine ? (const ccv_nnc_tensor_view_t*)inputs[4] : 0);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[elementwise_affine ? 7 : 5]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_mean = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[elementwise_affine ? 7 : 5]);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[elementwise_affine ? 8 : 6]));
	const ccv_nnc_cudnn_tensor_view_descriptor_t saved_inv_std = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)inputs[elementwise_affine ? 8 : 6]);
	if (output_size > 1 && outputs[1])
		{ assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[1])); }
	const ccv_nnc_cudnn_tensor_view_descriptor_t dscale = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, output_size > 1 ? (const ccv_nnc_tensor_view_t*)outputs[1] : 0);
	if (output_size > 2 && outputs[2])
		{ assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[2])); }
	const ccv_nnc_cudnn_tensor_view_descriptor_t dbias = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, output_size > 2 ? (const ccv_nnc_tensor_view_t*)outputs[2] : 0);
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[0], gdim);
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[elementwise_affine ? 7 : 5], rdim);
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)inputs[elementwise_affine ? 8 : 6], rdim));
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)inputs[3], gdim));
	assert(ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)outputs[0], gdim));
	static const float one = 1, zero = 0, neg_one = -1;
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
	const int use_bfloat = inputs[0]->info.datatype == CCV_16BF || inputs[3]->info.datatype == CCV_16BF || inputs[elementwise_affine ? 7 : 5]->info.datatype == CCV_16BF || inputs[elementwise_affine ? 8 : 6]->info.datatype == CCV_16BF || outputs[0]->info.datatype == CCV_16BF || (elementwise_affine && inputs[4]->info.datatype == CCV_16BF) || (output_size > 1 && outputs[1] && outputs[1]->info.datatype == CCV_16BF) || (output_size > 2 && outputs[2] && outputs[2]->info.datatype == CCV_16BF);
	size_t scale_workspace_size = 0;
	if (!use_bfloat && dscale.descriptor)
		{ CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, g.descriptor, dscale.descriptor, &scale_workspace_size)); }
	size_t mean_workspace_size = 0;
	if (!use_bfloat)
		{ CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, g.descriptor, saved_mean.descriptor, &mean_workspace_size)); }
	const size_t workspace_size = ccv_max(scale_workspace_size, mean_workspace_size);
	if (use_bfloat)
	{
		ccv_nnc_tensor_param_t g32_info = inputs[0]->info;
		g32_info.datatype = CCV_32F;
		ccv_nnc_tensor_param_t a32_info = inputs[3]->info;
		a32_info.datatype = CCV_32F;
		ccv_nnc_tensor_param_t h32_info = outputs[0]->info;
		h32_info.datatype = CCV_32F;
		ccv_nnc_tensor_param_t saved_mean32_info = inputs[elementwise_affine ? 7 : 5]->info;
		saved_mean32_info.datatype = CCV_32F;
		ccv_nnc_tensor_param_t saved_inv_std32_info = inputs[elementwise_affine ? 8 : 6]->info;
		saved_inv_std32_info.datatype = CCV_32F;
		size_t placeholder = 0;
		ccv_nnc_tensor_t g32t = ccv_nnc_tensor(&placeholder, g32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t g32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&g32t);
		ccv_nnc_tensor_t a32t = ccv_nnc_tensor(&placeholder, a32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t a32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&a32t);
		ccv_nnc_tensor_t h32t = ccv_nnc_tensor(&placeholder, h32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t h32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&h32t);
		ccv_nnc_tensor_t saved_mean32t = ccv_nnc_tensor(&placeholder, saved_mean32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t saved_mean32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&saved_mean32t);
		ccv_nnc_tensor_t saved_inv_std32t = ccv_nnc_tensor(&placeholder, saved_inv_std32_info, 0);
		ccv_nnc_cudnn_tensor_view_descriptor_t saved_inv_std32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&saved_inv_std32t);
		ccv_nnc_cudnn_tensor_view_descriptor_t scale32 = {};
		int scale_count = 0;
		if (elementwise_affine)
		{
			ccv_nnc_tensor_param_t scale32_info = inputs[4]->info;
			scale32_info.datatype = CCV_32F;
			scale_count = ccv_nnc_tensor_count(scale32_info);
			ccv_nnc_tensor_t scale32t = ccv_nnc_tensor(&placeholder, scale32_info, 0);
			scale32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&scale32t);
		}
		ccv_nnc_cudnn_tensor_view_descriptor_t dscale32 = {};
		int dscale_count = 0;
		if (dscale.descriptor)
		{
			ccv_nnc_tensor_param_t dscale32_info = outputs[1]->info;
			dscale32_info.datatype = CCV_32F;
			dscale_count = ccv_nnc_tensor_count(dscale32_info);
			ccv_nnc_tensor_t dscale32t = ccv_nnc_tensor(&placeholder, dscale32_info, 0);
			dscale32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&dscale32t);
		}
		ccv_nnc_cudnn_tensor_view_descriptor_t dbias32 = {};
		int dbias_count = 0;
		if (dbias.descriptor)
		{
			ccv_nnc_tensor_param_t dbias32_info = outputs[2]->info;
			dbias32_info.datatype = CCV_32F;
			dbias_count = ccv_nnc_tensor_count(dbias32_info);
			ccv_nnc_tensor_t dbias32t = ccv_nnc_tensor(&placeholder, dbias32_info, 0);
			dbias32 = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&dbias32t);
		}
		size_t bfloat_scale_workspace_size = 0;
		if (dscale32.descriptor)
			{ CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, g32.descriptor, dscale32.descriptor, &bfloat_scale_workspace_size)); }
		size_t bfloat_bias_workspace_size = 0;
		if (dbias32.descriptor)
			{ CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, g32.descriptor, dbias32.descriptor, &bfloat_bias_workspace_size)); }
		size_t bfloat_mean_workspace_size = 0;
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce, g32.descriptor, saved_mean32.descriptor, &bfloat_mean_workspace_size));
		const size_t bfloat_workspace_size = ccv_max(ccv_max(bfloat_scale_workspace_size, bfloat_bias_workspace_size), bfloat_mean_workspace_size);
		uint8_t* const bfloat_workspace = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, bfloat_workspace_size + sizeof(float) * (gcount * 6 + rcount * 4 + scale_count + dscale_count + dbias_count), CCV_TENSOR_GPU_MEMORY);
		float* const g32p = (float*)(bfloat_workspace + bfloat_workspace_size);
		g32.data.u8 = (uint8_t*)g32p;
		float* const a32p = g32p + gcount;
		a32.data.u8 = (uint8_t*)a32p;
		float* const h32p = a32p + gcount;
		h32.data.u8 = (uint8_t*)h32p;
		float* const saved_mean32p = h32p + gcount;
		saved_mean32.data.u8 = (uint8_t*)saved_mean32p;
		float* const saved_inv_std32p = saved_mean32p + rcount;
		saved_inv_std32.data.u8 = (uint8_t*)saved_inv_std32p;
		float* scale32p = saved_inv_std32p + rcount;
		if (elementwise_affine)
		{
			scale32.data.u8 = (uint8_t*)scale32p;
			scale32p += scale_count;
		}
		float* dscale32p = scale32p;
		if (dscale32.descriptor)
		{
			dscale32.data.u8 = (uint8_t*)dscale32p;
			dscale32p += dscale_count;
		}
		float* dbias32p = dscale32p;
		if (dbias32.descriptor)
		{
			dbias32.data.u8 = (uint8_t*)dbias32p;
			dbias32p += dbias_count;
		}
		float* const ahp = dbias32p;
		const ccv_nnc_tensor_t aht = ccv_nnc_tensor(ahp, g32_info, 0);
		const ccv_nnc_cudnn_tensor_view_descriptor_t ah = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&aht);
		float* const gssp = ahp + gcount;
		const ccv_nnc_tensor_t gsst = ccv_nnc_tensor(gssp, g32_info, 0);
		const ccv_nnc_cudnn_tensor_view_descriptor_t gss = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&gsst);
		float* const ahgssp = gssp + gcount;
		const ccv_nnc_tensor_t ahgsst = ccv_nnc_tensor(ahgssp, g32_info, 0);
		const ccv_nnc_cudnn_tensor_view_descriptor_t ahgss = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&ahgsst);
		float* const gssrp = ahgssp + gcount;
		const ccv_nnc_tensor_t gssrt = ccv_nnc_tensor(gssrp, saved_mean32_info, 0);
		const ccv_nnc_cudnn_tensor_view_descriptor_t gssr = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&gssrt);
		float* const ahgssrp = gssrp + rcount;
		const ccv_nnc_tensor_t ahgssrt = ccv_nnc_tensor(ahgssrp, saved_mean32_info, 0);
		const ccv_nnc_cudnn_tensor_view_descriptor_t ahgssr = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&ahgssrt);
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, g.descriptor, g.data.u8, &zero, g32.descriptor, g32.data.u8));
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, a.descriptor, a.data.u8, &zero, a32.descriptor, a32.data.u8));
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, saved_mean.descriptor, saved_mean.data.u8, &zero, saved_mean32.descriptor, saved_mean32.data.u8));
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, saved_inv_std32.descriptor, saved_inv_std32.data.u8));
		if (elementwise_affine)
			{ CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, scale.descriptor, scale.data.u8, &zero, scale32.descriptor, scale32.data.u8)); }
		if (dbias32.descriptor)
			{ CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, bfloat_workspace, bfloat_workspace_size, &one, g32.descriptor, g32.data.u8, &zero, dbias32.descriptor, dbias32.data.u8)); }
		cudnnOpTensorDescriptor_t op = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
		cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, a32.descriptor, a32.data.u8, &neg_one, saved_mean32.descriptor, saved_mean32.data.u8, &zero, ah.descriptor, ah.data.u8));
		cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, saved_inv_std32.descriptor, saved_inv_std32.data.u8, &zero, ah.descriptor, ah.data.u8));
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, g32.descriptor, g32.data.u8, &zero, ahgss.descriptor, ahgss.data.u8));
		if (dscale32.descriptor)
			{ CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, bfloat_workspace, bfloat_workspace_size, &one, ahgss.descriptor, ahgss.data.u8, &zero, dscale32.descriptor, dscale32.data.u8)); }
		if (elementwise_affine)
		{
			CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, g32.descriptor, g32.data.u8, &one, scale32.descriptor, scale32.data.u8, &zero, gss.descriptor, gss.data.u8));
			CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, gss.descriptor, gss.data.u8, &one, saved_inv_std32.descriptor, saved_inv_std32.data.u8, &zero, gss.descriptor, gss.data.u8));
		} else {
			CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, g32.descriptor, g32.data.u8, &one, saved_inv_std32.descriptor, saved_inv_std32.data.u8, &zero, gss.descriptor, gss.data.u8));
		}
		CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, bfloat_workspace, bfloat_workspace_size, &one, gss.descriptor, gss.data.u8, &zero, gssr.descriptor, gssr.data.u8));
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, gss.descriptor, gss.data.u8, &zero, ahgss.descriptor, ahgss.data.u8));
		CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, bfloat_workspace, bfloat_workspace_size, &one, ahgss.descriptor, ahgss.data.u8, &zero, ahgssr.descriptor, ahgssr.data.u8));
		ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, ahgssr.descriptor, ahgssr.data.u8, &zero, ah.descriptor, ah.data.u8));
		cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, gssr.descriptor, gssr.data.u8, &zero, ah.descriptor, ah.data.u8));
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, gss.descriptor, gss.data.u8, &neg_inv_n, ah.descriptor, ah.data.u8, &zero, h32.descriptor, h32.data.u8));
		CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, h32.descriptor, h32.data.u8, &zero, h.descriptor, h.data.u8));
		if (dscale32.descriptor)
			{ CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, dscale32.descriptor, dscale32.data.u8, &zero, dscale.descriptor, dscale.data.u8)); }
		if (dbias32.descriptor)
			{ CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, dbias32.descriptor, dbias32.data.u8, &zero, dbias.descriptor, dbias.data.u8)); }
		ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_mean);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(dscale);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(dbias);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(g32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(a32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(h32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_mean32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(dscale32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(dbias32);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(ah);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(gss);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(ahgss);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(gssr);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(ahgssr);
		return CCV_NNC_EXEC_SUCCESS;
	}
	uint8_t* const workspace = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, workspace_size + sizeof(float) * gcount * 3 + sizeof(float) * rcount * 2, CCV_TENSOR_GPU_MEMORY);
	if (dbias.descriptor)
		{ CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &one, g.descriptor, g.data.u8, &zero, dbias.descriptor, dbias.data.u8)); }
	float* const ahp = (float*)(workspace + workspace_size);
	const ccv_nnc_tensor_t aht = ccv_nnc_tensor(ahp, inputs[0]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t ah = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&aht);
	float* const gssp = ahp + gcount;
	const ccv_nnc_tensor_t gsst = ccv_nnc_tensor(gssp, inputs[0]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t gss = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&gsst);
	float* const ahgssp = gssp + gcount;
	const ccv_nnc_tensor_t ahgsst = ccv_nnc_tensor(ahgssp, inputs[0]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t ahgss = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&ahgsst);
	float* const gssrp = ahgssp + gcount;
	const ccv_nnc_tensor_t gssrt = ccv_nnc_tensor(gssrp, inputs[elementwise_affine ? 7 : 5]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t gssr = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&gssrt);
	float* const ahgssrp = gssrp + rcount;
	const ccv_nnc_tensor_t ahgssrt = ccv_nnc_tensor(ahgssrp, inputs[elementwise_affine ? 7 : 5]->info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t ahgssr = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&ahgssrt);
	cudnnOpTensorDescriptor_t op = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, a.descriptor, a.data.u8, &neg_one, saved_mean.descriptor, saved_mean.data.u8, &zero, ah.descriptor, ah.data.u8));
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, ah.descriptor, ah.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, g.descriptor, g.data.u8, &zero, ahgss.descriptor, ahgss.data.u8));
	if (dscale.descriptor)
		{ CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &one, ahgss.descriptor, ahgss.data.u8, &zero, dscale.descriptor, dscale.data.u8)); }
	if (elementwise_affine)
	{
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, g.descriptor, g.data.u8, &one, scale.descriptor, scale.data.u8, &zero, gss.descriptor, gss.data.u8));
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, gss.descriptor, gss.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, gss.descriptor, gss.data.u8));
	} else {
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, g.descriptor, g.data.u8, &one, saved_inv_std.descriptor, saved_inv_std.data.u8, &zero, gss.descriptor, gss.data.u8));
	}
	CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &one, gss.descriptor, gss.data.u8, &zero, gssr.descriptor, gssr.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, gss.descriptor, gss.data.u8, &zero, ahgss.descriptor, ahgss.data.u8));
	CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce, 0, 0, workspace, workspace_size, &one, ahgss.descriptor, ahgss.data.u8, &zero, ahgssr.descriptor, ahgssr.data.u8));
	ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, ahgssr.descriptor, ahgssr.data.u8, &zero, ah.descriptor, ah.data.u8));
	cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, ah.descriptor, ah.data.u8, &one, gssr.descriptor, gssr.data.u8, &zero, ah.descriptor, ah.data.u8));
	CUDNN_ENFORCE(cudnnOpTensor(cudnn, op, &one, gss.descriptor, gss.data.u8, &neg_inv_n, ah.descriptor, ah.data.u8, &zero, h.descriptor, h.data.u8));
	ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, op);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(scale);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_mean);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(saved_inv_std);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(dscale);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(dbias);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(ah);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(gss);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(ahgss);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(gssr);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(ahgssr);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_back;
#endif
}
