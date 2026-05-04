#pragma once

#include <limits.h>

#if defined(HAVE_CUDNN) && defined(HAVE_CUDA_SM80)

#include <nnc/gpu/3rdparty/flash_attn/layer_norm/ln_api.h>

typedef struct {
	int rows;
	int cols;
	int input_datatype;
	int weight_datatype;
	layer_norm::DataType input_type;
	layer_norm::DataType weight_type;
} ccv_nnc_flash_norm_info_t;

typedef struct {
	int valid;
	int device;
	int rows;
	int cols;
	int input_datatype;
	int weight_datatype;
	size_t elts_per_thread;
	size_t workspace_bytes;
	size_t barrier_size;
	int ctas_per_col;
} ccv_nnc_flash_norm_launch_config_t;

static inline unsigned char* _ccv_nnc_flash_norm_data(const ccv_nnc_tensor_t* const tensor)
{
	return tensor->data.u8 + tensor->dataof;
}

static int _ccv_nnc_flash_norm_datatype(const int datatype, layer_norm::DataType* const type)
{
	switch (datatype)
	{
		case CCV_16F:
			*type = layer_norm::DATA_TYPE_FP16;
			return 1;
		case CCV_16BF:
			*type = layer_norm::DATA_TYPE_BF16;
			return 1;
		case CCV_32F:
			*type = layer_norm::DATA_TYPE_FP32;
			return 1;
		default:
			return 0;
	}
}

static int _ccv_nnc_flash_norm_tail_axes(const int* const axis, const int axis_count, const int nd)
{
	if (axis_count <= 0 || axis_count > nd)
		return 0;
	int seen[CCV_NNC_MAX_DIM_ALLOC] = {};
	int i;
	for (i = 0; i < axis_count; i++)
	{
		if (axis[i] < 0 || axis[i] >= nd || seen[axis[i]])
			return 0;
		seen[axis[i]] = 1;
	}
	for (i = 0; i < nd - axis_count; i++)
		if (seen[i])
			return 0;
	for (i = nd - axis_count; i < nd; i++)
		if (!seen[i])
			return 0;
	return 1;
}

static int _ccv_nnc_flash_norm_check(const int* const axis, const int axis_count, ccv_nnc_tensor_t* const input, ccv_nnc_tensor_t* const output, ccv_nnc_tensor_t* const saved_a, ccv_nnc_tensor_t* const saved_b, ccv_nnc_tensor_t* const scale, ccv_nnc_tensor_t* const bias, const int elementwise_affine, ccv_nnc_flash_norm_info_t* const info)
{
	if (!CCV_IS_TENSOR_CONTIGUOUS(input) || !CCV_IS_TENSOR_CONTIGUOUS(output) || !CCV_IS_TENSOR_CONTIGUOUS(saved_a))
		return 0;
	if (saved_b && !CCV_IS_TENSOR_CONTIGUOUS(saved_b))
		return 0;
	if (input->info.datatype != output->info.datatype)
		return 0;
	if (!_ccv_nnc_flash_norm_datatype(input->info.datatype, &info->input_type))
		return 0;
	if (saved_a->info.datatype != CCV_32F && saved_a->info.datatype != CCV_16F && saved_a->info.datatype != CCV_16BF)
		return 0;
	if (saved_b && saved_b->info.datatype != CCV_32F && saved_b->info.datatype != CCV_16F && saved_b->info.datatype != CCV_16BF)
		return 0;
	const int nd = ccv_nnc_tensor_nd(input->info.dim);
	if (nd <= 0 || nd > CCV_NNC_MAX_DIM + 2)
		return 0;
	if (!_ccv_nnc_flash_norm_tail_axes(axis, axis_count, nd))
		return 0;
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)input, adim);
	if (!ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)output, adim))
		return 0;
	size_t rows = 1;
	size_t cols = 1;
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC; i++)
		rdim[i] = adim[i];
	const int dim_offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (i = 0; i < nd - axis_count; i++)
		rows *= adim[dim_offset + i];
	for (i = nd - axis_count; i < nd; i++)
	{
		cols *= adim[dim_offset + i];
		rdim[dim_offset + i] = 1;
	}
	if (rows == 0 || rows > INT_MAX || cols == 0 || cols > INT_MAX)
		return 0;
	if (cols % 8 != 0 || layer_norm::round_hidden_size((uint32_t)cols) > 8192)
		return 0;
	if (!ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)saved_a, rdim))
		return 0;
	if (saved_b && !ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)saved_b, rdim))
		return 0;
	info->rows = (int)rows;
	info->cols = (int)cols;
	info->input_datatype = input->info.datatype;
	info->weight_datatype = input->info.datatype;
	info->weight_type = info->input_type;
	if (elementwise_affine)
	{
		if (!scale || !CCV_IS_TENSOR_CONTIGUOUS(scale) || ccv_nnc_tensor_count(scale->info) != cols)
			return 0;
		if (bias && (!CCV_IS_TENSOR_CONTIGUOUS(bias) || ccv_nnc_tensor_count(bias->info) != cols || bias->info.datatype != scale->info.datatype))
			return 0;
		if (scale->info.datatype != input->info.datatype && scale->info.datatype != CCV_32F)
			return 0;
		if (!_ccv_nnc_flash_norm_datatype(scale->info.datatype, &info->weight_type))
			return 0;
		info->weight_datatype = scale->info.datatype;
	}
	return 1;
}

static int _ccv_nnc_flash_norm_configure(layer_norm::LaunchParams<layer_norm::FwdParams>& launch_params, const ccv_nnc_flash_norm_info_t info)
{
	int device = 0;
	CUDA_ENFORCE(cudaGetDevice(&device));
	static __thread ccv_nnc_flash_norm_launch_config_t config_cache[16];
	int i;
	for (i = 0; i < 16; i++)
		if (config_cache[i].valid &&
			config_cache[i].device == device &&
			config_cache[i].rows == info.rows &&
			config_cache[i].cols == info.cols &&
			config_cache[i].input_datatype == info.input_datatype &&
			config_cache[i].weight_datatype == info.weight_datatype)
		{
			launch_params.elts_per_thread = config_cache[i].elts_per_thread;
			launch_params.workspace_bytes = config_cache[i].workspace_bytes;
			launch_params.barrier_size = config_cache[i].barrier_size;
			launch_params.params.ctas_per_col = config_cache[i].ctas_per_col;
			return 1;
		}
	cudaDeviceProp props = {};
	CUDA_ENFORCE(cudaGetDeviceProperties(&props, device));
	launch_params.props = &props;
	if (!layer_norm::run_layer_norm_fwd(launch_params, info.weight_type, info.input_type, info.input_type, info.input_type, info.cols, true))
		return 0;
	const size_t slot = ((size_t)device * 13 + (size_t)info.rows * 7 + (size_t)info.cols * 5 + (size_t)info.input_datatype * 3 + (size_t)info.weight_datatype) & 15;
	config_cache[slot].valid = 1;
	config_cache[slot].device = device;
	config_cache[slot].rows = info.rows;
	config_cache[slot].cols = info.cols;
	config_cache[slot].input_datatype = info.input_datatype;
	config_cache[slot].weight_datatype = info.weight_datatype;
	config_cache[slot].elts_per_thread = launch_params.elts_per_thread;
	config_cache[slot].workspace_bytes = launch_params.workspace_bytes;
	config_cache[slot].barrier_size = launch_params.barrier_size;
	config_cache[slot].ctas_per_col = launch_params.params.ctas_per_col;
	return 1;
}

static void _ccv_nnc_flash_norm_transform_stats(ccv_nnc_stream_context_t* const stream_context, const float* const src, ccv_nnc_tensor_t* const dst)
{
	static const float one = 1, zero = 0;
	ccv_nnc_tensor_param_t src_info = dst->info;
	src_info.datatype = CCV_32F;
	ccv_nnc_tensor_t src_tensor = ccv_nnc_tensor((void*)src, src_info, 0);
	const ccv_nnc_cudnn_tensor_view_descriptor_t src_desc = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&src_tensor);
	const ccv_nnc_cudnn_tensor_view_descriptor_t dst_desc = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)dst);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	CUDNN_ENFORCE(cudnnTransformTensor(cudnn, &one, src_desc.descriptor, src_desc.data.u8, &zero, dst_desc.descriptor, dst_desc.data.u8));
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(src_desc);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(dst_desc);
}

static int _ccv_nnc_flash_norm_forw(ccv_nnc_stream_context_t* const stream_context, ccv_nnc_tensor_t* const input, ccv_nnc_tensor_t* const scale, ccv_nnc_tensor_t* const bias, ccv_nnc_tensor_t* const output, ccv_nnc_tensor_t* const saved_mean, ccv_nnc_tensor_t* const saved_inv_std, const ccv_nnc_flash_norm_info_t info, const float epsilon, const int is_rms_norm)
{
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	layer_norm::LaunchParams<layer_norm::FwdParams> launch_params = {};
	launch_params.stream = stream;
	size_t stats_workspace_count = 0;
	const int mean_direct = !is_rms_norm && saved_mean && saved_mean->info.datatype == CCV_32F;
	const int inv_std_direct = saved_inv_std->info.datatype == CCV_32F;
	stats_workspace_count += mean_direct ? 0 : info.rows;
	stats_workspace_count += inv_std_direct ? 0 : info.rows;
	if (is_rms_norm)
		stats_workspace_count += info.rows;
	const size_t stats_workspace_size = sizeof(float) * stats_workspace_count;
	launch_params.params.rows = info.rows;
	launch_params.params.cols = info.cols;
	launch_params.params.x0 = _ccv_nnc_flash_norm_data(input);
	launch_params.params.x1 = 0;
	launch_params.params.residual = 0;
	launch_params.params.x = 0;
	launch_params.params.dmask = 0;
	launch_params.params.dmask1 = 0;
	launch_params.params.mu = 0;
	launch_params.params.rs = 0;
	launch_params.params.gamma = scale ? _ccv_nnc_flash_norm_data(scale) : ccv_nnc_stream_context_get_ones(stream_context, info.cols, info.input_datatype);
	launch_params.params.gamma1 = 0;
	launch_params.params.rowscale = 0;
	launch_params.params.colscale = 0;
	launch_params.params.x0_subset = 0;
	launch_params.params.z_subset = 0;
	launch_params.params.z = _ccv_nnc_flash_norm_data(output);
	launch_params.params.z1 = 0;
	launch_params.params.beta = bias ? _ccv_nnc_flash_norm_data(bias) : 0;
	launch_params.params.beta1 = 0;
	launch_params.params.epsilon = epsilon;
	launch_params.params.dropout_keep_p = 1.f;
	launch_params.params.dropout_scale = 1.f;
	launch_params.params.inverse_cols = 1.f / (float)info.cols;
	launch_params.params.rowscale_const = 1.f;
	launch_params.params.is_rms_norm = is_rms_norm;
	launch_params.params.workspace = 0;
	launch_params.params.barrier = 0;
	if (!_ccv_nnc_flash_norm_configure(launch_params, info))
		return 0;
	const size_t barrier_size = launch_params.barrier_size * sizeof(int);
	const size_t total_workspace_size = stats_workspace_size + launch_params.workspace_bytes + barrier_size;
	uint8_t* workspace = total_workspace_size > 0 ? (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, total_workspace_size, CCV_TENSOR_GPU_MEMORY) : 0;
	float* stats_workspace = (float*)workspace;
	float* mean = 0;
	if (is_rms_norm)
	{
		mean = stats_workspace;
		stats_workspace += info.rows;
	} else if (mean_direct) {
		mean = (float*)_ccv_nnc_flash_norm_data(saved_mean);
	} else {
		mean = stats_workspace;
		stats_workspace += info.rows;
	}
	float* inv_std = 0;
	if (inv_std_direct)
		inv_std = (float*)_ccv_nnc_flash_norm_data(saved_inv_std);
	else {
		inv_std = stats_workspace;
		stats_workspace += info.rows;
	}
	launch_params.params.mu = mean;
	launch_params.params.rs = inv_std;
	launch_params.params.workspace = launch_params.workspace_bytes > 0 ? workspace + stats_workspace_size : 0;
	launch_params.params.barrier = barrier_size > 0 ? (int*)(workspace + stats_workspace_size + launch_params.workspace_bytes) : 0;
	if (barrier_size > 0)
		CUDA_ENFORCE(cudaMemsetAsync(launch_params.params.barrier, 0, barrier_size, stream));
	if (!layer_norm::run_layer_norm_fwd(launch_params, info.weight_type, info.input_type, info.input_type, info.input_type, info.cols, false))
		return 0;
	CUDA_ENFORCE(cudaGetLastError());
	if (!is_rms_norm && !mean_direct)
		_ccv_nnc_flash_norm_transform_stats(stream_context, mean, saved_mean);
	if (!inv_std_direct)
		_ccv_nnc_flash_norm_transform_stats(stream_context, inv_std, saved_inv_std);
	return 1;
}

#endif
