extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

struct ccv_nnc_rmsnorm_gated_info_t {
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	int ndim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int sstride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
};

static __device__ float _ccv_nnc_rmsnorm_gated_load(const void* const data, const int datatype, const size_t index)
{
	if (datatype == CCV_16F)
		return (float)((const __half*)data)[index];
	if (datatype == CCV_16BF)
		return (float)((const __nv_bfloat16*)data)[index];
	return ((const float*)data)[index];
}

static __device__ int _ccv_nnc_rmsnorm_gated_offset(int index, const int dim[4], const int stride[4])
{
	int offset = 0;
	for (int i = 3; i >= 0; i--)
	{
		offset += (index % dim[i]) * stride[i];
		index /= dim[i];
	}
	return offset;
}

template<typename NUM>
static __global__ void _ccv_nnc_rmsnorm_gated_forw_kernel(const int groups, const int n, const float epsilon, const ccv_nnc_rmsnorm_gated_info_t info, const NUM* const a, const void* const gate, const int gate_datatype, const void* const scale, const int scale_datatype, NUM* const b)
{
	__shared__ float sums[256];
	for (int group = blockIdx.x; group < groups; group += gridDim.x)
	{
		const int a_base = _ccv_nnc_rmsnorm_gated_offset(group, info.rdim, info.astride);
		float sum = 0;
		for (int i = threadIdx.x; i < n; i += blockDim.x)
		{
			const float value = (float)a[a_base + _ccv_nnc_rmsnorm_gated_offset(i, info.ndim, info.astride)];
			sum += value * value;
		}
		sums[threadIdx.x] = sum;
		__syncthreads();
		for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
		{
			if (threadIdx.x < stride)
				sums[threadIdx.x] += sums[threadIdx.x + stride];
			__syncthreads();
		}
		const float inv_std = rsqrtf(sums[0] / n + epsilon);
		const int gate_base = _ccv_nnc_rmsnorm_gated_offset(group, info.rdim, info.gstride);
		const int scale_base = _ccv_nnc_rmsnorm_gated_offset(group, info.rdim, info.sstride);
		const int b_base = _ccv_nnc_rmsnorm_gated_offset(group, info.rdim, info.bstride);
		for (int i = threadIdx.x; i < n; i += blockDim.x)
		{
			const float z = _ccv_nnc_rmsnorm_gated_load(gate, gate_datatype, gate_base + _ccv_nnc_rmsnorm_gated_offset(i, info.ndim, info.gstride));
			const float ez = expf(-fabsf(z));
			const float sigmoid = z >= 0 ? 1.f / (1.f + ez) : ez / (1.f + ez);
			const float gamma = scale ? _ccv_nnc_rmsnorm_gated_load(scale, scale_datatype, scale_base + _ccv_nnc_rmsnorm_gated_offset(i, info.ndim, info.sstride)) : 1;
			const float value = (float)a[a_base + _ccv_nnc_rmsnorm_gated_offset(i, info.ndim, info.astride)];
			b[b_base + _ccv_nnc_rmsnorm_gated_offset(i, info.ndim, info.bstride)] = (NUM)(value * inv_std * gamma * z * sigmoid);
		}
		__syncthreads();
	}
}

// Contiguous tensors normalized over trailing axes: group rows are [group * n, (group + 1) * n). One warp owns a
// row and keeps its values in registers between the reduction and the output pass.
template<typename NUM, int VALUES_PER_LANE>
static __global__ void _ccv_nnc_rmsnorm_gated_contiguous_forw_kernel(const int groups, const int n, const float epsilon, const NUM* const a, const void* const gate, const int gate_datatype, const void* const scale, const int scale_datatype, NUM* const b)
{
	const int warps = blockDim.x / 32;
	const int lane = threadIdx.x % 32;
	for (int group = blockIdx.x * warps + threadIdx.x / 32; group < groups; group += gridDim.x * warps)
	{
		const size_t base = (size_t)group * n;
		float values[VALUES_PER_LANE];
		float sum = 0;
#pragma unroll
		for (int j = 0; j < VALUES_PER_LANE; j++)
		{
			const int i = lane + j * 32;
			values[j] = i < n ? (float)a[base + i] : 0;
			sum += values[j] * values[j];
		}
		for (int offset = 16; offset > 0; offset /= 2)
			sum += __shfl_xor_sync(0xffffffff, sum, offset);
		const float inv_std = rsqrtf(sum / n + epsilon);
#pragma unroll
		for (int j = 0; j < VALUES_PER_LANE; j++)
		{
			const int i = lane + j * 32;
			if (i < n)
			{
				const float z = _ccv_nnc_rmsnorm_gated_load(gate, gate_datatype, base + i);
				const float ez = expf(-fabsf(z));
				const float sigmoid = z >= 0 ? 1.f / (1.f + ez) : ez / (1.f + ez);
				const float gamma = scale ? _ccv_nnc_rmsnorm_gated_load(scale, scale_datatype, i) : 1;
				b[base + i] = (NUM)(values[j] * inv_std * gamma * z * sigmoid);
			}
		}
	}
}

template<typename NUM>
static void _ccv_nnc_rmsnorm_gated_contiguous_forw_launch(cudaStream_t stream, const int groups, const int n, const float epsilon, const NUM* const a, const void* const gate, const int gate_datatype, const void* const scale, const int scale_datatype, NUM* const b)
{
	// 8 warps per block, one row per warp.
	const int blocks = ccv_min((groups + 7) / 8, 4096);
	if (n <= 128)
		_ccv_nnc_rmsnorm_gated_contiguous_forw_kernel<NUM, 4><<<blocks, 256, 0, stream>>>(groups, n, epsilon, a, gate, gate_datatype, scale, scale_datatype, b);
	else if (n <= 256)
		_ccv_nnc_rmsnorm_gated_contiguous_forw_kernel<NUM, 8><<<blocks, 256, 0, stream>>>(groups, n, epsilon, a, gate, gate_datatype, scale, scale_datatype, b);
	else if (n <= 512)
		_ccv_nnc_rmsnorm_gated_contiguous_forw_kernel<NUM, 16><<<blocks, 256, 0, stream>>>(groups, n, epsilon, a, gate, gate_datatype, scale, scale_datatype, b);
	else
		_ccv_nnc_rmsnorm_gated_contiguous_forw_kernel<NUM, 32><<<blocks, 256, 0, stream>>>(groups, n, epsilon, a, gate, gate_datatype, scale, scale_datatype, b);
}

static int _ccv_nnc_rmsnorm_gated_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int affine = cmd.info.rmsnorm_gated.elementwise_affine;
	assert(input_size == (affine ? 3 : 2));
	assert(output_size == 1);
	if (outputs[0]->info.dim[0] == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const gate = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const scale = affine ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int datatype = a->info.datatype;
	if ((datatype != CCV_32F && datatype != CCV_16F && datatype != CCV_16BF) || b->info.datatype != datatype ||
		(gate->info.datatype != CCV_32F && gate->info.datatype != CCV_16F && gate->info.datatype != CCV_16BF) ||
		(scale && scale->info.datatype != CCV_32F && scale->info.datatype != CCV_16F && scale->info.datatype != CCV_16BF))
		return CCV_NNC_EXEC_INVALID;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd >= 1 && a_nd <= 4);
	assert(CCV_NNC_MAX_DIM == 2);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	assert(ccv_nnc_tensor_view_check_dim(gate, adim));
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	ccv_nnc_rmsnorm_gated_info_t info = {};
	ccv_nnc_tensor_view_get_stride(a, info.astride);
	ccv_nnc_tensor_view_get_stride(gate, info.gstride);
	ccv_nnc_tensor_view_get_stride(b, info.bstride);
	for (int i = 0; i < 4; i++)
	{
		info.rdim[i] = adim[i];
		info.ndim[i] = 1;
	}
	for (int i = 0; i < cmd.info.rmsnorm_gated.count; i++)
	{
		const int axis = cmd.info.rmsnorm_gated.axis[i];
		assert(axis >= 0 && axis < a_nd);
		info.rdim[axis + 4 - a_nd] = 1;
		info.ndim[axis + 4 - a_nd] = adim[axis + 4 - a_nd];
	}
	int sdim[CCV_NNC_MAX_DIM_ALLOC];
	if (scale)
	{
		ccv_nnc_tensor_view_get_dim(scale, sdim);
		ccv_nnc_tensor_view_get_stride(scale, info.sstride);
		for (int i = 0; i < 4; i++)
		{
			assert(sdim[i] == 1 || sdim[i] == adim[i]);
			if (sdim[i] == 1)
				info.sstride[i] = 0;
		}
	}
	int groups = 1, n = 1;
	for (int i = 0; i < 4; i++)
	{
		groups *= info.rdim[i];
		n *= info.ndim[i];
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const void* const gamma = scale ? scale->data.u8 : 0;
	const int scale_datatype = scale ? scale->info.datatype : CCV_32F;
	// The contiguous kernel indexes linearly, which needs the normalized axes to be the trailing ones, and the
	// scale to span exactly those axes.
	int contiguous = n <= 1024 && CCV_IS_TENSOR_CONTIGUOUS(a) && CCV_IS_TENSOR_CONTIGUOUS(gate) && CCV_IS_TENSOR_CONTIGUOUS(b) && (!scale || CCV_IS_TENSOR_CONTIGUOUS(scale));
	for (int i = 0; contiguous && i < 4; i++)
	{
		if (scale && sdim[i] != info.ndim[i])
			contiguous = 0;
		for (int j = i + 1; contiguous && j < 4; j++)
			if (info.ndim[i] > 1 && info.rdim[j] > 1)
				contiguous = 0;
	}
	if (contiguous)
	{
		const float epsilon = cmd.info.rmsnorm_gated.epsilon;
		if (datatype == CCV_32F)
			_ccv_nnc_rmsnorm_gated_contiguous_forw_launch(stream, groups, n, epsilon, a->data.f32, gate->data.u8, gate->info.datatype, gamma, scale_datatype, b->data.f32);
		else if (datatype == CCV_16F)
			_ccv_nnc_rmsnorm_gated_contiguous_forw_launch(stream, groups, n, epsilon, (const __half*)a->data.f16, gate->data.u8, gate->info.datatype, gamma, scale_datatype, (__half*)b->data.f16);
		else
			_ccv_nnc_rmsnorm_gated_contiguous_forw_launch(stream, groups, n, epsilon, (const __nv_bfloat16*)a->data.f16, gate->data.u8, gate->info.datatype, gamma, scale_datatype, (__nv_bfloat16*)b->data.f16);
		return CCV_NNC_EXEC_SUCCESS;
	}
#define RMSNORM_GATED_LAUNCH(NUM, a_data, b_data) \
	_ccv_nnc_rmsnorm_gated_forw_kernel<NUM><<<ccv_min(groups, 4096), 256, 0, stream>>>(groups, n, cmd.info.rmsnorm_gated.epsilon, info, a_data, gate->data.u8, gate->info.datatype, gamma, scale_datatype, b_data)
	if (datatype == CCV_32F)
		RMSNORM_GATED_LAUNCH(float, a->data.f32, b->data.f32);
	else if (datatype == CCV_16F)
		RMSNORM_GATED_LAUNCH(__half, (const __half*)a->data.f16, (__half*)b->data.f16);
	else
		RMSNORM_GATED_LAUNCH(__nv_bfloat16, (const __nv_bfloat16*)a->data.f16, (__nv_bfloat16*)b->data.f16);
#undef RMSNORM_GATED_LAUNCH
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_gated_forw;
#endif
}
