extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#if defined(HAVE_CUDA) && defined(HAVE_CUB)

#include <cub/util_type.cuh>
#include <cub/device/device_radix_sort.cuh>

struct float5 {
	float v[5];
};

__global__ void _ccv_nnc_scatter_rank_kernel(const int n, const float* const a, float* const b, float* const rank)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		rank[i] = a[i * 5];
		((int *)b)[i * 5] = i;
		b[i * 5 + 1] = a[i * 5 + 1];
		b[i * 5 + 2] = a[i * 5 + 2];
		b[i * 5 + 3] = a[i * 5 + 3];
		b[i * 5 + 4] = a[i * 5 + 4];
	}
}

__global__ void _ccv_nnc_merge_rank_kernel(const int n, float* const b, float* const rank, int* const c)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		c[i] = ((int*)b)[i * 5];
		b[i * 5] = rank[i];
	}
}

static int _ccv_nnc_nms_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 2);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[1];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(a_nd == b_nd);
	int i;
	for (i = 0; i < a_nd; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int* cinc = CCV_IS_TENSOR_VIEW(c) ? c->inc : c->info.dim;
	const int n = a_nd >= 3 ? a->info.dim[0] : 1;
	const int aninc = a_nd >= 3 ? ainc[1] * ainc[2] : 0;
	const int bninc = b_nd >= 3 ? binc[1] * binc[2] : 0;
	const int cninc = c_nd >= 2 ? cinc[1] : 0;
	const int m = a_nd >= 3 ? a->info.dim[1] : a->info.dim[0];
	if (c_nd == 1)
		{ assert(m == c->info.dim[0]); }
	else
		{ assert(c_nd == 2 && n == c->info.dim[0] && m == c->info.dim[1]); }
	const float iou_threshold = cmd.info.nms.iou_threshold;
	assert((a_nd <= 1 ? 1 : a->info.dim[a_nd - 1]) == 5 && ainc[a_nd - 1] == 5 && ainc[a_nd - 1] == binc[b_nd - 1]);
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairsDescending(0, temp_storage_bytes, a->data.f32, b->data.f32, (float5*)b->data.f32, (float5*)b->data.i32, m, 0, sizeof(float) * 8, 0);
	size_t aligned_temp_storage_bytes = ((temp_storage_bytes + 511) / 512) * 512;
	uint8_t* const d_temp_storage = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, aligned_temp_storage_bytes + sizeof(float) * m, CCV_TENSOR_GPU_MEMORY);
	float* const rank = (float*)(d_temp_storage + aligned_temp_storage_bytes);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	for (i = 0; i < n; i++)
	{
		const float* const ap = a->data.f32 + i * aninc;
		float* const bp = b->data.f32 + i * bninc;
		int* const cp = c->data.i32 + i * cninc;
		_ccv_nnc_scatter_rank_kernel<<<CUDA_GET_BLOCKS(m), CUDA_NUM_THREADS, 0, stream>>>(m, ap, bp, rank);
		cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, rank, rank, (float5*)bp, (float5*)bp, m, 0, sizeof(float) * 8, stream);
		_ccv_nnc_merge_rank_kernel<<<CUDA_GET_BLOCKS(m), CUDA_NUM_THREADS, 0, stream>>>(m, bp, rank, cp);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

__global__ void _ccv_nnc_nms_zero_kernel(const int n, float* const b)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		b[i * 5] = 0;
		b[i * 5 + 1] = 0;
		b[i * 5 + 2] = 0;
		b[i * 5 + 3] = 0;
		b[i * 5 + 4] = 0;
	}
}

__global__ void _ccv_nnc_nms_back_kernel(const int n, const float* const a, const int* const idx, float* const b)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		const int j = idx[i];
		if (j >= 0)
		{
			b[j * 5] = a[i * 5];
			b[j * 5 + 1] = a[i * 5 + 1];
			b[j * 5 + 2] = a[i * 5 + 2];
			b[j * 5 + 3] = a[i * 5 + 3];
			b[j * 5 + 4] = a[i * 5 + 4];
		}
	}
}

static int _ccv_nnc_nms_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 5);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)inputs[4];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(a_nd == b_nd);
	int i;
	for (i = 0; i < a_nd; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int* cinc = CCV_IS_TENSOR_VIEW(c) ? c->inc : c->info.dim;
	const int n = a_nd >= 3 ? a->info.dim[0] : 1;
	const int aninc = a_nd >= 3 ? ainc[1] * ainc[2] : 0;
	const int bninc = b_nd >= 3 ? binc[1] * binc[2] : 0;
	const int cninc = c_nd >= 2 ? cinc[1] : 0;
	const int m = a_nd >= 3 ? a->info.dim[1] : a->info.dim[0];
	if (c_nd == 1)
		{ assert(m == c->info.dim[0]); }
	else
		{ assert(c_nd == 2 && n == c->info.dim[0] && m == c->info.dim[1]); }
	assert((a_nd <= 1 ? 1 : a->info.dim[a_nd - 1]) == 5 && ainc[a_nd - 1] == 5 && ainc[a_nd - 1] == binc[b_nd - 1]);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	for (i = 0; i < n; i++)
	{
		const float* const ap = a->data.f32 + i * aninc;
		float* const bp = b->data.f32 + i * bninc;
		int* const cp = c->data.i32 + i * cninc;
		_ccv_nnc_nms_zero_kernel<<<CUDA_GET_BLOCKS(m), CUDA_NUM_THREADS, 0, stream>>>(m, bp);
		_ccv_nnc_nms_back_kernel<<<CUDA_GET_BLOCKS(m), CUDA_NUM_THREADS, 0, stream>>>(m, ap, cp, bp);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_NMS_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#if defined(HAVE_CUDA) && defined(HAVE_CUB)
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_nms_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_NMS_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#if defined(HAVE_CUDA) && defined(HAVE_CUB)
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_nms_back;
#endif
}
