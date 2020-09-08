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
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>

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

template<int threadsPerBlock>
__global__ void _ccv_nnc_iou_mask_kernel(const int gm, const int m, const float iou_threshold, const float* const b, uint64_t* const iou_mask)
{
	// Compute only upper-left triangle.
	int row_start = blockIdx.x / (gm + 1);
	int col_start = blockIdx.x % (gm + 1);
	if (col_start > row_start)
	{
		col_start = col_start - row_start - 1;
		row_start = gm - 1 - row_start;
	}
	const int row_size = min(m - row_start * threadsPerBlock, threadsPerBlock);
	const int col_size = min(m - col_start * threadsPerBlock, threadsPerBlock);
	__shared__ float boxes[threadsPerBlock * 4];
	if (threadIdx.x < col_size)
	{
		boxes[threadIdx.x * 4] = b[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
		boxes[threadIdx.x * 4 + 1] = b[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
		boxes[threadIdx.x * 4 + 2] = b[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
		boxes[threadIdx.x * 4 + 3] = b[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
	}
	__syncthreads();
	if (threadIdx.x < row_size)
	{
		const int row_idx = threadsPerBlock * row_start + threadIdx.x;
		const float* const bp = b + row_idx * 5;
		int i;
		int end = (row_start == col_start) ? threadIdx.x : col_size;
		uint64_t t = 0;
		const float area1 = bp[3] * bp[4];
		for (i = 0; i < end; i++)
		{
			const float area2 = boxes[i * 4 + 2] * boxes[i * 4 + 3];
			const float xdiff = ccv_max(0, ccv_min(bp[1] + bp[3], boxes[i * 4] + boxes[i * 4 + 2]) - ccv_max(bp[1], boxes[i * 4]));
			const float ydiff = ccv_max(0, ccv_min(bp[2] + bp[4], boxes[i * 4 + 1] + boxes[i * 4 + 3]) - ccv_max(bp[2], boxes[i * 4 + 1]));
			const float intersection = xdiff * ydiff;
			const float iou = intersection / (area1 + area2 - intersection);
			if (iou >= iou_threshold)
				t |= (1ULL << i);
		}
		iou_mask[row_idx * gm + col_start] = t;
	}
}

__global__ void _ccv_nnc_nms_zero_flags(const int n, int* const flags)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		flags[i] = 0;
	}
}

template<int threadsPerBlock>
__global__ void _ccv_nnc_iou_postproc_kernel(const int gm, const int m, const uint64_t* const iou_mask, int* const flags, float* const b, int* const c)
{
	const int row_idx = threadsPerBlock * blockIdx.x + threadIdx.x;
	int i;
	int suppressed = (row_idx >= m);
	for (i = 0; i < blockIdx.x; i++) // Compute whether we depends on these, for each of them.
	{
		const uint64_t ious = row_idx < m ? iou_mask[row_idx * gm + i] : 0;
		if (threadIdx.x == 0) // Wait flags to turn to 1.
			while (cub::ThreadLoad<cub::LOAD_CG>(flags + i) == 0)
				__threadfence_block();
		__syncthreads(); // Now it is available. Sync all threads to this point.
		if (suppressed)
			continue;
		int j;
		const int col_size = min(m - i * threadsPerBlock, threadsPerBlock);
		for (j = 0; j < col_size; j++)
			if (ious & (1ULL << j)) // And it overlaps. Mark this one as not good.
				if (c[i * threadsPerBlock + j] != -1) // If this is not marked as unavailable.
					c[row_idx] = -1, suppressed = 1;
	}
	__shared__ int bc[threadsPerBlock];
	bc[threadIdx.x] = row_idx < m ? c[row_idx] : 0;
	// Now, go over it internally.
	const uint64_t ious = row_idx < m ? iou_mask[row_idx * gm + blockIdx.x] : 0;
	for (i = 0; i < threadsPerBlock; i++)
	{
		__syncthreads(); // Need to sync on every round.
		if (i >= threadIdx.x)
			continue;
		if (ious & (1ULL << i)) // And it overlaps. Mark this one as not good.
			if (bc[i] != -1) // If this is not marked as unavailable.
				bc[threadIdx.x] = -1;
	}
	// Write back.
	if (row_idx < m)
		c[row_idx] = bc[threadIdx.x];
	// Done mine. Mark it visible for other blocks. Store the flag.
	__syncthreads();
	if (threadIdx.x == 0)
		cub::ThreadStore<cub::STORE_CG>(flags + blockIdx.x, 1);
	// If I am the last one, I am responsible for removing suppressed values.
	if (blockIdx.x == gm - 1 && threadIdx.x == 0)
	{
		int j;
		for (i = 0, j = 0; i < m; i++)
			if (c[i] != -1)
			{
				int k;
				if (i != j)
				{
					for (k = 0; k < 5; k++)
						b[j * 5 + k] = b[i * 5 + k];
					c[j] = c[i];
				}
				++j;
			}
		for (i = j; i < m; i++)
			c[i] = -1, b[i * 5] = -FLT_MAX;
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
	const int gm = (m + 63) / 64;
	// Use full parallelism to compute whether it overlaps or not (iou >= iou_threshold).
	size_t iou_bytes = ((sizeof(uint64_t) * (m * gm) + 511) / 512) * 512;
	size_t flag_bytes = sizeof(int) * gm;
	size_t total_bytes = ccv_max(iou_bytes + flag_bytes, aligned_temp_storage_bytes + sizeof(float) * m);
	uint8_t* const d_temp_storage = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, total_bytes, CCV_TENSOR_GPU_MEMORY);
	float* const rank = (float*)(d_temp_storage + aligned_temp_storage_bytes);
	uint64_t* const d_ious = (uint64_t*)d_temp_storage;
	int* const d_flags = (int*)((uint8_t*)d_ious + iou_bytes);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	for (i = 0; i < n; i++)
	{
		const float* const ap = a->data.f32 + i * aninc;
		float* const bp = b->data.f32 + i * bninc;
		int* const cp = c->data.i32 + i * cninc;
		// Scatter to ranks, so we can sort by these floating-points.
		_ccv_nnc_scatter_rank_kernel<<<CUDA_GET_BLOCKS(m), CUDA_NUM_THREADS, 0, stream>>>(m, ap, bp, rank);
		// Sorting.
		cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, rank, rank, (float5*)bp, (float5*)bp, m, 0, sizeof(float) * 8, stream);
		// Merging back into respective arrays.
		_ccv_nnc_merge_rank_kernel<<<CUDA_GET_BLOCKS(m), CUDA_NUM_THREADS, 0, stream>>>(m, bp, rank, cp);
		// Compute whether it overlaps or not with the other. There is no dependencies between them.
		const int block_size = (gm + 1) * gm / 2;
		_ccv_nnc_iou_mask_kernel<64><<<block_size, 64, 0, stream>>>(gm, m, iou_threshold, bp, d_ious);
		_ccv_nnc_nms_zero_flags<<<CUDA_GET_BLOCKS(gm), CUDA_NUM_THREADS, 0, stream>>>(gm, d_flags);
		// Remove overlap items. There are dependencies, because we only remove items that overlap with existing items.
		_ccv_nnc_iou_postproc_kernel<64><<<gm, 64, 0, stream>>>(gm, m, d_ious, d_flags, bp, cp);
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
