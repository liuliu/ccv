extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

#include <cub/util_type.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>

__global__ void _ccv_nnc_arange(const int n, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		indices[i] = i;
	}
}

static int _ccv_nnc_argsort_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == a_nd);
	assert(indices->info.datatype == CCV_32S);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	const int count = ccv_nnc_tensor_count(a->info);
	assert(a_nd == 1);
	size_t temp_storage_bytes = 0;
	if (cmd.info.argsort.descending)
	{
		if (a->info.datatype == CCV_32F)
			cub::DeviceRadixSort::SortPairsDescending(0, temp_storage_bytes, a->data.f32, a->data.f32, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, 0);
		else
			cub::DeviceRadixSort::SortPairsDescending(0, temp_storage_bytes, a->data.i32, a->data.i32, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, 0);
		const size_t aligned_temp_storage_bytes = ((temp_storage_bytes + 511) / 512) * 512;
		// Use full parallelism to compute whether it overlaps or not (iou >= iou_threshold).
		const size_t total_bytes = aligned_temp_storage_bytes + ((a->info.datatype == CCV_32F) ? sizeof(float) : sizeof(int)) * count;
		uint8_t* const d_temp_storage = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, total_bytes, CCV_TENSOR_GPU_MEMORY);
		void* workmem = d_temp_storage + aligned_temp_storage_bytes;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		// Fill in indices.
		_ccv_nnc_arange<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, indices->data.i32);
		// Sorting.
		if (a->info.datatype == CCV_32F)
			cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, a->data.f32, (float*)workmem, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, stream);
		else
			cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, a->data.i32, (int*)workmem, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, stream);
	} else {
		if (a->info.datatype == CCV_32F)
			cub::DeviceRadixSort::SortPairs(0, temp_storage_bytes, a->data.f32, a->data.f32, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, 0);
		else
			cub::DeviceRadixSort::SortPairs(0, temp_storage_bytes, a->data.i32, a->data.i32, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, 0);
		const size_t aligned_temp_storage_bytes = ((temp_storage_bytes + 511) / 512) * 512;
		// Use full parallelism to compute whether it overlaps or not (iou >= iou_threshold).
		const size_t total_bytes = aligned_temp_storage_bytes + ((a->info.datatype == CCV_32F) ? sizeof(float) : sizeof(int)) * count;
		uint8_t* const d_temp_storage = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, total_bytes, CCV_TENSOR_GPU_MEMORY);
		void* workmem = d_temp_storage + aligned_temp_storage_bytes;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		// Fill in indices.
		_ccv_nnc_arange<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, indices->data.i32);
		// Sorting.
		if (a->info.datatype == CCV_32F)
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, a->data.f32, (float*)workmem, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, stream);
		else
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, a->data.i32, (int*)workmem, indices->data.i32, indices->data.i32, count, 0, sizeof(float) * 8, stream);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_argsort_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_ARGSORT_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_argsort_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ARGSORT_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_argsort_back;
#endif
}
