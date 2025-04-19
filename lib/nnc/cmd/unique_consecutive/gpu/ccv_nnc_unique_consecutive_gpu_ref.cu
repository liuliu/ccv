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
#include <cub/device/device_run_length_encode.cuh>

template<typename NUM>
__global__ void _ccv_nnc_fill_zeros(const int n, const int* d_num_runs_out, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		if (i >= d_num_runs_out[0])
			b[i] = -1, indices[i] = 0;
	}
}

static int _ccv_nnc_unique_consecutive_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 2);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(b->info.dim) == a_nd);
	ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == a_nd);
	assert(indices->info.datatype == CCV_32S);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(a->info.datatype == b->info.datatype);
	const int count = ccv_nnc_tensor_count(a->info);
	assert(a_nd == 1); // Can only handle 1d tensor for this.
	const int bincount = b->info.dim[0];
	assert(bincount > 0);
	assert(bincount == indices->info.dim[0]);
	size_t temp_storage_bytes = 0;
	if (a->info.datatype == CCV_32F)
		cub::DeviceRunLengthEncode::Encode(0, temp_storage_bytes, a->data.f32, b->data.f32, indices->data.i32, indices->data.i32, count, 0);
	else
		cub::DeviceRunLengthEncode::Encode(0, temp_storage_bytes, a->data.i32, b->data.i32, indices->data.i32, indices->data.i32, count, 0);
	const size_t aligned_temp_storage_bytes = ((temp_storage_bytes + 511) / 512) * 512;
	// Use full parallelism to compute whether it overlaps or not (iou >= iou_threshold).
	const size_t total_bytes = aligned_temp_storage_bytes + sizeof(int);
	uint8_t* const d_temp_storage = (uint8_t*)ccv_nnc_stream_context_get_workspace(stream_context, total_bytes, CCV_TENSOR_GPU_MEMORY);
	int* d_num_runs_out = (int*)(d_temp_storage + aligned_temp_storage_bytes);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (a->info.datatype == CCV_32F)
	{
		cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, a->data.f32, b->data.f32, indices->data.i32, d_num_runs_out, count, stream);
		// Note that this potentially could overflow if the bincount is smaller than d_num_runs_out.
		_ccv_nnc_fill_zeros<<<CUDA_GET_BLOCKS(bincount), CUDA_NUM_THREADS, 0, stream>>>(bincount, d_num_runs_out, b->data.f32, indices->data.i32);
	} else {
		cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, a->data.i32, b->data.i32, indices->data.i32, d_num_runs_out, count, stream);
		// Note that this potentially could overflow if the bincount is smaller than d_num_runs_out.
		_ccv_nnc_fill_zeros<<<CUDA_GET_BLOCKS(bincount), CUDA_NUM_THREADS, 0, stream>>>(bincount, d_num_runs_out, b->data.i32, indices->data.i32);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_unique_consecutive_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_unique_consecutive_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_unique_consecutive_back;
#endif
}
