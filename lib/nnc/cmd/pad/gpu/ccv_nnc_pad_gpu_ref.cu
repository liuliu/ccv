extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

template<typename NUM>
__global__ void _ccv_nnc_pad_zero_forw_1d(const NUM* const ap, const int begin0, const int adim0, NUM* const bp, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim0) {
		const int x = i;
		if (x - begin0 >= 0 && x - begin0 < adim0)
			bp[i] = ap[x - begin0];
		else
			bp[i] = 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_zero_forw_2d(const NUM* const ap, const int begin1, const int begin0, const int adim1, const int adim0, NUM* const bp, const int bdim10, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim10) {
		const int x = i % bdim0;
		const int y = i / bdim0;
		if (x - begin0 >= 0 && x - begin0 < adim0 && y - begin1 >= 0 && y - begin1 < adim1)
			bp[i] = ap[(y - begin1) * adim0 + x - begin0];
		else
			bp[i] = 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_zero_forw_3d(const NUM* const ap, const int begin2, const int begin1, const int begin0, const int adim2, const int adim1, const int adim0, NUM* const bp, const int bdim210, const int bdim1, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim210) {
		const int x = i % bdim0;
		int y = i / bdim0;
		const int z = y / bdim1;
		y = y % bdim1;
		if (x - begin0 >= 0 && x - begin0 < adim0 && y - begin1 >= 0 && y - begin1 < adim1 && z - begin2 >= 0 && z - begin2 < adim2)
			bp[i] = ap[((z - begin2) * adim1 + (y - begin1)) * adim0 + x - begin0];
		else
			bp[i] = 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_zero_forw_4d(const NUM* const ap, const int begin3, const int begin2, const int begin1, const int begin0, const int adim3, const int adim2, const int adim1, const int adim0, NUM* const bp, const int bdim3210, const int bdim2, const int bdim1, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim3210) {
		const int x = i % bdim0;
		int y = i / bdim0;
		int z = y / bdim1;
		y = y % bdim1;
		const int u = z / bdim2;
		z = z % bdim2;
		if (x - begin0 >= 0 && x - begin0 < adim0 && y - begin1 >= 0 && y - begin1 < adim1 && z - begin2 >= 0 && z - begin2 < adim2 && u - begin3 >= 0 && u - begin3 < adim3)
			bp[i] = ap[(((u - begin3) * adim2 + (z - begin2)) * adim1 + (y - begin1)) * adim0 + x - begin0];
		else
			bp[i] = 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_replicate_forw_1d(const NUM* const ap, const int begin0, const int adim0, NUM* const bp, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim0) {
		const int x = i;
		const int ax = min(max(x - begin0, 0), adim0 - 1);
		bp[i] = ap[ax];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_replicate_forw_2d(const NUM* const ap, const int begin1, const int begin0, const int adim1, const int adim0, NUM* const bp, const int bdim10, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim10) {
		const int x = i % bdim0;
		const int y = i / bdim0;
		const int ax = min(max(x - begin0, 0), adim0 - 1);
		const int ay = min(max(y - begin1, 0), adim1 - 1);
		bp[i] = ap[ay * adim0 + ax];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_replicate_forw_3d(const NUM* const ap, const int begin2, const int begin1, const int begin0, const int adim2, const int adim1, const int adim0, NUM* const bp, const int bdim210, const int bdim1, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim210) {
		const int x = i % bdim0;
		int y = i / bdim0;
		const int z = y / bdim1;
		y = y % bdim1;
		const int ax = min(max(x - begin0, 0), adim0 - 1);
		const int ay = min(max(y - begin1, 0), adim1 - 1);
		const int az = min(max(z - begin2, 0), adim2 - 1);
		bp[i] = ap[(az * adim1 + ay) * adim0 + ax];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_replicate_forw_4d(const NUM* const ap, const int begin3, const int begin2, const int begin1, const int begin0, const int adim3, const int adim2, const int adim1, const int adim0, NUM* const bp, const int bdim3210, const int bdim2, const int bdim1, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim3210) {
		const int x = i % bdim0;
		int y = i / bdim0;
		int z = y / bdim1;
		y = y % bdim1;
		const int u = z / bdim2;
		z = z % bdim2;
		const int ax = min(max(x - begin0, 0), adim0 - 1);
		const int ay = min(max(y - begin1, 0), adim1 - 1);
		const int az = min(max(z - begin2, 0), adim2 - 1);
		const int au = min(max(u - begin3, 0), adim3 - 1);
		bp[i] = ap[((au * adim2 + az) * adim1 + ay) * adim0 + ax];
	}
}

static int _ccv_nnc_pad_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(a_nd == b_nd);
	assert(a->info.datatype == b->info.datatype);
	assert(a->info.format == b->info.format);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int* const begin = cmd.info.size.dim;
	if (cmd.info.pad.type == CCV_NNC_PAD_ZERO)
	{
		if (a_nd == 1)
		{
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_zero_forw_1d<<<CUDA_GET_BLOCKS(b->info.dim[0]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], a->info.dim[0], b->data.f32, b->info.dim[0]);
			else
				_ccv_nnc_pad_zero_forw_1d<<<CUDA_GET_BLOCKS(b->info.dim[0]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], a->info.dim[0], (__half*)b->data.f16, b->info.dim[0]);
		} else if (a_nd == 2) {
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_zero_forw_2d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], a->info.dim[0], a->info.dim[1], b->data.f32, b->info.dim[1] * b->info.dim[0], b->info.dim[1]);
			else
				_ccv_nnc_pad_zero_forw_2d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], a->info.dim[0], a->info.dim[1], (__half*)b->data.f16, b->info.dim[1] * b->info.dim[0], b->info.dim[1]);
		} else if (a_nd == 3) {
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_zero_forw_3d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], begin[2], a->info.dim[0], a->info.dim[1], a->info.dim[2], b->data.f32, b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2]);
			else
				_ccv_nnc_pad_zero_forw_3d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], begin[2], a->info.dim[0], a->info.dim[1], a->info.dim[2], (__half*)b->data.f16, b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2]);
		} else if (a_nd == 4) {
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_zero_forw_4d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2] * b->info.dim[3]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], begin[2], begin[3], a->info.dim[0], a->info.dim[1], a->info.dim[2], a->info.dim[3], b->data.f32, b->info.dim[3] * b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2], b->info.dim[3]);
			else
				_ccv_nnc_pad_zero_forw_4d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2] * b->info.dim[3]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], begin[2], begin[3], a->info.dim[0], a->info.dim[1], a->info.dim[2], a->info.dim[3], (__half*)b->data.f16, b->info.dim[3] * b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2], b->info.dim[3]);
		} else {
			assert(0);
		}
	} else {
		assert(cmd.info.pad.type == CCV_NNC_PAD_REPLICATE);
		if (a_nd == 1)
		{
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_replicate_forw_1d<<<CUDA_GET_BLOCKS(b->info.dim[0]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], a->info.dim[0], b->data.f32, b->info.dim[0]);
			else
				_ccv_nnc_pad_replicate_forw_1d<<<CUDA_GET_BLOCKS(b->info.dim[0]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], a->info.dim[0], (__half*)b->data.f16, b->info.dim[0]);
		} else if (a_nd == 2) {
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_replicate_forw_2d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], a->info.dim[0], a->info.dim[1], b->data.f32, b->info.dim[1] * b->info.dim[0], b->info.dim[1]);
			else
				_ccv_nnc_pad_replicate_forw_2d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], a->info.dim[0], a->info.dim[1], (__half*)b->data.f16, b->info.dim[1] * b->info.dim[0], b->info.dim[1]);
		} else if (a_nd == 3) {
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_replicate_forw_3d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], begin[2], a->info.dim[0], a->info.dim[1], a->info.dim[2], b->data.f32, b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2]);
			else
				_ccv_nnc_pad_replicate_forw_3d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], begin[2], a->info.dim[0], a->info.dim[1], a->info.dim[2], (__half*)b->data.f16, b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2]);
		} else if (a_nd == 4) {
			if (a->info.datatype == CCV_32F)
				_ccv_nnc_pad_replicate_forw_4d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2] * b->info.dim[3]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], begin[2], begin[3], a->info.dim[0], a->info.dim[1], a->info.dim[2], a->info.dim[3], b->data.f32, b->info.dim[3] * b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2], b->info.dim[3]);
			else
				_ccv_nnc_pad_replicate_forw_4d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2] * b->info.dim[3]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], begin[2], begin[3], a->info.dim[0], a->info.dim[1], a->info.dim[2], a->info.dim[3], (__half*)b->data.f16, b->info.dim[3] * b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2], b->info.dim[3]);
		} else {
			assert(0);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_pad_back_1d(const NUM* const ap, const int begin0, NUM* const bp, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim0) {
		const int x = i;
		bp[i] = ap[x + begin0];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_back_2d(const NUM* const ap, const int begin1, const int begin0, const int adim0, NUM* const bp, const int bdim10, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim10) {
		const int x = i % bdim0;
		const int y = i / bdim0;
		bp[i] = ap[(y + begin1) * adim0 + x + begin0];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_back_3d(const NUM* const ap, const int begin2, const int begin1, const int begin0, const int adim1, const int adim0, NUM* const bp, const int bdim210, const int bdim1, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim210) {
		const int x = i % bdim0;
		int y = i / bdim0;
		const int z = y / bdim1;
		y = y % bdim1;
		bp[i] = ap[((z + begin2) * adim1 + (y + begin1)) * adim0 + x + begin0];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_pad_back_4d(const NUM* const ap, const int begin3, const int begin2, const int begin1, const int begin0, const int adim2, const int adim1, const int adim0, NUM* const bp, const int bdim3210, const int bdim2, const int bdim1, const int bdim0)
{
	CUDA_1D_KERNEL_LOOP(i, bdim3210) {
		const int x = i % bdim0;
		int y = i / bdim0;
		int z = y / bdim1;
		y = y % bdim1;
		const int u = z / bdim2;
		z = z % bdim2;
		bp[i] = ap[(((u + begin3) * adim2 + (z + begin2)) * adim1 + (y + begin1)) * adim0 + x + begin0];
	}
}

static int _ccv_nnc_pad_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(a_nd == b_nd);
	assert(a->info.datatype == b->info.datatype);
	assert(a->info.format == b->info.format);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int* const begin = cmd.info.size.dim;
	int i;
	for (i = 0; i < a_nd; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i] + begin[i] + cmd.info.pad.end[i]);
		// We don't support negative pad.
		assert(begin[i] >= 0);
		assert(cmd.info.pad.end[i] >= 0);
	}
	if (a_nd == 1)
	{
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_pad_back_1d<<<CUDA_GET_BLOCKS(b->info.dim[0]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], b->data.f32, b->info.dim[0]);
		else
			_ccv_nnc_pad_back_1d<<<CUDA_GET_BLOCKS(b->info.dim[0]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], (__half*)b->data.f16, b->info.dim[0]);
	} else if (a_nd == 2) {
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_pad_back_2d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], a->info.dim[1], b->data.f32, b->info.dim[1] * b->info.dim[0], b->info.dim[1]);
		else
			_ccv_nnc_pad_back_2d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], a->info.dim[1], (__half*)b->data.f16, b->info.dim[1] * b->info.dim[0], b->info.dim[1]);
	} else if (a_nd == 3) {
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_pad_back_3d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], begin[2], a->info.dim[1], a->info.dim[2], b->data.f32, b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2]);
		else
			_ccv_nnc_pad_back_3d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], begin[2], a->info.dim[1], a->info.dim[2], (__half*)b->data.f16, b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2]);
	} else if (a_nd == 4) {
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_pad_back_4d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2] * b->info.dim[3]), CUDA_NUM_THREADS, 0, stream>>>(a->data.f32, begin[0], begin[1], begin[2], begin[3], a->info.dim[1], a->info.dim[2], a->info.dim[3], b->data.f32, b->info.dim[3] * b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2], b->info.dim[3]);
		else
			_ccv_nnc_pad_back_4d<<<CUDA_GET_BLOCKS(b->info.dim[0] * b->info.dim[1] * b->info.dim[2] * b->info.dim[3]), CUDA_NUM_THREADS, 0, stream>>>((__half*)a->data.f16, begin[0], begin[1], begin[2], begin[3], a->info.dim[1], a->info.dim[2], a->info.dim[3], (__half*)b->data.f16, b->info.dim[3] * b->info.dim[2] * b->info.dim[1] * b->info.dim[0], b->info.dim[1], b->info.dim[2], b->info.dim[3]);
	} else {
		assert(0);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_pad_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_pad_back;
#endif
}
