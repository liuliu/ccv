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
__global__ void _ccv_nnc_top_1(const size_t count, const int dim, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		NUM k = a[i * dim];
		int v = 0;
		for (int j = 1; j < dim; j++)
			if (a[i * dim + j] < k)
				k = a[i * dim + j], v = j;
		b[i] = k;
		indices[i] = v;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_top_1_descending(const size_t count, const int dim, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		NUM k = a[i * dim];
		int v = 0;
		for (int j = 1; j < dim; j++)
			if (a[i * dim + j] > k)
				k = a[i * dim + j], v = j;
		b[i] = k;
		indices[i] = v;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_top_1_stride(const size_t count, const int dim, const int stride, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(n, count) {
		const int i = n / stride;
		const int k = n % stride;
		NUM key = a[i * dim * stride + k];
		int v = 0;
		for (int j = 1; j < dim; j++)
			if (a[(i * dim + j) * stride + k] < key)
				key = a[(i * dim + j) * stride + k], v = j;
		b[i * stride + k] = key;
		indices[i * stride + k] = v;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_top_1_descending_stride(const size_t count, const int dim, const int stride, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(n, count) {
		const int i = n / stride;
		const int k = n % stride;
		NUM key = a[i * dim * stride + k];
		int v = 0;
		for (int j = 1; j < dim; j++)
			if (a[(i * dim + j) * stride + k] > key)
				key = a[(i * dim + j) * stride + k], v = j;
		b[i * stride + k] = key;
		indices[i * stride + k] = v;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_top_2(const size_t count, const int dim, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		NUM k0 = a[i * dim];
		int v0 = 0;
		NUM k1 = a[i * dim + 1];
		int v1 = 1;
		if (k1 < k0)
		{
			NUM kt = k0;
			k0 = k1;
			k1 = kt;
			int vt = v0;
			v0 = v1;
			v1 = vt;
		}
		for (int j = 2; j < dim; j++)
			if (a[i * dim + j] < k0)
			{
				k1 = k0, v1 = v0;
				k0 = a[i * dim + j], v0 = j;
			} else if (a[i * dim + j] < k1)
				k1 = a[i * dim + j], v1 = j;
		b[i * 2] = k0;
		indices[i * 2] = v0;
		b[i * 2 + 1] = k1;
		indices[i * 2 + 1] = v1;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_top_2_descending(const size_t count, const int dim, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		NUM k0 = a[i * dim];
		int v0 = 0;
		NUM k1 = a[i * dim + 1];
		int v1 = 1;
		if (k1 > k0)
		{
			NUM kt = k0;
			k0 = k1;
			k1 = kt;
			int vt = v0;
			v0 = v1;
			v1 = vt;
		}
		for (int j = 2; j < dim; j++)
			if (a[i * dim + j] > k0)
			{
				k1 = k0, v1 = v0;
				k0 = a[i * dim + j], v0 = j;
			} else if (a[i * dim + j] > k1)
				k1 = a[i * dim + j], v1 = j;
		b[i * 2] = k0;
		indices[i * 2] = v0;
		b[i * 2 + 1] = k1;
		indices[i * 2 + 1] = v1;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_top_2_stride(const size_t count, const int dim, const int stride, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(n, count) {
		const int i = n / stride;
		const int k = n % stride;
		NUM k0 = a[i * dim * stride + k];
		int v0 = 0;
		NUM k1 = a[(i * dim + 1) * stride + k];
		int v1 = 1;
		if (k1 < k0)
		{
			NUM kt = k0;
			k0 = k1;
			k1 = kt;
			int vt = v0;
			v0 = v1;
			v1 = vt;
		}
		for (int j = 2; j < dim; j++)
			if (a[(i * dim + j) * stride + k] < k0)
			{
				k1 = k0, v1 = v0;
				k0 = a[(i * dim + j) * stride + k], v0 = j;
			} else if (a[(i * dim + j) * stride + k] < k1)
				k1 = a[(i * dim + j) * stride + k], v1 = j;
		b[i * 2 * stride + k] = k0;
		indices[i * 2 * stride + k] = v0;
		b[(i * 2 + 1) * stride + k] = k1;
		indices[(i * 2 + 1) * stride + k] = v1;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_top_2_descending_stride(const size_t count, const int dim, const int stride, const NUM* const a, NUM* const b, int* const indices)
{
	CUDA_1D_KERNEL_LOOP(n, count) {
		const int i = n / stride;
		const int k = n % stride;
		NUM k0 = a[i * dim * stride + k];
		int v0 = 0;
		NUM k1 = a[(i * dim + 1) * stride + k];
		int v1 = 1;
		if (k1 > k0)
		{
			NUM kt = k0;
			k0 = k1;
			k1 = kt;
			int vt = v0;
			v0 = v1;
			v1 = vt;
		}
		for (int j = 2; j < dim; j++)
			if (a[(i * dim + j) * stride + k] > k0)
			{
				k1 = k0, v1 = v0;
				k0 = a[(i * dim + j) * stride + k], v0 = j;
			} else if (a[(i * dim + j) * stride + k] > k1)
				k1 = a[(i * dim + j) * stride + k], v1 = j;
		b[i * 2 * stride + k] = k0;
		indices[i * 2 * stride + k] = v0;
		b[(i * 2 + 1) * stride + k] = k1;
		indices[(i * 2 + 1) * stride + k] = v1;
	}
}

static int _ccv_nnc_partition_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 2);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(b->info.dim) == a_nd);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == a_nd);
	assert(indices->info.datatype == CCV_32S);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(a->info.datatype == b->info.datatype);
	const int count = ccv_nnc_tensor_count(a->info);
	const int kth = cmd.info.partition.kth;
	assert(kth <= 2); // We can only do top 2 or top 1.
	// Currently, this is only optimized for small dimensions.
	int i, j, k, f;
	int sort_runs = 1;
	int sort_stride = 1;
	for (i = 0; i < a_nd; i++)
	{
		if (i < cmd.info.partition.along_axis) // Skip this.
			sort_runs *= a->info.dim[i];
		else if (i > cmd.info.partition.along_axis)
			sort_stride *= a->info.dim[i];
	}
	const int dim = a->info.dim[cmd.info.partition.along_axis];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (sort_stride == 1) // If this is the last axis, it is simpler to call.
	{
		if (kth == 1)
		{
			if (a->info.datatype == CCV_32F)
			{
				if (cmd.info.partition.descending)
					_ccv_nnc_top_1_descending<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.f32, b->data.f32, indices->data.i32);
				else
					_ccv_nnc_top_1<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.f32, b->data.f32, indices->data.i32);
			} else if (a->info.datatype == CCV_16F) {
				if (cmd.info.partition.descending)
					_ccv_nnc_top_1_descending<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
				else
					_ccv_nnc_top_1<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
			} else {
				assert(a->info.datatype == CCV_32S);
				if (cmd.info.partition.descending)
					_ccv_nnc_top_1_descending<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.i32, b->data.i32, indices->data.i32);
				else
					_ccv_nnc_top_1<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.i32, b->data.i32, indices->data.i32);
			}
		} else {
			assert(kth == 2);
			if (a->info.datatype == CCV_32F)
			{
				if (cmd.info.partition.descending)
					_ccv_nnc_top_2_descending<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.f32, b->data.f32, indices->data.i32);
				else
					_ccv_nnc_top_2<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.f32, b->data.f32, indices->data.i32);
			} else if (a->info.datatype == CCV_16F) {
				if (cmd.info.partition.descending)
					_ccv_nnc_top_2_descending<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
				else
					_ccv_nnc_top_2<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
			} else {
				assert(a->info.datatype == CCV_32S);
				if (cmd.info.partition.descending)
					_ccv_nnc_top_2_descending<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.i32, b->data.i32, indices->data.i32);
				else
					_ccv_nnc_top_2<<<CUDA_GET_BLOCKS(sort_runs), CUDA_NUM_THREADS, 0, stream>>>(sort_runs, dim, a->data.i32, b->data.i32, indices->data.i32);
			}
		}
	} else {
		if (kth == 1)
		{
			if (a->info.datatype == CCV_32F)
			{
				if (cmd.info.partition.descending)
					_ccv_nnc_top_1_descending_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.f32, b->data.f32, indices->data.i32);
				else
					_ccv_nnc_top_1_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.f32, b->data.f32, indices->data.i32);
			} else if (a->info.datatype == CCV_16F) {
				if (cmd.info.partition.descending)
					_ccv_nnc_top_1_descending_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
				else
					_ccv_nnc_top_1_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
			} else {
				assert(a->info.datatype == CCV_32S);
				if (cmd.info.partition.descending)
					_ccv_nnc_top_1_descending_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.i32, b->data.i32, indices->data.i32);
				else
					_ccv_nnc_top_1_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.i32, b->data.i32, indices->data.i32);
			}
		} else {
			assert(kth == 2);
			if (a->info.datatype == CCV_32F)
			{
				if (cmd.info.partition.descending)
					_ccv_nnc_top_2_descending_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.f32, b->data.f32, indices->data.i32);
				else
					_ccv_nnc_top_2_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.f32, b->data.f32, indices->data.i32);
			} else if (a->info.datatype == CCV_16F) {
				if (cmd.info.partition.descending)
					_ccv_nnc_top_2_descending_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
				else
					_ccv_nnc_top_2_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, (__half*)a->data.f16, (__half*)b->data.f16, indices->data.i32);
			} else {
				assert(a->info.datatype == CCV_32S);
				if (cmd.info.partition.descending)
					_ccv_nnc_top_2_descending_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.i32, b->data.i32, indices->data.i32);
				else
					_ccv_nnc_top_2_stride<<<CUDA_GET_BLOCKS(sort_runs * sort_stride), CUDA_NUM_THREADS, 0, stream>>>(sort_runs * sort_stride, dim, sort_stride, a->data.i32, b->data.i32, indices->data.i32);
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_partition_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_PARTITION_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_16F | CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_partition_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PARTITION_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_16F | CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_partition_back;
#endif
}
