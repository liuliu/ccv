#include <cuda.h>
#include <cublas_v2.h>
extern "C" {
#include "../cwc.h"
#include "../cwc_internal.h"
}
#include "../../inl/ccv_convnet_inl.h"

template <int input_per_thread, int filter_per_thread, int filter_per_block>
__global__ static void _cwc_kern_convolutional_forward_propagate(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels_per_partition, const int partition,
		float* out, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count,
		float* const biases)
{
	assert(gridDim.x * partition * filter_per_block == out_cols * count);
	assert(gridDim.y == out_rows);
	assert(gridDim.z == partition);
	extern __shared__ float shared[];
	float* shared_block = &shared[0];
	float* shared_weights = &shared[batch];
	float* shared_bias = &shared[batch + filter_per_block];
	float prod[filter_per_thread][input_per_thread];
	assert(batch == input_per_thread * blockDim.x);
	assert(filter_per_block == filter_per_thread * blockDim.y);
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	int c, i, j, x, y;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			prod[i][j] = 0;
	const int origin_x = blockIdx.x % out_cols;
	const int origin_y = blockIdx.y;
	const int filter_group_idx = blockIdx.z * count / (filter_per_block * partition) + blockIdx.x / out_cols; // for the partitioned filter group
	input += (blockIdx.z * channels_per_partition * rows * cols +  origin_y * strides * cols + origin_x * strides) * batch;
	assert(thcnt >= batch);
	assert(thcnt >= filter_per_block);
	if (thidx < filter_per_block)
		shared_bias[thidx] = biases[filter_group_idx * filter_per_block + thidx];
	const int start_x = max(origin_x * strides - border, 0) - (origin_x * strides - border);
	const int end_x = min(origin_x * strides - border + filter_cols, cols) - (origin_x * strides - border);
	const int start_y = max(origin_y * strides - border, 0) - (origin_y * strides - border);
	const int end_y = min(origin_y * strides - border + filter_rows, rows) - (origin_y * strides - border);
	filter += filter_group_idx * filter_per_block;
	for (c = 0; c < channels_per_partition; c++)
	{
		for (y = start_y; y < end_y; y++)
			for (x = start_x; x < end_x; x++)
			{
				if (thidx < batch)
					shared_block[thidx] = input[((y - border) * cols + x - border) * batch + thidx];
				if (thidx < filter_per_block)
					shared_weights[thidx] = filter[(y * filter_cols + x) * count + thidx];
				__syncthreads();
				#pragma unroll
				for (i = 0; i < filter_per_thread; i++)
					#pragma unroll
					for (j = 0; j < input_per_thread; j++)
						prod[i][j] += shared_block[j + threadIdx.x * input_per_thread] * shared_weights[i + threadIdx.y * filter_per_thread];
				__syncthreads();
			}
		input += rows * cols * batch;
		filter += filter_rows * filter_cols * count;
	}
	const int outcnt = out_rows * out_cols * batch;
	out += (filter_group_idx * filter_per_block + threadIdx.y * filter_per_thread) * outcnt + (origin_y * out_cols + origin_x) * batch + threadIdx.x * input_per_thread;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
	{
		const float bias = shared_bias[i + threadIdx.y * filter_per_thread];
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			out[j] = max(0.0, prod[i][j] + bias);
		out += outcnt;
	}
}

static int _cwc_convnet_convolutional_forward_propagate_vary(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, const cudaStream_t& stream,
		int x, int y, int z) // these are the dynamic configurations
{
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
	// first do configuration validation
	if (!(batch % x == 0 && z % y == 0 && layer->net.convolutional.count % (z * out_partition) == 0 &&
				batch / x * z / y <= 1024 && /* thread number constraint */
				batch / x * z / y >= batch && batch / x * z / y >= z && /* kernel internal loading constraint */
				sizeof(float) * (batch + z * 2) <= 48 * 1024 /* shared memory size constraint */))
		return -1;
	assert(b);
#define vary_block(_x, _y, _z) do { \
		dim3 threads_per_block(batch / _x, _z / _y); \
		assert(threads_per_block.x * threads_per_block.y <= 1024); \
		dim3 num_blocks(out_cols * layer->net.convolutional.count / (_z * out_partition), out_rows, out_partition); \
		int shared_memory_size = sizeof(float) * (batch + _z * 2); \
		cudaFuncSetCacheConfig(_cwc_kern_convolutional_forward_propagate<_x, _y, _z>, cudaFuncCachePreferShared); \
		_cwc_kern_convolutional_forward_propagate \
			<_x, _y, _z> \
			<<<num_blocks, threads_per_block, shared_memory_size, stream>>> \
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
			 a, rows, cols, layer->input.matrix.channels / out_partition, out_partition, \
			 b, out_rows, out_cols, \
			 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count, \
			 layer->bias); \
	} while (0)
	cwc_vary_4_a(x, 1, 2, 4, 8, cwc_vary_5_b, y, 1, 2, 4, 6, 8, cwc_vary_6_c, z, 16, 24, 32, 36, 64, 72, vary_block);
#undef vary_block
	cudaError_t error = cudaGetLastError();
	if (cudaErrorInvalidConfiguration == error)
		return -1;
	assert(error == cudaSuccess);
	return 0;
}

void cwc_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, const cudaStream_t& stream)
{
	static int vary_x[] = { 1, 2, 4, 8 };
	static int vary_y[] = { 1, 2, 4, 6, 8 };
	static int vary_z[] = { 16, 24, 32, 36, 64, 72 };
	CWC_IMPLEMENT_VARY_STUB(EXTRA(layer)->vary.convolutional.forward, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_forward_propagate_vary, layer, rows, cols, batch, a, b, stream);
}

template <int channel_per_thread, int filter_per_thread, int channel_per_block, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coefficient_default(const int strides, const int border, const int batch, const int batch_group_count,
		float* input, const int rows, const int cols, const int channels_per_partition, const int partition,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count_per_partition)
{
	assert(gridDim.x == filter_cols);
	assert(gridDim.y == filter_rows);
	assert(gridDim.z * channel_per_block * batch_per_block == channels_per_partition * partition * batch);
	assert(batch == batch_per_block * batch_group_count);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[channel_per_block];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(blockDim.x * filter_per_thread == count_per_partition);
	assert(blockDim.y * channel_per_thread == channel_per_block);
	assert(thcnt >= channel_per_block);
	assert(thcnt >= count_per_partition);
	const int origin_x = blockIdx.x;
	const int origin_y = blockIdx.y;
	const int channel_group_count = channels_per_partition / channel_per_block;
	const int partition_idx = blockIdx.z / (channel_group_count * batch_group_count);
	const int batch_group_idx = (blockIdx.z % (channel_group_count * batch_group_count)) / channel_group_count;
	const int channel_group_idx = blockIdx.z % channel_group_count;
	const int start_x = max(origin_x - border, 0) - (origin_x - border);
	const int end_x = min(out_cols, (cols + border - origin_x + strides - 1) / strides);
	const int start_y = max(origin_y - border, 0) - (origin_y - border);
	const int end_y = min(out_rows, (rows + border - origin_y + strides - 1) / strides);
	input += (partition_idx * batch + batch_group_idx * batch_per_block) * rows * cols * channels_per_partition + (origin_y * cols + origin_x) * channels_per_partition + channel_group_idx * channel_per_block;
	out_grad += (partition_idx * batch + batch_group_idx * batch_per_block) * out_rows * out_cols * count_per_partition;
	int i, j, c, x, y;
	float prod[channel_per_thread][filter_per_thread];
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			prod[i][j] = 0;
	for (c = 0; c < batch_per_block; c++)
	{
		for (y = start_y; y < end_y; y++)
			for (x = start_x; x < end_x; x++)
			{
				if (thidx < count_per_partition)
					shared_out_grad[thidx] = out_grad[(y * out_cols + x) * count_per_partition + thidx];
				if (thidx < channel_per_block)
					shared_input[thidx] = input[((y * strides - border) * cols + x * strides - border) * channels_per_partition + thidx];
				__syncthreads();
				#pragma unroll
				for (i = 0; i < channel_per_thread; i++)
					#pragma unroll
					for (j = 0; j < filter_per_thread; j++)
						prod[i][j] += shared_input[i + threadIdx.y * channel_per_thread] * shared_out_grad[j + threadIdx.x * filter_per_thread];
				__syncthreads();
			}
		input += rows * cols * channels_per_partition;
		out_grad += out_rows * out_cols * count_per_partition;
	}
	const int cocnt = filter_cols * filter_rows * count_per_partition * partition;
	coeff += cocnt * (channels_per_partition * batch_group_idx + channel_group_idx * channel_per_block) + (origin_y * filter_cols + origin_x) * count_per_partition * partition + partition_idx * count_per_partition;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			coeff[(i + threadIdx.y * channel_per_thread) * cocnt + j + threadIdx.x * filter_per_thread] = prod[i][j];
}

template <int channel_per_thread, int filter_per_thread, int static_filter_rows, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coefficient_rows(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count)
{
	assert(gridDim.x == filter_cols);
	assert(gridDim.y == out_rows);
	assert(static_filter_rows == filter_rows);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[filter_rows * channels * batch_per_block];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(blockDim.x * filter_per_thread == count);
	assert(blockDim.y * channel_per_thread == channels);
	assert(thcnt >= channels * batch_per_block);
	assert(thcnt >= count);
	const int origin_x = blockIdx.x;
	const int batch_group_idx = blockIdx.z;
	const int start_x = max(origin_x - border, 0) - (origin_x - border);
	const int end_x = min(out_cols, (cols + border - origin_x + strides - 1) / strides);
	input += (rows * cols * channels * batch_group_idx + origin_x * channels) * batch_per_block;
	out_grad += out_rows * out_cols * count * batch_group_idx * batch_per_block;
	int i, j, k, c, x;
	const int y = blockIdx.y;
	float prod[static_filter_rows][channel_per_thread][filter_per_thread];
	#pragma unroll
	for (i = 0; i < static_filter_rows; i++)
		#pragma unroll
		for (j = 0; j < channel_per_thread; j++)
			#pragma unroll
			for (k = 0; k < filter_per_thread; k++)
				prod[i][j][k] = 0;
	const int iy = y * strides - border;
	input += y * strides * cols * channels * batch_per_block;
	out_grad += y * out_cols * count * batch_per_block;
	for (x = start_x; x < end_x; x++)
	{
		if (thidx < channels * batch_per_block)
			#pragma unroll
			for (i = 0; i < static_filter_rows; i++)
				shared_input[i * channels * batch_per_block + thidx] = (i + iy >= 0 && i + iy < rows) ? input[((i - border) * cols + x * strides - border) * channels * batch_per_block + thidx] : 0;
		if (thidx < count)
			#pragma unroll
			for (c = 0; c < batch_per_block; c++)
				shared_out_grad[c * count + thidx] = out_grad[x * count * batch_per_block + c * count + thidx];
		__syncthreads();
		#pragma unroll
		for (i = 0; i < static_filter_rows; i++)
			#pragma unroll
			for (j = 0; j < channel_per_thread; j++)
				#pragma unroll
				for (k = 0; k < filter_per_thread; k++)
				{
					float sum = 0;
					#pragma unroll
					for (c = 0; c < batch_per_block; c++)
						sum += shared_input[i * channels * batch_per_block + c * channels + j + threadIdx.y * channel_per_thread] * shared_out_grad[c * count + k + threadIdx.x * filter_per_thread];
					prod[i][j][k] += sum;
				}
		__syncthreads();
	}
	const int cocnt = filter_cols * filter_rows * count;
	coeff += cocnt * channels * (blockIdx.y + blockIdx.z * out_rows) + origin_x * count;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < static_filter_rows; j++)
			#pragma unroll
			for (k = 0; k < filter_per_thread; k++)
				coeff[(i + threadIdx.y * channel_per_thread) * cocnt + j * filter_cols * count + k + threadIdx.x * filter_per_thread] = prod[j][i][k];
}

template <int input_per_thread, int channel_per_thread, int channel_per_block, int strides>
__global__ static void _cwc_kern_convolutional_backward_propagate_error(const int border, const int batch,
		float* input_grad, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count_per_partition, const int partition)
{
	assert(gridDim.z == partition);
	extern __shared__ float shared[];
	float* shared_grad = &shared[0];
	float* shared_weights = &shared[batch];
	float prod[input_per_thread][channel_per_thread];
	assert(batch == input_per_thread * blockDim.x);
	assert(channel_per_block == channel_per_thread * blockDim.y);
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(thcnt >= batch);
	assert(thcnt >= channel_per_block);
	const int origin_x = blockIdx.x % cols;
	const int origin_y = blockIdx.y;
	const int channel_group_idx = blockIdx.z * channels / (channel_per_block * partition) + blockIdx.x / cols;
	int i, j, k, c, x, y;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		#pragma unroll
		for (j = 0; j < channel_per_thread; j++)
			prod[i][j] = 0;
	const int ycnt = (filter_rows - 1 - (origin_x + border) % strides) / strides + 1;
	const int xcnt = (filter_cols - 1 - (origin_y + border) % strides) / strides + 1;
	const int filter_y = (ycnt - 1) * strides + (origin_x + border) % strides;
	assert(filter_y < filter_rows);
	const int filter_x = (xcnt - 1) * strides + (origin_y + border) % strides;
	assert(filter_x < filter_cols);
	const int out_y = (origin_x + border) / strides - ycnt + 1;
	const int out_x = (origin_y + border) / strides - xcnt + 1;
	const int out_start_y = max(out_y, 0);
	const int out_start_x = max(out_x, 0);
	const int filter_start_y = filter_y - (out_start_y - out_y) * strides;
	const int filter_start_x = filter_x - (out_start_x - out_x) * strides;
	out_grad += (blockIdx.z * count_per_partition * out_rows * out_cols + out_start_y * out_cols + out_start_x) * batch;
	const int out_end_y = out_y + ycnt - 1;
	const int out_end_x = out_x + xcnt - 1;
	const int filter_end_y = (origin_x + border) % strides + (out_end_y - min(out_end_y, out_rows - 1)) * strides;
	const int filter_end_x = (origin_y + border) % strides + (out_end_x - min(out_end_x, out_cols - 1)) * strides;
	const int outcnt = out_rows * out_cols * batch;
	filter += channel_group_idx * channel_per_block;
	for (k = 0; k < count_per_partition; k++)
	{
		float* out_grad_per_filter = out_grad + k * outcnt;
		for (y = filter_start_y; y >= filter_end_y; y -= strides)
		{
			for (x = filter_start_x, c = 0; x >= filter_end_x; x -= strides, c++)
			{
				if (thidx < batch)
					shared_grad[thidx] = out_grad_per_filter[c * batch + thidx];
				if (thidx < channel_per_block)
					shared_weights[thidx] = filter[(y * filter_cols + x) * channels + thidx];
				__syncthreads();
				#pragma unroll
				for (i = 0; i < input_per_thread; i++)
					#pragma unroll
					for (j = 0; j < channel_per_thread; j++)
						prod[i][j] += shared_grad[i + threadIdx.x * input_per_thread] * shared_weights[j + threadIdx.y * channel_per_thread];
				__syncthreads();
			}
			out_grad_per_filter += out_cols * batch;
		}
		filter += filter_rows * filter_cols * channels;
	}
	const int incnt = rows * cols * batch;
	input_grad += channel_group_idx * channel_per_block * incnt + (origin_x * cols + origin_y) * batch;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			input_grad[(i + threadIdx.y * channel_per_thread) * incnt + j + threadIdx.x * input_per_thread] = prod[j][i];
}

// this method rewinds a matrix
template <int reorder_per_block>
__global__ static void _cwc_kern_reorder_matrix_major(float* a, float* b, const int count, const int channels_per_partition, const int partition, const int batch)
{
	assert(blockDim.x == reorder_per_block);
	const int batch_group_idx = blockIdx.y % (batch / reorder_per_block);
	const int channel_group_idx = blockIdx.y / (batch / reorder_per_block);
	a += (blockIdx.z * count * channels_per_partition + blockIdx.x + channel_group_idx * reorder_per_block * count) * batch + batch_group_idx * reorder_per_block;
	b += (blockIdx.z * count * batch + batch_group_idx * reorder_per_block * count + blockIdx.x) * channels_per_partition + channel_group_idx * reorder_per_block;
	__shared__ float prod[reorder_per_block][reorder_per_block];
	int i;
	#pragma unroll
	for (i = 0; i < reorder_per_block; i++)
		prod[i][threadIdx.x] = a[i * count * batch + threadIdx.x];
	__syncthreads();
	#pragma unroll
	for (i = 0; i < reorder_per_block; i++)
		b[i * count * channels_per_partition + threadIdx.x] = prod[threadIdx.x][i];
}

// this method rewinds a matrix
__global__ static void _cwc_kern_reorder_matrix_major_parted(float* a, float* b, const int count, const int channels, const int batch, const int channels_per_partition, const int batch_per_partition, const int partition)
{
	b[(threadIdx.x * count + blockIdx.x) * channels + blockIdx.y + threadIdx.y * channels_per_partition] = a[(blockIdx.y * count + blockIdx.x) * batch + threadIdx.x + threadIdx.y * batch_per_partition];
}

// this method rewinds a matrix
template <int batch_per_block>
__global__ static void _cwc_kern_reorder_matrix_major_per_block_rows(float* a, float* b, const int count, const int channels, const int batch)
{
	const int thidx = blockIdx.y * batch_per_block + threadIdx.y;
	b[(blockIdx.y * count + blockIdx.x) * channels * batch_per_block + threadIdx.y * channels + threadIdx.x] = a[(threadIdx.x * count + blockIdx.x) * batch + thidx];
}

// this method rewinds a matrix
template <int channel_per_block, int batch_per_block, int batch_group_per_block>
__global__ static void _cwc_kern_reorder_matrix_major_per_block(float* a, float* b, const int count, const int channels, const int batch)
{
	const int batch_group_idx = blockIdx.y % (batch / (batch_per_block * batch_group_per_block));
	const int channel_group_idx = blockIdx.y / (batch / (batch_per_block * batch_group_per_block));
	a += (channel_group_idx * channel_per_block * count + blockIdx.x) * batch + batch_group_idx * batch_per_block * batch_group_per_block;
	b += (batch_group_idx * batch_group_per_block * count + blockIdx.x) * channels * batch_per_block + channel_group_idx * channel_per_block;
	__shared__ float prod[channel_per_block][batch_per_block * batch_group_per_block];
	int i, j;
	#pragma unroll
	for (i = 0; i < channel_per_block; i++)
		prod[i][threadIdx.x] = a[i * count * batch + threadIdx.x];
	__syncthreads();
	if (threadIdx.x < channel_per_block)
		#pragma unroll
		for (i = 0; i < batch_group_per_block; i++)
			#pragma unroll
			for (j = 0; j < batch_per_block; j++)
				b[(i * count * batch_per_block + j) * channels + threadIdx.x] = prod[threadIdx.x][i * batch_per_block + j];
}

static void _cwc_convnet_reorder_matrix_major_per_block(float* a, float* b, const int count, const int channels, const int batch, const cudaStream_t& stream)
{
	// this is by experience, ideally, this can be profile-guided too
	const int batch_group_count = batch / BATCH_PER_BLOCK;
	if (channels < 8)
	{
		assert(batch % BATCH_PER_BLOCK == 0);
		assert(channels * BATCH_PER_BLOCK <= 1024);
		_cwc_kern_reorder_matrix_major_per_block_rows
		<BATCH_PER_BLOCK>
		<<<dim3(count, batch_group_count), dim3(channels, BATCH_PER_BLOCK), 0, stream>>>
		(a, b, count, channels, batch);
	} else {
		assert(channels % THREAD_PER_BLOCK == 0);
		assert(THREAD_PER_BLOCK % BATCH_PER_BLOCK == 0);
		assert(batch % THREAD_PER_BLOCK == 0);
		_cwc_kern_reorder_matrix_major_per_block
		<THREAD_PER_BLOCK, BATCH_PER_BLOCK, THREAD_PER_BLOCK / BATCH_PER_BLOCK>
		<<<dim3(count, (channels / THREAD_PER_BLOCK) * (batch / THREAD_PER_BLOCK)), THREAD_PER_BLOCK, sizeof(float) * THREAD_PER_BLOCK * THREAD_PER_BLOCK, stream>>>
		(a, b, count, channels, batch);
	}
}

static int _cwc_convnet_convolutional_backward_propagate_coefficient_rows_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle,
		int x, int y, int z)
{
	if (!(layer->net.convolutional.count % y == 0 && layer->input.matrix.channels % x == 0 &&
				layer->net.convolutional.count / y * layer->input.matrix.channels / x <= 1024 && /* thread per block constraint */
				layer->net.convolutional.count / y * layer->input.matrix.channels / x >= layer->input.matrix.channels * BATCH_PER_BLOCK &&
				layer->net.convolutional.count / y * layer->input.matrix.channels / x >= layer->net.convolutional.count && /* shared loading constraint */
				sizeof(float) * BATCH_PER_BLOCK * (layer->net.convolutional.rows * layer->input.matrix.channels + layer->net.convolutional.count) <= 48 * 1024 /* shared memory size constraint */))
		return -1;
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	assert(out_partition == 1); // this cannot handle partition
	float* chm = scratch;
	float* cha = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch;
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	const int batch_group_count = batch / BATCH_PER_BLOCK;
	_cwc_convnet_reorder_matrix_major_per_block
	(m, chm, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels, batch, stream);
	_cwc_convnet_reorder_matrix_major_per_block
	(a, cha, out_rows * out_cols, layer->net.convolutional.count, batch, stream);
#define vary_block(_x, _y, _z) do { \
		dim3 threads_per_block_for_coeff(layer->net.convolutional.count / _y, layer->input.matrix.channels / _x); \
		assert(threads_per_block_for_coeff.x * threads_per_block_for_coeff.y <= 1024); \
		dim3 num_blocks_for_coeff(layer->net.convolutional.cols, out_rows, batch_group_count); \
		int shared_memory_size = sizeof(float) * BATCH_PER_BLOCK * (layer->net.convolutional.rows * layer->input.matrix.channels + layer->net.convolutional.count); \
		cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coefficient_rows<_x, _y, _z, BATCH_PER_BLOCK>, cudaFuncCachePreferShared); \
		_cwc_kern_convolutional_backward_propagate_coefficient_rows \
		<_x, _y, _z, BATCH_PER_BLOCK> \
		<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>> \
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
			chm, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
			cha, out_rows, out_cols, \
			cbw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count); \
	} while (0)
	// special casing for image
	cwc_vary_4_a(x, 1, 2, 3, 4, cwc_vary_4_b, y, 1, 2, 3, 4, cwc_vary_5_c, layer->net.convolutional.rows, 3, 5, 7, 9, 11, vary_block);
#undef vary_block
	cudaError_t error = cudaGetLastError();
	if (cudaErrorInvalidConfiguration == error)
		return -1;
	assert(error == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_backward_propagate_coefficient_rows(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	static int vary_x[] = { 1, 2, 3, 4 };
	static int vary_y[] = { 1, 2, 3, 4 };
	static int vary_z[] = { 1 };
	// benchmarking requires it has no side effect
	CWC_IMPLEMENT_VARY_STUB(EXTRA(layer)->vary.convolutional.backward.coefficient, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_backward_propagate_coefficient_rows_vary, layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	int count = layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->input.matrix.channels;
	const int batch_group_count = batch / BATCH_PER_BLOCK;
	// this has side-effect since it is accumulation
	cublasSgemv(handle, CUBLAS_OP_N, count, out_rows * batch_group_count, &one, cbw, count, unit, 1, &one, configuration->w, 1);
}

static int _cwc_convnet_convolutional_backward_propagate_coefficient_default_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle,
		int x, int y, int z)
{
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	if (!(layer->net.convolutional.count % (y * out_partition) == 0 && z % x == 0 && layer->net.convolutional.channels % (z * out_partition) == 0 &&
		  layer->net.convolutional.count / (y * out_partition) * z / x <= 1024 && /* thread per block constraint */
		  layer->net.convolutional.count / (y * out_partition) * z / x >= z && layer->net.convolutional.count / (y * out_partition) * z / x >= layer->net.convolutional.count / out_partition && /* shared loading constraint */
				sizeof(float) * (z + layer->net.convolutional.count / out_partition) <= 32 * 1024 /* shared memory size constraint */))
		return -1;
	float* chm = scratch;
	float* cha = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch;
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	const int batch_group_count = batch / BATCH_PER_BLOCK;
	assert((layer->input.matrix.channels / out_partition) % THREAD_PER_BLOCK == 0);
	assert((layer->net.convolutional.count / out_partition) % THREAD_PER_BLOCK == 0);
	assert(batch % THREAD_PER_BLOCK == 0);
	_cwc_kern_reorder_matrix_major
	<THREAD_PER_BLOCK>
	<<<dim3(layer->input.matrix.rows * layer->input.matrix.cols, (layer->input.matrix.channels / out_partition / THREAD_PER_BLOCK) * (batch / THREAD_PER_BLOCK), out_partition), THREAD_PER_BLOCK, sizeof(float) * THREAD_PER_BLOCK * THREAD_PER_BLOCK, stream>>>
	(m, chm, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels / out_partition, out_partition, batch);
	_cwc_kern_reorder_matrix_major
	<THREAD_PER_BLOCK>
	<<<dim3(out_rows * out_cols, (layer->net.convolutional.count / out_partition / THREAD_PER_BLOCK) * (batch / THREAD_PER_BLOCK), out_partition), THREAD_PER_BLOCK, sizeof(float) * THREAD_PER_BLOCK * THREAD_PER_BLOCK, stream>>>
	(a, cha, out_rows * out_cols, layer->net.convolutional.count / out_partition, out_partition, batch);
#define vary_block(_x, _y, _z) do { \
		dim3 threads_per_block_for_coeff(layer->net.convolutional.count / (_y * out_partition), _z / _x); \
		assert(threads_per_block_for_coeff.x * threads_per_block_for_coeff.y <= 1024); \
		dim3 num_blocks_for_coeff(layer->net.convolutional.cols, layer->net.convolutional.rows, layer->net.convolutional.channels / _z * batch_group_count); \
		int shared_memory_size = sizeof(float) * (_z + layer->net.convolutional.count / out_partition); \
		_cwc_kern_convolutional_backward_propagate_coefficient_default \
		<_x, _y, _z, BATCH_PER_BLOCK> \
		<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>> \
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch, batch_group_count, \
			chm, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / out_partition, out_partition, \
			cha, out_rows, out_cols, \
			cbw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count / out_partition); \
	} while (0)
	cwc_vary_6_a(x, 1, 2, 3, 4, 6, 8, cwc_vary_6_b, y, 1, 2, 3, 4, 6, 8, cwc_vary_4_c, z, 16, 24, 32, 36, vary_block);
#undef vary_block
	cudaError_t error = cudaGetLastError();
	if (cudaErrorInvalidConfiguration == error)
		return -1;
	assert(error == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_backward_propagate_coefficient_default(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	static int vary_x[] = { 1, 2, 3, 4, 6, 8 };
	static int vary_y[] = { 1, 2, 3, 4, 6, 8 };
	static int vary_z[] = { 16, 24, 32, 36 };
	// benchmarking requires it has no side effect
	CWC_IMPLEMENT_VARY_STUB(EXTRA(layer)->vary.convolutional.backward.coefficient, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_backward_propagate_coefficient_default_vary, layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	int count = layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->input.matrix.channels / out_partition;
	const int batch_group_count = batch / BATCH_PER_BLOCK;
	// this has side-effect since it is accumulation
	cublasSgemv(handle, CUBLAS_OP_N, count, batch_group_count, &one, cbw, count, unit, 1, &one, configuration->w, 1);
}

static int _cwc_convnet_convolutional_backward_propagate_error_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle,
		int x, int y, int z)
{
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	if (!(batch % x == 0 && z % y == 0 &&
				layer->input.matrix.channels % (z * out_partition) == 0 &&
				batch / x * z / y <= 1024 && /* thread per block constraint */
				batch / x * z / y >= batch && batch / x * z / y >= z && /* shared memory loading constraint */
				sizeof(float) * (batch + z) <= 48 * 1024 /* shared memory size constraint */))
		return -1;
	float* chw = scratch;
	_cwc_kern_reorder_matrix_major_parted
	<<<dim3(layer->net.convolutional.rows * layer->net.convolutional.cols, layer->input.matrix.channels / out_partition), dim3(layer->net.convolutional.count / out_partition, out_partition), 0, stream>>>
	(layer->w, chw, layer->net.convolutional.rows * layer->net.convolutional.cols, layer->input.matrix.channels, layer->net.convolutional.count, layer->input.matrix.channels / out_partition, layer->net.convolutional.count / out_partition, out_partition);
#define vary_block(_x, _y, _z, _s) do { \
		dim3 threads_per_block(batch / _x, _z / _y); \
		assert(threads_per_block.x * threads_per_block.y <= 1024); \
		dim3 num_blocks(layer->input.matrix.cols * layer->input.matrix.channels / (_z * out_partition), layer->input.matrix.rows, out_partition); \
		int shared_memory_size = sizeof(float) * (batch + _z); \
		cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_error<_x, _y, _z, _s>, cudaFuncCachePreferShared); \
		_cwc_kern_convolutional_backward_propagate_error \
		<_x, _y, _z, _s> \
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>> \
		(layer->net.convolutional.border, batch, \
		 b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
		 a, out_rows, out_cols, \
		 chw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count / out_partition, out_partition); \
	} while (0)
	cwc_vary_4_a(x, 1, 2, 4, 8, cwc_vary_5_b, y, 1, 2, 4, 6, 8, cwc_vary_6_c, z, 16, 24, 32, 36, 64, 72, cwc_vary_4_d, layer->net.convolutional.strides, 1, 2, 3, 4, vary_block);
#undef vary_block
	cudaError_t error = cudaGetLastError();
	if (cudaErrorInvalidConfiguration == error)
		return -1;
	assert(error == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_backward_propagate_error(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	static int vary_x[] = { 1, 2, 4, 8 };
	static int vary_y[] = { 1, 2, 4, 6, 8 };
	static int vary_z[] = { 16, 24, 32, 36, 64, 72 };
	CWC_IMPLEMENT_VARY_STUB(EXTRA(layer)->vary.convolutional.backward.gradient, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_backward_propagate_error_vary, layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
}

void cwc_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	assert(layer->net.convolutional.count % 4 == 0);
	assert(batch % BATCH_PER_BLOCK == 0);
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	// it turns out that first apply relu would save us a lot of computation because no need to low both out and out_grad any more
	cwc_kern_relu_backward_propagate
	<<<dim3(out_cols, out_rows, layer->net.convolutional.count), batch, 0, stream>>>
	(batch, n, a, out_rows, out_cols, layer->net.convolutional.count);
	assert(cudaGetLastError() == cudaSuccess);
	if (cwc_convnet_layer_use_rows(layer))
		_cwc_convnet_convolutional_backward_propagate_coefficient_rows(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
	else
		_cwc_convnet_convolutional_backward_propagate_coefficient_default(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
	// compute the bias directly using gemv routine
	cublasSgemv(handle, CUBLAS_OP_T, out_rows * out_cols * batch, layer->net.convolutional.count, &one, a, out_rows * out_cols * batch, unit, 1, &one, configuration->bias, 1);
	assert(cudaGetLastError() == cudaSuccess);
	if (b)
		_cwc_convnet_convolutional_backward_propagate_error(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
}
