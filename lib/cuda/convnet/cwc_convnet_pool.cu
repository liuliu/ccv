#include <cuda.h>
#include <cublas_v2.h>
extern "C" {
#include "../cwc.h"
#include "../cwc_internal.h"
}
#include "../../inl/ccv_convnet_inl.h"

template <int input_per_thread>
__global__ static void _cwc_kern_max_pool_forward_propagate(const int strides, const int border, const int size, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols)
{
	assert(gridDim.x == out_cols);
	assert(gridDim.y == out_rows);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	assert(blockDim.x == batch);
	const int thidx = threadIdx.x;
	int i, x, y;
	input += blockIdx.z * rows * cols * batch + (blockIdx.y * strides * cols + blockIdx.x * strides) * batch;
	float prod[input_per_thread];
	const int input_y = blockIdx.y * strides - border;
	const int input_x = blockIdx.x * strides - border;
	const int input_start_y = max(input_y, 0);
	const int input_start_x = max(input_x, 0);
	const int input_end_y = min(input_y + size, rows);
	const int input_end_x = min(input_x + size, cols);
	const int size_start_y = input_start_y - input_y - border;
	const int size_start_x = input_start_x - input_x - border;
	const int size_end_y = size - border + (input_end_y - (input_y + size));
	const int size_end_x = size - border + (input_end_x - (input_x + size));
	// this is equal to iterating over 0 to size, and then compute the input origin by blockIdx.y * strides - border + y
	#pragma unroll
	for (y = size_start_y; y < size_end_y; y++)
		#pragma unroll
		for (x = size_start_x; x < size_end_x; x++)
		{
			shared_input[thidx] = input[(y * cols + x) * batch + thidx];
			__syncthreads();
			if (x == size_start_x && y == size_start_y)
				#pragma unroll
				for (i = 0; i < input_per_thread; i++)
					prod[i] = shared_input[i + threadIdx.x * input_per_thread];
			else
				#pragma unroll
				for (i = 0; i < input_per_thread; i++)
					prod[i] = max(prod[i], shared_input[i + threadIdx.x * input_per_thread]);
			__syncthreads();
		}
	out += blockIdx.z * out_rows * out_cols * batch + (blockIdx.y * out_cols + blockIdx.x) * batch;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		out[i + threadIdx.x * input_per_thread] = prod[i];
}

void cwc_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
	dim3 num_blocks(out_cols, out_rows, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_max_pool_forward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 a, rows, cols, layer->input.matrix.channels,
	 b, out_rows, out_cols);
}

template <int input_per_thread>
__global__ static void _cwc_kern_average_pool_forward_propagate(const int strides, const int border, const int size, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols)
{
	assert(gridDim.x == out_rows);
	assert(gridDim.y == out_cols);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	assert(thcnt >= batch);
	int i, x, y;
	input += blockIdx.z * rows * cols * batch + (blockIdx.x * strides * cols + blockIdx.y * strides) * batch;
	float prod[input_per_thread];
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		prod[i] = 0;
	const int input_y = blockIdx.x * strides - border;
	const int input_x = blockIdx.y * strides - border;
	const int input_start_y = max(input_y, 0);
	const int input_start_x = max(input_x, 0);
	const int input_end_y = min(input_y + size, rows);
	const int input_end_x = min(input_x + size, cols);
	const int size_start_y = input_start_y - input_y - border;
	const int size_start_x = input_start_x - input_x - border;
	const int size_end_y = size - border + (input_end_y - (input_y + size));
	const int size_end_x = size - border + (input_end_x - (input_x + size));
	// this is equal to iterating over 0 to size, and then compute the input origin by blockIdx.x * strides - border + y
	#pragma unroll
	for (y = size_start_y; y < size_end_y; y++)
		#pragma unroll
		for (x = size_start_x; x < size_end_x; x++)
		{
			if (thidx < batch)
				shared_input[thidx] = input[(y * cols + x) * batch + thidx];
			__syncthreads();
			#pragma unroll
			for (i = 0; i < input_per_thread; i++)
				prod[i] += shared_input[i + threadIdx.x * input_per_thread];
			__syncthreads();
		}
	float inv_size = 1.0 / ((input_end_y - input_start_y) * (input_end_x - input_start_x));
	out += blockIdx.z * out_rows * out_cols * batch + (blockIdx.x * out_cols + blockIdx.y) * batch;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		out[i + threadIdx.x * input_per_thread] = prod[i] * inv_size;
}

void cwc_convnet_average_pool_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
	dim3 num_blocks(out_rows, out_cols, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_average_pool_forward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 a, rows, cols, layer->input.matrix.channels,
	 b, out_rows, out_cols);
}

template <int input_per_thread>
__global__ static void _cwc_kern_max_pool_backward_propagate(const int strides, const int border, const int size, const int batch,
		float* input, float* input_grad, const int rows, const int cols, const int channels,
		float* out, float* out_grad, const int out_rows, int out_cols)
{
	assert(gridDim.x == cols);
	assert(gridDim.y == rows);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out = &shared[batch];
	float* shared_grad = &shared[batch * 2];
	assert(blockDim.x == batch);
	const int thidx = threadIdx.x;
	float prod[input_per_thread];
	int i, x, y;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		prod[i] = 0;
	const int ycnt = (size - 1 - (blockIdx.y + border) % strides) / strides + 1;
	const int xcnt = (size - 1 - (blockIdx.x + border) % strides) / strides + 1;
	const int out_y = (blockIdx.y + border) / strides - ycnt + 1;
	const int out_x = (blockIdx.x + border) / strides - xcnt + 1;
	const int out_start_y = max(out_y, 0);
	const int out_start_x = max(out_x, 0);
	out += (blockIdx.z * out_rows * out_cols + out_start_y * out_cols) * batch;
	out_grad += (blockIdx.z * out_rows * out_cols + out_start_y * out_cols) * batch;
	const int out_end_y = min(out_y + ycnt, out_rows);
	const int out_end_x = min(out_x + xcnt, out_cols);
	input += (blockIdx.z * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	if (thidx < batch)
		shared_input[thidx] = input[thidx];
	for (y = out_start_y; y < out_end_y; y++)
	{
		for (x = out_start_x; x < out_end_x; x++)
		{
			shared_out[thidx] = out[x * batch + thidx],
			shared_grad[thidx] = out_grad[x * batch + thidx];
			__syncthreads();
			#pragma unroll
			for (i = 0; i < input_per_thread; i++)
				// we have to do direct comparison otherwise it will contribute to too many cells
				// and the propagation won't work. But CPU will have different result comparing with GPU
				if (shared_out[i + threadIdx.x * input_per_thread] == shared_input[i + threadIdx.x * input_per_thread])
					prod[i] += shared_grad[i + threadIdx.x * input_per_thread];
			__syncthreads();
		}
		out += out_cols * batch;
		out_grad += out_cols * batch;
	}
	input_grad += (blockIdx.z * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		input_grad[i + threadIdx.x * input_per_thread] = prod[i];
}

void cwc_convnet_max_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	dim3 num_blocks(layer->input.matrix.cols, layer->input.matrix.rows, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch * 3;
	_cwc_kern_max_pool_backward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 m, b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
	 n, a, out_rows, out_cols);
}

template <int input_per_thread>
__global__ static void _cwc_kern_average_pool_backward_propagate(const int strides, const int border, const int size, const int batch,
		float* input_grad, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, int out_cols)
{
	assert(gridDim.x == cols);
	assert(gridDim.y == rows);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_grad = &shared[0];
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	assert(thcnt >= batch);
	float prod[input_per_thread];
	int i, x, y;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		prod[i] = 0;
	const int ycnt = (size - 1 - (blockIdx.y + border) % strides) / strides + 1;
	const int xcnt = (size - 1 - (blockIdx.x + border) % strides) / strides + 1;
	const int out_y = (blockIdx.y + border) / strides - ycnt + 1;
	const int out_x = (blockIdx.x + border) / strides - xcnt + 1;
	const int out_start_y = max(out_y, 0);
	const int out_start_x = max(out_x, 0);
	out_grad += (blockIdx.z * out_rows * out_cols + out_start_y * out_cols) * batch;
	const int out_end_y = min(out_y + ycnt, out_rows);
	const int out_end_x = min(out_x + xcnt, out_cols);
	for (y = out_start_y; y < out_end_y; y++)
	{
		for (x = out_start_x; x < out_end_x; x++)
		{
			if (thidx < batch)
				shared_grad[thidx] = out_grad[x * batch + thidx];
			__syncthreads();
			float inv_size = 1.0 / ((min(y * strides + size - border, rows) - max(y * strides - border, 0)) * (min(x * strides + size - border, cols) - max(x * strides - border, 0)));
			#pragma unroll
			for (i = 0; i < input_per_thread; i++)
				prod[i] += shared_grad[i + threadIdx.x * input_per_thread] * inv_size;
			__syncthreads();
		}
		out_grad += out_cols * batch;
	}
	input_grad += (blockIdx.z * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		input_grad[i + threadIdx.x * input_per_thread] = prod[i];
}

void cwc_convnet_average_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	dim3 num_blocks(layer->input.matrix.cols, layer->input.matrix.rows, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_average_pool_backward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
	 a, out_rows, out_cols);
}
