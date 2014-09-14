#include <cuda.h>
#include <cublas_v2.h>
extern "C" {
#include "../cwc.h"
#include "../cwc_internal.h"
}
#include "../../inl/ccv_convnet_inl.h"

template <int input_per_thread, int size>
__global__ static void _cwc_kern_rnorm_forward_propagate(const int batch,
		float* input, const int rows, const int cols, const int channels_per_partition, const int partition,
		float* out, float* denoms, const float kappa, const float alpha, const float beta)
{
	assert(gridDim.x == cols);
	assert(gridDim.y == rows);
	assert(gridDim.z == partition);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	const int way = size / 2;
	const int thidx = threadIdx.x;
	int i, j, c;
	float prod[input_per_thread];
	const int incnt = rows * cols * batch;
	input += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	out += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	denoms += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	const int end_way = min(way, channels_per_partition - 1);
	for (c = 0; c < end_way; c++)
	{
		shared_input[c * batch + thidx] = input[thidx];
		input += incnt;
	}
	for (c = 0; c < channels_per_partition; c++)
	{
		const int start_way = max(c - way, 0);
		const int end_way = min(c + way, channels_per_partition - 1);
		if (c + way < channels_per_partition)
		{
			shared_input[(end_way % size) * batch + thidx] = input[thidx];
			input += incnt;
		}
		__syncthreads();
		#pragma unroll
		for (i = 0; i < input_per_thread; i++)
			prod[i] = 0;
		#pragma unroll 5
		for (i = start_way; i <= end_way; i++)
			#pragma unroll
			for (j = 0; j < input_per_thread; j++)
				prod[j] += shared_input[(i % size) * batch + j + threadIdx.x * input_per_thread] * shared_input[(i % size) * batch + j + threadIdx.x * input_per_thread];
		#pragma unroll
		for (i = 0; i < input_per_thread; i++)
			prod[i] = kappa + alpha * prod[i];
		#pragma unroll
		for (i = 0; i < input_per_thread; i++)
		{
			denoms[i + threadIdx.x * input_per_thread] = prod[i];
			out[i + threadIdx.x * input_per_thread] = shared_input[(c % size) * batch + i + threadIdx.x * input_per_thread] *  powf(prod[i], -beta);
		}
		denoms += incnt;
		out += incnt;
		__syncthreads();
	}
}

void cwc_convnet_rnorm_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, float* denoms, const cudaStream_t& stream)
{
	dim3 num_blocks(cols, rows, layer->input.matrix.partition);
	dim3 threads_per_block(batch);
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch * layer->net.rnorm.size;
#define vary_block(_, _x) \
	cudaFuncSetCacheConfig(_cwc_kern_rnorm_forward_propagate<1, _x>, cudaFuncCachePreferShared); \
	_cwc_kern_rnorm_forward_propagate \
	<1, _x> \
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>> \
	(batch, \
	 a, rows, cols, layer->input.matrix.channels / layer->input.matrix.partition, layer->input.matrix.partition, \
	 b, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	cwc_vary_2_a(layer->net.rnorm.size, 3, 5, vary_block);
#undef vary_block
}

template <int input_per_thread, int size>
__global__ static void _cwc_kern_rnorm_backward_propagate(const int batch,
		float* input, float* input_grad, const int rows, const int cols, const int channels_per_partition, const int partition,
		float* out, float* out_grad, float* denoms, const float kappa, const float alpha, const float beta)
{
	assert(gridDim.x == cols);
	assert(gridDim.y == rows);
	assert(gridDim.z == partition);
	extern __shared__ float shared[];
	float* shared_out_grad = &shared[0];
	float* shared_out = &shared[batch * size];
	float* shared_denoms = &shared[batch * size * 2];
	float* shared_input = &shared[batch * size * 3];
	const int way = size / 2;
	assert(blockDim.x == batch);
	const int thidx = threadIdx.x;
	int i, j, c;
	float prod[input_per_thread];
	const int incnt = rows * cols * batch;
	out += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	out_grad += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	denoms += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	input += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	input_grad += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	const int end_way = min(way, channels_per_partition - 1);
	for (c = 0; c < end_way; c++)
	{
		shared_out_grad[c * batch + thidx] = out_grad[thidx],
		shared_out[c * batch + thidx] = out[thidx],
		shared_denoms[c * batch + thidx] = denoms[thidx];
		out_grad += incnt;
		out += incnt;
		denoms += incnt;
	}
	for (c = 0; c < channels_per_partition; c++)
	{
		const int start_way = max(c - way, 0);
		const int end_way = min(c + way, channels_per_partition - 1);
		if (c + way < channels_per_partition)
		{
			shared_out_grad[(end_way % size) * batch + thidx] = out_grad[thidx],
			shared_out[(end_way % size) * batch + thidx] = out[thidx],
			shared_denoms[(end_way % size) * batch + thidx] = denoms[thidx];
			out_grad += incnt;
			out += incnt;
			denoms += incnt;
		}
		shared_input[thidx] = input[thidx];
		__syncthreads();
		#pragma unroll
		for (i = 0; i < input_per_thread; i++)
			prod[i] = 0;
		#pragma unroll 5
		for (i = start_way; i <= end_way; i++)
			#pragma unroll
			for (j = 0; j < input_per_thread; j++)
				prod[j] += -2 * alpha * beta * shared_out_grad[(i % size) * batch + j + threadIdx.x * input_per_thread] * shared_out[(i % size) * batch + j + threadIdx.x * input_per_thread] / shared_denoms[(i % size) * batch + j + threadIdx.x * input_per_thread];
		#pragma unroll
		for (i = 0; i < input_per_thread; i++)
			input_grad[i + threadIdx.x * input_per_thread] = shared_input[i + threadIdx.x * input_per_thread] * prod[i] + shared_out_grad[(c % size) * batch + i + threadIdx.x * input_per_thread] *  powf(shared_denoms[(c % size) * batch + i + threadIdx.x * input_per_thread], -beta);
		input += incnt;
		input_grad += incnt;
		__syncthreads();
	}
}

void cwc_convnet_rnorm_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* denoms, float* b, const cudaStream_t& stream)
{
	dim3 num_blocks(layer->input.matrix.cols, layer->input.matrix.rows, layer->input.matrix.partition);
	dim3 threads_per_block(batch);
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch * (layer->net.rnorm.size * 3 + 1);
#define vary_block(_, _x) \
	cudaFuncSetCacheConfig(_cwc_kern_rnorm_backward_propagate<1, _x>, cudaFuncCachePreferShared); \
	_cwc_kern_rnorm_backward_propagate \
	<1, _x> \
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>> \
	(batch, \
	 m, b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / layer->input.matrix.partition, layer->input.matrix.partition, \
	 n, a, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	cwc_vary_2_a(layer->net.rnorm.size, 3, 5, vary_block);
#undef vary_block
}
