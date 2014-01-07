extern "C" {
#include "cwc.h"
#include "../ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
}
#include "cublas_v2.h"

// this structure holds intermediate on-device memory representation of convnet

typedef struct {
	cudaStream_t stream;
	cublasHandle_t cublas;
	ccv_convnet_layer_t* configurations;
	float** forwards;
	float** backwards;
	float* batch_unit;
} cwc_convnet_context_t;

typedef struct {
	int batch;
	ccv_convnet_layer_t* layers;
	cwc_convnet_context_t contexts[2];
} cwc_convnet_t;

#define GPU(x) ((cwc_convnet_t*)((x)->reserved))

inline static void _ccv_convnet_layer_deduce_output_format(ccv_convnet_layer_t* layer, int* rows, int* cols)
{
	assert(rows != 0 && cols != 0);
	switch(layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			assert(layer->net.convolutional.rows % 2); // as of now, don't support even number of kernel size
			assert(layer->net.convolutional.cols % 2);
			assert((layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) % layer->net.convolutional.strides == 0);
			assert((layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) % layer->net.convolutional.strides == 0);
			*rows = (layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) / layer->net.convolutional.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) / layer->net.convolutional.strides + 1;
			break;
		case CCV_CONVNET_FULL_CONNECT:
			*rows = layer->net.full_connect.count;
			*cols = 1;
			break;
		case CCV_CONVNET_MAX_POOL:
		case CCV_CONVNET_AVERAGE_POOL:
			assert((layer->input.matrix.rows + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			assert((layer->input.matrix.cols + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			*rows = (layer->input.matrix.rows + layer->net.pool.border * 2 - layer->net.pool.size) / layer->net.pool.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.pool.border * 2 - layer->net.pool.size) / layer->net.pool.strides + 1;
			break;
	}
}

static void _cwc_convnet_rewind_convolutional_weights_onto_device(float* w, float* ow, int wnum, int filters, int channels)
{
	assert(wnum % (filters * channels) == 0);
	float* iw = (float*)ccmalloc(sizeof(float) * wnum);
	int count = wnum / (filters * channels);
	int i, j, k;
	for (i = 0; i < channels; i++)
		for (j = 0; j < count; j++)
			for (k = 0; k < filters; k++)
				iw[i * count * filters + j * filters + k] = w[k * count * channels + j * channels + i];
	cudaMemcpy(ow, iw, sizeof(float) * wnum, cudaMemcpyHostToDevice);
	ccfree(iw);
}

static void _cwc_convnet_rewind_full_connect_weights_onto_device(float* w, float* ow, int wnum, int count, int channels)
{
	assert(wnum % (count * channels) == 0);
	float* iw = (float*)ccmalloc(sizeof(float) * wnum);
	int rows = wnum / (count * channels);
	int i, j, k;
	for (i = 0; i < rows; i++)
		for (j = 0; j < channels; j++)
			for (k = 0; k < count; k++)
				iw[i * channels * count + j * count + k] = w[i * channels * count + k * channels + j];
	cudaMemcpy(ow, iw, sizeof(float) * wnum, cudaMemcpyHostToDevice);
	ccfree(iw);
}

static void _cwc_convnet_reserve_onto_device(ccv_convnet_t* convnet, int batch)
{
	assert(GPU(convnet) == 0);
	convnet->reserved = (cwc_convnet_t*)ccmalloc(sizeof(cwc_convnet_t) + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 4);
	GPU(convnet)->batch = batch;
	GPU(convnet)->layers = (ccv_convnet_layer_t*)(GPU(convnet) + 1);
	int i, j, out_rows, out_cols;
	memcpy(GPU(convnet)->layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	// configurations (the backprop deltas)
	float* batch_unit = 0;
	cudaMallocHost(&batch_unit, sizeof(float) * batch);
	for (i = 0; i < batch; i++)
		batch_unit[i] = 1;
	for (i = 0; i < 2; i++)
	{
		GPU(convnet)->contexts[i].configurations = GPU(convnet)->layers + convnet->count * (i + 1);
		memcpy(GPU(convnet)->contexts[i].configurations, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
		GPU(convnet)->contexts[i].forwards = (float**)(GPU(convnet)->layers + convnet->count * 2) + convnet->count * i;
		GPU(convnet)->contexts[i].backwards = (float**)(GPU(convnet)->layers + convnet->count * 2) + convnet->count * 2 + convnet->count * i;
		GPU(convnet)->contexts[i].batch_unit = 0;
		cudaMalloc(&GPU(convnet)->contexts[i].batch_unit, sizeof(float) * batch);
		cudaMemcpy(GPU(convnet)->contexts[i].batch_unit, batch_unit, sizeof(float) * batch, cudaMemcpyHostToDevice);
	}
	cudaFreeHost(batch_unit);
	ccv_convnet_layer_t* layers = GPU(convnet)->layers;
	int batch_per_block = batch / 16;
	for (i = 0; i < 2; i++)
	{
		cudaStreamCreate(&GPU(convnet)->contexts[i].stream);
		cublasCreate(&GPU(convnet)->contexts[i].cublas);
		cublasSetStream(GPU(convnet)->contexts[i].cublas, GPU(convnet)->contexts[i].stream);
	}
	for (i = 0; i < convnet->count; i++)
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				for (j = 0; j < 2; j++)
					assert(GPU(convnet)->contexts[j].configurations[i].type == CCV_CONVNET_CONVOLUTIONAL);
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				_cwc_convnet_rewind_convolutional_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].net.convolutional.count, layers[i].net.convolutional.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.convolutional.count, cudaMemcpyHostToDevice);
				_ccv_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols);
				for (j = 0; j < 2; j++)
				{
					GPU(convnet)->contexts[j].configurations[i].w = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].configurations[i].w, sizeof(float) * (layers[i].wnum * batch_per_block * out_rows + layers[i].net.convolutional.count));
					assert(GPU(convnet)->contexts[j].configurations[i].w);
					GPU(convnet)->contexts[j].configurations[i].bias = GPU(convnet)->contexts[j].configurations[i].w + layers[i].wnum * batch_per_block * out_rows;
					GPU(convnet)->contexts[j].forwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * batch);
					if (i > 0) // if it is the input layer, no need to backprop to outmost one
					{
						GPU(convnet)->contexts[j].backwards[i] = 0;
						cudaMalloc(&GPU(convnet)->contexts[j].backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
					}
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				for (j = 0; j < 2; j++)
					assert(GPU(convnet)->contexts[j].configurations[i].type == CCV_CONVNET_FULL_CONNECT);
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				_cwc_convnet_rewind_full_connect_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].input.matrix.rows * layers[i].input.matrix.cols, layers[i].input.matrix.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.full_connect.count, cudaMemcpyHostToDevice);
				for (j = 0; j < 2; j++)
				{
					GPU(convnet)->contexts[j].configurations[i].w = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].configurations[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
					GPU(convnet)->contexts[j].configurations[i].bias = GPU(convnet)->contexts[j].configurations[i].w + layers[i].wnum;
					GPU(convnet)->contexts[j].forwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].forwards[i], sizeof(float) * layers[i].net.full_connect.count * batch);
					GPU(convnet)->contexts[j].backwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				}
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				_ccv_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols);
				for (j = 0; j < 2; j++)
				{
					assert(GPU(convnet)->contexts[j].configurations[i].type == layers[i].type);
					GPU(convnet)->contexts[j].configurations[i].w = GPU(convnet)->contexts[j].configurations[i].bias = 0;
					GPU(convnet)->contexts[j].forwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch);
					GPU(convnet)->contexts[j].backwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				}
				layers[i].w = layers[i].bias = 0;
				break;
		}
}

// =========================================== KERNEL CODE ===================================================

template <int input_per_thread, int filter_per_thread>
__global__ void _cwc_kern_convolutional_forward_propagate(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count,
		float* const biases)
{
	// gridDim.x == out_rows
	// gridDim.y == out_cols
	extern __shared__ float shared[];
	float* shared_block = &shared[0];
	float* shared_weights = &shared[batch];
	float* shared_bias = &shared[batch + count];
	float prod[filter_per_thread][input_per_thread];
	assert(batch == input_per_thread * blockDim.x);
	assert(count == filter_per_thread * blockDim.y);
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	const int filter_loads = (count + thcnt - 1) / thcnt;
	int c, i, j, x, y;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			prod[i][j] = 0;
	input += (blockIdx.x * strides * cols + blockIdx.y * strides) * batch;
	#pragma unroll
	for (i = 0; i < filter_loads; i++)
		if (i * thcnt + thidx < count)
			shared_bias[i * thcnt + thidx] = biases[i * thcnt + thidx];
	for (c = 0; c < channels; c++)
	{
		for (y = 0; y < filter_rows; y++)
		{
			const int iy = y + blockIdx.x * strides - border;
			if (iy >= 0 && iy < rows)
				for (x = 0; x < filter_cols; x++)
				{
					const int ix = x + blockIdx.y * strides - border;
					if (ix >= 0 && ix < cols)
					{
						#pragma unroll
						for (i = 0; i < input_loads; i++)
							if (i * thcnt + thidx < batch)
								shared_block[i * thcnt + thidx] = input[((y - border) * cols + x - border) * batch + i * thcnt + thidx];
						#pragma unroll
						for (i = 0; i < filter_loads; i++)
							if (i * thcnt + thidx < count)
								shared_weights[i * thcnt + thidx] = filter[(y * filter_cols + x) * count + i * thcnt + thidx];
						__syncthreads();
						#pragma unroll
						for (i = 0; i < filter_per_thread; i++)
							#pragma unroll
							for (j = 0; j < input_per_thread; j++)
								prod[i][j] += shared_block[j + threadIdx.x * input_per_thread] * shared_weights[i + threadIdx.y * filter_per_thread];
						__syncthreads();
					}
				}
		}
		input += rows * cols * batch;
		filter += filter_rows * filter_cols * count;
	}
	const int outcnt = out_rows * out_cols * batch;
	out += (blockIdx.x * out_cols + blockIdx.y) * batch;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
	{
		const float bias = shared_bias[i + threadIdx.y * filter_per_thread];
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			out[(i + threadIdx.y * filter_per_thread) * outcnt + j + threadIdx.x * input_per_thread] = max(0.0, prod[i][j] + bias);
	}
}

static void _cwc_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	assert(b);
	dim3 threads_per_block(batch / 8, layer->net.convolutional.count / 4);
	dim3 num_blocks(out_rows, out_cols);
	int shared_memory_size = sizeof(float) * (batch + layer->net.convolutional.count);
	_cwc_kern_convolutional_forward_propagate
		<8, 4>
		<<<num_blocks, threads_per_block, shared_memory_size + /* need extra space for bias */ sizeof(float) * layer->net.convolutional.count, stream>>>
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
		 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 b, out_rows, out_cols,
		 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count,
		 layer->bias);
}

template <int channel_per_thread, int filter_per_thread, int batch_per_block, int filter_strides>
__global__ void _cwc_kern_convolutional_backward_propagate_delta(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, float* out_grad, const int out_rows, const int out_cols,
		float* delta,
		float* filter, const int filter_rows, const int filter_cols, const int count)
{
	// gridDim.x == filter_rows
	// gridDim.y == filter_cols
	assert(gridDim.z == out_rows * batch / batch_per_block);
	extern __shared__ float shared[];
	float* shared_block = &shared[0];
	float* shared_out = &shared[batch_per_block * channels * filter_strides];
	float* shared_grad = &shared[batch_per_block * (channels * filter_strides + count)];
	float prod[filter_strides][channel_per_thread][filter_per_thread];
	// channel_per_thread * blockDim.x == channels
	// filter_per_thread * blockDim.y == filter_count
	assert(channel_per_thread * blockDim.x == channels);
	assert(filter_per_thread * blockDim.y == count);
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(batch % batch_per_block == 0);
	assert(thcnt % batch_per_block == 0);
	int i, j, k, q, x;
	#pragma unroll
	for (q = 0; q < filter_strides; q++)
		#pragma unroll
		for (i = 0; i < channel_per_thread; i++)
			#pragma unroll
			for (j = 0; j < filter_per_thread; j++)
				prod[q][i][j] = 0;
	const int bxidx = thidx % batch_per_block;
	const int byidx = thidx / batch_per_block;
	const int batch_idx = blockIdx.z % (batch / batch_per_block);
	const int incnt = rows * cols * batch;
	input += (blockIdx.x * cols + blockIdx.y * filter_strides) * batch + batch_idx * batch_per_block + byidx * incnt + bxidx;
	const int outcnt = out_rows * out_cols * batch;
	const int block_loads = (batch_per_block * channels + thcnt - 1) / thcnt;
	const int out_loads = (batch_per_block * count + thcnt - 1) / thcnt;
	const int block_loads_factor = (thcnt / batch_per_block) * incnt;
	const int out_loads_factor = (thcnt / batch_per_block) * outcnt;
	const int filter_idx = threadIdx.y * filter_per_thread;
	const int channel_idx = threadIdx.x * channel_per_thread;
	const int y = blockIdx.z / (batch / batch_per_block);
	const int max_q = min(blockIdx.y * filter_strides + filter_strides, filter_cols) - blockIdx.y * filter_strides;
	out += batch_idx * batch_per_block + byidx * outcnt + bxidx + y * out_cols * batch;
	out_grad += batch_idx * batch_per_block + byidx * outcnt + bxidx + y * out_cols * batch;
	const int iy = blockIdx.x + y * strides - border;
	if (iy >= 0 && iy < rows)
	{
		input += (y * strides - border) * cols * batch;
		for (x = 0; x < out_cols; x++)
		{
			const int ix = blockIdx.y * filter_strides + x * strides - border;
			if (ix + max_q > 0 && ix < cols)
			{
				const int start_q = max(ix, 0) - ix;
				const int end_q = min(ix + max_q, cols) - ix;
				#pragma unroll
				for (q = start_q; q < end_q; q++)
					#pragma unroll
					for (i = 0; i < block_loads; i++)
						if (thidx + i * thcnt < batch_per_block * channels)
							shared_block[q * batch_per_block * channels + thidx + i * thcnt] = input[(q + x * strides - border) * batch + i * block_loads_factor];
				#pragma unroll
				for (i = 0; i < out_loads; i++)
					if (thidx + i * thcnt < batch_per_block * count)
						shared_out[thidx + i * thcnt] = out[x * batch + i * out_loads_factor],
						shared_grad[thidx + i * thcnt] = out_grad[x * batch + i * out_loads_factor];
				__syncthreads();
				#pragma unroll
				for (k = 0; k < batch_per_block; k++)
					#pragma unroll
					for (i = 0; i < filter_per_thread; i++)
						if (shared_out[(i + filter_idx) * batch_per_block + k] > 0)
							#pragma unroll
							for (q = start_q; q < end_q; q++)
								#pragma unroll
								for (j = 0; j < channel_per_thread; j++)
									prod[q][j][i] += shared_block[q * batch_per_block * channels + (j + channel_idx) * batch_per_block + k] * shared_grad[(i + filter_idx) * batch_per_block + k];
				__syncthreads();
			}
		}
	}
	delta += (blockIdx.x * filter_cols + blockIdx.y * filter_strides) * count + blockIdx.z * filter_rows * filter_cols * count * channels;
	const int deltacnt = filter_rows * filter_cols * count;
	#pragma unroll
	for (q = 0; q < max_q; q++)
		#pragma unroll
		for (i = 0; i < channel_per_thread; i++)
			#pragma unroll
			for (j = 0; j < filter_per_thread; j++)
				delta[(i + channel_idx) * deltacnt + q * count + j + filter_idx] = prod[q][i][j];
}

template <int out_per_thread>
__global__ void _cwc_kern_convolutional_backward_propagate_bias(const int batch,
		float* out, float* out_grad, const int out_rows, const int out_cols,
		float* bias, const int count)
{
	// gridDim.x == count
	assert(gridDim.x == count);
	const int skip_pixels = blockDim.y;
	extern __shared__ float shared[];
	float* shared_bias = &shared[0];
	float* shared_out = &shared[1];
	float* shared_grad = &shared[1 + batch * skip_pixels];
	int i, x;
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	const int out_loads = (batch * skip_pixels + thcnt - 1) / thcnt;
	assert(thcnt % batch == 0);
	out += blockIdx.x * out_rows * out_cols * batch + thidx;
	out_grad += blockIdx.x * out_rows * out_cols * batch + thidx;
	const int out_load_factor = thcnt;
	const int out_load_pixels = thcnt / batch;
	if (thidx == 0)
		shared_bias[0] = 0;
	for (x = 0; x < out_rows * out_cols; x += skip_pixels)
	{
		for (i = 0; i < out_loads; i++)
			if (i * thcnt + thidx < batch * skip_pixels && x + i * out_load_pixels < out_rows * out_cols)
				shared_out[i * thcnt + thidx] = out[x * batch + i * out_load_factor],
				shared_grad[i * thcnt + thidx] = out_grad[x * batch + i * out_load_factor];
		__syncthreads();
		// because I branched out with threadIdx, therefore, synchronization must happen outside of the if clause
		if (threadIdx.y + x < out_rows * out_cols)
		{
			if (shared_out[threadIdx.y * batch + threadIdx.x * out_per_thread] <= 0)
				shared_grad[threadIdx.y * batch + threadIdx.x * out_per_thread] = 0;
			#pragma unroll
			for (i = 1; i < out_per_thread; i++)
				if (shared_out[threadIdx.y * batch + threadIdx.x * out_per_thread + i] > 0)
					shared_grad[threadIdx.y * batch + threadIdx.x * out_per_thread] += shared_grad[threadIdx.y * batch + threadIdx.x * out_per_thread + i];
		}
		__syncthreads();
		// I can do better here, but bias computation is not the bottleneck
		if (threadIdx.y + x < out_rows * out_cols && threadIdx.x == 0)
			#pragma unroll
			for (i = 1; i < blockDim.x; i++)
				shared_grad[threadIdx.y * batch] += shared_grad[threadIdx.y * batch + i * out_per_thread];
		__syncthreads();
		// because I branched out with threadIdx, therefore, synchronization must happen outside of the if clause, thus, this if clause appeared repeatedly
		if (threadIdx.y + x < out_rows * out_cols && thidx == 0)
		{
			#pragma unroll
			for (i = 1; i < blockDim.y && i + x < out_rows * out_cols; i++)
				shared_grad[0] += shared_grad[i * batch];
			shared_bias[0] += shared_grad[0];
		}
		__syncthreads();
	}
	if (thidx == 0)
		bias[blockIdx.x] = shared_bias[0];
}

template <int input_per_thread, int channel_per_thread, int filter_per_iteration>
__global__ void _cwc_kern_convolutional_backward_propagate(const int strides, const int border, const int batch,
		float* input_grad, const int rows, const int cols, const int channels,
		float* out, float* out_grad, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count)
{
	// gridDim.x = rows
	// gridDim.y = cols
	extern __shared__ float shared[];
	float* shared_out = &shared[0];
	float* shared_grad = &shared[batch];
	float* shared_weights = &shared[batch * 2];
	float prod[input_per_thread][channel_per_thread];
	assert(batch == input_per_thread * blockDim.x);
	assert(channels == channel_per_thread * blockDim.y);
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	const int channel_filter_loads = (channels * filter_per_iteration + thcnt - 1) / thcnt;
	int i, j, k, c, x, y;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		#pragma unroll
		for (j = 0; j < channel_per_thread; j++)
			prod[i][j] = 0;
	const int ycnt = (filter_rows - 1 - (blockIdx.x + border) % strides) / strides + 1;
	const int xcnt = (filter_cols - 1 - (blockIdx.y + border) % strides) / strides + 1;
	const int filter_y = (ycnt - 1) * strides + (blockIdx.x + border) % strides;
	assert(filter_y < filter_rows);
	const int filter_x = (xcnt - 1) * strides + (blockIdx.y + border) % strides;
	assert(filter_x < filter_cols);
	const int out_y = (blockIdx.x + border) / strides - ycnt + 1;
	const int out_x = (blockIdx.y + border) / strides - xcnt + 1;
	const int out_start_y = max(out_y, 0);
	const int out_start_x = max(out_x, 0);
	const int filter_start_y = filter_y - (out_start_y - out_y) * strides;
	const int filter_start_x = filter_x - (out_start_x - out_x) * strides;
	out += (out_start_y * out_cols + out_start_x) * batch;
	out_grad += (out_start_y * out_cols + out_start_x) * batch;
	const int out_end_y = out_y + ycnt - 1;
	const int out_end_x = out_x + xcnt - 1;
	const int filter_end_y = (blockIdx.x + border) % strides + (out_end_y - min(out_end_y, out_rows - 1)) * strides;
	const int filter_end_x = (blockIdx.y + border) % strides + (out_end_x - min(out_end_x, out_cols - 1)) * strides;
	const int outcnt = out_rows * out_cols * batch;
	for (y = filter_start_y; y >= filter_end_y; y -= strides)
	{
		for (x = filter_start_x, c = 0; x >= filter_end_x; x -= strides, c++)
		{
			#pragma unroll
			for (k = 0; k < count; k++)
			{
				if (k % filter_per_iteration == 0)
				{
					const int min_channel_filter_count = channels * min(filter_per_iteration, count - k);
					#pragma unroll
					for (i = 0; i < channel_filter_loads; i++)
						if (i * thcnt + thidx < min_channel_filter_count)
						{
							const int channel_idx = (i * thcnt + thidx) / filter_per_iteration;
							const int filter_idx = (i * thcnt + thidx) % filter_per_iteration + k;
							shared_weights[i * thcnt + thidx] = filter[(channel_idx * filter_rows * filter_cols + y * filter_cols + x) * count + filter_idx];
						}
				}
				float* out_per_filter = out + k * outcnt;
				float* out_grad_per_filter = out_grad + k * outcnt;
				#pragma unroll
				for (i = 0; i < input_loads; i++)
					if (i * thcnt + thidx < batch)
						shared_out[i * thcnt + thidx] = out_per_filter[c * batch + i * thcnt + thidx],
						shared_grad[i * thcnt + thidx] = out_grad_per_filter[c * batch + i * thcnt + thidx];
				__syncthreads();
				const int k_idx = k % filter_per_iteration;
				#pragma unroll
				for (i = 0; i < input_per_thread; i++)
					if (shared_out[i + threadIdx.x * input_per_thread] > 0)
						#pragma unroll
						for (j = 0; j < channel_per_thread; j++)
							prod[i][j] += shared_grad[i + threadIdx.x * input_per_thread] * shared_weights[(j + threadIdx.y * channel_per_thread) * filter_per_iteration + k_idx];
				__syncthreads();
			}
		}
		out += out_cols * batch;
		out_grad += out_cols * batch;
	}
	const int incnt = rows * cols * batch;
	input_grad += (blockIdx.x * cols + blockIdx.y) * batch;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			input_grad[(i + threadIdx.y * channel_per_thread) * incnt + j + threadIdx.x * input_per_thread] = prod[j][i];
}

static void _cwc_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, const cudaStream_t& stream)
{
	assert(layer->net.convolutional.count % 4 == 0);
	int out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	dim3 threads_per_block_for_delta(layer->input.matrix.channels, layer->net.convolutional.count);
	assert(batch % 16 == 0);
	dim3 num_blocks_for_delta(layer->net.convolutional.rows, (layer->net.convolutional.cols + 5) / 6, out_rows * batch / 16);
	int shared_memory_size = sizeof(float) * (16 * (layer->input.matrix.channels * 6 + layer->net.convolutional.count * 2));
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_delta<1, 1, 16, 6>, cudaFuncCachePreferShared);
	_cwc_kern_convolutional_backward_propagate_delta
	<1, 1, 16, 6>
	<<<num_blocks_for_delta, threads_per_block_for_delta, shared_memory_size, stream>>>
	(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
		m, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		n, a, out_rows, out_cols,
		configuration->w,
		layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
	dim3 threads_per_block_for_bias(batch / 8, 8);
	dim3 num_blocks_for_bias(layer->net.convolutional.count);
	shared_memory_size = sizeof(float) * (1 + batch * 8 * 2);
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_bias<8>, cudaFuncCachePreferShared);
	_cwc_kern_convolutional_backward_propagate_bias
	<8>
	<<<num_blocks_for_bias, threads_per_block_for_bias, shared_memory_size, stream>>>
	(batch,
		n, a, out_rows, out_cols,
		configuration->bias, layer->net.convolutional.count);
	if (b)
	{
		dim3 threads_per_block(batch, 1);
		dim3 num_blocks(layer->input.matrix.rows, layer->input.matrix.cols);
		shared_memory_size = sizeof(float) * (batch * 2 + layer->input.matrix.channels * 48);
		_cwc_kern_convolutional_backward_propagate
		<1, 3, 48>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
		 b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 n, a, out_rows, out_cols,
		 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
	}
}

template <int input_per_thread>
__global__ void _cwc_kern_max_pool_forward_propagate(const int strides, const int border, const int size, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols)
{
	// gridDim.x == out_rows
	// gridDim.y == out_cols
	// gridDim.z == channels
	assert(gridDim.x == out_rows);
	assert(gridDim.y == out_cols);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	int i, x, y;
	input += blockIdx.z * rows * cols * batch + (blockIdx.x * strides * cols + blockIdx.y * strides) * batch;
	float prod[input_per_thread];
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
			#pragma unroll
			for (i = 0; i < input_loads; i++)
				if (i * thcnt + thidx < batch)
					shared_input[i * thcnt + thidx] = input[(y * cols + x) * batch + i * thcnt + thidx];
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
	out += blockIdx.z * out_rows * out_cols * batch + (blockIdx.x * out_cols + blockIdx.y) * batch;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		out[i + threadIdx.x * input_per_thread] = prod[i];
}

static void _cwc_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	dim3 num_blocks(out_rows, out_cols, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_max_pool_forward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
	 b, out_rows, out_cols);
}

template <int input_per_thread>
__global__ void _cwc_kern_max_pool_backward_propagate(const int strides, const int border, const int size, const int batch,
		float* input, float* input_grad, const int rows, const int cols, const int channels,
		float* out, float* out_grad, const int out_rows, int out_cols)
{
	// gridDim.x == rows
	// gridDim.y == cols
	// gridDim.z == channels
	assert(gridDim.x == rows);
	assert(gridDim.y == cols);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out = &shared[batch];
	float* shared_grad = &shared[batch * 2];
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	float prod[input_per_thread];
	int i, x, y;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		prod[i] = 0;
	const int ycnt = (size - 1 - (blockIdx.x + border) % strides) / strides + 1;
	const int xcnt = (size - 1 - (blockIdx.y + border) % strides) / strides + 1;
	const int out_y = (blockIdx.x + border) / strides - ycnt + 1;
	const int out_x = (blockIdx.y + border) / strides - xcnt + 1;
	const int out_start_y = max(out_y, 0);
	const int out_start_x = max(out_x, 0);
	out += (blockIdx.z * out_rows * out_cols + out_start_y * out_cols) * batch;
	out_grad += (blockIdx.z * out_rows * out_cols + out_start_y * out_cols) * batch;
	const int out_end_y = min(out_y + ycnt, out_rows);
	const int out_end_x = min(out_x + xcnt, out_cols);
	input += (blockIdx.z * rows * cols + blockIdx.x * cols + blockIdx.y) * batch;
	for (i = 0; i < input_loads; i++)
		if (i * thcnt + thidx < batch)
			shared_input[i * thcnt + thidx] = input[i * thcnt + thidx];
	for (y = out_start_y; y < out_end_y; y++)
	{
		for (x = out_start_x; x < out_end_x; x++)
		{
			#pragma unroll
			for (i = 0; i < input_loads; i++)
				if (i * thcnt + thidx < batch)
					shared_out[i * thcnt + thidx] = out[x * batch + i * thcnt + thidx],
					shared_grad[i * thcnt + thidx] = out_grad[x * batch + i * thcnt + thidx];
			__syncthreads();
			#pragma unroll
			for (i = 0; i < input_per_thread; i++)
			{
				float vi = fabsf(shared_input[i + threadIdx.x * input_per_thread]);
				float vo = fabsf(shared_out[i + threadIdx.x * input_per_thread]);
				float delta = fabsf(vi - vo) / max(max(vi, vo), 1e-5);
				if (delta < 1e-5) // there seems to be a bug that the direct comparison of these two float number will have different result on GPU comparing with CPU result
				// if (shared_out[i + threadIdx.x * input_per_thread] == shared_input[i + threadIdx.x * input_per_thread]) // if we don't care of accuracy and needs that extra 4ms per batch, we can change to this line
					prod[i] += shared_grad[i + threadIdx.x * input_per_thread];
			}
			__syncthreads();
		}
		out += out_cols * batch;
		out_grad += out_cols * batch;
	}
	input_grad += (blockIdx.z * rows * cols + blockIdx.x * cols + blockIdx.y) * batch;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		input_grad[i + threadIdx.x * input_per_thread] = prod[i];
}

static void _cwc_convnet_max_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	dim3 num_blocks(layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels);
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
__global__ void _cwc_kern_average_pool_forward_propagate(const int strides, const int border, const int size, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols)
{
	// gridDim.x == out_rows
	// gridDim.y == out_cols
	// gridDim.z == channels
	assert(gridDim.x == out_rows);
	assert(gridDim.y == out_cols);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
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
			#pragma unroll
			for (i = 0; i < input_loads; i++)
				if (i * thcnt + thidx < batch)
					shared_input[i * thcnt + thidx] = input[(y * cols + x) * batch + i * thcnt + thidx];
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

static void _cwc_convnet_average_pool_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	dim3 num_blocks(out_rows, out_cols, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_average_pool_forward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
	 b, out_rows, out_cols);
}

template <int input_per_thread>
__global__ void _cwc_kern_average_pool_backward_propagate(const int strides, const int border, const int size, const int batch,
		float* input_grad, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, int out_cols)
{
	// gridDim.x == rows
	// gridDim.y == cols
	// gridDim.z == channels
	assert(gridDim.x == rows);
	assert(gridDim.y == cols);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_grad = &shared[0];
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	float prod[input_per_thread];
	int i, x, y;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		prod[i] = 0;
	const int ycnt = (size - 1 - (blockIdx.x + border) % strides) / strides + 1;
	const int xcnt = (size - 1 - (blockIdx.y + border) % strides) / strides + 1;
	const int out_y = (blockIdx.x + border) / strides - ycnt + 1;
	const int out_x = (blockIdx.y + border) / strides - xcnt + 1;
	const int out_start_y = max(out_y, 0);
	const int out_start_x = max(out_x, 0);
	out_grad += (blockIdx.z * out_rows * out_cols + out_start_y * out_cols) * batch;
	const int out_end_y = min(out_y + ycnt, out_rows);
	const int out_end_x = min(out_x + xcnt, out_cols);
	for (y = out_start_y; y < out_end_y; y++)
	{
		for (x = out_start_x; x < out_end_x; x++)
		{
			#pragma unroll
			for (i = 0; i < input_loads; i++)
				if (i * thcnt + thidx < batch)
					shared_grad[i * thcnt + thidx] = out_grad[x * batch + i * thcnt + thidx];
			__syncthreads();
			float inv_size = 1.0 / ((min(y * strides + size - border, rows) - max(y * strides - border, 0)) * (min(x * strides + size - border, cols) - max(x * strides - border, 0)));
			#pragma unroll
			for (i = 0; i < input_per_thread; i++)
				prod[i] += shared_grad[i + threadIdx.x * input_per_thread] * inv_size;
			__syncthreads();
		}
		out_grad += out_cols * batch;
	}
	input_grad += (blockIdx.z * rows * cols + blockIdx.x * cols + blockIdx.y) * batch;
	#pragma unroll
	for (i = 0; i < input_per_thread; i++)
		input_grad[i + threadIdx.x * input_per_thread] = prod[i];
}

static void _cwc_convnet_average_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	dim3 num_blocks(layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch * 3;
	_cwc_kern_average_pool_backward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
	 a, out_rows, out_cols);
}

static void _cwc_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, float* batch_unit /* this is just 1's in device */, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	out_cols = batch;
	rows = layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels;
	float alpha = 1;
	float beta = 0;
	// make copies of bias into db's columns, note that for cublas, it is row-major matrix
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch, out_rows, 1, &alpha, batch_unit, batch, layer->bias, 1, &beta, b, batch);
	beta = 1;
	// and then do the GEMM by adding bias
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch, out_rows, rows, &alpha, a, batch, layer->w, rows, &beta, b, batch);
}

static void _cwc_convnet_full_connect_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* m, float* b, float* batch_unit, ccv_convnet_layer_t* configuration, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	out_cols = batch;
	rows = layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels;
	float alpha = 1;
	float beta = 0;
	// propagate bias
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, out_rows, batch, &alpha, batch_unit, 1, a, batch, &beta, configuration->bias, 1);
	// propagate error
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch, rows, out_rows, &alpha, a, batch, layer->w, rows, &beta, b, batch);
	// propagate weights
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, out_rows, batch, &alpha, m, batch, a, batch, &beta, configuration->w, rows);
}

template <int input_per_thread>
__global__ void _cwc_kern_convnet_softmax_with_logistic_loss(const int batch, const int count, float* a, int* c)
{
	int i;
	extern float shared[];
	const int thidx = threadIdx.x;
	float max_val = a[thidx];
	for (i = 0; i < count; i++)
	{
		shared[thidx] = a[i * batch + thidx];
		if (shared[thidx] > max_val)
			max_val = shared[thidx];
	}
	__syncthreads();
	float val = 0;
	for (i = 0; i < count; i++)
	{
		shared[thidx] = a[i * batch + thidx];
		val += shared[thidx] = expf(shared[thidx] - max_val);
		a[i * batch + thidx] = shared[thidx];
	}
	__syncthreads();
	val = 1.0 / val;
	for (i = 0; i < count; i++)
		a[i * batch + thidx] = (i == c[thidx]) - (a[i * batch + thidx] * val);
}

static void _cwc_convnet_softmax_with_logistic_loss(int batch, int count, float* a, int* c, const cudaStream_t& stream)
{
	dim3 num_blocks(1);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_convnet_softmax_with_logistic_loss
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(batch, count, a, c);
}

/* assuming a is in device memory */
static void _cwc_convnet_encode_impl(ccv_convnet_t* convnet, float* a, int batch, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_cwc_convnet_convolutional_forward_propagate(layer, batch, i == 0 ? a : context->forwards[i - 1], context->forwards[i], context->stream);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_cwc_convnet_full_connect_forward_propagate(layer, batch, context->forwards[i - 1], context->forwards[i], context->batch_unit, context->cublas);
				assert(i > 0);
				break;
			case CCV_CONVNET_MAX_POOL:
				assert(i > 0);
				_cwc_convnet_max_pool_forward_propagate(layer, batch, context->forwards[i - 1], context->forwards[i], context->stream);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				_cwc_convnet_average_pool_forward_propagate(layer, batch,  context->forwards[i - 1], context->forwards[i], context->stream);
				break;
		}
	}
}

static void _cwc_convnet_backwards_propagate_error(ccv_convnet_t* convnet, float* a, int batch, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	for (i = convnet->count - 1; i >= 0; i--)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
		ccv_convnet_layer_t* configuration = context->configurations + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_cwc_convnet_convolutional_backward_propagate(layer, batch, i == convnet->count - 1 ? a : context->backwards[i + 1], context->forwards[i - 1], context->forwards[i], context->backwards[i], configuration, context->stream);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_cwc_convnet_full_connect_backward_propagate(layer, batch,  i == convnet->count - 1 ? a : context->backwards[i + 1], context->forwards[i - 1], context->backwards[i], context->batch_unit, configuration, context->cublas);
				break;
			case CCV_CONVNET_MAX_POOL:
				_cwc_convnet_max_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : context->backwards[i + 1], context->forwards[i - 1], context->forwards[i], context->backwards[i], context->stream);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_cwc_convnet_average_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : context->backwards[i + 1], context->backwards[i], context->stream);
				break;
		}
	}
}

void cwc_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch)
{
}

void cwc_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int* labels, int batch)
{
}

void cwc_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, ccv_convnet_train_param_t params)
{
	assert(params.mini_batch % 16 == 0);
	if (!GPU(convnet))
		_cwc_convnet_reserve_onto_device(convnet, params.mini_batch);
	int i, j, t, x, k;
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	int aligned_padding = categorizeds->rnum % params.mini_batch;
	int aligned_rnum = categorizeds->rnum - aligned_padding;
	int aligned_batches = categorizeds->rnum / params.mini_batch;
	int* idx = (int*)ccmalloc(sizeof(int) * (categorizeds->rnum + aligned_padding));
	for (i = 0; i < categorizeds->rnum; i++)
		idx[i] = i;
	gsl_ran_shuffle(rng, idx, categorizeds->rnum, sizeof(int));
	// the last layer has to be full connect, thus we can use it as softmax layer
	assert(convnet->layers[convnet->count - 1].type == CCV_CONVNET_FULL_CONNECT);
	int category_count = convnet->layers[convnet->count - 1].net.full_connect.count;
	float* input_batch_on_host[2] = {
		0, 0
	};
	for (i = 0; i < 2; i++)
		cudaMallocHost(&input_batch_on_host[i], sizeof(float) * convnet->rows * convnet->cols * convnet->channels * params.mini_batch); 
	assert(input_batch_on_host[0] && input_batch_on_host[1]);
	float* input_batch_on_device[2] = {
		0, 0
	};
	for (i = 0; i < 2; i++)
		cudaMalloc(&input_batch_on_device[i], sizeof(float) * convnet->rows * convnet->cols * convnet->channels * params.mini_batch);
	assert(input_batch_on_device[0] && input_batch_on_device[1]);
	int* c_on_host[2] = {
		0, 0
	};
	for (i = 0; i < 2; i++)
		cudaMallocHost(&c_on_host[i], sizeof(int) * params.mini_batch); 
	assert(c_on_host[0] && c_on_host[1]);
	int* c_on_device[2] = {
		0, 0
	};
	for (i = 0; i < 2; i++)
		cudaMalloc(&c_on_device[i], sizeof(int) * params.mini_batch); 
	assert(c_on_device[0] && c_on_device[1]);
	for (t = 0; t < params.max_epoch; t++)
	{
		printf(" - at epoch %d / %d\n", t + 1, params.max_epoch);
		for (i = 0; i < aligned_batches; i++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + (i % 2);
			float* current_input_batch = input_batch_on_host[i % 2];
			int* current_c = c_on_host[i % 2];
			for (j = 0; j < params.mini_batch; j++)
			{
				ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, idx[i * params.mini_batch + j]);
				current_c[j] = categorized->c;
				switch (categorized->type)
				{
					case CCV_CATEGORIZED_DENSE_MATRIX:
						for (k = 0; k < convnet->channels; k++)
							for (x = 0; x < convnet->rows * convnet->cols; x++)
								current_input_batch[(k * convnet->rows * convnet->cols + x) * params.mini_batch + j] = categorized->matrix->data.f32[x * convnet->channels + k];
						break;
					case CCV_CATEGORIZED_FILE:
						// TODO: implement this support
						break;
				}
			}
			cudaMemcpyAsync(input_batch_on_device[i % 2], current_input_batch, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->stream);
			cudaMemcpyAsync(c_on_device[i % 2], current_c, sizeof(float) * params.mini_batch, cudaMemcpyHostToDevice, context->stream);
			// sync with the other stream core so that we can compute on the single true layer parameters
			cudaStreamSynchronize(GPU(convnet)->contexts[(i + 1) % 2].stream);
			_cwc_convnet_encode_impl(convnet, input_batch_on_device[i % 2], params.mini_batch, context);
			_cwc_convnet_softmax_with_logistic_loss(params.mini_batch, category_count, context->forwards[convnet->count - 1], c_on_device[i % 2], context->stream);
			_cwc_convnet_backwards_propagate_error(convnet, context->forwards[convnet->count - 1], params.mini_batch, context);
		}
		cudaDeviceSynchronize(); // synchronize at this point
		if (t + 1 < params.max_epoch)
		{
			// reshuffle the parts we visited and move the rest to the beginning
			memcpy(idx + categorizeds->rnum, idx + aligned_rnum, sizeof(int) * aligned_padding);
			memmove(idx + aligned_padding, idx, sizeof(int) * aligned_rnum);
			memcpy(idx, idx + categorizeds->rnum, sizeof(int) * aligned_padding);
			gsl_ran_shuffle(rng, idx + aligned_padding, aligned_rnum, sizeof(int));
		}
	}
	for (i = 0; i < 2; i++)
		cudaFree(input_batch_on_device[i]);
	for (i = 0; i < 2; i++)
		cudaFreeHost(input_batch_on_host[i]);
	for (i = 0; i < 2; i++)
		cudaFree(c_on_device[i]);
	for (i = 0; i < 2; i++)
		cudaFreeHost(c_on_host[i]);
	ccfree(idx);
	gsl_rng_free(rng);
}

void cwc_convnet_free(ccv_convnet_t* convnet)
{
	if (GPU(convnet))
	{
		int i;
		ccv_convnet_layer_t* layers = GPU(convnet)->layers;
		for (i = 0; i < convnet->count; i++)
			cudaFree(layers[i].w);
	}
	ccfree(convnet);
}
