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
	float** forwards;
	float** backwards;
	float** denoms;
	float* batch_unit;
} cwc_convnet_context_t;

typedef struct {
	int batch;
	ccv_convnet_layer_t* layers;
	ccv_convnet_layer_t* configurations;
	ccv_convnet_layer_t* momentums;
	cwc_convnet_context_t contexts[2];
} cwc_convnet_t;

#define GPU(x) ((cwc_convnet_t*)((x)->reserved))
#define CWC_COEFF_SPREAD (8)

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
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			*rows = layer->input.matrix.rows;
			*cols = layer->input.matrix.cols;
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
	convnet->reserved = (cwc_convnet_t*)ccmalloc(sizeof(cwc_convnet_t) + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 6);
	GPU(convnet)->batch = batch;
	GPU(convnet)->layers = (ccv_convnet_layer_t*)(GPU(convnet) + 1);
	int i, j, out_rows, out_cols;
	memcpy(GPU(convnet)->layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	// configurations (the backprop coeffs)
	float* batch_unit = 0;
	cudaMallocHost(&batch_unit, sizeof(float) * batch);
	for (i = 0; i < batch; i++)
		batch_unit[i] = 1;
	GPU(convnet)->configurations = GPU(convnet)->layers + convnet->count;
	memcpy(GPU(convnet)->configurations, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	GPU(convnet)->momentums = GPU(convnet)->layers + convnet->count * 2;
	memcpy(GPU(convnet)->momentums, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	for (i = 0; i < 2; i++)
	{
		GPU(convnet)->contexts[i].denoms = (float**)(GPU(convnet)->layers + convnet->count * 3) + convnet->count * i;
		GPU(convnet)->contexts[i].forwards = (float**)(GPU(convnet)->layers + convnet->count * 3) + convnet->count * 2 + convnet->count * i;
		GPU(convnet)->contexts[i].backwards = (float**)(GPU(convnet)->layers + convnet->count * 3) + convnet->count * 4 + convnet->count * i;
		GPU(convnet)->contexts[i].batch_unit = 0;
		cudaMalloc(&GPU(convnet)->contexts[i].batch_unit, sizeof(float) * batch);
		cudaMemcpy(GPU(convnet)->contexts[i].batch_unit, batch_unit, sizeof(float) * batch, cudaMemcpyHostToDevice);
	}
	cudaFreeHost(batch_unit);
	ccv_convnet_layer_t* layers = GPU(convnet)->layers;
	int batch_per_block = batch / CWC_COEFF_SPREAD;
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
				assert(GPU(convnet)->configurations[i].type == CCV_CONVNET_CONVOLUTIONAL);
				assert(GPU(convnet)->momentums[i].type == CCV_CONVNET_CONVOLUTIONAL);
				// allocating for layer
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				_cwc_convnet_rewind_convolutional_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].net.convolutional.count, layers[i].net.convolutional.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.convolutional.count, cudaMemcpyHostToDevice);
				_ccv_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols);
				// allocating for configurations 
				GPU(convnet)->configurations[i].w = 0;
				cudaMalloc(&GPU(convnet)->configurations[i].w, sizeof(float) * (layers[i].wnum * batch_per_block * out_rows + layers[i].net.convolutional.count));
				assert(GPU(convnet)->configurations[i].w);
				GPU(convnet)->configurations[i].bias = GPU(convnet)->configurations[i].w + layers[i].wnum * batch_per_block * out_rows;
				// allocating for momentums
				GPU(convnet)->momentums[i].w = 0;
				cudaMalloc(&GPU(convnet)->momentums[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				assert(GPU(convnet)->momentums[i].w);
				GPU(convnet)->momentums[i].bias = GPU(convnet)->momentums[i].w + layers[i].wnum;
				for (j = 0; j < 2; j++)
				{
					GPU(convnet)->contexts[j].denoms[i] = 0;
					GPU(convnet)->contexts[j].forwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * batch);
					GPU(convnet)->contexts[j].backwards[i] = 0;
					if (i > 0) // if it is the input layer, no need to backprop to outmost one
						cudaMalloc(&GPU(convnet)->contexts[j].backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				assert(GPU(convnet)->configurations[i].type == CCV_CONVNET_FULL_CONNECT);
				assert(GPU(convnet)->momentums[i].type == CCV_CONVNET_FULL_CONNECT);
				// allocating for layer
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				_cwc_convnet_rewind_full_connect_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].input.matrix.rows * layers[i].input.matrix.cols, layers[i].input.matrix.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.full_connect.count, cudaMemcpyHostToDevice);
				// allocating for configurations 
				GPU(convnet)->configurations[i].w = 0;
				cudaMalloc(&GPU(convnet)->configurations[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->configurations[i].bias = GPU(convnet)->configurations[i].w + layers[i].wnum;
				// allocating for momentums
				GPU(convnet)->momentums[i].w = 0;
				cudaMalloc(&GPU(convnet)->momentums[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->momentums[i].bias = GPU(convnet)->momentums[i].w + layers[i].wnum;
				for (j = 0; j < 2; j++)
				{
					GPU(convnet)->contexts[j].denoms[i] = 0;
					GPU(convnet)->contexts[j].forwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].forwards[i], sizeof(float) * layers[i].net.full_connect.count * batch);
					GPU(convnet)->contexts[j].backwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				}
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				assert(i > 0);
				assert(GPU(convnet)->configurations[i].type == CCV_CONVNET_LOCAL_RESPONSE_NORM);
				assert(GPU(convnet)->momentums[i].type == CCV_CONVNET_LOCAL_RESPONSE_NORM);
				GPU(convnet)->configurations[i].w = GPU(convnet)->configurations[i].bias = 0;
				assert(GPU(convnet)->momentums[i].type == layers[i].type);
				GPU(convnet)->momentums[i].w = GPU(convnet)->momentums[i].bias = 0;
				for (j = 0; j < 2; j++)
				{
					GPU(convnet)->contexts[j].denoms[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].denoms[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch);
					GPU(convnet)->contexts[j].forwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch);
					GPU(convnet)->contexts[j].backwards[i] = 0;
					cudaMalloc(&GPU(convnet)->contexts[j].backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				}
				layers[i].w = layers[i].bias = 0;
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				_ccv_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols);
				assert(GPU(convnet)->configurations[i].type == layers[i].type);
				GPU(convnet)->configurations[i].w = GPU(convnet)->configurations[i].bias = 0;
				assert(GPU(convnet)->momentums[i].type == layers[i].type);
				GPU(convnet)->momentums[i].w = GPU(convnet)->momentums[i].bias = 0;
				for (j = 0; j < 2; j++)
				{
					GPU(convnet)->contexts[j].denoms[i] = 0;
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
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_forward_propagate<8, 4>, cudaFuncCachePreferShared);
	_cwc_kern_convolutional_forward_propagate
		<8, 4>
		<<<num_blocks, threads_per_block, shared_memory_size + /* need extra space for bias */ sizeof(float) * layer->net.convolutional.count, stream>>>
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
		 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 b, out_rows, out_cols,
		 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count,
		 layer->bias);
}

__global__ void _cwc_kern_convolutional_relu_backward_propagate(const int batch,
		float* out, float* out_grad, const int out_rows, const int out_cols,
		const int count)
{
	assert(gridDim.x == out_rows);
	assert(gridDim.y == out_cols);
	assert(gridDim.z == count);
	assert(blockDim.x == batch);
	out += (blockIdx.z * out_rows * out_cols + blockIdx.x * out_cols + blockIdx.y) * batch;
	out_grad += (blockIdx.z * out_rows * out_cols + blockIdx.x * out_cols + blockIdx.y) * batch;
	if (out[threadIdx.x] <= 0)
		out_grad[threadIdx.x] = 0;
}

template <int channel_per_thread, int filter_per_thread, int batch_per_block, int filter_strides>
__global__ void _cwc_kern_convolutional_backward_propagate_coeff(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff,
		float* filter, const int filter_rows, const int filter_cols, const int count)
{
	// gridDim.x == filter_rows
	// gridDim.y == filter_cols
	assert(gridDim.z == out_rows * batch / batch_per_block);
	extern __shared__ float shared[];
	float* shared_block = &shared[0];
	float* shared_grad = &shared[batch_per_block * channels * filter_strides];
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
						shared_grad[thidx + i * thcnt] = out_grad[x * batch + i * out_loads_factor];
				__syncthreads();
				#pragma unroll
				for (q = start_q; q < end_q; q++)
					#pragma unroll
					for (i = 0; i < channel_per_thread; i++)
						#pragma unroll
						for (j = 0; j < filter_per_thread; j++)
							#pragma unroll
							for (k = 0; k < batch_per_block; k++)
								prod[q][i][j] += shared_block[q * batch_per_block * channels + (i + channel_idx) * batch_per_block + k] * shared_grad[(j + filter_idx) * batch_per_block + k];
				__syncthreads();
			}
		}
	}
	coeff += (blockIdx.x * filter_cols + blockIdx.y * filter_strides) * count + blockIdx.z * filter_rows * filter_cols * count * channels;
	const int coeffcnt = filter_rows * filter_cols * count;
	#pragma unroll
	for (q = 0; q < max_q; q++)
		#pragma unroll
		for (i = 0; i < channel_per_thread; i++)
			#pragma unroll
			for (j = 0; j < filter_per_thread; j++)
				coeff[(i + channel_idx) * coeffcnt + q * count + j + filter_idx] = prod[q][i][j];
}

template <int out_per_thread>
__global__ void _cwc_kern_convolutional_backward_propagate_bias(const int batch,
		float* out_grad, const int out_rows, const int out_cols,
		float* bias, const int count)
{
	// gridDim.x == count
	assert(gridDim.x == count);
	const int skip_pixels = blockDim.y;
	extern __shared__ float shared[];
	float* shared_bias = &shared[0];
	float* shared_grad = &shared[1];
	int i, x;
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	const int out_loads = (batch * skip_pixels + thcnt - 1) / thcnt;
	assert(thcnt % batch == 0);
	out_grad += blockIdx.x * out_rows * out_cols * batch + thidx;
	const int out_load_factor = thcnt;
	const int out_load_pixels = thcnt / batch;
	if (thidx == 0)
		shared_bias[0] = 0;
	for (x = 0; x < out_rows * out_cols; x += skip_pixels)
	{
		for (i = 0; i < out_loads; i++)
			if (i * thcnt + thidx < batch * skip_pixels && x + i * out_load_pixels < out_rows * out_cols)
				shared_grad[i * thcnt + thidx] = out_grad[x * batch + i * out_load_factor];
		__syncthreads();
		// because I branched out with threadIdx, therefore, synchronization must happen outside of the if clause
		if (threadIdx.y + x < out_rows * out_cols)
		{
			#pragma unroll
			for (i = 1; i < out_per_thread; i++)
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
		float* out_grad, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count)
{
	// gridDim.x = rows
	// gridDim.y = cols
	extern __shared__ float shared[];
	float* shared_grad = &shared[0];
	float* shared_weights = &shared[batch];
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
				const int k_idx = k % filter_per_iteration;
				if (k_idx == 0)
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
				float* out_grad_per_filter = out_grad + k * outcnt;
				#pragma unroll
				for (i = 0; i < input_loads; i++)
					if (i * thcnt + thidx < batch)
						shared_grad[i * thcnt + thidx] = out_grad_per_filter[c * batch + i * thcnt + thidx];
				__syncthreads();
				#pragma unroll
				for (i = 0; i < input_per_thread; i++)
					#pragma unroll
					for (j = 0; j < channel_per_thread; j++)
						prod[i][j] += shared_grad[i + threadIdx.x * input_per_thread] * shared_weights[(j + threadIdx.y * channel_per_thread) * filter_per_iteration + k_idx];
				__syncthreads();
			}
		}
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
	assert(batch % 16 == 0);
	int out_rows, out_cols, shared_memory_size;
	_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	// it turns out that first apply relu would save us a lot of computation because no need to low both out and out_grad any more
	_cwc_kern_convolutional_relu_backward_propagate
	<<<dim3(out_rows, out_cols, layer->net.convolutional.count), batch, 0, stream>>>
	(batch, n, a, out_rows, out_cols, layer->net.convolutional.count);
	assert(cudaGetLastError() == cudaSuccess);
	if (layer->input.matrix.channels == 3)
	{
		dim3 threads_per_block_for_coeff(layer->input.matrix.channels, layer->net.convolutional.count);
		if (layer->net.convolutional.cols <= 5)
		{
			dim3 num_blocks_for_coeff(layer->net.convolutional.rows, (layer->net.convolutional.cols + 4) / 5, out_rows * batch / CWC_COEFF_SPREAD);
			shared_memory_size = sizeof(float) * (CWC_COEFF_SPREAD * (layer->input.matrix.channels * 5 + layer->net.convolutional.count));
			cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coeff<1, 1, CWC_COEFF_SPREAD, 5>, cudaFuncCachePreferShared);
			_cwc_kern_convolutional_backward_propagate_coeff
			<1, 1, CWC_COEFF_SPREAD, 5>
			<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>>
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
				m, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
				a, out_rows, out_cols,
				configuration->w,
				layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
		} else {
			dim3 num_blocks_for_coeff(layer->net.convolutional.rows, (layer->net.convolutional.cols + 5) / 6, out_rows * batch / CWC_COEFF_SPREAD);
			shared_memory_size = sizeof(float) * (CWC_COEFF_SPREAD * (layer->input.matrix.channels * 6 + layer->net.convolutional.count));
			cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coeff<1, 1, CWC_COEFF_SPREAD, 6>, cudaFuncCachePreferShared);
			_cwc_kern_convolutional_backward_propagate_coeff
			<1, 1, CWC_COEFF_SPREAD, 6>
			<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>>
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
				m, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
				a, out_rows, out_cols,
				configuration->w,
				layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
		}
	} else if (layer->net.convolutional.count * layer->input.matrix.channels <= 1024 * 8) {
		dim3 threads_per_block_for_coeff(layer->input.matrix.channels / 2, layer->net.convolutional.count / 4);
		if (layer->net.convolutional.cols <= 5)
		{
			dim3 num_blocks_for_coeff(layer->net.convolutional.rows, (layer->net.convolutional.cols + 4) / 5, out_rows * batch / CWC_COEFF_SPREAD);
			shared_memory_size = sizeof(float) * (CWC_COEFF_SPREAD * (layer->input.matrix.channels * 5 + layer->net.convolutional.count));
			cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coeff<2, 4, CWC_COEFF_SPREAD, 5>, cudaFuncCachePreferShared);
			_cwc_kern_convolutional_backward_propagate_coeff
			<2, 4, CWC_COEFF_SPREAD, 5>
			<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>>
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
				m, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
				a, out_rows, out_cols,
				configuration->w,
				layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
		} else {
			dim3 num_blocks_for_coeff(layer->net.convolutional.rows, (layer->net.convolutional.cols + 5) / 6, out_rows * batch / CWC_COEFF_SPREAD);
			shared_memory_size = sizeof(float) * (CWC_COEFF_SPREAD * (layer->input.matrix.channels * 6 + layer->net.convolutional.count));
			cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coeff<2, 4, CWC_COEFF_SPREAD, 6>, cudaFuncCachePreferShared);
			_cwc_kern_convolutional_backward_propagate_coeff
			<2, 4, CWC_COEFF_SPREAD, 6>
			<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>>
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
				m, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
				a, out_rows, out_cols,
				configuration->w,
				layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
		}
	}
	dim3 threads_per_block_for_bias(batch / 8, 8);
	dim3 num_blocks_for_bias(layer->net.convolutional.count);
	shared_memory_size = sizeof(float) * (1 + batch * 8);
	_cwc_kern_convolutional_backward_propagate_bias
	<8>
	<<<num_blocks_for_bias, threads_per_block_for_bias, shared_memory_size, stream>>>
	(batch,
		a, out_rows, out_cols,
		configuration->bias, layer->net.convolutional.count);
	assert(cudaGetLastError() == cudaSuccess);
	if (b)
	{
		dim3 threads_per_block(batch / 8, layer->input.matrix.channels / 2);
		dim3 num_blocks(layer->input.matrix.rows, layer->input.matrix.cols);
		shared_memory_size = sizeof(float) * (batch + layer->input.matrix.channels * 16);
		cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate<8, 2, 16>, cudaFuncCachePreferShared);
		_cwc_kern_convolutional_backward_propagate
		<8, 2, 16>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
		 b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 a, out_rows, out_cols,
		 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
		assert(cudaGetLastError() == cudaSuccess);
	}
}

template <int input_per_thread, int size>
__global__ void _cwc_kern_rnorm_forward_propagate(const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, float* denoms, const float kappa, const float alpha, const float beta)
{
	assert(gridDim.x == rows);
	assert(gridDim.y == cols);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	const int way = size / 2;
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	int i, j, c;
	float prod[input_per_thread];
	const int incnt = rows * cols * batch;
	input += (blockIdx.x * cols + blockIdx.y) * batch;
	out += (blockIdx.x * cols + blockIdx.y) * batch;
	denoms += (blockIdx.x * cols + blockIdx.y) * batch;
	const int end_way = min(way, channels - 1);
	for (c = 0; c < end_way; c++)
	{
		#pragma unroll
		for (i = 0; i < input_loads; i++)
			if (i * thcnt + thidx < batch)
				shared_input[c * batch + i * thcnt + thidx] = input[i * thcnt + thidx];
		input += incnt;
	}
	for (c = 0; c < channels; c++)
	{
		const int start_way = max(c - way, 0);
		const int end_way = min(c + way, channels - 1);
		if (c + way < channels)
		{
			#pragma unroll
			for (i = 0; i < input_loads; i++)
				if (i * thcnt + thidx < batch)
					shared_input[(end_way % size) * batch + i * thcnt + thidx] = input[i * thcnt + thidx];
			input += incnt;
		}
		__syncthreads();
		#pragma unroll
		for (i = 0; i < input_per_thread; i++)
			prod[i] = 0;
		#pragma unroll
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

static void _cwc_convnet_rnorm_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, float* denoms, const cudaStream_t& stream)
{
	dim3 num_blocks(layer->input.matrix.rows, layer->input.matrix.cols);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch * layer->net.rnorm.size;
	if (layer->net.rnorm.size == 3)
	{
		cudaFuncSetCacheConfig(_cwc_kern_rnorm_forward_propagate<1, 3>, cudaFuncCachePreferShared);
		_cwc_kern_rnorm_forward_propagate
		<1, 3>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(batch,
		 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 b, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	} else if (layer->net.rnorm.size == 5) {
		cudaFuncSetCacheConfig(_cwc_kern_rnorm_forward_propagate<1, 5>, cudaFuncCachePreferShared);
		_cwc_kern_rnorm_forward_propagate
		<1, 5>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(batch,
		 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 b, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	}
}

template <int input_per_thread, int size>
__global__ void _cwc_kern_rnorm_backward_propagate(const int batch,
		float* input, float* input_grad, const int rows, const int cols, const int channels,
		float* out, float* out_grad, float* denoms, const float kappa, const float alpha, const float beta)
{
	assert(gridDim.x == rows);
	assert(gridDim.y == cols);
	extern __shared__ float shared[];
	float* shared_out_grad = &shared[0];
	float* shared_out = &shared[batch * size];
	float* shared_denoms = &shared[batch * size * 2];
	float* shared_input = &shared[batch * size * 3];
	const int way = size / 2;
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	int i, j, c;
	float prod[input_per_thread];
	const int incnt = rows * cols * batch;
	out += (blockIdx.x * cols + blockIdx.y) * batch;
	out_grad += (blockIdx.x * cols + blockIdx.y) * batch;
	denoms += (blockIdx.x * cols + blockIdx.y) * batch;
	input += (blockIdx.x * cols + blockIdx.y) * batch;
	input_grad += (blockIdx.x * cols + blockIdx.y) * batch;
	const int end_way = min(way, channels - 1);
	for (c = 0; c < end_way; c++)
	{
		#pragma unroll
		for (i = 0; i < input_loads; i++)
			if (i * thcnt + thidx < batch)
				shared_out_grad[c * batch + i * thcnt + thidx] = out_grad[i * thcnt + thidx],
				shared_out[c * batch + i * thcnt + thidx] = out[i * thcnt + thidx],
				shared_denoms[c * batch + i * thcnt + thidx] = denoms[i * thcnt + thidx];
		out_grad += incnt;
		out += incnt;
		denoms += incnt;
	}
	for (c = 0; c < channels; c++)
	{
		const int start_way = max(c - way, 0);
		const int end_way = min(c + way, channels - 1);
		if (c + way < channels)
		{
			#pragma unroll
			for (i = 0; i < input_loads; i++)
				if (i * thcnt + thidx < batch)
					shared_out_grad[(end_way % size) * batch + i * thcnt + thidx] = out_grad[i * thcnt + thidx],
					shared_out[(end_way % size) * batch + i * thcnt + thidx] = out[i * thcnt + thidx],
					shared_denoms[(end_way % size) * batch + i * thcnt + thidx] = denoms[i * thcnt + thidx];
			out_grad += incnt;
			out += incnt;
			denoms += incnt;
		}
		for (i = 0; i < input_loads; i++)
			if (i * thcnt + thidx < batch)
				shared_input[i * thcnt + thidx] = input[i * thcnt + thidx],
		__syncthreads();
		#pragma unroll
		for (i = 0; i < input_per_thread; i++)
			prod[i] = 0;
		#pragma unroll
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

static void _cwc_convnet_rnorm_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* denoms, float* b, const cudaStream_t& stream)
{
	dim3 num_blocks(layer->input.matrix.rows, layer->input.matrix.cols);
	dim3 threads_per_block(batch);
	int shared_memory_size = sizeof(float) * batch * (layer->net.rnorm.size * 3 + 1);
	if (layer->net.rnorm.size == 3)
	{
		cudaFuncSetCacheConfig(_cwc_kern_rnorm_backward_propagate<1, 3>, cudaFuncCachePreferShared);
		_cwc_kern_rnorm_backward_propagate
		<1, 3>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(batch,
		 m, b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 n, a, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	} else if (layer->net.rnorm.size == 5) {
		cudaFuncSetCacheConfig(_cwc_kern_rnorm_backward_propagate<1, 5>, cudaFuncCachePreferShared);
		_cwc_kern_rnorm_backward_propagate
		<1, 5>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(batch,
		 m, b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
		 n, a, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
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
				// we have to do direct comparison otherwise it will contribute to too many cells
				// and the propagation won't work. But CPU will have different result comparing with GPU
				if (shared_out[i + threadIdx.x * input_per_thread] == shared_input[i + threadIdx.x * input_per_thread])
					prod[i] += shared_grad[i + threadIdx.x * input_per_thread];
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
	int shared_memory_size = sizeof(float) * batch;
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
	for (i = 1; i < count; i++)
	{
		shared[thidx] = a[i * batch + thidx];
		if (shared[thidx] > max_val)
			max_val = shared[thidx];
	}
	float val = 0;
	for (i = 0; i < count; i++)
	{
		shared[thidx] = a[i * batch + thidx];
		val += (shared[thidx] = expf(shared[thidx] - max_val));
		a[i * batch + thidx] = shared[thidx];
	}
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

template <int input_per_thread>
__global__ void _cwc_kern_convnet_tests_return(const int batch, const int count, float* a, int* c)
{
	int i;
	const int thidx = threadIdx.x;
	float max_val = a[thidx];
	int max_idx = 0;
	for (i = 1; i < count; i++)
	{
		float val = a[i * batch + thidx];
		if (val > max_val)
			max_val = val, max_idx = i;
	}
	c[thidx] = max_idx;
}

static void _cwc_convnet_tests_return(int batch, int count, float* a, int* c, const cudaStream_t& stream)
{
	dim3 num_blocks(1);
	dim3 threads_per_block(batch);
	_cwc_kern_convnet_tests_return
	<1>
	<<<num_blocks, threads_per_block, 0, stream>>>
	(batch, count, a, c);
}

template <int momentum_read>
__global__ void _cwc_kern_convolutional_descent_coeff(float* coeff, float* grad, float* momentum,
		const int count, const int strides,
		const float learn_rate, const float momentum_rate, const float decay)
{
	if (blockIdx.x * blockDim.x + threadIdx.x < count)
	{
		coeff += blockIdx.x * blockDim.x;
		grad += blockIdx.x * blockDim.x;
		momentum += blockIdx.x * blockDim.x;
		int i;
		float grad_sum = 0;
		const int thidx = threadIdx.x;
		#pragma unroll
		for (i = 0; i < strides; i++)
			grad_sum += grad[i * count + thidx];
		float old_coeff = coeff[thidx];
		float velocity = (momentum_read ? momentum_rate * momentum[thidx] : 0) - decay * learn_rate * old_coeff + learn_rate * grad_sum;
		coeff[thidx] = velocity + old_coeff;
		momentum[thidx] = velocity;
	}
}

template <int momentum_read>
__global__ void _cwc_kern_net_descent(float* a, float* grad, float* momentum,
		const int count,
		const float learn_rate, const float momentum_rate, const float decay)
{
	if (blockIdx.x * blockDim.x + threadIdx.x < count)
	{
		a += blockIdx.x * blockDim.x;
		grad += blockIdx.x * blockDim.x;
		momentum += blockIdx.x * blockDim.x;
		const int thidx = threadIdx.x;
		float old_a = a[thidx];
		float velocity = (momentum_read ? momentum_rate * momentum[thidx] : 0) - decay * learn_rate * old_a + learn_rate * grad[thidx];
		a[thidx] = velocity + old_a;
		momentum[thidx] = velocity;
	}
}

static void _cwc_convnet_net_descent(ccv_convnet_t* convnet, int momentum_read, int batch, float learn_rate, float momentum_rate, float decay, cwc_convnet_context_t* context)
{
	int i, out_rows, out_cols;
	dim3 threads_per_block(128);
	dim3 num_blocks_for_coeff;
	dim3 num_blocks_for_bias;
	int batch_per_block = batch / CWC_COEFF_SPREAD;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
		ccv_convnet_layer_t* configuration = GPU(convnet)->configurations + i;
		ccv_convnet_layer_t* momentum = GPU(convnet)->momentums + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
				num_blocks_for_coeff = (layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->net.convolutional.channels + 127) / 128;
				num_blocks_for_bias = (layer->net.convolutional.count + 127) / 128;
				if (momentum_read)
				{
					_cwc_kern_convolutional_descent_coeff
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum, out_rows * batch_per_block, learn_rate, momentum_rate, decay);
					_cwc_kern_net_descent
					<1>
					<<<num_blocks_for_bias, threads_per_block, 0, context->stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count, learn_rate, momentum_rate, decay);
				} else {
					_cwc_kern_convolutional_descent_coeff
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum, out_rows * batch_per_block, learn_rate, momentum_rate, decay);
					_cwc_kern_net_descent
					<0>
					<<<num_blocks_for_bias, threads_per_block, 0, context->stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count, learn_rate, momentum_rate, decay);
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				// assume coeff and bias in the same continuous memory region
				num_blocks_for_coeff = (layer->wnum + layer->net.full_connect.count + 127) / 128;
				if (momentum_read)
				{
					_cwc_kern_net_descent
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum + layer->net.full_connect.count, learn_rate, momentum_rate, decay);
				} else {
					_cwc_kern_net_descent
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum + layer->net.full_connect.count, learn_rate, momentum_rate, decay);
				}
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				break;
		}
	}
}

static void _cwc_convnet_batch_formation(ccv_array_t* categorizeds, int* idx, int rows, int cols, int channels, int batch, int offset, int size, float* b, int* c)
{
	int i, k, x;
	assert(size <= batch);
	for (i = 0; i < size; i++)
	{
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, idx ? idx[offset + i] : offset + i);
		if (c)
			c[i] = categorized->c;
		switch (categorized->type)
		{
			case CCV_CATEGORIZED_DENSE_MATRIX:
				for (k = 0; k < channels; k++)
					for (x = 0; x < rows * cols; x++)
						b[(k * rows * cols + x) * batch + i] = categorized->matrix->data.f32[x * channels + k];
				break;
			case CCV_CATEGORIZED_FILE:
				// TODO: implement this support
				break;
		}
	}
}

// assuming a is in device memory
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
				assert(i > 0);
				_cwc_convnet_full_connect_forward_propagate(layer, batch, context->forwards[i - 1], context->forwards[i], context->batch_unit, context->cublas);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				assert(i > 0);
				_cwc_convnet_rnorm_forward_propagate(layer, batch, context->forwards[i - 1], context->forwards[i], context->denoms[i], context->stream);
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

static void _cwc_convnet_backwards_propagate_error(ccv_convnet_t* convnet, float* a, float* m, int batch, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	for (i = convnet->count - 1; i >= 0; i--)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
		ccv_convnet_layer_t* configuration = GPU(convnet)->configurations + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_cwc_convnet_convolutional_backward_propagate(layer, batch, i == convnet->count - 1 ? a : context->backwards[i + 1], context->forwards[i], i > 0 ? context->forwards[i - 1] : m, context->backwards[i], configuration, context->stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_cwc_convnet_full_connect_backward_propagate(layer, batch,  i == convnet->count - 1 ? a : context->backwards[i + 1], i > 0 ? context->forwards[i - 1] : m, context->backwards[i], context->batch_unit, configuration, context->cublas);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				_cwc_convnet_rnorm_backward_propagate(layer, batch, i == convnet->count - 1 ? a : context->backwards[i + 1], context->forwards[i], i > 0 ? context->forwards[i - 1] : m, context->denoms[i], context->backwards[i], context->stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_MAX_POOL:
				_cwc_convnet_max_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : context->backwards[i + 1], context->forwards[i], i > 0 ? context->forwards[i - 1] : m, context->backwards[i], context->stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_cwc_convnet_average_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : context->backwards[i + 1], context->backwards[i], context->stream);
				assert(cudaGetLastError() == cudaSuccess);
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
	int i, j, t;
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
	int* test_returns_on_host = 0;
	cudaMallocHost(&test_returns_on_host, sizeof(int) * tests->rnum);
	int* test_returns_on_device = 0;
	cudaMalloc(&test_returns_on_device, sizeof(int) * ((tests->rnum + params.mini_batch - 1) / params.mini_batch * params.mini_batch));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for (t = 0; t < params.max_epoch; t++)
	{
		cudaEventRecord(start, 0);
		// run updates
		for (i = 0; i < aligned_batches; i++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + (i % 2);
			float* current_input_batch = input_batch_on_host[i % 2];
			int* current_c = c_on_host[i % 2];
			_cwc_convnet_batch_formation(categorizeds, idx, convnet->rows, convnet->cols, convnet->channels, params.mini_batch, i * params.mini_batch, params.mini_batch, current_input_batch, current_c);
			FLUSH(" - at epoch %03d / %d => stochastic gradient descent at %d / %d", t + 1, params.max_epoch, i + 1, aligned_batches);
			cudaMemcpyAsync(input_batch_on_device[i % 2], current_input_batch, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->stream);
			assert(cudaGetLastError() == cudaSuccess);
			cudaMemcpyAsync(c_on_device[i % 2], current_c, sizeof(int) * params.mini_batch, cudaMemcpyHostToDevice, context->stream);
			assert(cudaGetLastError() == cudaSuccess);
			// sync with the other stream core so that we can compute on the single true layer parameters
			cudaStreamSynchronize(GPU(convnet)->contexts[(i + 1) % 2].stream);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_encode_impl(convnet, input_batch_on_device[i % 2], params.mini_batch, context);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_softmax_with_logistic_loss(params.mini_batch, category_count, context->forwards[convnet->count - 1], c_on_device[i % 2], context->stream);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_backwards_propagate_error(convnet, context->forwards[convnet->count - 1], input_batch_on_device[i % 2], params.mini_batch, context);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_net_descent(convnet, i > 0, params.mini_batch, params.learn_rate, params.momentum, params.decay, context);
			assert(cudaGetLastError() == cudaSuccess);
		}
		cudaDeviceSynchronize(); // synchronize at this point
		// run tests
		for (i = j = 0; i < tests->rnum; i += params.mini_batch, j++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + (j % 2);
			float* current_input_batch = input_batch_on_host[j % 2];
			_cwc_convnet_batch_formation(tests, 0, convnet->rows, convnet->cols, convnet->channels, params.mini_batch, i, ccv_min(params.mini_batch, tests->rnum - i), current_input_batch, 0);
			cudaMemcpyAsync(input_batch_on_device[j % 2], current_input_batch, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->stream);
			assert(cudaGetLastError() == cudaSuccess);
			// sync with the other stream core so that we can compute on the single true layer parameters
			cudaStreamSynchronize(GPU(convnet)->contexts[(j + 1) % 2].stream);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_encode_impl(convnet, input_batch_on_device[j % 2], params.mini_batch, context);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_tests_return(params.mini_batch, category_count, context->forwards[convnet->count - 1], test_returns_on_device + i, context->stream);
		}
		cudaDeviceSynchronize(); // synchronize at this point
		cudaMemcpy(test_returns_on_host, test_returns_on_device, sizeof(int) * tests->rnum, cudaMemcpyDeviceToHost);
		int miss = 0;
		for (i = 0; i < tests->rnum; i++)
		{
			ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, i);
			if (test->c != test_returns_on_host[i])
				++miss;
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsed_time = 0;
		cudaEventElapsedTime(&elapsed_time, start, stop);
		FLUSH(" - at epoch %03d / %d => with miss rate %.2f%% (%.3f sec)\n", t + 1, params.max_epoch, miss * 100.0f / tests->rnum, elapsed_time / 1000);
		if (t + 1 < params.max_epoch)
		{
			// reshuffle the parts we visited and move the rest to the beginning
			memcpy(idx + categorizeds->rnum, idx + aligned_rnum, sizeof(int) * aligned_padding);
			memmove(idx + aligned_padding, idx, sizeof(int) * aligned_rnum);
			memcpy(idx, idx + categorizeds->rnum, sizeof(int) * aligned_padding);
			gsl_ran_shuffle(rng, idx + aligned_padding, aligned_rnum, sizeof(int));
		}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	for (i = 0; i < 2; i++)
		cudaFree(input_batch_on_device[i]);
	for (i = 0; i < 2; i++)
		cudaFreeHost(input_batch_on_host[i]);
	for (i = 0; i < 2; i++)
		cudaFree(c_on_device[i]);
	for (i = 0; i < 2; i++)
		cudaFreeHost(c_on_host[i]);
	cudaFree(test_returns_on_device);
	cudaFreeHost(test_returns_on_host);
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
