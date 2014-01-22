extern "C" {
#include "cwc.h"
#include "../ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
}
#include <cuda.h>
#include <cublas_v2.h>

// this structure holds intermediate on-device memory representation of convnet

typedef struct {
	// on host
	struct {
		float* input; // input per batch
		int* c; // class
		float** dor; // dropout regulator, in this version I generate dor on CPU because it is lightweight and gsl has shuffle method, which is better suited for this (and faster than per-node randomization)
	} host;
	// on device
	struct {
		cudaStream_t stream;
		cublasHandle_t cublas;
		float* input;
		int* c;
		float** dor;
	} device;
} cwc_convnet_context_t;

typedef struct {
	size_t memory_usage;
} cwc_convnet_stats_t;

typedef struct {
	int batch;
	ccv_convnet_layer_t* layers;
	ccv_convnet_layer_t* configurations;
	ccv_convnet_layer_t* momentums;
	float** forwards; // the forward output layers
	float** backwards; // the backwards output layer
	float** denoms; // the denominator for rnorm layer, thus, backprop can reuse the value
	float* unit; // the unit vector for a batch, ease the GEMM on full-connect layer
	float* scratch; // the scratch space for temporary reuse, it will be max(wnum, input rows * cols * channels + output rows * cols * channels)
	cwc_convnet_context_t contexts[2];
	cwc_convnet_stats_t device;
} cwc_convnet_t;

#define GPU(x) ((cwc_convnet_t*)((x)->reserved))
#define BATCH_PER_BLOCK (8)

inline static void _cwc_convnet_layer_deduce_output_format(ccv_convnet_layer_t* layer, int* rows, int* cols)
{
	assert(rows != 0 && cols != 0);
	switch(layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			assert(layer->net.convolutional.rows % 2); // as of now, don't support even number of kernel size
			assert(layer->net.convolutional.cols % 2);
			assert((layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) % layer->net.convolutional.strides == 0);
			assert((layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) % layer->net.convolutional.strides == 0);
			*rows = (layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
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
			*rows = (layer->input.matrix.rows + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			break;
	}
}

static int _cwc_convnet_layer_use_multi_way(ccv_convnet_layer_t* layer)
{
	return layer->input.matrix.channels <= 8;
}

static void _cwc_convnet_reorder_convolutional_weights_onto_device(float* w, float* ow, int wnum, int filters, int channels)
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

static void _cwc_convnet_reorder_full_connect_weights_onto_device(float* w, float* ow, int wnum, int count, int channels)
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

static void _cwc_convnet_alloc_reserved(ccv_convnet_t* convnet, int batch, ccv_convnet_layer_train_param_t* layer_params)
{
	if (GPU(convnet) && GPU(convnet)->batch != batch)
		ccv_convnet_compact(convnet);
	else if (GPU(convnet))
		return; // it is allocated properly, no-op
	convnet->reserved = (cwc_convnet_t*)ccmalloc(sizeof(cwc_convnet_t) + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 10);
	GPU(convnet)->batch = batch;
	GPU(convnet)->device.memory_usage = 0;
	GPU(convnet)->layers = (ccv_convnet_layer_t*)(GPU(convnet) + 1);
	int i, j, out_rows, out_cols;
	memcpy(GPU(convnet)->layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	ccv_convnet_layer_t* layers = GPU(convnet)->layers;
	// configurations (the backprop coeffs)
	size_t scratch_space = 0;
	size_t unit_size = batch;
	for (i = 0; i < convnet->count; i++)
		if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
		{
			int use_multi_way = _cwc_convnet_layer_use_multi_way(layers + i);
			_cwc_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols);
			scratch_space = ccv_max(scratch_space, layers[i].wnum);
			scratch_space = ccv_max(scratch_space,
					out_rows * out_cols * layers[i].net.convolutional.count * batch + // output layer reorder
					layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch + // input layer reorder
					layers[i].net.convolutional.count * layers[i].input.matrix.channels * layers[i].net.convolutional.rows * layers[i].net.convolutional.cols * (use_multi_way ? out_rows : 1) * (batch / BATCH_PER_BLOCK)); // unconsolidated weights output
			if (use_multi_way)
				unit_size = ccv_max(unit_size, out_rows * (batch / BATCH_PER_BLOCK));
		}
	GPU(convnet)->scratch = 0;
	cudaMalloc(&GPU(convnet)->scratch, sizeof(float) * scratch_space);
	assert(GPU(convnet)->scratch);
	GPU(convnet)->device.memory_usage += sizeof(float) * scratch_space;
	float* unit = 0;
	cudaMallocHost(&unit, sizeof(float) * unit_size);
	for (i = 0; i < unit_size; i++)
		unit[i] = 1;
	GPU(convnet)->unit = 0;
	cudaMalloc(&GPU(convnet)->unit, sizeof(float) * unit_size);
	GPU(convnet)->device.memory_usage += sizeof(float) * unit_size;
	cudaMemcpy(GPU(convnet)->unit, unit, sizeof(float) * unit_size, cudaMemcpyHostToDevice);
	cudaFreeHost(unit);
	GPU(convnet)->configurations = GPU(convnet)->layers + convnet->count;
	memcpy(GPU(convnet)->configurations, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	GPU(convnet)->momentums = GPU(convnet)->layers + convnet->count * 2;
	memcpy(GPU(convnet)->momentums, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	GPU(convnet)->forwards = (float**)(GPU(convnet)->layers + convnet->count * 3);
	GPU(convnet)->backwards = (float**)(GPU(convnet)->layers + convnet->count * 3) + convnet->count;
	GPU(convnet)->denoms = (float**)(GPU(convnet)->layers + convnet->count * 3) + convnet->count * 2;
	for (i = 0; i < 2; i++)
	{
		cwc_convnet_context_t* context = GPU(convnet)->contexts + i;
		context->host.dor = (float**)(GPU(convnet)->layers + convnet->count * 3) + convnet->count * 3 + convnet->count * i;
		context->device.dor = (float**)(GPU(convnet)->layers + convnet->count * 3) + convnet->count * 5 + convnet->count * i;
		context->host.input = 0;
		cudaMallocHost(&context->host.input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch); 
		assert(context->host.input);
		context->device.input = 0;
		cudaMalloc(&context->device.input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch);
		GPU(convnet)->device.memory_usage += sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch;
		assert(context->device.input);
		context->host.c = 0;
		cudaMallocHost(&context->host.c, sizeof(int) * batch); 
		assert(context->host.c);
		context->device.c = 0;
		cudaMalloc(&context->device.c, sizeof(int) * batch); 
		GPU(convnet)->device.memory_usage += sizeof(int) * batch;
		cudaStreamCreate(&context->device.stream);
		cublasCreate(&context->device.cublas);
		cublasSetStream(context->device.cublas, context->device.stream);
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
				GPU(convnet)->device.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count);
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				_cwc_convnet_reorder_convolutional_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].net.convolutional.count, layers[i].net.convolutional.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.convolutional.count, cudaMemcpyHostToDevice);
				_cwc_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols);
				// allocating for configurations 
				GPU(convnet)->configurations[i].w = 0;
				cudaMalloc(&GPU(convnet)->configurations[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				GPU(convnet)->device.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count);
				assert(GPU(convnet)->configurations[i].w);
				GPU(convnet)->configurations[i].bias = GPU(convnet)->configurations[i].w + layers[i].wnum;
				// allocating for momentums
				GPU(convnet)->momentums[i].w = 0;
				cudaMalloc(&GPU(convnet)->momentums[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				GPU(convnet)->device.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count);
				assert(GPU(convnet)->momentums[i].w);
				GPU(convnet)->momentums[i].bias = GPU(convnet)->momentums[i].w + layers[i].wnum;
				GPU(convnet)->denoms[i] = 0;
				GPU(convnet)->forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->forwards[i], sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * batch;
				assert(GPU(convnet)->forwards[i]);
				GPU(convnet)->backwards[i] = 0;
				if (i > 0) // if it is the input layer, no need to backprop to outmost one
				{
					cudaMalloc(&GPU(convnet)->backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
					GPU(convnet)->device.memory_usage += sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch;
					assert(GPU(convnet)->backwards[i]);
				}
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					context->host.dor[i] = 0;
					context->device.dor[i] = 0;
					if (layer_params[i].dor > 0)
					{
						assert(i > 0);
						cudaMallocHost(&context->host.dor[i], sizeof(float) * batch * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels);
						assert(context->host.dor[i]);
						cudaMalloc(&context->device.dor[i], sizeof(float) * batch * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels);
						GPU(convnet)->device.memory_usage += sizeof(float) * batch * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels;
						assert(context->device.dor[i]);
					}
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				assert(GPU(convnet)->configurations[i].type == CCV_CONVNET_FULL_CONNECT);
				assert(GPU(convnet)->momentums[i].type == CCV_CONVNET_FULL_CONNECT);
				// allocating for layer
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->device.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count);
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				_cwc_convnet_reorder_full_connect_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].input.matrix.rows * layers[i].input.matrix.cols, layers[i].input.matrix.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.full_connect.count, cudaMemcpyHostToDevice);
				// allocating for configurations 
				GPU(convnet)->configurations[i].w = 0;
				cudaMalloc(&GPU(convnet)->configurations[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->device.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count);
				assert(GPU(convnet)->configurations[i].w);
				GPU(convnet)->configurations[i].bias = GPU(convnet)->configurations[i].w + layers[i].wnum;
				// allocating for momentums
				GPU(convnet)->momentums[i].w = 0;
				cudaMalloc(&GPU(convnet)->momentums[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->device.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count);
				assert(GPU(convnet)->momentums[i].w);
				GPU(convnet)->momentums[i].bias = GPU(convnet)->momentums[i].w + layers[i].wnum;
				GPU(convnet)->denoms[i] = 0;
				GPU(convnet)->forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->forwards[i], sizeof(float) * layers[i].net.full_connect.count * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * layers[i].net.full_connect.count * batch;
				assert(GPU(convnet)->forwards[i]);
				GPU(convnet)->backwards[i] = 0;
				cudaMalloc(&GPU(convnet)->backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch;
				assert(GPU(convnet)->backwards[i]);
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					context->host.dor[i] = 0;
					context->device.dor[i] = 0;
					if (layer_params[i].dor > 0)
					{
						cudaMallocHost(&context->host.dor[i], sizeof(float) * batch * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels);
						assert(context->host.dor[i]);
						cudaMalloc(&context->device.dor[i], sizeof(float) * batch * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels);
						GPU(convnet)->device.memory_usage += sizeof(float) * batch * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels;
						assert(context->device.dor[i]);
					}
				}
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				assert(i > 0);
				assert(GPU(convnet)->configurations[i].type == CCV_CONVNET_LOCAL_RESPONSE_NORM);
				assert(GPU(convnet)->momentums[i].type == CCV_CONVNET_LOCAL_RESPONSE_NORM);
				GPU(convnet)->configurations[i].w = GPU(convnet)->configurations[i].bias = 0;
				assert(GPU(convnet)->momentums[i].type == layers[i].type);
				GPU(convnet)->momentums[i].w = GPU(convnet)->momentums[i].bias = 0;
				GPU(convnet)->denoms[i] = 0;
				cudaMalloc(&GPU(convnet)->denoms[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch;
				assert(GPU(convnet)->denoms[i]);
				GPU(convnet)->forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch;
				assert(GPU(convnet)->forwards[i]);
				GPU(convnet)->backwards[i] = 0;
				cudaMalloc(&GPU(convnet)->backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch;
				assert(GPU(convnet)->backwards[i]);
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					context->host.dor[i] = 0;
					context->device.dor[i] = 0;
				}
				layers[i].w = layers[i].bias = 0;
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				_cwc_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols);
				assert(GPU(convnet)->configurations[i].type == layers[i].type);
				GPU(convnet)->configurations[i].w = GPU(convnet)->configurations[i].bias = 0;
				assert(GPU(convnet)->momentums[i].type == layers[i].type);
				GPU(convnet)->momentums[i].w = GPU(convnet)->momentums[i].bias = 0;
				GPU(convnet)->denoms[i] = 0;
				GPU(convnet)->forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch;
				assert(GPU(convnet)->forwards[i]);
				GPU(convnet)->backwards[i] = 0;
				cudaMalloc(&GPU(convnet)->backwards[i], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				GPU(convnet)->device.memory_usage += sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch;
				assert(GPU(convnet)->backwards[i]);
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					context->host.dor[i] = 0;
					context->device.dor[i] = 0;
				}
				layers[i].w = layers[i].bias = 0;
				break;
		}
}

// =========================================== KERNEL CODE ===================================================

template <int input_per_thread, int filter_per_thread, int filter_per_block>
__global__ static void _cwc_kern_convolutional_forward_propagate(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count,
		float* const biases)
{
	assert(gridDim.x * filter_per_block == out_rows * count);
	assert(gridDim.y == out_cols);
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
	const int filter_group_idx = blockIdx.x / out_cols;
	input += (origin_y * strides * cols + origin_x * strides) * batch;
	assert(thcnt >= batch);
	assert(thcnt >= filter_per_block);
	if (thidx < filter_per_block)
		shared_bias[thidx] = biases[filter_group_idx * filter_per_block + thidx];
	const int start_x = max(origin_x * strides - border, 0) - (origin_x * strides - border);
	const int end_x = min(origin_x * strides - border + filter_cols, cols) - (origin_x * strides - border);
	const int start_y = max(origin_y * strides - border, 0) - (origin_y * strides - border);
	const int end_y = min(origin_y * strides - border + filter_rows, rows) - (origin_y * strides - border);
	filter += filter_group_idx * filter_per_block;
	for (c = 0; c < channels; c++)
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
	out += filter_group_idx * filter_per_block * outcnt + (origin_y * out_cols + origin_x) * batch;
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
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	assert(b);
#define vary_block(_x, _y, _z) do { \
		dim3 threads_per_block(batch / _x, _z / _y); \
		assert(threads_per_block.x * threads_per_block.y <= 1024); \
		dim3 num_blocks(out_cols * layer->net.convolutional.count / _z, out_rows); \
		int shared_memory_size = sizeof(float) * (batch + _z * 2); \
		cudaFuncSetCacheConfig(_cwc_kern_convolutional_forward_propagate<_x, _y, _z>, cudaFuncCachePreferShared); \
		_cwc_kern_convolutional_forward_propagate \
			<_x, _y, _z> \
			<<<num_blocks, threads_per_block, shared_memory_size + /* need extra space for bias */ sizeof(float) * layer->net.convolutional.count, stream>>> \
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
			 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
			 b, out_rows, out_cols, \
			 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count, \
			 layer->bias); \
	} while (0)
	if (layer->net.convolutional.count % 72 == 0)
		vary_block(4, 8, 72);
	else if (layer->net.convolutional.count % 36 == 0)
		vary_block(4, 8, 36);
	else if (layer->net.convolutional.count % 32 == 0)
		vary_block(4, 8, 32);
	else
		vary_block(4, 4, 16);
#undef vary_block
	assert(cudaGetLastError() == cudaSuccess);
}

__global__ static void _cwc_kern_convolutional_relu_backward_propagate(const int batch,
		float* out, float* out_grad, const int out_rows, const int out_cols,
		const int count)
{
	assert(gridDim.x == out_cols);
	assert(gridDim.y == out_rows);
	assert(gridDim.z == count);
	assert(blockDim.x == batch);
	out += (blockIdx.z * out_rows * out_cols + blockIdx.y * out_cols + blockIdx.x) * batch;
	out_grad += (blockIdx.z * out_rows * out_cols + blockIdx.y * out_cols + blockIdx.x) * batch;
	if (out[threadIdx.x] <= 0)
		out_grad[threadIdx.x] = 0;
}

template <int channel_per_thread, int filter_per_thread, int channel_per_block, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coeff_default(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count)
{
	assert(gridDim.x == filter_cols);
	assert(gridDim.y == filter_rows);
	assert(gridDim.z * channel_per_block * batch_per_block == channels * batch);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[channel_per_block];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(blockDim.x * filter_per_thread == count);
	assert(blockDim.y * channel_per_thread == channel_per_block);
	assert(thcnt >= channel_per_block);
	assert(thcnt >= count);
	const int origin_x = blockIdx.x;
	const int origin_y = blockIdx.y;
	const int channel_group_count = channels / channel_per_block;
	const int channel_group_idx = blockIdx.z % channel_group_count;
	const int batch_group_idx = blockIdx.z / channel_group_count;
	const int start_x = max(origin_x - border, 0) - (origin_x - border);
	const int end_x = min(out_cols, (cols + border - origin_x + strides - 1) / strides);
	const int start_y = max(origin_y - border, 0) - (origin_y - border);
	const int end_y = min(out_rows, (rows + border - origin_y + strides - 1) / strides);
	input += rows * cols * channels * batch_group_idx * batch_per_block + (origin_y * cols + origin_x) * channels + channel_group_idx * channel_per_block;
	out_grad += out_rows * out_cols * count * batch_group_idx * batch_per_block;
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
				if (thidx < count)
					shared_out_grad[thidx] = out_grad[(y * out_cols + x) * count + thidx];
				if (thidx < channel_per_block)
					shared_input[thidx] = input[((y * strides - border) * cols + x * strides - border) * channels + thidx];
				__syncthreads();
				#pragma unroll
				for (i = 0; i < channel_per_thread; i++)
					#pragma unroll
					for (j = 0; j < filter_per_thread; j++)
						prod[i][j] += shared_input[i + threadIdx.y * channel_per_thread] * shared_out_grad[j + threadIdx.x * filter_per_thread];
				__syncthreads();
			}
		input += rows * cols * channels;
		out_grad += out_rows * out_cols * count;
	}
	const int cocnt = filter_cols * filter_rows * count;
	coeff += cocnt * (channels * batch_group_idx + channel_group_idx * channel_per_block) + (origin_y * filter_cols + origin_x) * count;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			coeff[(i + threadIdx.y * channel_per_thread) * cocnt + j + threadIdx.x * filter_per_thread] = prod[i][j];
}

template <int channel_per_thread, int filter_per_thread, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coeff_multi_way(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count)
{
	assert(gridDim.x == filter_cols);
	assert(gridDim.y == filter_rows);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[channels * batch_per_block];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(blockDim.x * filter_per_thread == count);
	assert(blockDim.y * channel_per_thread == channels);
	assert(thcnt >= channels * batch_per_block);
	assert(thcnt >= count);
	const int origin_x = blockIdx.x;
	const int origin_y = blockIdx.y;
	const int batch_group_idx = blockIdx.z / out_rows;
	const int start_x = max(origin_x - border, 0) - (origin_x - border);
	const int end_x = min(out_cols, (cols + border - origin_x + strides - 1) / strides);
	input += (rows * cols * channels * batch_group_idx + (origin_y * cols + origin_x) * channels) * batch_per_block;
	out_grad += out_rows * out_cols * count * batch_group_idx * batch_per_block;
	int i, j, c, x;
	const int y = blockIdx.z % out_rows;
	float prod[channel_per_thread][filter_per_thread];
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			prod[i][j] = 0;
	const int iy = origin_y + y * strides - border;
	const int chidx = thidx < channels * batch_per_block ? thidx : channels * batch_per_block - 1;
	if (iy >= 0 && iy < rows)
	{
		input += (y * strides - border) * cols * channels * batch_per_block;
		out_grad += y * out_cols * count * batch_per_block;
		for (x = start_x; x < end_x; x++)
		{
			if (thidx < count)
				#pragma unroll
				for (c = 0; c < batch_per_block; c++)
					shared_out_grad[c * count + thidx] = out_grad[x * count * batch_per_block + c * count + thidx];
			shared_input[chidx] = input[(x * strides - border) * channels * batch_per_block + chidx]; // no need for a conditional
			__syncthreads();
			#pragma unroll
			for (i = 0; i < channel_per_thread; i++)
				#pragma unroll
				for (j = 0; j < filter_per_thread; j++)
				{
					float sum = 0;
					#pragma unroll
					for (c = 0; c < batch_per_block; c++)
						sum += shared_input[c * channels + i + threadIdx.y * channel_per_thread] * shared_out_grad[c * count + j + threadIdx.x * filter_per_thread];
					prod[i][j] += sum;
				}
			__syncthreads();
		}
	}
	const int cocnt = filter_cols * filter_rows * count;
	coeff += cocnt * channels * blockIdx.z + (origin_y * filter_cols + origin_x) * count;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			coeff[(i + threadIdx.y * channel_per_thread) * cocnt + j + threadIdx.x * filter_per_thread] = prod[i][j];
}

template <int out_per_thread>
__global__ static void _cwc_kern_convolutional_backward_propagate_bias(const int batch,
		float* out_grad, const int out_rows, const int out_cols,
		float* bias, const int count)
{
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

template <int input_per_thread, int channel_per_thread, int channel_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate(const int strides, const int border, const int batch,
		float* input_grad, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count)
{
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
	const int channel_group_idx = blockIdx.x / cols;
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
	out_grad += (out_start_y * out_cols + out_start_x) * batch;
	const int out_end_y = out_y + ycnt - 1;
	const int out_end_x = out_x + xcnt - 1;
	const int filter_end_y = (origin_x + border) % strides + (out_end_y - min(out_end_y, out_rows - 1)) * strides;
	const int filter_end_x = (origin_y + border) % strides + (out_end_x - min(out_end_x, out_cols - 1)) * strides;
	const int outcnt = out_rows * out_cols * batch;
	filter += channel_group_idx * channel_per_block;
	for (k = 0; k < count; k++)
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
__global__ static void _cwc_kern_reorder_matrix_major(float* a, float* b, const int count, const int channels, const int batch)
{
	b[(threadIdx.x * count + blockIdx.x) * channels + blockIdx.y] = a[(blockIdx.y * count + blockIdx.x) * batch + threadIdx.x];
}

// this method rewinds a matrix
__global__ static void _cwc_kern_reorder_matrix_major_per_block(float* a, float* b, const int count, const int channels, const int batch, const int batch_per_block)
{
	const int thidx = threadIdx.x + threadIdx.y * batch_per_block;
	b[(threadIdx.y * count + blockIdx.x) * channels * batch_per_block + threadIdx.x * channels + blockIdx.y] = a[(blockIdx.y * count + blockIdx.x) * batch + thidx];
}

static void _cwc_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	assert(layer->net.convolutional.count % 4 == 0);
	assert(batch % BATCH_PER_BLOCK == 0);
	int out_rows, out_cols, shared_memory_size;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	// it turns out that first apply relu would save us a lot of computation because no need to low both out and out_grad any more
	_cwc_kern_convolutional_relu_backward_propagate
	<<<dim3(out_cols, out_rows, layer->net.convolutional.count), batch, 0, stream>>>
	(batch, n, a, out_rows, out_cols, layer->net.convolutional.count);
	assert(cudaGetLastError() == cudaSuccess);
	float* chm = scratch;
	float* cha = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch;
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	float alpha = 1, beta = 0;
	int count = layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->input.matrix.channels;
	if (_cwc_convnet_layer_use_multi_way(layer))
	{
		const int batch_group_count = batch / BATCH_PER_BLOCK;
		_cwc_kern_reorder_matrix_major_per_block
		<<<dim3(layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels), dim3(BATCH_PER_BLOCK, batch_group_count), 0, stream>>>
		(m, chm, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels, batch, BATCH_PER_BLOCK);
		_cwc_kern_reorder_matrix_major_per_block
		<<<dim3(out_rows * out_cols, layer->net.convolutional.count), dim3(BATCH_PER_BLOCK, batch_group_count), 0, stream>>>
		(a, cha, out_rows * out_cols, layer->net.convolutional.count, batch, BATCH_PER_BLOCK);
#define vary_block(_x, _y) do { \
			dim3 threads_per_block_for_coeff(layer->net.convolutional.count / _y, layer->input.matrix.channels / _x); \
			assert(threads_per_block_for_coeff.x * threads_per_block_for_coeff.y < 1024); \
			int batch_group_count = batch / BATCH_PER_BLOCK; \
			dim3 num_blocks_for_coeff(layer->net.convolutional.cols, layer->net.convolutional.rows, out_rows * batch_group_count); \
			shared_memory_size = sizeof(float) * BATCH_PER_BLOCK * (layer->input.matrix.channels + layer->net.convolutional.count); \
			cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coeff_multi_way<_x, _y, BATCH_PER_BLOCK>, cudaFuncCachePreferShared); \
			_cwc_kern_convolutional_backward_propagate_coeff_multi_way \
			<_x, _y, BATCH_PER_BLOCK> \
			<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>> \
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
				chm, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
				cha, out_rows, out_cols, \
				cbw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count); \
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, count, 1, out_rows * batch_group_count, &alpha, cbw, count, unit, out_rows * batch_group_count, &beta, configuration->w, count); \
		} while (0)
		// special casing for image
		if (layer->input.matrix.channels == 3)
			vary_block(3, 1);
		else if (layer->net.convolutional.count % 2 == 0)
			vary_block(1, 2);
#undef vary_block
	} else {
		_cwc_kern_reorder_matrix_major
		<<<dim3(layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels), batch, 0, stream>>>
		(m, chm, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels, batch);
		_cwc_kern_reorder_matrix_major
		<<<dim3(out_rows * out_cols, layer->net.convolutional.count), batch, 0, stream>>>
		(a, cha, out_rows * out_cols, layer->net.convolutional.count, batch);
#define vary_block(_x, _y, _z) do { \
			dim3 threads_per_block_for_coeff(layer->net.convolutional.count / _y, _z / _x); \
			assert(threads_per_block_for_coeff.x * threads_per_block_for_coeff.y < 1024); \
			int batch_group_count = batch / BATCH_PER_BLOCK; \
			dim3 num_blocks_for_coeff(layer->net.convolutional.cols, layer->net.convolutional.rows, layer->net.convolutional.channels / _z * batch_group_count); \
			shared_memory_size = sizeof(float) * (_z + layer->net.convolutional.count); \
			_cwc_kern_convolutional_backward_propagate_coeff_default \
			<_x, _y, _z, BATCH_PER_BLOCK> \
			<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>> \
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
				chm, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
				cha, out_rows, out_cols, \
				cbw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count); \
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, count, 1, batch_group_count, &alpha, cbw, count, unit, batch_group_count, &beta, configuration->w, count); \
		} while (0)
		if (layer->input.matrix.channels % 36 == 0)
			vary_block(4, 8, 36);
		else if (layer->input.matrix.channels % 32 == 0)
			vary_block(4, 8, 32);
#undef vary_block
	}
	assert(cudaGetLastError() == cudaSuccess);
	dim3 threads_per_block_for_bias(batch / 16, 16);
	assert(threads_per_block_for_bias.x * threads_per_block_for_bias.y <= 1024);
	dim3 num_blocks_for_bias(layer->net.convolutional.count);
	shared_memory_size = sizeof(float) * (1 + batch * 16);
	_cwc_kern_convolutional_backward_propagate_bias
	<16>
	<<<num_blocks_for_bias, threads_per_block_for_bias, shared_memory_size, stream>>>
	(batch,
		a, out_rows, out_cols,
		configuration->bias, layer->net.convolutional.count);
	assert(cudaGetLastError() == cudaSuccess);
	if (b)
	{
		float* chw = scratch;
		_cwc_kern_reorder_matrix_major
		<<<dim3(layer->net.convolutional.rows * layer->net.convolutional.cols, layer->input.matrix.channels), layer->net.convolutional.count, 0, stream>>>
		(layer->w, chw, layer->net.convolutional.rows * layer->net.convolutional.cols, layer->input.matrix.channels, layer->net.convolutional.count);
#define vary_block(_x, _y, _z) do { \
			dim3 threads_per_block(batch / _x, _z / _y); \
			assert(threads_per_block.x * threads_per_block.y <= 1024); \
			dim3 num_blocks(layer->input.matrix.cols * layer->input.matrix.channels / _z, layer->input.matrix.rows); \
			shared_memory_size = sizeof(float) * (batch + _z); \
			cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate<_x, _y, _z>, cudaFuncCachePreferShared); \
			_cwc_kern_convolutional_backward_propagate \
			<_x, _y, _z> \
			<<<num_blocks, threads_per_block, shared_memory_size, stream>>> \
			(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
			 b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
			 a, out_rows, out_cols, \
			 chw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count); \
		} while (0)
		if (layer->input.matrix.channels % 72 == 0)
			vary_block(4, 8, 72);
		else if (layer->input.matrix.channels % 36 == 0)
			vary_block(4, 8, 36);
		else if (layer->input.matrix.channels % 32 == 0)
			vary_block(4, 8, 32);
#undef vary_block
		assert(cudaGetLastError() == cudaSuccess);
	}
}

template <int input_per_thread, int size>
__global__ static void _cwc_kern_rnorm_forward_propagate(const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, float* denoms, const float kappa, const float alpha, const float beta)
{
	assert(gridDim.x == cols);
	assert(gridDim.y == rows);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	const int way = size / 2;
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	int i, j, c;
	float prod[input_per_thread];
	const int incnt = rows * cols * batch;
	input += (blockIdx.y * cols + blockIdx.x) * batch;
	out += (blockIdx.y * cols + blockIdx.x) * batch;
	denoms += (blockIdx.y * cols + blockIdx.x) * batch;
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
	dim3 num_blocks(layer->input.matrix.cols, layer->input.matrix.rows);
	dim3 threads_per_block(batch);
	assert(threads_per_block.x <= 1024);
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
__global__ static void _cwc_kern_rnorm_backward_propagate(const int batch,
		float* input, float* input_grad, const int rows, const int cols, const int channels,
		float* out, float* out_grad, float* denoms, const float kappa, const float alpha, const float beta)
{
	assert(gridDim.x == cols);
	assert(gridDim.y == rows);
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
	out += (blockIdx.y * cols + blockIdx.x) * batch;
	out_grad += (blockIdx.y * cols + blockIdx.x) * batch;
	denoms += (blockIdx.y * cols + blockIdx.x) * batch;
	input += (blockIdx.y * cols + blockIdx.x) * batch;
	input_grad += (blockIdx.y * cols + blockIdx.x) * batch;
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
	dim3 num_blocks(layer->input.matrix.cols, layer->input.matrix.rows);
	dim3 threads_per_block(batch);
	assert(threads_per_block.x <= 1024);
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
__global__ static void _cwc_kern_max_pool_forward_propagate(const int strides, const int border, const int size, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols)
{
	assert(gridDim.x == out_cols);
	assert(gridDim.y == out_rows);
	assert(gridDim.z == channels);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	assert(thcnt >= batch);
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
			if (thidx < batch)
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

static void _cwc_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
	dim3 num_blocks(out_cols, out_rows, layer->input.matrix.channels);
	dim3 threads_per_block(batch);
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_max_pool_forward_propagate
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(layer->net.pool.strides, layer->net.pool.border, layer->net.pool.size, batch,
	 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels,
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
			if (thidx < batch)
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

static void _cwc_convnet_max_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
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

static void _cwc_convnet_average_pool_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
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

static void _cwc_convnet_average_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
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

static void _cwc_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, float* batch_unit /* this is just 1's in device */, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
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
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
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
__global__ static void _cwc_kern_convnet_softmax_with_logistic_loss(const int batch, const int count, float* a, int* c)
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
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_convnet_softmax_with_logistic_loss
	<1>
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(batch, count, a, c);
}

template <int input_per_thread>
__global__ static void _cwc_kern_convnet_tests_return(const int batch, const int count, float* a, int* c)
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
	assert(threads_per_block.x <= 1024);
	_cwc_kern_convnet_tests_return
	<1>
	<<<num_blocks, threads_per_block, 0, stream>>>
	(batch, count, a, c);
}

template <int momentum_read>
__global__ static void _cwc_kern_net_sgd(float* a, float* grad, float* momentum,
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

static void _cwc_convnet_net_sgd(ccv_convnet_t* convnet, int momentum_read, int batch, ccv_convnet_layer_train_param_t* layer_params, cwc_convnet_context_t* context)
{
	int i, out_rows, out_cols;
	dim3 threads_per_block(128);
	assert(threads_per_block.x <= 1024);
	dim3 num_blocks_for_coeff;
	dim3 num_blocks_for_bias;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
		ccv_convnet_layer_t* configuration = GPU(convnet)->configurations + i;
		ccv_convnet_layer_t* momentum = GPU(convnet)->momentums + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols);
				num_blocks_for_coeff = (layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->net.convolutional.channels + 127) / 128;
				num_blocks_for_bias = (layer->net.convolutional.count + 127) / 128;
				if (momentum_read)
				{
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate, layer_params[i].w.momentum, layer_params[i].w.decay);
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count,
					 layer_params[i].bias.learn_rate, layer_params[i].bias.momentum, layer_params[i].bias.decay);
				} else {
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate, layer_params[i].w.momentum, layer_params[i].w.decay);
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count,
					 layer_params[i].bias.learn_rate, layer_params[i].bias.momentum, layer_params[i].bias.decay);
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				// assume coeff and bias in the same continuous memory region
				num_blocks_for_coeff = (layer->wnum + layer->net.full_connect.count + 127) / 128;
				if (momentum_read)
				{
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate, layer_params[i].w.momentum, layer_params[i].w.decay);
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.full_connect.count,
					 layer_params[i].bias.learn_rate, layer_params[i].bias.momentum, layer_params[i].bias.decay);
				} else {
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate, layer_params[i].w.momentum, layer_params[i].w.decay);
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.full_connect.count,
					 layer_params[i].bias.learn_rate, layer_params[i].bias.momentum, layer_params[i].bias.decay);
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
				assert(rows == categorized->matrix->rows && cols == categorized->matrix->cols && channels == CCV_GET_CHANNEL(categorized->matrix->type));
				for (k = 0; k < channels; k++)
					for (x = 0; x < rows * cols; x++)
						b[(k * rows * cols + x) * batch + i] = categorized->matrix->data.f32[x * channels + k];
				break;
			case CCV_CATEGORIZED_FILE:
			{
				ccv_dense_matrix_t* image = 0;
				ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				ccv_dense_matrix_t* norm = 0;
				if (image->rows > 251 && image->cols > 251)
					ccv_resample(image, &norm, 0, ccv_max(251, (int)(image->rows * 251.0 / image->cols + 0.5)), ccv_max(251, (int)(image->cols * 251.0 / image->rows + 0.5)), CCV_INTER_AREA);
				else if (image->rows < 251 || image->cols < 251)
					ccv_resample(image, &norm, 0, ccv_max(251, (int)(image->rows * 251.0 / image->cols + 0.5)), ccv_max(251, (int)(image->cols * 251.0 / image->rows + 0.5)), CCV_INTER_CUBIC);
				else
					norm = image;
				if (norm != image)
					ccv_matrix_free(image);
				ccv_dense_matrix_t* patch = 0;
				ccv_slice(norm, (ccv_matrix_t**)&patch, CCV_32F, 0, 0, 225, 225);
				ccv_matrix_free(norm);
				assert(channels == CCV_GET_CHANNEL(patch->type));
				for (k = 0; k < channels; k++)
					for (x = 0; x < rows * cols; x++)
						b[(k * rows * cols + x) * batch + i] = patch->data.f32[x * channels + k] / 255.0 * 2 - 1;
				ccv_matrix_free(patch);
				break;
			}
		}
	}
}

static void _cwc_convnet_dor_mean_net(ccv_convnet_t* convnet, ccv_convnet_layer_train_param_t* layer_params, const cublasHandle_t& handle)
{
	int i;
	for (i = 0; i < convnet->count; i++)
		if (layer_params[i].dor > 0)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
			float dor = 1.0 - layer_params[i].dor;
			cublasSscal(handle, layer->wnum, &dor, layer->w, 1);
		}
}

static void _cwc_convnet_dor_mean_net_undo(ccv_convnet_t* convnet, ccv_convnet_layer_train_param_t* layer_params, const cublasHandle_t& handle)
{
	int i;
	for (i = 0; i < convnet->count; i++)
		if (layer_params[i].dor > 0)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
			float inv_dor = 1.0 / (1.0 - layer_params[i].dor);
			cublasSscal(handle, layer->wnum, &inv_dor, layer->w, 1);
		}
}

static void _cwc_convnet_dor_formation(ccv_convnet_t* convnet, int batch, gsl_rng* rng, ccv_convnet_layer_train_param_t* layer_params, cwc_convnet_context_t* context)
{
	int i, j;
	for (i = 0; i < convnet->count; i++)
		if (context->host.dor[i])
		{
			assert(context->device.dor[i]);
			assert(layer_params[i].dor > 0);
			ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
			int count = layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels;
			for (j = 0; j < batch * count; j++)
				context->host.dor[i][j] = (gsl_rng_uniform(rng) >= layer_params[i].dor) ? 1.0 : 0.0;
			cudaMemcpyAsync(context->device.dor[i], context->host.dor[i], sizeof(float) * count * batch, cudaMemcpyHostToDevice, context->device.stream);
			assert(cudaGetLastError() == cudaSuccess);
		}
}

__global__ static void _cwc_kern_mute_neuron(float* a, float* d)
{
	a += blockIdx.x * blockDim.x;
	d += blockIdx.x * blockDim.x;
	const int thidx = threadIdx.x;
	a[thidx] = a[thidx] * d[thidx];
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
				if (context->device.dor[i])
					_cwc_kern_mute_neuron
					<<<layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels, batch, 0, context->device.stream>>>
					(i == 0 ? a : GPU(convnet)->forwards[i - 1], context->device.dor[i]);
				_cwc_convnet_convolutional_forward_propagate(layer, batch, i == 0 ? a : GPU(convnet)->forwards[i - 1], GPU(convnet)->forwards[i], context->device.stream);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				if (context->device.dor[i])
					_cwc_kern_mute_neuron
					<<<layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels, batch, 0, context->device.stream>>>
					(GPU(convnet)->forwards[i - 1], context->device.dor[i]);
				_cwc_convnet_full_connect_forward_propagate(layer, batch, GPU(convnet)->forwards[i - 1], GPU(convnet)->forwards[i], GPU(convnet)->unit, context->device.cublas);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				assert(i > 0);
				_cwc_convnet_rnorm_forward_propagate(layer, batch, GPU(convnet)->forwards[i - 1], GPU(convnet)->forwards[i], GPU(convnet)->denoms[i], context->device.stream);
				break;
			case CCV_CONVNET_MAX_POOL:
				assert(i > 0);
				_cwc_convnet_max_pool_forward_propagate(layer, batch, GPU(convnet)->forwards[i - 1], GPU(convnet)->forwards[i], context->device.stream);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				_cwc_convnet_average_pool_forward_propagate(layer, batch,  GPU(convnet)->forwards[i - 1], GPU(convnet)->forwards[i], context->device.stream);
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
				_cwc_convnet_convolutional_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], GPU(convnet)->forwards[i], i > 0 ? GPU(convnet)->forwards[i - 1] : m, GPU(convnet)->backwards[i], configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
				if (context->device.dor[i] && GPU(convnet)->backwards[i])
					_cwc_kern_mute_neuron
					<<<layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels, batch, 0, context->device.stream>>>
					(GPU(convnet)->backwards[i], context->device.dor[i]);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_cwc_convnet_full_connect_backward_propagate(layer, batch,  i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], i > 0 ? GPU(convnet)->forwards[i - 1] : m, GPU(convnet)->backwards[i], GPU(convnet)->unit, configuration, context->device.cublas);
				if (context->device.dor[i])
					_cwc_kern_mute_neuron
					<<<layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels, batch, 0, context->device.stream>>>
					(GPU(convnet)->backwards[i], context->device.dor[i]);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				_cwc_convnet_rnorm_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], GPU(convnet)->forwards[i], i > 0 ? GPU(convnet)->forwards[i - 1] : m, GPU(convnet)->denoms[i], GPU(convnet)->backwards[i], context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_MAX_POOL:
				_cwc_convnet_max_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], GPU(convnet)->forwards[i], i > 0 ? GPU(convnet)->forwards[i - 1] : m, GPU(convnet)->backwards[i], context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_cwc_convnet_average_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], GPU(convnet)->backwards[i], context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
		}
	}
}

#ifndef CASE_TESTS

void cwc_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch)
{
}

void cwc_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int* labels, int batch)
{
}

void cwc_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, ccv_convnet_train_param_t params)
{
	assert(params.mini_batch % BATCH_PER_BLOCK == 0);
	_cwc_convnet_alloc_reserved(convnet, params.mini_batch, params.layer_params);
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
	struct {
		int* host;
		int* device;
	} test_returns = {
		.host = 0,
		.device = 0,
	};
	cudaMallocHost(&test_returns.host, sizeof(int) * tests->rnum);
	cudaMalloc(&test_returns.device, sizeof(int) * ((tests->rnum + params.mini_batch - 1) / params.mini_batch * params.mini_batch));
	assert(test_returns.device);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for (t = 0; t < params.max_epoch; t++)
	{
		cudaEventRecord(start, 0);
		// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in updating
		if (t > 0) // undo the mean network for further training
			_cwc_convnet_dor_mean_net_undo(convnet, params.layer_params, GPU(convnet)->contexts[1].device.cublas);
		// run updates
		for (i = 0; i < aligned_batches; i++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + (i % 2);
			_cwc_convnet_batch_formation(categorizeds, idx, convnet->rows, convnet->cols, convnet->channels, params.mini_batch, i * params.mini_batch, params.mini_batch, context->host.input, context->host.c);
			FLUSH(" - at epoch %03d / %d => stochastic gradient descent at %d / %d", t + 1, params.max_epoch, i + 1, aligned_batches);
			cudaMemcpyAsync(context->device.input, context->host.input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device.stream);
			assert(cudaGetLastError() == cudaSuccess);
			cudaMemcpyAsync(context->device.c, context->host.c, sizeof(int) * params.mini_batch, cudaMemcpyHostToDevice, context->device.stream);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_dor_formation(convnet, params.mini_batch, rng, params.layer_params, context);
			assert(cudaGetLastError() == cudaSuccess);
			// sync with the other stream core so that we can compute on the single true layer parameters
			cudaStreamSynchronize(GPU(convnet)->contexts[(i + 1) % 2].device.stream);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_encode_impl(convnet, context->device.input, params.mini_batch, context);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_softmax_with_logistic_loss(params.mini_batch, category_count, GPU(convnet)->forwards[convnet->count - 1], context->device.c, context->device.stream);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_backwards_propagate_error(convnet, GPU(convnet)->forwards[convnet->count - 1], context->device.input, params.mini_batch, context);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_net_sgd(convnet, i > 0, params.mini_batch, params.layer_params, context);
			assert(cudaGetLastError() == cudaSuccess);
		}
		cudaDeviceSynchronize(); // synchronize at this point
		// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in testing
		_cwc_convnet_dor_mean_net(convnet, params.layer_params, GPU(convnet)->contexts[1].device.cublas);
		// run tests
		for (i = j = 0; i < tests->rnum; i += params.mini_batch, j++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + (j % 2);
			_cwc_convnet_batch_formation(tests, 0, convnet->rows, convnet->cols, convnet->channels, params.mini_batch, i, ccv_min(params.mini_batch, tests->rnum - i), context->host.input, 0);
			cudaMemcpyAsync(context->device.input, context->host.input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device.stream);
			assert(cudaGetLastError() == cudaSuccess);
			// sync with the other stream core so that we can compute on the single true layer parameters
			cudaStreamSynchronize(GPU(convnet)->contexts[(j + 1) % 2].device.stream);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_encode_impl(convnet, context->device.input, params.mini_batch, context);
			assert(cudaGetLastError() == cudaSuccess);
			_cwc_convnet_tests_return(params.mini_batch, category_count, GPU(convnet)->forwards[convnet->count - 1], test_returns.device + i, context->device.stream);
		}
		cudaDeviceSynchronize(); // synchronize at this point
		cudaMemcpy(test_returns.host, test_returns.device, sizeof(int) * tests->rnum, cudaMemcpyDeviceToHost);
		int miss = 0;
		for (i = 0; i < tests->rnum; i++)
		{
			ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, i);
			if (test->c != test_returns.host[i])
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
	cudaFree(test_returns.device);
	cudaFreeHost(test_returns.host);
	ccfree(idx);
	gsl_rng_free(rng);
}

void cwc_convnet_compact(ccv_convnet_t* convnet)
{
	if (GPU(convnet))
	{
		cudaFree(GPU(convnet)->scratch);
		cudaFree(GPU(convnet)->unit);
		int i, j;
		for (i = 0; i < 2; i++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + i;
			cudaFreeHost(context->host.input);
			cudaFree(context->device.input);
			cudaFreeHost(context->host.c);
			cudaFree(context->device.c);
			cudaStreamDestroy(context->device.stream);
			cublasDestroy(context->device.cublas);
		}
		for (i = 0; i < convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
			if (layer->w)
				cudaFree(layer->w);
			ccv_convnet_layer_t* configuration = GPU(convnet)->configurations + i;
			if (configuration->w)
				cudaFree(configuration->w);
			ccv_convnet_layer_t* momentum = GPU(convnet)->momentums + i;
			if (momentum->w)
				cudaFree(momentum->w);
			if (GPU(convnet)->denoms[i])
				cudaFree(GPU(convnet)->denoms[i]);
			if (GPU(convnet)->forwards[i])
				cudaFree(GPU(convnet)->forwards[i]);
			if (GPU(convnet)->backwards[i])
				cudaFree(GPU(convnet)->backwards[i]);
			for (j = 0; j < 2; j++)
			{
				cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
				if (context->host.dor[i])
					cudaFreeHost(context->host.dor[i]);
				if (context->device.dor[i])
					cudaFree(context->device.dor[i]);
			}
		}
		ccfree(convnet->reserved);
	}
}

#endif
