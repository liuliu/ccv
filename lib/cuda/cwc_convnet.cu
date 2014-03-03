extern "C" {
#include "cwc.h"
#include "cwc_internal.h"
#include "../ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
}
#include <cuda.h>
#include <cublas_v2.h>
#include "../3rdparty/sqlite3/sqlite3.h"

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
	ccv_convnet_layer_train_param_t* layer_params;
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

typedef struct {
	int x, y, z;
} cwc_convnet_kernel_vary_t;

typedef struct {
	struct {
		cwc_convnet_kernel_vary_t forward;
		struct {
			cwc_convnet_kernel_vary_t coefficient;
			cwc_convnet_kernel_vary_t gradient;
		} backward;
	} convolutional;
} cwc_convnet_layer_vary_t;

#define VARY(x) ((cwc_convnet_layer_vary_t*)((x)->reserved))
#define GPU(x) ((cwc_convnet_t*)((x)->reserved))
#define BATCH_PER_BLOCK (8)

inline static void _cwc_convnet_layer_deduce_output_format(ccv_convnet_layer_t* layer, int* rows, int* cols, int* partition)
{
	assert(rows != 0 && cols != 0 && partition != 0);
	switch(layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			assert(layer->net.convolutional.rows % 2); // as of now, don't support even number of kernel size
			assert(layer->net.convolutional.cols % 2);
			assert((layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) % layer->net.convolutional.strides == 0);
			assert((layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) % layer->net.convolutional.strides == 0);
			*rows = (layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
			*partition = layer->input.matrix.partition;
			break;
		case CCV_CONVNET_FULL_CONNECT:
			*rows = layer->net.full_connect.count;
			*cols = 1;
			*partition = 1;
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			*rows = layer->input.matrix.rows;
			*cols = layer->input.matrix.cols;
			*partition = layer->input.matrix.partition;
			break;
		case CCV_CONVNET_MAX_POOL:
		case CCV_CONVNET_AVERAGE_POOL:
			assert((layer->input.matrix.rows + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			assert((layer->input.matrix.cols + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			*rows = (layer->input.matrix.rows + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			*partition = layer->input.matrix.partition;
			break;
	}
}

static int _cwc_convnet_layer_use_multi_way(ccv_convnet_layer_t* layer)
{
	return layer->input.matrix.channels <= 8 && layer->input.matrix.partition == 1;
}

static void _cwc_convnet_reorder_convolutional_weights_onto_device(float* w, float* ow, int wnum, int filters, int channels, int channel_partition)
{
	int channels_per_partition = channels / channel_partition;
	assert(wnum % (filters * channels_per_partition) == 0);
	float* iw = (float*)ccmalloc(sizeof(float) * wnum);
	int count = wnum / (filters * channels_per_partition);
	int i, j, k;
	for (i = 0; i < channels_per_partition; i++)
		for (j = 0; j < count; j++)
			for (k = 0; k < filters; k++)
				iw[i * count * filters + j * filters + k] = w[k * count * channels_per_partition + j * channels_per_partition + i];
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
	if (GPU(convnet) && (GPU(convnet)->batch != batch || GPU(convnet)->layer_params != layer_params))
		ccv_convnet_compact(convnet);
	else if (GPU(convnet))
		return; // it is allocated properly, no-op
	convnet->reserved = (cwc_convnet_t*)ccmalloc(sizeof(cwc_convnet_t) + sizeof(cwc_convnet_layer_vary_t) * convnet->count + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 10);
	GPU(convnet)->batch = batch;
	GPU(convnet)->layer_params = layer_params;
	GPU(convnet)->device.memory_usage = 0;
	cwc_convnet_layer_vary_t* layer_vary = (cwc_convnet_layer_vary_t*)(GPU(convnet) + 1);
	memset(layer_vary, 0, sizeof(cwc_convnet_layer_vary_t) * convnet->count);
	GPU(convnet)->layers = (ccv_convnet_layer_t*)(layer_vary + convnet->count);
	int i, j, out_rows, out_cols, out_partition /* this is not useful for allocation */;
	memcpy(GPU(convnet)->layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	ccv_convnet_layer_t* layers = GPU(convnet)->layers;
	// configurations (the backprop coefficients)
	size_t scratch_space = 0;
	size_t unit_size = batch;
	for (i = 0; i < convnet->count; i++)
		if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
		{
			int use_multi_way = _cwc_convnet_layer_use_multi_way(layers + i);
			layers[i].reserved = layer_vary + i;
			_cwc_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols, &out_partition);
			scratch_space = ccv_max(scratch_space, layers[i].wnum);
			scratch_space = ccv_max(scratch_space,
					out_rows * out_cols * layers[i].net.convolutional.count * batch + // output layer reorder
					layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch + // input layer reorder
					layers[i].wnum * (use_multi_way ? out_rows : 1) * (batch / BATCH_PER_BLOCK)); // unconsolidated weights output
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
				_cwc_convnet_reorder_convolutional_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].net.convolutional.count, layers[i].net.convolutional.channels, layers[i].input.matrix.partition);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.convolutional.count, cudaMemcpyHostToDevice);
				_cwc_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols, &out_partition);
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
					if (layer_params && layer_params[i].dor > 0)
					{
						assert(i > 0);
						cudaMallocHost(&context->host.dor[i], sizeof(float) * batch * out_rows * out_cols * layers[i].net.convolutional.count);
						assert(context->host.dor[i]);
						cudaMalloc(&context->device.dor[i], sizeof(float) * batch * out_rows * out_cols * layers[i].net.convolutional.count);
						GPU(convnet)->device.memory_usage += sizeof(float) * batch * out_rows * out_cols * layers[i].net.convolutional.count;
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
					if (layer_params && layer_params[i].dor > 0)
					{
						cudaMallocHost(&context->host.dor[i], sizeof(float) * batch * layers[i].net.full_connect.count);
						assert(context->host.dor[i]);
						cudaMalloc(&context->device.dor[i], sizeof(float) * batch * layers[i].net.full_connect.count);
						GPU(convnet)->device.memory_usage += sizeof(float) * batch * layers[i].net.full_connect.count;
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
				_cwc_convnet_layer_deduce_output_format(layers + i, &out_rows, &out_cols, &out_partition);
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
		float* input, const int rows, const int cols, const int channels_per_partition, const int partition,
		float* out, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count,
		float* const biases)
{
	assert(gridDim.x * partition * filter_per_block == out_rows * count);
	assert(gridDim.y == out_cols);
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

static int _cwc_convnet_convolutional_forward_propagate_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream,
		int x, int y, int z) // these are the dynamic configurations
{
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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
			 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / out_partition, out_partition, \
			 b, out_rows, out_cols, \
			 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count, \
			 layer->bias); \
	} while (0)
	cwc_vary_4_a(x, 1, 2, 4, 8, cwc_vary_5_b, y, 1, 2, 4, 6, 8, cwc_vary_6_c, z, 16, 24, 32, 36, 64, 72, vary_block);
#undef vary_block
	assert(cudaGetLastError() == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	static int vary_x[] = { 1, 2, 4, 8 };
	static int vary_y[] = { 1, 2, 4, 6, 8 };
	static int vary_z[] = { 16, 24, 32, 36, 64, 72 };
	CWC_IMPLEMENT_VARY_STUB(VARY(layer)->convolutional.forward, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_forward_propagate_vary, layer, batch, a, b, stream);
}

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
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	int i, j, c;
	float prod[input_per_thread];
	const int incnt = rows * cols * batch;
	input += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	out += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	denoms += (blockIdx.z * channels_per_partition * rows * cols + blockIdx.y * cols + blockIdx.x) * batch;
	const int end_way = min(way, channels_per_partition - 1);
	for (c = 0; c < end_way; c++)
	{
		#pragma unroll
		for (i = 0; i < input_loads; i++)
			if (i * thcnt + thidx < batch)
				shared_input[c * batch + i * thcnt + thidx] = input[i * thcnt + thidx];
		input += incnt;
	}
	for (c = 0; c < channels_per_partition; c++)
	{
		const int start_way = max(c - way, 0);
		const int end_way = min(c + way, channels_per_partition - 1);
		if (c + way < channels_per_partition)
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
	dim3 num_blocks(layer->input.matrix.cols, layer->input.matrix.rows, layer->input.matrix.partition);
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
		 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / layer->input.matrix.partition, layer->input.matrix.partition,
		 b, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	} else if (layer->net.rnorm.size == 5) {
		cudaFuncSetCacheConfig(_cwc_kern_rnorm_forward_propagate<1, 5>, cudaFuncCachePreferShared);
		_cwc_kern_rnorm_forward_propagate
		<1, 5>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(batch,
		 a, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / layer->input.matrix.partition, layer->input.matrix.partition,
		 b, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
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
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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

static void _cwc_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, float* batch_unit /* this is just 1's in device */, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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

__global__ static void _cwc_kern_mute_neuron(float* a, float* d)
{
	a += blockIdx.x * blockDim.x;
	d += blockIdx.x * blockDim.x;
	const int thidx = threadIdx.x;
	a[thidx] = a[thidx] * d[thidx];
}

// assuming a is in device memory
static void _cwc_convnet_encode_impl(ccv_convnet_t* convnet, float* a, int batch, int dor, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_cwc_convnet_convolutional_forward_propagate(layer, batch, i == 0 ? a : GPU(convnet)->forwards[i - 1], GPU(convnet)->forwards[i], context->device.stream);
				if (dor && context->device.dor[i])
				{
					int out_rows, out_cols, out_partition;
					_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
					_cwc_kern_mute_neuron
					<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device.stream>>>
					(GPU(convnet)->forwards[i], context->device.dor[i]);
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				_cwc_convnet_full_connect_forward_propagate(layer, batch, GPU(convnet)->forwards[i - 1], GPU(convnet)->forwards[i], GPU(convnet)->unit, context->device.cublas);
				if (dor && context->device.dor[i])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device.stream>>>
					(GPU(convnet)->forwards[i], context->device.dor[i]);
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

#ifdef HAVE_GSL

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

template <int channel_per_thread, int filter_per_thread, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coefficient_multi_way(const int strides, const int border, const int batch,
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
__global__ static void _cwc_kern_convolutional_backward_propagate_error(const int strides, const int border, const int batch,
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
__global__ static void _cwc_kern_reorder_matrix_major(float* a, float* b, const int count, const int channels_per_partition, const int partition, const int batch)
{
	b += blockIdx.z * count * channels_per_partition * batch;
	a += blockIdx.z * count * channels_per_partition * batch;
	b[(threadIdx.x * count + blockIdx.x) * channels_per_partition + blockIdx.y] = a[(blockIdx.y * count + blockIdx.x) * batch + threadIdx.x];
}
// this method rewinds a matrix
__global__ static void _cwc_kern_reorder_matrix_major_parted(float* a, float* b, const int count, const int channels, const int batch, const int channels_per_partition, const int batch_per_partition, const int partition)
{
	b[(threadIdx.x * count + blockIdx.x) * channels + blockIdx.y + threadIdx.y * channels_per_partition] = a[(blockIdx.y * count + blockIdx.x) * batch + threadIdx.x + threadIdx.y * batch_per_partition];
}

// this method rewinds a matrix
__global__ static void _cwc_kern_reorder_matrix_major_per_block(float* a, float* b, const int count, const int channels, const int batch, const int batch_per_block)
{
	const int thidx = threadIdx.x + threadIdx.y * batch_per_block;
	b[(threadIdx.y * count + blockIdx.x) * channels * batch_per_block + threadIdx.x * channels + blockIdx.y] = a[(blockIdx.y * count + blockIdx.x) * batch + thidx];
}

static int _cwc_convnet_convolutional_backward_propagate_coefficient_multi_way_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle,
		int x, int y, int z)
{
	if (!(layer->net.convolutional.count % y == 0 && layer->input.matrix.channels % x == 0 &&
				layer->net.convolutional.count / y * layer->input.matrix.channels / x <= 1024 && /* thread per block constraint */
				layer->net.convolutional.count / y * layer->input.matrix.channels / x >= layer->input.matrix.channels * BATCH_PER_BLOCK &&
				layer->net.convolutional.count / y * layer->input.matrix.channels / x >= layer->net.convolutional.count && /* shared loading constraint */
				sizeof(float) * BATCH_PER_BLOCK * (layer->input.matrix.channels + layer->net.convolutional.count) <= 48 * 1024 /* shared memory size constraint */))
		return -1;
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
	assert(out_partition == 1); // this cannot handle partition
	float* chm = scratch;
	float* cha = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch;
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	float alpha = 1, beta = 0;
	int count = layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->input.matrix.channels;
	const int batch_group_count = batch / BATCH_PER_BLOCK;
	_cwc_kern_reorder_matrix_major_per_block
	<<<dim3(layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels), dim3(BATCH_PER_BLOCK, batch_group_count), 0, stream>>>
	(m, chm, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels, batch, BATCH_PER_BLOCK);
	_cwc_kern_reorder_matrix_major_per_block
	<<<dim3(out_rows * out_cols, layer->net.convolutional.count), dim3(BATCH_PER_BLOCK, batch_group_count), 0, stream>>>
	(a, cha, out_rows * out_cols, layer->net.convolutional.count, batch, BATCH_PER_BLOCK);
#define vary_block(_x, _y) do { \
		dim3 threads_per_block_for_coeff(layer->net.convolutional.count / _y, layer->input.matrix.channels / _x); \
		assert(threads_per_block_for_coeff.x * threads_per_block_for_coeff.y <= 1024); \
		int batch_group_count = batch / BATCH_PER_BLOCK; \
		dim3 num_blocks_for_coeff(layer->net.convolutional.cols, layer->net.convolutional.rows, out_rows * batch_group_count); \
		int shared_memory_size = sizeof(float) * BATCH_PER_BLOCK * (layer->input.matrix.channels + layer->net.convolutional.count); \
		cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coefficient_multi_way<_x, _y, BATCH_PER_BLOCK>, cudaFuncCachePreferShared); \
		_cwc_kern_convolutional_backward_propagate_coefficient_multi_way \
		<_x, _y, BATCH_PER_BLOCK> \
		<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>> \
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
			chm, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
			cha, out_rows, out_cols, \
			cbw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count); \
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, count, 1, out_rows * batch_group_count, &alpha, cbw, count, unit, out_rows * batch_group_count, &beta, configuration->w, count); \
	} while (0)
	// special casing for image
	cwc_vary_4_a(x, 1, 2, 3, 4, cwc_vary_2_c, y, 1, 2, vary_block);
#undef vary_block
	assert(cudaGetLastError() == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_backward_propagate_coefficient_multi_way(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	static int vary_x[] = { 1, 2, 3, 4 };
	static int vary_y[] = { 1, 2 };
	static int vary_z[] = { 1 };
	CWC_IMPLEMENT_VARY_STUB(VARY(layer)->convolutional.backward.coefficient, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_backward_propagate_coefficient_multi_way_vary, layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
}

static int _cwc_convnet_convolutional_backward_propagate_coefficient_default_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle,
		int x, int y, int z)
{
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
	if (!(layer->net.convolutional.count % (y * out_partition) == 0 && z % x == 0 && layer->net.convolutional.channels % (z * out_partition) == 0 &&
		  layer->net.convolutional.count / (y * out_partition) * z / x <= 1024 && /* thread per block constraint */
		  layer->net.convolutional.count / (y * out_partition) * z / x >= z && layer->net.convolutional.count / (y * out_partition) * z / x >= layer->net.convolutional.count / out_partition && /* shared loading constraint */
				sizeof(float) * (z + layer->net.convolutional.count / out_partition) <= 32 * 1024 /* shared memory size constraint */))
		return -1;
	float* chm = scratch;
	float* cha = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch;
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	float alpha = 1, beta = 0;
	int count = layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->input.matrix.channels / out_partition;
	_cwc_kern_reorder_matrix_major
	<<<dim3(layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels / out_partition, out_partition), batch, 0, stream>>>
	(m, chm, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels / out_partition, out_partition, batch);
	_cwc_kern_reorder_matrix_major
	<<<dim3(out_rows * out_cols, layer->net.convolutional.count / out_partition, out_partition), batch, 0, stream>>>
	(a, cha, out_rows * out_cols, layer->net.convolutional.count / out_partition, out_partition, batch);
#define vary_block(_x, _y, _z) do { \
		dim3 threads_per_block_for_coeff(layer->net.convolutional.count / (_y * out_partition), _z / _x); \
		assert(threads_per_block_for_coeff.x * threads_per_block_for_coeff.y <= 1024); \
		int batch_group_count = batch / BATCH_PER_BLOCK; \
		dim3 num_blocks_for_coeff(layer->net.convolutional.cols, layer->net.convolutional.rows, layer->net.convolutional.channels / _z * batch_group_count); \
		int shared_memory_size = sizeof(float) * (_z + layer->net.convolutional.count / out_partition); \
		_cwc_kern_convolutional_backward_propagate_coefficient_default \
		<_x, _y, _z, BATCH_PER_BLOCK> \
		<<<num_blocks_for_coeff, threads_per_block_for_coeff, shared_memory_size, stream>>> \
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch, batch_group_count, \
			chm, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / out_partition, out_partition, \
			cha, out_rows, out_cols, \
			cbw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count / out_partition); \
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, count, 1, batch_group_count, &alpha, cbw, count, unit, batch_group_count, &beta, configuration->w, count); \
	} while (0)
	cwc_vary_6_a(x, 1, 2, 3, 4, 6, 8, cwc_vary_6_b, y, 1, 2, 3, 4, 6, 8, cwc_vary_4_c, z, 16, 24, 32, 36, vary_block);
#undef vary_block
	assert(cudaGetLastError() == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_backward_propagate_coefficient_default(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	static int vary_x[] = { 1, 2, 3, 4, 6, 8 };
	static int vary_y[] = { 1, 2, 3, 4, 6, 8 };
	static int vary_z[] = { 16, 24, 32, 36 };
	CWC_IMPLEMENT_VARY_STUB(VARY(layer)->convolutional.backward.coefficient, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_backward_propagate_coefficient_default_vary, layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
}

static int _cwc_convnet_convolutional_backward_propagate_error_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle,
		int x, int y, int z)
{
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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
#define vary_block(_x, _y, _z) do { \
		dim3 threads_per_block(batch / _x, _z / _y); \
		assert(threads_per_block.x * threads_per_block.y <= 1024); \
		dim3 num_blocks(layer->input.matrix.cols * layer->input.matrix.channels / (_z * out_partition), layer->input.matrix.rows, out_partition); \
		int shared_memory_size = sizeof(float) * (batch + _z); \
		cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_error<_x, _y, _z>, cudaFuncCachePreferShared); \
		_cwc_kern_convolutional_backward_propagate_error \
		<_x, _y, _z> \
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>> \
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch, \
		 b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels, \
		 a, out_rows, out_cols, \
		 chw, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count / out_partition, out_partition); \
	} while (0)
	cwc_vary_4_a(x, 1, 2, 4, 8, cwc_vary_5_b, y, 1, 2, 4, 6, 8, cwc_vary_6_c, z, 16, 24, 32, 36, 64, 72, vary_block);
#undef vary_block
	assert(cudaGetLastError() == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_backward_propagate_error(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	static int vary_x[] = { 1, 2, 4, 8 };
	static int vary_y[] = { 1, 2, 4, 6, 8 };
	static int vary_z[] = { 16, 24, 32, 36, 64, 72 };
	CWC_IMPLEMENT_VARY_STUB(VARY(layer)->convolutional.backward.gradient, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_backward_propagate_error_vary, layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
}

static void _cwc_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	assert(layer->net.convolutional.count % 4 == 0);
	assert(batch % BATCH_PER_BLOCK == 0);
	int out_rows, out_cols, out_partition, shared_memory_size;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
	// it turns out that first apply relu would save us a lot of computation because no need to low both out and out_grad any more
	_cwc_kern_convolutional_relu_backward_propagate
	<<<dim3(out_cols, out_rows, layer->net.convolutional.count), batch, 0, stream>>>
	(batch, n, a, out_rows, out_cols, layer->net.convolutional.count);
	assert(cudaGetLastError() == cudaSuccess);
	if (_cwc_convnet_layer_use_multi_way(layer))
		_cwc_convnet_convolutional_backward_propagate_coefficient_multi_way(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
	else
		_cwc_convnet_convolutional_backward_propagate_coefficient_default(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
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
		_cwc_convnet_convolutional_backward_propagate_error(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
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
	const int thcnt = blockDim.x;
	const int thidx = threadIdx.x;
	const int input_loads = (batch + thcnt - 1) / thcnt;
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
	for (c = 0; c < channels_per_partition; c++)
	{
		const int start_way = max(c - way, 0);
		const int end_way = min(c + way, channels_per_partition - 1);
		if (c + way < channels_per_partition)
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
				shared_input[i * thcnt + thidx] = input[i * thcnt + thidx];
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
	dim3 num_blocks(layer->input.matrix.cols, layer->input.matrix.rows, layer->input.matrix.partition);
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
		 m, b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / layer->input.matrix.partition, layer->input.matrix.partition,
		 n, a, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	} else if (layer->net.rnorm.size == 5) {
		cudaFuncSetCacheConfig(_cwc_kern_rnorm_backward_propagate<1, 5>, cudaFuncCachePreferShared);
		_cwc_kern_rnorm_backward_propagate
		<1, 5>
		<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
		(batch,
		 m, b, layer->input.matrix.rows, layer->input.matrix.cols, layer->input.matrix.channels / layer->input.matrix.partition, layer->input.matrix.partition,
		 n, a, denoms, layer->net.rnorm.kappa, layer->net.rnorm.alpha, layer->net.rnorm.beta);
	}
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
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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

static void _cwc_convnet_average_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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

static void _cwc_convnet_full_connect_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* m, float* b, float* batch_unit, ccv_convnet_layer_t* configuration, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols, out_partition;
	_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
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
		const float learn_rate, const float momentum_rate, const float decay_and_learn)
{
	if (blockIdx.x * blockDim.x + threadIdx.x < count)
	{
		a += blockIdx.x * blockDim.x;
		grad += blockIdx.x * blockDim.x;
		momentum += blockIdx.x * blockDim.x;
		const int thidx = threadIdx.x;
		float old_a = a[thidx];
		float velocity = (momentum_read ? momentum_rate * momentum[thidx] : 0) - decay_and_learn * old_a + learn_rate * grad[thidx];
		a[thidx] = velocity + old_a;
		momentum[thidx] = velocity;
	}
}

static void _cwc_convnet_net_sgd(ccv_convnet_t* convnet, int momentum_read, int batch, ccv_convnet_layer_train_param_t* layer_params, cwc_convnet_context_t* context)
{
	int i, out_rows, out_cols, out_partition;
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
				_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
				num_blocks_for_coeff = (layer->wnum + 127) / 128;
				num_blocks_for_bias = (layer->net.convolutional.count + 127) / 128;
				if (momentum_read)
				{
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				} else {
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				// assume coeff and bias in the same continuous memory region
				num_blocks_for_coeff = (layer->wnum + 127) / 128;
				num_blocks_for_bias = (layer->net.full_connect.count + 127) / 128;
				if (momentum_read)
				{
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.full_connect.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				} else {
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device.stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device.stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.full_connect.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				}
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				break;
		}
	}
}

static void _cwc_convnet_batch_formation(gsl_rng* rng, ccv_array_t* categorizeds, ccv_dense_matrix_t* mean_activity, ccv_dense_matrix_t* eigenvectors, ccv_dense_matrix_t* eigenvalues, float color_gain, int* idx, ccv_size_t dim, int rows, int cols, int channels, int symmetric, int batch, int offset, int size, float* b, int* c)
{
	int i, k, x;
	assert(size <= batch);
	float* channel_gains = (float*)alloca(sizeof(float) * channels);
	memset(channel_gains, 0, sizeof(float) * channels);
	for (i = 0; i < size; i++)
	{
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, idx ? idx[offset + i] : offset + i);
		if (c)
			c[i] = categorized->c;
		ccv_dense_matrix_t* image;
		switch (categorized->type)
		{
			case CCV_CATEGORIZED_DENSE_MATRIX:
				image = categorized->matrix;
				break;
			case CCV_CATEGORIZED_FILE:
				image = 0;
				ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				if (!image)
				{
					printf("cannot load %s.\n", categorized->file.filename);
					continue;
				}
				break;
		}
		assert(image->rows == dim.height || image->cols == dim.width);
		ccv_dense_matrix_t* input = 0;
		if (image->cols != dim.width || image->rows != dim.height)
		{
			int x = rng ? gsl_rng_uniform_int(rng, image->cols - dim.width + 1) : (image->cols - dim.width + 1) / 2;
			int y = rng ? gsl_rng_uniform_int(rng, image->rows - dim.height + 1) : (image->rows - dim.height + 1) / 2;
			assert(x == 0 || y == 0);
			ccv_slice(image, (ccv_matrix_t**)&input, CCV_32F, y, x, dim.height, dim.width);
		} else
			ccv_shift(image, (ccv_matrix_t**)&input, CCV_32F, 0, 0); // converting to 32f
		// we loaded it in, deallocate it now
		if (categorized->type != CCV_CATEGORIZED_DENSE_MATRIX)
			ccv_matrix_free(image);
		// random horizontal reflection
		if (symmetric && rng && gsl_rng_uniform_int(rng, 2) == 0)
			ccv_flip(input, &input, 0, CCV_FLIP_X);
		ccv_subtract(input, mean_activity, (ccv_matrix_t**)&input, 0);
		ccv_dense_matrix_t* patch = 0;
		if (input->cols != cols || input->rows != rows)
		{
			int x = rng ? gsl_rng_uniform_int(rng, input->cols - cols + 1) : (input->cols - cols + 1) / 2;
			int y = rng ? gsl_rng_uniform_int(rng, input->rows - rows + 1) : (input->rows - rows + 1) / 2;
			ccv_slice(input, (ccv_matrix_t**)&patch, CCV_32F, y, x, rows, cols);
			ccv_matrix_free(input);
		} else
			patch = input;
		assert(channels == CCV_GET_CHANNEL(patch->type));
		if (color_gain > 0 && rng && eigenvectors && eigenvalues)
		{
			assert(channels == 3); // only support RGB color gain
			memset(channel_gains, 0, sizeof(float) * channels);
			for (k = 0; k < channels; k++)
			{
				float alpha = gsl_ran_gaussian(rng, color_gain) * eigenvalues->data.f64[k];
				for (x = 0; x < channels; x++)
					channel_gains[x] += eigenvectors->data.f64[k * channels + x] * alpha;
			}
		}
		for (k = 0; k < channels; k++)
			for (x = 0; x < rows * cols; x++)
				b[(k * rows * cols + x) * batch + i] = patch->data.f32[x * channels + k] + channel_gains[k];
		ccv_matrix_free(patch);
	}
}

static void _cwc_convnet_mean_formation(ccv_array_t* categorizeds, ccv_size_t dim, int channels, int symmetric, ccv_dense_matrix_t** b)
{
	int i, count = 0;
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(dim.height, dim.width, channels | CCV_64F, 0, 0);
	ccv_zero(c);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, dim.height, dim.width, channels | CCV_32F, channels | CCV_32F, 0);
	for (i = 0; i < categorizeds->rnum; i++)
	{
		if (i % 23 == 0 || i == categorizeds->rnum - 1)
			FLUSH(" - compute mean activity %d / %d", i + 1, categorizeds->rnum);
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);
		ccv_dense_matrix_t* image;
		switch (categorized->type)
		{
			case CCV_CATEGORIZED_DENSE_MATRIX:
				image = categorized->matrix;
				break;
			case CCV_CATEGORIZED_FILE:
				image = 0;
				ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				if (!image)
				{
					printf("cannot load %s.\n", categorized->file.filename);
					continue;
				}
				break;
		}
		ccv_dense_matrix_t* patch = 0;
		if (image->cols != dim.width || image->rows != dim.height)
		{
			int x = (image->cols - dim.width + 1) / 2;
			int y = (image->rows - dim.height + 1) / 2;
			assert(x == 0 || y == 0);
			ccv_slice(image, (ccv_matrix_t**)&patch, CCV_32F, y, x, dim.height, dim.width);
		} else
			ccv_shift(image, (ccv_matrix_t**)&patch, CCV_32F, 0, 0); // converting to 32f
		if (categorized->type != CCV_CATEGORIZED_DENSE_MATRIX)
			ccv_matrix_free(image);
		ccv_add(patch, c, (ccv_matrix_t**)&c, CCV_64F);
		++count;
		ccv_matrix_free(patch);
	}
	if (symmetric)
	{
		int j, k;
		double p = 0.5 / count;
		double* cptr = c->data.f64;
		float* dbptr = db->data.f32;
		for (i = 0; i < db->rows; i++)
		{
			for (j = 0; j < db->cols; j++)
				for (k = 0; k < channels; k++)
					dbptr[j * channels + k] = p * (cptr[j * channels + k] + cptr[(c->cols - j - 1) * channels + k]);
			dbptr += db->cols * channels;
			cptr += c->cols * channels;
		}
	} else {
		double p = 1.0 / count;
		for (i = 0; i < dim.height * dim.width * channels; i++)
			db->data.f32[i] = p * c->data.f64[i];
	}
	ccv_matrix_free(c);
	printf("\n");
}

static void _cwc_convnet_channel_eigen(ccv_array_t* categorizeds, ccv_dense_matrix_t* mean_activity, ccv_size_t dim, int channels, ccv_dense_matrix_t** eigenvectors, ccv_dense_matrix_t** eigenvalues)
{
	assert(channels == 3); // this function cannot handle anything other than 3x3 covariance matrix
	double* mean_value = (double*)alloca(sizeof(double) * channels);
	memset(mean_value, 0, sizeof(double) * channels);
	assert(CCV_GET_CHANNEL(mean_activity->type) == channels);
	assert(mean_activity->rows == dim.height);
	assert(mean_activity->cols == dim.width);
	int i, j, k, c, count = 0;
	for (i = 0; i < dim.height * dim.width; i++)
		for (k = 0; k < channels; k++)
			mean_value[k] += mean_activity->data.f32[i * channels + k];
	for (i = 0; i < channels; i++)
		mean_value[i] = mean_value[i] / (dim.height * dim.width);
	double* covariance = (double*)alloca(sizeof(double) * channels * channels);
	memset(covariance, 0, sizeof(double) * channels * channels);
	for (c = 0; c < categorizeds->rnum; c++)
	{
		if (c % 23 == 0 || c == categorizeds->rnum - 1)
			FLUSH(" - compute covariance matrix for data augmentation (color gain) %d / %d", c + 1, categorizeds->rnum);
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, c);
		ccv_dense_matrix_t* image;
		switch (categorized->type)
		{
			case CCV_CATEGORIZED_DENSE_MATRIX:
				image = categorized->matrix;
				break;
			case CCV_CATEGORIZED_FILE:
				image = 0;
				ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				if (!image)
				{
					printf("cannot load %s.\n", categorized->file.filename);
					continue;
				}
				break;
		}
		ccv_dense_matrix_t* patch = 0;
		if (image->cols != dim.width || image->rows != dim.height)
		{
			int x = (image->cols - dim.width + 1) / 2;
			int y = (image->rows - dim.height + 1) / 2;
			assert(x == 0 || y == 0);
			ccv_slice(image, (ccv_matrix_t**)&patch, CCV_32F, y, x, dim.height, dim.width);
		} else
			ccv_shift(image, (ccv_matrix_t**)&patch, CCV_32F, 0, 0); // converting to 32f
		if (categorized->type != CCV_CATEGORIZED_DENSE_MATRIX)
			ccv_matrix_free(image);
		for (i = 0; i < dim.width * dim.height; i++)
			for (j = 0; j < channels; j++)
				for (k = j; k < channels; k++)
					covariance[j * channels + k] += (patch->data.f32[i * channels + j] - mean_value[j]) * (patch->data.f32[i * channels + k] - mean_value[k]);
		++count;
		ccv_matrix_free(patch);
	}
	for (i = 0; i < channels; i++)
		for (j = 0; j < i; j++)
			covariance[i * channels + j] = covariance[j * channels + i];
	double p = 1.0 / ((double)count * dim.height * dim.width);
	for (i = 0; i < channels; i++)
		for (j = 0; j < channels; j++)
			covariance[i * channels + j] *= p; // scale down
	ccv_dense_matrix_t covm = ccv_dense_matrix(3, 3, CCV_64F | CCV_C1, covariance, 0);
	ccv_eigen(&covm, eigenvectors, eigenvalues, CCV_64F, 1e-8);
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
			assert(layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT);
			switch (layer->type)
			{
				case CCV_CONVNET_CONVOLUTIONAL:
					cublasSscal(handle, layer->net.convolutional.count, &dor, layer->bias, 1);
					break;
				case CCV_CONVNET_FULL_CONNECT:
					cublasSscal(handle, layer->net.full_connect.count, &dor, layer->bias, 1);
					break;
			}
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
			assert(layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT);
			switch (layer->type)
			{
				case CCV_CONVNET_CONVOLUTIONAL:
					cublasSscal(handle, layer->net.convolutional.count, &inv_dor, layer->bias, 1);
					break;
				case CCV_CONVNET_FULL_CONNECT:
					cublasSscal(handle, layer->net.full_connect.count, &inv_dor, layer->bias, 1);
					break;
			}
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
			int out_rows, out_cols, out_partition;
			_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
			assert(layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT);
			int count = layer->type == CCV_CONVNET_FULL_CONNECT ? layer->net.full_connect.count : out_rows * out_cols * layer->net.convolutional.count;
			for (j = 0; j < batch * count; j++)
				context->host.dor[i][j] = (gsl_rng_uniform(rng) >= layer_params[i].dor) ? 1.0 : 0.0;
			cudaMemcpyAsync(context->device.dor[i], context->host.dor[i], sizeof(float) * count * batch, cudaMemcpyHostToDevice, context->device.stream);
			assert(cudaGetLastError() == cudaSuccess);
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
				if (context->device.dor[i])
				{
					int out_rows, out_cols, out_partition;
					_cwc_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
					_cwc_kern_mute_neuron
					<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device.stream>>>
					(i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], context->device.dor[i]);
				}
				_cwc_convnet_convolutional_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], GPU(convnet)->forwards[i], i > 0 ? GPU(convnet)->forwards[i - 1] : m, GPU(convnet)->backwards[i], configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				if (context->device.dor[i])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device.stream>>>
					(i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], context->device.dor[i]);
				_cwc_convnet_full_connect_backward_propagate(layer, batch,  i == convnet->count - 1 ? a : GPU(convnet)->backwards[i + 1], i > 0 ? GPU(convnet)->forwards[i - 1] : m, GPU(convnet)->backwards[i], GPU(convnet)->unit, configuration, context->device.cublas);
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

typedef struct {
	int t, i;
	int inum;
	int* idx;
	ccv_convnet_t* convnet;
	// these are eigenvectors / values for color covariance matrix
	ccv_dense_matrix_t* eigenvectors;
	ccv_dense_matrix_t* eigenvalues;
	ccv_function_state_reserve_field
} cwc_convnet_supervised_train_function_state_t;

static void _cwc_convnet_supervised_train_function_state_read(const char* filename, cwc_convnet_supervised_train_function_state_t* z)
{
	ccv_convnet_t* convnet = ccv_convnet_read(1, filename);
	if (!convnet)
		return;
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(z->convnet)->layers + i;
		ccv_convnet_layer_t* z_layer = z->convnet->layers + i;
		ccv_convnet_layer_t* host_layer = convnet->layers + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_cwc_convnet_reorder_convolutional_weights_onto_device(host_layer->w, layer->w, layer->wnum, layer->net.convolutional.count, layer->net.convolutional.channels, layer->input.matrix.partition);
				cudaMemcpy(layer->bias, host_layer->bias, sizeof(float) * layer->net.convolutional.count, cudaMemcpyHostToDevice);
				memcpy(z_layer->w, host_layer->w, sizeof(float) * (layer->wnum + layer->net.convolutional.count));
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_cwc_convnet_reorder_full_connect_weights_onto_device(host_layer->w, layer->w, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
				cudaMemcpy(layer->bias, host_layer->bias, sizeof(float) * layer->net.full_connect.count, cudaMemcpyHostToDevice);
				memcpy(z_layer->w, host_layer->w, sizeof(float) * (layer->wnum + layer->net.full_connect.count));
				assert(cudaGetLastError() == cudaSuccess);
				break;
		}
	}
	assert(convnet->input.height == z->convnet->input.height);
	assert(convnet->input.width == z->convnet->input.width);
	assert(convnet->rows == z->convnet->rows);
	assert(convnet->cols == z->convnet->cols);
	assert(convnet->channels == z->convnet->channels);
	memcpy(z->convnet->mean_activity->data.f32, convnet->mean_activity->data.f32, sizeof(float) * z->convnet->input.height * z->convnet->input.width * z->convnet->channels);
	ccv_convnet_free(convnet);
	sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		z->line_no = 0;
		const char function_state_qs[] =
			"SELECT t, i, inum, line_no, idx, eigenvectors, eigenvalues FROM function_state WHERE fsid = 0;";
		sqlite3_stmt* function_state_stmt = 0;
		if (SQLITE_OK == sqlite3_prepare_v2(db, function_state_qs, sizeof(function_state_qs), &function_state_stmt, 0))
		{
			if (SQLITE_ROW == sqlite3_step(function_state_stmt))
			{
				z->t = sqlite3_column_int(function_state_stmt, 0);
				z->i = sqlite3_column_int(function_state_stmt, 1);
				int inum = sqlite3_column_int(function_state_stmt, 2);
				assert(inum == z->inum);
				z->line_no = sqlite3_column_int(function_state_stmt, 3);
				const void* idx = sqlite3_column_blob(function_state_stmt, 4);
				memcpy(z->idx, idx, sizeof(int) * z->inum);
				if (sqlite3_column_bytes(function_state_stmt, 5) == sizeof(double) * 3 * 3 &&
					sqlite3_column_bytes(function_state_stmt, 6) == sizeof(double) * 3)
				{
					const void* eigenvectors = sqlite3_column_blob(function_state_stmt, 5);
					const void* eigenvalues = sqlite3_column_blob(function_state_stmt, 6);
					if (!z->eigenvectors)
						z->eigenvectors = ccv_dense_matrix_new(3, 3, CCV_64F | CCV_C1, 0, 0);
					if (!z->eigenvalues)
						z->eigenvalues = ccv_dense_matrix_new(1, 3, CCV_64F | CCV_C1, 0, 0);
					memcpy(z->eigenvectors->data.u8, eigenvectors, sizeof(double) * 3 * 3);
					memcpy(z->eigenvalues->data.u8, eigenvalues, sizeof(double) * 3);
				}
			}
			sqlite3_finalize(function_state_stmt);
		}
		sqlite3_stmt* momentum_data_stmt = 0;
		const char momentum_data_qs[] =
			"SELECT layer, weight, bias FROM momentum_data;";
		if (SQLITE_OK == sqlite3_prepare_v2(db, momentum_data_qs, sizeof(momentum_data_qs), &momentum_data_stmt, 0))
		{
			while(sqlite3_step(momentum_data_stmt) == SQLITE_ROW)
			{
				ccv_convnet_layer_t* layer = GPU(z->convnet)->layers + sqlite3_column_int(momentum_data_stmt, 0);
				ccv_convnet_layer_t* momentum = GPU(z->convnet)->momentums + sqlite3_column_int(momentum_data_stmt, 0);
				int wnum = sqlite3_column_bytes(momentum_data_stmt, 1) / sizeof(float);
				int bnum = sqlite3_column_bytes(momentum_data_stmt, 2) / sizeof(float);
				if (wnum != layer->wnum)
					continue;
				const void* w = sqlite3_column_blob(momentum_data_stmt, 1);
				const void* bias = sqlite3_column_blob(momentum_data_stmt, 2);
				switch (layer->type)
				{
					case CCV_CONVNET_CONVOLUTIONAL:
						if (bnum != layer->net.convolutional.count)
							continue;
						_cwc_convnet_reorder_convolutional_weights_onto_device((float*)w, momentum->w, layer->wnum, layer->net.convolutional.count, layer->net.convolutional.channels, layer->input.matrix.partition);
						cudaMemcpy(momentum->bias, bias, sizeof(float) * layer->net.convolutional.count, cudaMemcpyHostToDevice);
						break;
					case CCV_CONVNET_FULL_CONNECT:
						if (bnum != layer->net.full_connect.count)
							continue;
						_cwc_convnet_reorder_full_connect_weights_onto_device((float*)w, momentum->w, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
						cudaMemcpy(momentum->bias, bias, sizeof(float) * layer->net.full_connect.count, cudaMemcpyHostToDevice);
						break;
				}
			}
			sqlite3_finalize(momentum_data_stmt);
		}
		sqlite3_close(db);
	}
}

static void _cwc_convnet_reorder_convolutional_weights_onto_host(float* w, float* hw, int wnum, int filters, int channels, int channel_partition)
{
	int channels_per_partition = channels / channel_partition;
	assert(wnum % (filters * channels_per_partition) == 0);
	float* iw = (float*)ccmalloc(sizeof(float) * wnum);
	cudaMemcpy(iw, w, sizeof(float) * wnum, cudaMemcpyDeviceToHost);
	int count = wnum / (filters * channels_per_partition);
	int i, j, k;
	for (i = 0; i < channels_per_partition; i++)
		for (j = 0; j < count; j++)
			for (k = 0; k < filters; k++)
				hw[k * count * channels_per_partition + j * channels_per_partition + i] = iw[i * count * filters + j * filters + k];
	ccfree(iw);
}

static void _cwc_convnet_reorder_full_connect_weights_onto_host(float* w, float* hw, int wnum, int count, int channels)
{
	assert(wnum % (count * channels) == 0);
	float* iw = (float*)ccmalloc(sizeof(float) * wnum);
	cudaMemcpy(iw, w, sizeof(float) * wnum, cudaMemcpyDeviceToHost);
	int rows = wnum / (count * channels);
	int i, j, k;
	for (i = 0; i < rows; i++)
		for (j = 0; j < channels; j++)
			for (k = 0; k < count; k++)
				hw[i * channels * count + k * channels + j] = iw[i * channels * count + j * count + k];
	ccfree(iw);
}

static void _cwc_convnet_host_synchronize(ccv_convnet_t* convnet)
{
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->layers + i;
		ccv_convnet_layer_t* host_layer = convnet->layers + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_cwc_convnet_reorder_convolutional_weights_onto_host(layer->w, host_layer->w, layer->wnum, layer->net.convolutional.count, layer->net.convolutional.channels, layer->input.matrix.partition);
				cudaMemcpy(host_layer->bias, layer->bias, sizeof(float) * layer->net.convolutional.count, cudaMemcpyDeviceToHost);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_cwc_convnet_reorder_full_connect_weights_onto_host(layer->w, host_layer->w, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
				cudaMemcpy(host_layer->bias, layer->bias, sizeof(float) * layer->net.full_connect.count, cudaMemcpyDeviceToHost);
				assert(cudaGetLastError() == cudaSuccess);
				break;
		}
	}
}

static void _cwc_convnet_supervised_train_function_state_write(cwc_convnet_supervised_train_function_state_t* z, const char* filename)
{
	_cwc_convnet_host_synchronize(z->convnet);
	ccv_convnet_write_param_t params;
	params.half_precision = 0;
	ccv_convnet_write(z->convnet, filename, params);
	sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		const char function_state_create_table_qs[] =
			"CREATE TABLE IF NOT EXISTS function_state "
			"(fsid INTEGER PRIMARY KEY ASC, t INTEGER, i INTEGER, inum INTEGER, line_no INTEGER, idx BLOB, eigenvectors BLOB, eigenvalues BLOB);"
			"CREATE TABLE IF NOT EXISTS momentum_data "
			"(layer INTEGER PRIMARY KEY ASC, weight BLOB, bias BLOB);";
		assert(SQLITE_OK == sqlite3_exec(db, function_state_create_table_qs, 0, 0, 0));
		const char function_state_insert_qs[] =
			"REPLACE INTO function_state "
			"(fsid, t, i, inum, line_no, idx, eigenvectors, eigenvalues) VALUES "
			"(0, $t, $i, $inum, $line_no, $idx, $eigenvectors, $eigenvalues);";
		sqlite3_stmt* function_state_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, function_state_insert_qs, sizeof(function_state_insert_qs), &function_state_insert_stmt, 0));
		sqlite3_bind_int(function_state_insert_stmt, 1, z->t);
		sqlite3_bind_int(function_state_insert_stmt, 2, z->i);
		sqlite3_bind_int(function_state_insert_stmt, 3, z->inum);
		sqlite3_bind_int(function_state_insert_stmt, 4, z->line_no);
		sqlite3_bind_blob(function_state_insert_stmt, 5, z->idx, sizeof(int) * z->inum, SQLITE_STATIC);
		if (z->eigenvectors)
			sqlite3_bind_blob(function_state_insert_stmt, 6, z->eigenvectors->data.u8, sizeof(double) * 3 * 3, SQLITE_STATIC);
		if (z->eigenvalues)
			sqlite3_bind_blob(function_state_insert_stmt, 7, z->eigenvalues->data.u8, sizeof(double) * 3, SQLITE_STATIC);
		assert(SQLITE_DONE == sqlite3_step(function_state_insert_stmt));
		sqlite3_finalize(function_state_insert_stmt);
		const char momentum_data_insert_qs[] =
			"REPLACE INTO momentum_data "
			"(layer, weight, bias) VALUES ($layer, $weight, $bias);";
		sqlite3_stmt* momentum_data_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, momentum_data_insert_qs, sizeof(momentum_data_insert_qs), &momentum_data_insert_stmt, 0));
		int i;
		for (i = 0; i < z->convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = GPU(z->convnet)->layers + i;
			ccv_convnet_layer_t* momentum = GPU(z->convnet)->momentums + i;
			// insert momentum data
			if (layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT)
			{
				sqlite3_bind_int(momentum_data_insert_stmt, 1, i);
				float* w = (float*)ccmalloc(sizeof(float) * (layer->wnum + (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count)));
				float* bias = w + layer->wnum;
				switch (layer->type)
				{
					case CCV_CONVNET_CONVOLUTIONAL:
						_cwc_convnet_reorder_convolutional_weights_onto_host(momentum->w, w, layer->wnum, layer->net.convolutional.count, layer->net.convolutional.channels, layer->input.matrix.partition);
						cudaMemcpy(bias, momentum->bias, sizeof(float) * layer->net.convolutional.count, cudaMemcpyDeviceToHost);
						assert(cudaGetLastError() == cudaSuccess);
						break;
					case CCV_CONVNET_FULL_CONNECT:
						_cwc_convnet_reorder_full_connect_weights_onto_host(momentum->w, w, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
						cudaMemcpy(bias, momentum->bias, sizeof(float) * layer->net.full_connect.count, cudaMemcpyDeviceToHost);
						assert(cudaGetLastError() == cudaSuccess);
						break;
				}
				sqlite3_bind_blob(momentum_data_insert_stmt, 2, w, sizeof(float) * layer->wnum, SQLITE_STATIC);
				sqlite3_bind_blob(momentum_data_insert_stmt, 3, bias, sizeof(float) * (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count), SQLITE_STATIC);
				assert(SQLITE_DONE == sqlite3_step(momentum_data_insert_stmt));
				sqlite3_reset(momentum_data_insert_stmt);
				sqlite3_clear_bindings(momentum_data_insert_stmt);
				ccfree(w);
			}
		}
		sqlite3_finalize(momentum_data_insert_stmt);
		sqlite3_close(db);
	}
}

#endif

#ifndef CASE_TESTS

void cwc_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch)
{
	_cwc_convnet_alloc_reserved(convnet, batch, 0);
	_cwc_convnet_encode_impl(convnet, 0, batch, 0, 0);
}

void cwc_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int symmetric, ccv_array_t** ranks, int tops, int batch)
{
	_cwc_convnet_alloc_reserved(convnet, batch, 0);
}

void cwc_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, const char* filename, ccv_convnet_train_param_t params)
{
#ifdef HAVE_GSL
	assert(params.mini_batch % BATCH_PER_BLOCK == 0);
	_cwc_convnet_alloc_reserved(convnet, params.mini_batch, params.layer_params);
	int i, j, k;
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	int aligned_padding = categorizeds->rnum % params.mini_batch;
	int aligned_rnum = categorizeds->rnum - aligned_padding;
	int aligned_batches = categorizeds->rnum / params.mini_batch;
	int* idx = (int*)ccmalloc(sizeof(int) * (categorizeds->rnum + aligned_padding));
	for (i = 0; i < categorizeds->rnum; i++)
		idx[i] = i;
	params.iterations = ccv_min(params.iterations, aligned_batches);
	gsl_ran_shuffle(rng, idx, categorizeds->rnum, sizeof(int));
	// the last layer has to be full connect, thus we can use it as softmax layer
	assert(convnet->layers[convnet->count - 1].type == CCV_CONVNET_FULL_CONNECT);
	int category_count = convnet->layers[convnet->count - 1].net.full_connect.count;
	struct {
		int* host;
		int* device;
	} test_returns[2];
	test_returns[0].host = test_returns[1].host = 0;
	test_returns[0].device = test_returns[1].device = 0;
	for (i = 0; i < 2; i++)
	{
		cudaMallocHost(&test_returns[i].host, sizeof(int) * params.mini_batch);
		assert(test_returns[i].host);
		cudaMalloc(&test_returns[i].device, sizeof(int) * params.mini_batch);
		assert(test_returns[i].device);
	}
	cudaEvent_t start, stop, iteration;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&iteration);
	cwc_convnet_supervised_train_function_state_t z;
	z.idx = idx;
	z.inum = categorizeds->rnum;
	z.convnet = convnet;
	z.eigenvectors = 0;
	z.eigenvalues = 0;
	z.line_no = 0;
	int miss;
	float elapsed_time;
	ccv_function_state_begin(_cwc_convnet_supervised_train_function_state_read, z, filename);
	_cwc_convnet_mean_formation(categorizeds, z.convnet->input, z.convnet->channels, params.symmetric, &z.convnet->mean_activity);
	ccv_function_state_resume(_cwc_convnet_supervised_train_function_state_write, z, filename);
	if (z.convnet->channels == 3 && params.color_gain > 0) // do this if we want color gain type of data augmentation, and it is RGB color
		_cwc_convnet_channel_eigen(categorizeds, z.convnet->mean_activity, z.convnet->input, z.convnet->channels, &z.eigenvectors, &z.eigenvalues);
	ccv_function_state_resume(_cwc_convnet_supervised_train_function_state_write, z, filename);
	for (z.t = 0; z.t < params.max_epoch; z.t++)
	{
		for (z.i = 0; z.i < aligned_batches; z.i += params.iterations)
		{
			cudaEventRecord(start, 0);
			// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in updating
			if (z.t > 0) // undo the mean network for further training
				_cwc_convnet_dor_mean_net_undo(z.convnet, params.layer_params, GPU(z.convnet)->contexts[(z.i + 1) % 2].device.cublas);
			miss = 0;
			// run updates
			for (i = z.i; i < ccv_min(z.i + params.iterations, aligned_batches); i++)
			{
				cwc_convnet_context_t* context = GPU(z.convnet)->contexts + (i % 2);
				_cwc_convnet_batch_formation(rng, categorizeds, z.convnet->mean_activity, z.eigenvectors, z.eigenvalues, params.color_gain, z.idx, z.convnet->input, z.convnet->rows, z.convnet->cols, z.convnet->channels, params.symmetric, params.mini_batch, i * params.mini_batch, params.mini_batch, context->host.input, context->host.c);
				cudaMemcpyAsync(context->device.input, context->host.input, sizeof(float) * z.convnet->rows * z.convnet->cols * z.convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				cudaMemcpyAsync(context->device.c, context->host.c, sizeof(int) * params.mini_batch, cudaMemcpyHostToDevice, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				_cwc_convnet_dor_formation(z.convnet, params.mini_batch, rng, params.layer_params, context);
				assert(cudaGetLastError() == cudaSuccess);
				// sync with the other stream core so that we can compute on the single true layer parameters
				if (i > z.i)
					cudaEventRecord(stop, GPU(z.convnet)->contexts[(i + 1) % 2].device.stream);
				cudaStreamSynchronize(GPU(z.convnet)->contexts[(i + 1) % 2].device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				if (i > z.i) // we have another result, pull these
				{
					int* c = GPU(z.convnet)->contexts[(i + 1) % 2].host.c;
					for (k = 0; k < params.mini_batch; k++)
						if (c[k] != test_returns[(i + 1) % 2].host[k])
							++miss;
					cudaEventElapsedTime(&elapsed_time, iteration, stop);
					FLUSH(" - at epoch %03d / %d => stochastic gradient descent with miss rate %.2f%% at %d / %d (%.3f sec)", z.t + 1, params.max_epoch, miss * 100.0f /((i - z.i) * params.mini_batch), i + 1, aligned_batches, elapsed_time / 1000);
				}
				cudaEventRecord(iteration, context->device.stream);
				_cwc_convnet_encode_impl(z.convnet, context->device.input, params.mini_batch, 1, context);
				assert(cudaGetLastError() == cudaSuccess);
				// compute miss rate on training data
				_cwc_convnet_tests_return(params.mini_batch, category_count, GPU(z.convnet)->forwards[z.convnet->count - 1], test_returns[i % 2].device, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				cudaMemcpyAsync(test_returns[i % 2].host, test_returns[i % 2].device, sizeof(int) * params.mini_batch, cudaMemcpyDeviceToHost, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				// do the logistic loss and backward propagate
				_cwc_convnet_softmax_with_logistic_loss(params.mini_batch, category_count, GPU(z.convnet)->forwards[z.convnet->count - 1], context->device.c, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				_cwc_convnet_backwards_propagate_error(z.convnet, GPU(z.convnet)->forwards[z.convnet->count - 1], context->device.input, params.mini_batch, context);
				assert(cudaGetLastError() == cudaSuccess);
				_cwc_convnet_net_sgd(z.convnet, z.t > 0 || i > 0, params.mini_batch, params.layer_params, context);
				assert(cudaGetLastError() == cudaSuccess);
			}
			cudaDeviceSynchronize(); // synchronize at this point
			// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in testing
			_cwc_convnet_dor_mean_net(z.convnet, params.layer_params, GPU(z.convnet)->contexts[1].device.cublas);
			// run tests
			miss = 0;
			for (i = j = 0; i < tests->rnum; i += params.mini_batch, j++)
			{
				cwc_convnet_context_t* context = GPU(z.convnet)->contexts + (j % 2);
				_cwc_convnet_batch_formation(0, tests, z.convnet->mean_activity, 0, 0, 0, 0, z.convnet->input, z.convnet->rows, z.convnet->cols, z.convnet->channels, params.symmetric, params.mini_batch, i, ccv_min(params.mini_batch, tests->rnum - i), context->host.input, 0);
				cudaMemcpyAsync(context->device.input, context->host.input, sizeof(float) * z.convnet->rows * z.convnet->cols * z.convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				if (j > 0)
					cudaEventRecord(stop, GPU(z.convnet)->contexts[(j + 1) % 2].device.stream);
				// sync with the other stream core so that we can compute on the single true layer parameters
				cudaStreamSynchronize(GPU(z.convnet)->contexts[(j + 1) % 2].device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				if (j > 0) // we have another result, pull these
				{
					for (k = 0; k < params.mini_batch; k++)
					{
						ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, k + i - params.mini_batch);
						if (test->c != test_returns[(j + 1) % 2].host[k])
							++miss;
					}
					cudaEventElapsedTime(&elapsed_time, iteration, stop);
					FLUSH(" - at epoch %03d / %d => with miss rate %.2f%% at %d / %d (%.3f sec)", z.t + 1, params.max_epoch, miss * 100.0f / i, j + 1, (tests->rnum + params.mini_batch - 1) / params.mini_batch, elapsed_time / 1000);
				}
				cudaEventRecord(iteration, context->device.stream);
				_cwc_convnet_encode_impl(z.convnet, context->device.input, params.mini_batch, 0, context);
				assert(cudaGetLastError() == cudaSuccess);
				_cwc_convnet_tests_return(params.mini_batch, category_count, GPU(z.convnet)->forwards[z.convnet->count - 1], test_returns[j % 2].device, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
				cudaMemcpyAsync(test_returns[j % 2].host, test_returns[j % 2].device, sizeof(int) * params.mini_batch, cudaMemcpyDeviceToHost, context->device.stream);
				assert(cudaGetLastError() == cudaSuccess);
			}
			cudaDeviceSynchronize(); // synchronize at this point
			for (i = 0; i <= (tests->rnum - 1) % params.mini_batch; i++)
			{
				ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, i + (tests->rnum - 1) / params.mini_batch * params.mini_batch);
				if (test->c != test_returns[(j + 1) % 2].host[i])
					++miss;
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			elapsed_time = 0;
			cudaEventElapsedTime(&elapsed_time, start, stop);
			FLUSH(" - at epoch %03d / %d (%03d - %d) => with miss rate %.2f%% (%.3f sec)\n", z.t + 1, params.max_epoch, z.i + 1, ccv_min(z.i + params.iterations, aligned_batches), miss * 100.0f / tests->rnum, elapsed_time / 1000);
			ccv_function_state_resume(_cwc_convnet_supervised_train_function_state_write, z, filename);
		}
		if (z.t + 1 < params.max_epoch)
		{
			// reshuffle the parts we visited and move the rest to the beginning
			memcpy(z.idx + categorizeds->rnum, z.idx + aligned_rnum, sizeof(int) * aligned_padding);
			memmove(z.idx + aligned_padding, z.idx, sizeof(int) * aligned_rnum);
			memcpy(z.idx, z.idx + categorizeds->rnum, sizeof(int) * aligned_padding);
			gsl_ran_shuffle(rng, z.idx + aligned_padding, aligned_rnum, sizeof(int));
		}
	}
	ccv_function_state_finish();
	cudaEventDestroy(start);
	cudaEventDestroy(iteration);
	cudaEventDestroy(stop);
	for (i = 0; i < 2; i++)
	{
		cudaFree(test_returns[i].device);
		cudaFreeHost(test_returns[i].host);
	}
	ccfree(z.idx);
	gsl_rng_free(rng);
	GPU(convnet)->layer_params = 0;
#else
	printf("cwc_convnet_supervised_train requires GSL library support");
	exit(-1);
#endif
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
		convnet->reserved = 0;
	}
}

#endif
