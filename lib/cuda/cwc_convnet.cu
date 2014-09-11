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
#include "../inl/ccv_convnet_inl.h"

// this structure holds intermediate on-device memory representation of convnet

typedef struct {
	// on host
	struct {
		float* input; // input per batch
		int* c; // class
		float* out; // confidence score
		float** dor; // dropout regulator, in this version I generate dor on CPU because it is lightweight and gsl has shuffle method, which is better suited for this (and faster than per-node randomization)
	} host[2];
	// on device
	struct {
		// this is modeled after Alex's "One Weird Trick", there are 3 join points for me: 1). forward pass from data parallelism to model parallelism; 2). compute logistic loss; 3). backward pass from model parallelism to data parallelism;
		cudaStream_t data_stream; // based on above description, we need 3 streams, one stream for data parallelism
		cudaStream_t model_stream[2]; // two streams for model parallelism (to overlap data transfer and computation
		// based on above description, we need 6 events (3 join points):
		// 0: in forward pass, when data parallelism is done, and model parallelism will start;
		// 1: in forward pass, the first stream's model parallelism is done;
		// 2: in forward pass, the second stream's model parallelism is done;
		// 3: in backward pass, when the error propagate starts, (thus, model parallelism starts);
		// 4: in backward pass, the first stream's model parallelism is done;
		// 5: in backward pass, the second stream's model parallelism is done;
		cudaEvent_t data_joint;
		cudaEvent_t model_joint[2];
		cublasHandle_t data_cublas; // the same, just cublas handle to stream
		cublasHandle_t model_cublas[2]; // the same, just cublas handle to stream
		float* input;
		int* c;
		float* out;
		float** dor;
	} device[2];
} cwc_convnet_context_t;

typedef struct {
	size_t memory_usage;
} cwc_convnet_stats_t;

typedef struct {
	int batch;
	int tops;
	int dual_device;
	ccv_convnet_layer_train_param_t* layer_params;
	struct {
		ccv_convnet_layer_t* layers;
		ccv_convnet_layer_t* configurations;
		ccv_convnet_layer_t* momentums;
		float** forwards; // the forward output layers
		float** backwards; // the backwards output layer
		float** denoms; // the denominator for rnorm layer, thus, backprop can reuse the value
		float** scans; // the scan layer to reformat outputs
		float* unit; // the unit vector for a batch, ease the GEMM on full-connect layer
		float* scratch; // the scratch space for temporary reuse, it will be max(wnum, input rows * cols * channels + output rows * cols * channels)
	} device[2];
	cwc_convnet_context_t contexts[2];
	cwc_convnet_stats_t stats;
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
#define THREAD_PER_BLOCK (16)

static int _cwc_convnet_layer_use_rows(ccv_convnet_layer_t* layer)
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

static void _cwc_convnet_alloc_layers(ccv_convnet_t* convnet, int device_id, int dual_device)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	for (i = 0; i < convnet->count; i++)
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				// allocating for layer
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count);
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				_cwc_convnet_reorder_convolutional_weights_onto_device(convnet->layers[i].w, layers[i].w, layers[i].wnum, layers[i].net.convolutional.count, layers[i].net.convolutional.channels, layers[i].input.matrix.partition);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias, sizeof(float) * layers[i].net.convolutional.count, cudaMemcpyHostToDevice);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				// allocating for layer
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count);
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				// if it is due device, rewind it from different parts of the model
				_cwc_convnet_reorder_full_connect_weights_onto_device(convnet->layers[i].w + (dual_device ? device_id * layers[i].wnum: 0), layers[i].w, layers[i].wnum, layers[i].input.matrix.rows * layers[i].input.matrix.cols, layers[i].input.matrix.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias + (dual_device ? device_id * layers[i].net.full_connect.count : 0), sizeof(float) * layers[i].net.full_connect.count, cudaMemcpyHostToDevice);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				layers[i].w = layers[i].bias = 0;
				break;
		}
}

static void _cwc_convnet_alloc_configurations(ccv_convnet_t* convnet, int device_id)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	ccv_convnet_layer_t* configurations = GPU(convnet)->device[device_id].configurations;
	for (i = 0; i < convnet->count; i++)
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				assert(configurations[i].type == CCV_CONVNET_CONVOLUTIONAL);
				// allocating for configurations 
				configurations[i].w = 0;
				cudaMalloc(&configurations[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count);
				assert(configurations[i].w);
				configurations[i].bias = configurations[i].w + layers[i].wnum;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				assert(configurations[i].type == CCV_CONVNET_FULL_CONNECT);
				// allocating for configurations 
				configurations[i].w = 0;
				cudaMalloc(&configurations[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count);
				assert(configurations[i].w);
				configurations[i].bias = configurations[i].w + layers[i].wnum;
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				assert(configurations[i].type == layers[i].type);
				configurations[i].w = configurations[i].bias = 0;
				break;
		}
}

static void _cwc_convnet_alloc_momentums(ccv_convnet_t* convnet, int device_id)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	ccv_convnet_layer_t* momentums = GPU(convnet)->device[device_id].momentums;
	for (i = 0; i < convnet->count; i++)
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				assert(momentums[i].type == CCV_CONVNET_CONVOLUTIONAL);
				// allocating for momentums
				momentums[i].w = 0;
				cudaMalloc(&momentums[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count);
				assert(momentums[i].w);
				momentums[i].bias = momentums[i].w + layers[i].wnum;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				assert(momentums[i].type == CCV_CONVNET_FULL_CONNECT);
				// allocating for momentums
				momentums[i].w = 0;
				cudaMalloc(&momentums[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count);
				assert(momentums[i].w);
				momentums[i].bias = momentums[i].w + layers[i].wnum;
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				assert(momentums[i].type == layers[i].type);
				momentums[i].w = momentums[i].bias = 0;
				break;
		}
}

static void _cwc_convnet_alloc_forwards(ccv_convnet_t* convnet, int device_id, int dual_device, int start, int length, int rows, int cols, int batch)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	assert(start >= 0 && start + length <= convnet->count);
	int out_rows, out_cols, out_partition;
	for (i = start; i < start + length; i++)
	{
		_ccv_convnet_layer_derive_output(layers + i, rows, cols, &out_rows, &out_cols, &out_partition);
		// if the next layer is full connect (model parallelism), the forwards neuron needs to hold 2 batches rather than 1
		int dual_batch = (i + 1 < start + length && layers[i + 1].type == CCV_CONVNET_FULL_CONNECT) ? batch * (dual_device + 1) : batch;
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				GPU(convnet)->device[device_id].forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * dual_batch);
				GPU(convnet)->stats.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * dual_batch;
				assert(GPU(convnet)->device[device_id].forwards[i]);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				GPU(convnet)->device[device_id].forwards[i] = 0;
				// for full connect layer, because it uses model parallelism, each layer needs to hold 2 batches and full outputs
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * layers[i].net.full_connect.count * batch * (dual_device + 1) * (dual_device + 1));
				GPU(convnet)->stats.memory_usage += sizeof(float) * layers[i].net.full_connect.count * batch * (dual_device + 1) * (dual_device + 1);
				assert(GPU(convnet)->device[device_id].forwards[i]);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				assert(i > 0);
				GPU(convnet)->device[device_id].forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * dual_batch);
				GPU(convnet)->stats.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * dual_batch;
				assert(GPU(convnet)->device[device_id].forwards[i]);
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				GPU(convnet)->device[device_id].forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * dual_batch);
				GPU(convnet)->stats.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * dual_batch;
				assert(GPU(convnet)->device[device_id].forwards[i]);
				break;
		}
		rows = out_rows, cols = out_cols;
	}
}

static void _cwc_convnet_alloc_denoms(ccv_convnet_t* convnet, int device_id, int start, int length, int rows, int cols, int batch)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	assert(start >= 0 && start + length <= convnet->count);
	int out_rows, out_cols, out_partition;
	for (i = start; i < start + length; i++)
	{
		_ccv_convnet_layer_derive_output(layers + i, rows, cols, &out_rows, &out_cols, &out_partition);
		switch (layers[i].type)
		{
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				GPU(convnet)->device[device_id].denoms[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].denoms[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch);
				GPU(convnet)->stats.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * batch;
				assert(GPU(convnet)->device[device_id].denoms[i]);
				break;
			case CCV_CONVNET_CONVOLUTIONAL:
			case CCV_CONVNET_FULL_CONNECT:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				GPU(convnet)->device[device_id].denoms[i] = 0;
				break;
		}
		rows = out_rows, cols = out_cols;
	}
}

static void _cwc_convnet_alloc_backwards(ccv_convnet_t* convnet, int device_id, int dual_device, int batch)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	// find the layer with max memory usage, and then allocate two of them, because for backward propagate, no need to preserve the results
	size_t max_memory_usage = 0;
	for (i = 0; i < convnet->count; i++)
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				max_memory_usage = ccv_max(max_memory_usage, sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				// for full connect layer, because it uses model parallelism, each layer needs to hold 2 batches
				max_memory_usage = ccv_max(max_memory_usage, sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch * (dual_device + 1));
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				max_memory_usage = ccv_max(max_memory_usage, sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				max_memory_usage = ccv_max(max_memory_usage, sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				break;
		}
	assert(convnet->count > 2);
	// allocate two layers
	GPU(convnet)->device[device_id].backwards[0] = 0;
	for (i = 1; i < 3; i++)
	{
		GPU(convnet)->device[device_id].backwards[i] = 0;
		cudaMalloc(&GPU(convnet)->device[device_id].backwards[i], max_memory_usage);
		GPU(convnet)->stats.memory_usage += max_memory_usage;
		assert(GPU(convnet)->device[device_id].backwards[i]);
	}
	for (i = 3; i < convnet->count; i += 2)
		GPU(convnet)->device[device_id].backwards[i] = GPU(convnet)->device[device_id].backwards[1];
	for (i = 4; i < convnet->count; i += 2)
		GPU(convnet)->device[device_id].backwards[i] = GPU(convnet)->device[device_id].backwards[2];
}

static void _cwc_convnet_alloc_dor(ccv_convnet_t* convnet, int device_id, int batch, ccv_convnet_layer_train_param_t* layer_params)
{
	int i, j;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	int rows = convnet->rows;
	int cols = convnet->cols;
	int out_rows, out_cols, out_partition;
	for (i = 0; i < convnet->count; i++)
	{
		_ccv_convnet_layer_derive_output(layers + i, rows, cols, &out_rows, &out_cols, &out_partition);
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					context->host[device_id].dor[i] = 0;
					context->device[device_id].dor[i] = 0;
					if (layer_params && layer_params[i].dor > 0)
					{
						assert(i > 0);
						cudaMallocHost(&context->host[device_id].dor[i], sizeof(float) * batch * out_rows * out_cols * layers[i].net.convolutional.count);
						assert(context->host[device_id].dor[i]);
						cudaMalloc(&context->device[device_id].dor[i], sizeof(float) * batch * out_rows * out_cols * layers[i].net.convolutional.count);
						GPU(convnet)->stats.memory_usage += sizeof(float) * batch * out_rows * out_cols * layers[i].net.convolutional.count;
						assert(context->device[device_id].dor[i]);
					}
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					context->host[device_id].dor[i] = 0;
					context->device[device_id].dor[i] = 0;
					if (layer_params && layer_params[i].dor > 0)
					{
						cudaMallocHost(&context->host[device_id].dor[i], sizeof(float) * batch * layers[i].net.full_connect.count);
						assert(context->host[device_id].dor[i]);
						cudaMalloc(&context->device[device_id].dor[i], sizeof(float) * batch * layers[i].net.full_connect.count);
						GPU(convnet)->stats.memory_usage += sizeof(float) * batch * layers[i].net.full_connect.count;
						assert(context->device[device_id].dor[i]);
					}
				}
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					context->host[device_id].dor[i] = 0;
					context->device[device_id].dor[i] = 0;
				}
				break;
		}
		rows = out_rows, cols = out_cols;
	}
}

static void _cwc_convnet_alloc_input(ccv_convnet_t* convnet, int device_id, int context_id, int rows, int cols, int batch)
{
	cwc_convnet_context_t* context = GPU(convnet)->contexts + context_id;
	context->host[device_id].input = 0;
	cudaMallocHost(&context->host[device_id].input, sizeof(float) * rows * cols * convnet->channels * batch); 
	assert(context->host[device_id].input);
	context->device[device_id].input = 0;
	cudaMalloc(&context->device[device_id].input, sizeof(float) * rows * cols * convnet->channels * batch);
	GPU(convnet)->stats.memory_usage += sizeof(float) * rows * cols * convnet->channels * batch;
	assert(context->device[device_id].input);
}

static void _cwc_convnet_alloc_c(ccv_convnet_t* convnet, int device_id, int context_id, int batch)
{
	cwc_convnet_context_t* context = GPU(convnet)->contexts + context_id;
	context->host[device_id].c = 0;
	cudaMallocHost(&context->host[device_id].c, sizeof(int) * batch); 
	assert(context->host[device_id].c);
	context->device[device_id].c = 0;
	cudaMalloc(&context->device[device_id].c, sizeof(int) * batch); 
	GPU(convnet)->stats.memory_usage += sizeof(int) * batch;
}

static void _cwc_convnet_alloc_out(ccv_convnet_t* convnet, int device_id, int context_id, int batch)
{
	cwc_convnet_context_t* context = GPU(convnet)->contexts + context_id;
	context->host[device_id].out = 0;
	cudaMallocHost(&context->host[device_id].out, sizeof(float) * batch); 
	assert(context->host[device_id].out);
	context->device[device_id].out = 0;
	cudaMalloc(&context->device[device_id].out, sizeof(float) * batch); 
	GPU(convnet)->stats.memory_usage += sizeof(float) * batch;
}

static void _cwc_convnet_alloc_context(ccv_convnet_t* convnet, int device_id, int context_id, int dual_device)
{
	cwc_convnet_context_t* context = GPU(convnet)->contexts + context_id;
	cudaStreamCreate(&context->device[device_id].data_stream);
	cublasCreate(&context->device[device_id].data_cublas);
	cublasSetStream(context->device[device_id].data_cublas, context->device[device_id].data_stream);
	int i;
	if (dual_device)
	{
		// only allocate model parallelism stream / cublas handle / joint events when dual device mode is on
		for (i = 0; i < 2; i++)
		{
			cudaStreamCreate(&context->device[device_id].model_stream[i]);
			cublasCreate(&context->device[device_id].model_cublas[i]);
			cublasSetStream(context->device[device_id].model_cublas[i], context->device[device_id].model_stream[i]);
			cudaEventCreateWithFlags(&context->device[device_id].model_joint[i], cudaEventDisableTiming);
		}
		cudaEventCreateWithFlags(&context->device[device_id].data_joint, cudaEventDisableTiming);
	} else {
		for (i = 0; i < 2; i++)
		{
			context->device[device_id].model_stream[i] = 0;
			context->device[device_id].model_cublas[i] = 0;
			context->device[device_id].model_joint[i] = 0;
		}
		context->device[device_id].data_joint = 0;
	}
}

static void _cwc_convnet_alloc_scratch(ccv_convnet_t* convnet, int device_id, int batch)
{
	int i;
	int out_rows, out_cols, out_partition;
	size_t scratch_space = 0;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	for (i = 0; i < convnet->count; i++)
		if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
		{
			int use_rows = _cwc_convnet_layer_use_rows(layers + i);
			_ccv_convnet_layer_derive_output(layers + i, layers[i].input.matrix.rows, layers[i].input.matrix.cols, &out_rows, &out_cols, &out_partition);
			scratch_space = ccv_max(scratch_space, layers[i].wnum);
			scratch_space = ccv_max(scratch_space,
					out_rows * out_cols * layers[i].net.convolutional.count * batch + // output layer reorder
					layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch + // input layer reorder
					layers[i].wnum * (use_rows ? out_rows : 1) * (batch / BATCH_PER_BLOCK)); // unconsolidated weights output
		}
	GPU(convnet)->device[device_id].scratch = 0;
	cudaMalloc(&GPU(convnet)->device[device_id].scratch, sizeof(float) * scratch_space);
	assert(GPU(convnet)->device[device_id].scratch);
	GPU(convnet)->stats.memory_usage += sizeof(float) * scratch_space;
}

static void _cwc_convnet_make_unit(ccv_convnet_t* convnet, int device_id, int batch)
{
	int i;
	int out_rows, out_cols, out_partition;
	size_t unit_size = batch;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	for (i = 0; i < convnet->count; i++)
		if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
		{
			_ccv_convnet_layer_derive_output(layers + i, layers[i].input.matrix.rows, layers[i].input.matrix.cols, &out_rows, &out_cols, &out_partition);
			if (_cwc_convnet_layer_use_rows(layers + i))
				unit_size = ccv_max(unit_size, out_rows * (batch / BATCH_PER_BLOCK));
			unit_size = ccv_max(unit_size, out_rows * out_cols * batch);
		}
	float* unit = 0;
	cudaMallocHost(&unit, sizeof(float) * unit_size);
	for (i = 0; i < unit_size; i++)
		unit[i] = 1;
	GPU(convnet)->device[device_id].unit = 0;
	cudaMalloc(&GPU(convnet)->device[device_id].unit, sizeof(float) * unit_size);
	GPU(convnet)->stats.memory_usage += sizeof(float) * unit_size;
	cudaMemcpy(GPU(convnet)->device[device_id].unit, unit, sizeof(float) * unit_size, cudaMemcpyHostToDevice);
	cudaFreeHost(unit);
}

static void _cwc_convnet_alloc_scans(ccv_convnet_t* convnet, int device_id, int offset, int batch)
{
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		GPU(convnet)->device[device_id].scans[i] = 0;
		if (i == offset)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + offset + 1;
			cudaMalloc(&GPU(convnet)->device[device_id].scans[i], sizeof(float) * layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch);
			GPU(convnet)->stats.memory_usage += sizeof(float) * layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch;
			assert(GPU(convnet)->device[device_id].scans[i]);
		}
	}
}

// find the layer for scanning (it is the last convolutional layer)
static int _cwc_convnet_find_scan(ccv_convnet_t* convnet, int device_id)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	for (i = convnet->count - 1; i >= 0; i--)
		if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
			return i;
	return -1;
}

// allocate reserved for only forward path, this is only interesting to single-device mode
static void _cwc_convnet_alloc_reserved_for_classify(ccv_convnet_t* convnet, int tops, int batch)
{
	if (GPU(convnet) && (GPU(convnet)->batch != batch || GPU(convnet)->tops != tops || GPU(convnet)->layer_params != 0))
		ccv_convnet_compact(convnet);
	else if (GPU(convnet))
		return; // it is allocated properly, no-op
	convnet->reserved = (cwc_convnet_t*)ccmalloc(sizeof(cwc_convnet_t) + sizeof(cwc_convnet_layer_vary_t) * convnet->count + sizeof(ccv_convnet_layer_t) * convnet->count + sizeof(float*) * convnet->count * 3);
	GPU(convnet)->batch = batch;
	GPU(convnet)->tops = tops;
	GPU(convnet)->dual_device = 0;
	GPU(convnet)->layer_params = 0;
	GPU(convnet)->stats.memory_usage = 0;
	cwc_convnet_layer_vary_t* layer_vary = (cwc_convnet_layer_vary_t*)(GPU(convnet) + 1);
	memset(layer_vary, 0, sizeof(cwc_convnet_layer_vary_t) * convnet->count);
	GPU(convnet)->device[0].layers = (ccv_convnet_layer_t*)(layer_vary + convnet->count);
	memcpy(GPU(convnet)->device[0].layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	ccv_convnet_layer_t* layers = GPU(convnet)->device[0].layers;
	// point reserved place to layer_vary
	int i;
	for (i = 0; i < convnet->count; i++)
		if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
			layers[i].reserved = layer_vary + i;
	// alloc and copy layers
	_cwc_convnet_alloc_layers(convnet, 0, 0);
	GPU(convnet)->device[0].configurations = 0;
	GPU(convnet)->device[0].momentums = 0;
	GPU(convnet)->device[0].scratch = 0;
	_cwc_convnet_make_unit(convnet, 0, batch * 30);
	int scan = _cwc_convnet_find_scan(convnet, 0);
	GPU(convnet)->device[0].forwards = (float**)(GPU(convnet)->device[0].layers + convnet->count);
	// alloc forwards until the scan layer (for initial 6 patches)
	_cwc_convnet_alloc_forwards(convnet, 0, 0, 0, scan + 1, convnet->input.height, convnet->input.width, batch * 6);
	// alloc forwards from scan layer (for scanned 30 patches)
	_cwc_convnet_alloc_forwards(convnet, 0, 0, scan + 1, convnet->count - scan - 1, GPU(convnet)->device[0].layers[scan + 1].input.matrix.rows, GPU(convnet)->device[0].layers[scan + 1].input.matrix.cols, batch * 30);
	GPU(convnet)->device[0].denoms = (float**)(GPU(convnet)->device[0].layers + convnet->count) + convnet->count;
	// alloc until the scan layer
	_cwc_convnet_alloc_denoms(convnet, 0, 0, scan + 1, convnet->input.height, convnet->input.width, batch * 6);
	// alloc denoms from scan layer to the end
	_cwc_convnet_alloc_denoms(convnet, 0, scan + 1, convnet->count - scan - 1, GPU(convnet)->device[0].layers[scan + 1].input.matrix.rows, GPU(convnet)->device[0].layers[scan + 1].input.matrix.cols, batch * 30);
	// alloc scan layer
	GPU(convnet)->device[0].scans = (float**)(GPU(convnet)->device[0].layers + convnet->count) + convnet->count * 2;
	_cwc_convnet_alloc_scans(convnet, 0, scan, batch * 30);
	GPU(convnet)->device[0].backwards = 0;
	GPU(convnet)->contexts[0].host[0].dor = GPU(convnet)->contexts[0].device[0].dor = 0;
	_cwc_convnet_alloc_input(convnet, 0, 0, convnet->input.height, convnet->input.width, batch * 6);
	_cwc_convnet_alloc_c(convnet, 0, 0, batch * tops);
	_cwc_convnet_alloc_out(convnet, 0, 0, batch * tops);
	_cwc_convnet_alloc_context(convnet, 0, 0, 0);
	GPU(convnet)->contexts[1].host[0].dor = GPU(convnet)->contexts[1].device[0].dor = 0;
	GPU(convnet)->contexts[1].host[0].input = GPU(convnet)->contexts[1].device[0].input = 0;
	GPU(convnet)->contexts[1].host[0].c = GPU(convnet)->contexts[1].device[0].c = 0;
	GPU(convnet)->contexts[1].host[0].out = GPU(convnet)->contexts[1].device[0].out = 0;
	GPU(convnet)->contexts[1].device[0].data_stream = 0;
	GPU(convnet)->contexts[1].device[0].data_cublas = 0;
	GPU(convnet)->contexts[1].device[0].data_joint = 0;
	for (i = 0; i < 2; i++)
	{
		GPU(convnet)->contexts[1].device[0].model_stream[i] = 0;
		GPU(convnet)->contexts[1].device[0].model_cublas[i] = 0;
		GPU(convnet)->contexts[1].device[0].model_joint[i] = 0;
	}
}

// allocate reserved for both forward and backward path
static void _cwc_convnet_alloc_reserved_both(ccv_convnet_t* convnet, int batch, int dual_device, ccv_convnet_layer_train_param_t* layer_params)
{
	if (GPU(convnet) && (GPU(convnet)->batch != batch || GPU(convnet)->tops != 0 || GPU(convnet)->dual_device != dual_device || GPU(convnet)->layer_params != layer_params))
		ccv_convnet_compact(convnet);
	else if (GPU(convnet))
		return; // it is allocated properly, no-op
	assert(dual_device == !!dual_device);
	uint8_t* reserved = (uint8_t*)ccmalloc(sizeof(cwc_convnet_t) + (sizeof(cwc_convnet_layer_vary_t) * convnet->count + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 10) * (dual_device + 1));
	convnet->reserved = (cwc_convnet_t*)reserved;
	GPU(convnet)->batch = batch;
	GPU(convnet)->tops = 0;
	GPU(convnet)->dual_device = dual_device;
	GPU(convnet)->layer_params = layer_params;
	GPU(convnet)->stats.memory_usage = 0;
	int i, device_id;
	for (device_id = 0; device_id < dual_device + 1; device_id++)
	{
		cudaSetDevice(device_id);
		GPU(convnet)->device[device_id].scans = 0;
		cwc_convnet_layer_vary_t* layer_vary = (cwc_convnet_layer_vary_t*)(reserved + sizeof(cwc_convnet_t) + (sizeof(cwc_convnet_layer_vary_t) * convnet->count + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 10) * device_id);
		memset(layer_vary, 0, sizeof(cwc_convnet_layer_vary_t) * convnet->count);
		GPU(convnet)->device[device_id].layers = (ccv_convnet_layer_t*)(layer_vary + convnet->count);
		memcpy(GPU(convnet)->device[device_id].layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
		ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
		for (i = 0; i < convnet->count; i++)
			// point reserved place to layer_vary
			if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
				layers[i].reserved = layer_vary + i;
			// depends on if it is dual_device or not, full_connect will use model parallelism, therefore, here we split the model into half
			else if (layers[i].type == CCV_CONVNET_FULL_CONNECT)
			{
				assert(convnet->layers[i].net.full_connect.count % (dual_device + 1) == 0);
				layers[i].net.full_connect.count = convnet->layers[i].net.full_connect.count / (dual_device + 1);
				layers[i].wnum = layers[i].net.full_connect.count * layers[i].input.node.count;
			}
		// hook up configurations (the backprop coefficients)
		GPU(convnet)->device[device_id].configurations = GPU(convnet)->device[device_id].layers + convnet->count;
		memcpy(GPU(convnet)->device[device_id].configurations, layers, sizeof(ccv_convnet_layer_t) * convnet->count);
		// hook up momentums
		GPU(convnet)->device[device_id].momentums = GPU(convnet)->device[device_id].layers + convnet->count * 2;
		memcpy(GPU(convnet)->device[device_id].momentums, layers, sizeof(ccv_convnet_layer_t) * convnet->count);
		// alloc and copy layers
		_cwc_convnet_alloc_layers(convnet, device_id, dual_device);
		// alloc scratch space (for backprop on convolutional layer)
		_cwc_convnet_alloc_scratch(convnet, device_id, batch);
		// alloc and make unit vector
		_cwc_convnet_make_unit(convnet, device_id, batch);
		// alloc & copy configurations (the backprop coefficients)
		_cwc_convnet_alloc_configurations(convnet, device_id);
		// alloc & copy momentums
		_cwc_convnet_alloc_momentums(convnet, device_id);
		// hook up forwards and alloc forwards
		GPU(convnet)->device[device_id].forwards = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3);
		_cwc_convnet_alloc_forwards(convnet, device_id, dual_device, 0, convnet->count, convnet->rows, convnet->cols, batch);
		// hook up denoms and alloc denoms
		GPU(convnet)->device[device_id].denoms = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3) + convnet->count * 2;
		_cwc_convnet_alloc_denoms(convnet, device_id, 0, convnet->count, convnet->rows, convnet->cols, batch);
		// hook up backwards and alloc backwards
		GPU(convnet)->device[device_id].backwards = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3) + convnet->count;
		// hook up dor and alloc dor
		_cwc_convnet_alloc_backwards(convnet, device_id, dual_device, batch);
		for (i = 0; i < 2; i++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + i;
			context->host[device_id].dor = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3) + convnet->count * 3 + convnet->count * i;
			context->device[device_id].dor = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3) + convnet->count * 5 + convnet->count * i;
		}
		_cwc_convnet_alloc_dor(convnet, device_id, batch, layer_params);
		// alloc contexts
		for (i = 0; i < 2; i++)
		{
			_cwc_convnet_alloc_input(convnet, device_id, i, convnet->rows, convnet->cols, batch);
			_cwc_convnet_alloc_c(convnet, device_id, i, batch);
			GPU(convnet)->contexts[i].host[device_id].out = 0;
			GPU(convnet)->contexts[i].device[device_id].out = 0;
			_cwc_convnet_alloc_context(convnet, device_id, i, dual_device);
		}
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
	_ccv_convnet_layer_derive_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
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
	assert(cudaGetLastError() == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, const cudaStream_t& stream)
{
	static int vary_x[] = { 1, 2, 4, 8 };
	static int vary_y[] = { 1, 2, 4, 6, 8 };
	static int vary_z[] = { 16, 24, 32, 36, 64, 72 };
	CWC_IMPLEMENT_VARY_STUB(VARY(layer)->convolutional.forward, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_forward_propagate_vary, layer, rows, cols, batch, a, b, stream);
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

static void _cwc_convnet_rnorm_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, float* denoms, const cudaStream_t& stream)
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

static void _cwc_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	_ccv_convnet_layer_derive_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
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

static void _cwc_convnet_average_pool_forward_propagate(ccv_convnet_layer_t* layer, int rows, int cols, int batch, float* a, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	_ccv_convnet_layer_derive_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
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

__global__ static void _cwc_kern_relu_forward_propagate(float* a)
{
	a += blockIdx.x * blockDim.x;
	const int thidx = threadIdx.x;
	a[thidx] = max(0.0, a[thidx]);
}

static void _cwc_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, float* batch_unit /* this is just 1's in device */, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols, out_partition;
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	out_cols = batch;
	rows = layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels;
	float alpha = 1;
	float beta = 0;
	// make copies of bias into db's columns, note that for cublas, it is row-major matrix
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch, out_rows, 1, &alpha, batch_unit, batch, layer->bias, 1, &beta, b, batch);
	beta = 1;
	// and then do the GEMM by adding bias
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch, out_rows, rows, &alpha, a, batch, layer->w, rows, &beta, b, batch);
	if (layer->net.full_connect.relu)
		_cwc_kern_relu_forward_propagate
		<<<layer->net.full_connect.count, batch, 0, stream>>>
		(b);

}

__global__ static void _cwc_kern_mute_neuron(float* a, float* d)
{
	a += blockIdx.x * blockDim.x;
	d += blockIdx.x * blockDim.x;
	const int thidx = threadIdx.x;
	a[thidx] = a[thidx] * d[thidx];
}

static void _cwc_convnet_layer_forward_propagate(ccv_convnet_layer_t* layer, int device_id, int k, int rows, int cols, int batch, int dor, float* a, float* b, float* denoms, float* batch_unit, cwc_convnet_context_t* context)
{
	switch (layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			_cwc_convnet_convolutional_forward_propagate(layer, rows, cols, batch, a, b, context->device[device_id].data_stream);
			if (dor && context->device[device_id].dor[k])
			{
				int out_rows, out_cols, out_partition;
				_ccv_convnet_layer_derive_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
				_cwc_kern_mute_neuron
				<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device[device_id].data_stream>>>
				(b, context->device[device_id].dor[k]);
			}
			break;
		case CCV_CONVNET_FULL_CONNECT:
			assert(k > 0);
			_cwc_convnet_full_connect_forward_propagate(layer, batch, a, b, batch_unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
			if (dor && context->device[device_id].dor[k])
				_cwc_kern_mute_neuron
				<<<layer->net.full_connect.count, batch, 0, context->device[device_id].data_stream>>>
				(b, context->device[device_id].dor[k]);
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			assert(k > 0);
			_cwc_convnet_rnorm_forward_propagate(layer, rows, cols, batch, a, b, denoms, context->device[device_id].data_stream);
			break;
		case CCV_CONVNET_MAX_POOL:
			assert(k > 0);
			_cwc_convnet_max_pool_forward_propagate(layer, rows, cols, batch, a, b, context->device[device_id].data_stream);
			break;
		case CCV_CONVNET_AVERAGE_POOL:
			assert(k > 0);
			_cwc_convnet_average_pool_forward_propagate(layer, rows, cols, batch,  a, b, context->device[device_id].data_stream);
			break;
	}
}

static int _cwc_convnet_first_full_connect(ccv_convnet_t* convnet)
{
	int i;
	for (i = 0; i < convnet->count; i++)
		if (convnet->layers[i].type == CCV_CONVNET_FULL_CONNECT)
			return i;
	return 0;
}

// assuming a is in device memory
static void _cwc_convnet_encode_impl(ccv_convnet_t* convnet, int dual_device, int batch, int dor, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	if (dual_device)
	{
		int device_id;
		int count = _cwc_convnet_first_full_connect(convnet);
		for (device_id = 0; device_id < dual_device + 1; device_id++)
		{
			cudaSetDevice(device_id);
			for (i = 0; i < count; i++)
			{
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				_cwc_convnet_layer_forward_propagate(layer, device_id, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, dor, i == 0 ? context->device[device_id].input : GPU(convnet)->device[device_id].forwards[i - 1], GPU(convnet)->device[device_id].forwards[i], GPU(convnet)->device[device_id].denoms[i], GPU(convnet)->device[device_id].unit, context);
			}
			cudaEventRecord(context->device[device_id].data_joint, context->device[device_id].data_stream);
		}
		// big synchronization point, need to wait for both device finished forward pass
		for (device_id = 0; device_id < dual_device + 1; device_id++)
		{
			cudaSetDevice(device_id);
			cudaStreamWaitEvent(context->device[device_id].model_stream[0], context->device[device_id].data_joint, 0);
			cudaStreamWaitEvent(context->device[device_id].model_stream[1], context->device[device_id].data_joint, 0);
			int other_device_id = (device_id + 1) & dual_device;
			cudaStreamWaitEvent(context->device[device_id].model_stream[0], context->device[other_device_id].data_joint, 0);
			cudaStreamWaitEvent(context->device[device_id].model_stream[1], context->device[other_device_id].data_joint, 0);
		}
		// the connecting layer from data parallelism to model parallelism
		// this is different because first, I need to copy the full batch from first GPU to the second GPU (and vice versa).
		for (device_id = 0; device_id < dual_device + 1; device_id++)
		{
			cudaSetDevice(device_id);
			int other_device_id = (device_id + 1) & dual_device;
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + count;
			assert(layer->type == CCV_CONVNET_FULL_CONNECT);
			_cwc_convnet_full_connect_forward_propagate(layer, batch,
					GPU(convnet)->device[device_id].forwards[count - 1],
					GPU(convnet)->device[device_id].forwards[count] + ((dual_device + 1) * device_id + device_id) * batch * layer->net.full_connect.count,
					GPU(convnet)->device[device_id].unit,
					context->device[device_id].model_stream[0], context->device[device_id].model_cublas[0]);
			if (dor && context->device[device_id].dor[count])
				_cwc_kern_mute_neuron
				<<<layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[0]>>>
				(GPU(convnet)->device[device_id].forwards[count] + ((dual_device + 1) * device_id + device_id) * batch * layer->net.full_connect.count,
				 context->device[device_id].dor[count]);
			// finished and copy to (device_id, device_id), both available on device_id and other_device_id with stream 0
			cudaMemcpyPeerAsync(GPU(convnet)->device[other_device_id].forwards[count] + ((dual_device + 1) * device_id + device_id) * batch * layer->net.full_connect.count, other_device_id, GPU(convnet)->device[device_id].forwards[count] + ((dual_device + 1) * device_id + device_id) * batch * layer->net.full_connect.count, device_id, sizeof(float) * batch * layer->net.full_connect.count, context->device[device_id].model_stream[0]);
			// record the event so the other device can wait on the copy to complete
			cudaEventRecord(context->device[device_id].model_joint[0], context->device[device_id].model_stream[0]);
			// overlap the memory copy and the gemm computation
			// although it is slightly faster to use source stream, but we sync on the destination stream because otherwise we need to signal an event, and in that case, total cost is higher
			cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].forwards[count - 1] + batch * layer->input.node.count, device_id, GPU(convnet)->device[other_device_id].forwards[count - 1], other_device_id, sizeof(float) * batch * layer->input.node.count, context->device[device_id].model_stream[1]);
			// the last compute issued on this device
			_cwc_convnet_full_connect_forward_propagate(layer, batch,
					GPU(convnet)->device[device_id].forwards[count - 1] + batch * layer->input.node.count,
					GPU(convnet)->device[device_id].forwards[count] + ((dual_device + 1) * other_device_id + device_id) * batch * layer->net.full_connect.count,
					GPU(convnet)->device[device_id].unit,
					context->device[device_id].model_stream[1], context->device[device_id].model_cublas[1]);
			if (dor && context->device[device_id].dor[count])
				_cwc_kern_mute_neuron
				<<<layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[1]>>>
				(GPU(convnet)->device[device_id].forwards[count] + ((dual_device + 1) * other_device_id + device_id) * batch * layer->net.full_connect.count, context->device[device_id].dor[count]);
			cudaEventRecord(context->device[device_id].model_joint[1], context->device[device_id].model_stream[1]);
			// finished (other_device_id, device_id), available on device_id with stream 1
		}
		for (i = count + 1; i < convnet->count; i++)
		{
			// now it is all model parallelism, in model parallelism, it does two things:
			// 1). copy source data to current device => compute on current device
			// 2). compute on current device => copy result data to another device
			// 1 and 2 happen in parallel on one device
			int first = (i - count) & dual_device;
			int second = (i - count - 1) & dual_device;
			// wait for the previous copy to finish
			for (device_id = 0; device_id < dual_device + 1; device_id++)
			{
				cudaSetDevice(device_id);
				int other_device_id = (device_id + 1) & dual_device;
				// device_id on stream second waiting other_device_id on stream first
				cudaStreamWaitEvent(context->device[device_id].model_stream[second], context->device[other_device_id].model_joint[first], 0);
				// device_id on stream first waiting other_device_id on stream second
				cudaStreamWaitEvent(context->device[device_id].model_stream[first], context->device[other_device_id].model_joint[second], 0);
			}
			for (device_id = 0; device_id < dual_device + 1; device_id++)
			{
				cudaSetDevice(device_id);
				int other_device_id = (device_id + 1) & dual_device;
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				assert(layer->type == CCV_CONVNET_FULL_CONNECT);
				int first_device_id = (device_id + first) & dual_device;
				int second_device_id = (device_id + second) & dual_device;
				// first do the computation on the device that we already have full data
				_cwc_convnet_full_connect_forward_propagate(layer, batch,
						GPU(convnet)->device[device_id].forwards[i - 1] + first_device_id * batch * layer->input.node.count,
						GPU(convnet)->device[device_id].forwards[i] + ((dual_device + 1) * first_device_id + device_id) * batch * layer->net.full_connect.count,
						GPU(convnet)->device[device_id].unit, context->device[device_id].model_stream[first], context->device[device_id].model_cublas[first]);
				if (dor && context->device[device_id].dor[i])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[first]>>>
					(GPU(convnet)->device[device_id].forwards[i] + ((dual_device + 1) * first_device_id + device_id) * batch * layer->net.full_connect.count,
					 context->device[device_id].dor[i]);
				// finished and copy to (first_device_id, device_id), both available on device_id and other_device_id with stream first
				cudaMemcpyPeerAsync(GPU(convnet)->device[other_device_id].forwards[i] + ((dual_device + 1) * first_device_id + device_id) * batch * layer->net.full_connect.count, other_device_id, GPU(convnet)->device[device_id].forwards[i] + ((dual_device + 1) * first_device_id + device_id) * batch * layer->net.full_connect.count, device_id, sizeof(float) * batch * layer->net.full_connect.count, context->device[device_id].model_stream[first]);
				cudaEventRecord(context->device[device_id].model_joint[first], context->device[device_id].model_stream[first]);
				// finishing copy the 1 / 4th so that everyone can proceed
				int input_node_count = layer->input.node.count / 2;
				assert(layer->input.node.count % 2 == 0);
				cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].forwards[i - 1] + ((dual_device + 1) * second_device_id + other_device_id) * batch * input_node_count, device_id, GPU(convnet)->device[other_device_id].forwards[i - 1] + ((dual_device + 1) * second_device_id + other_device_id) * batch * input_node_count, other_device_id, sizeof(float) * batch * input_node_count, context->device[device_id].model_stream[second]);
				// copy (second_device_id, other_device_id), available on device_id now with stream second
				// the last compute issued on this device
				_cwc_convnet_full_connect_forward_propagate(layer, batch,
						GPU(convnet)->device[device_id].forwards[i - 1] + second_device_id * batch * layer->input.node.count,
						GPU(convnet)->device[device_id].forwards[i] + ((dual_device + 1) * second_device_id + device_id) * batch * layer->net.full_connect.count,
						GPU(convnet)->device[device_id].unit,
						context->device[device_id].model_stream[second], context->device[device_id].model_cublas[second]);
				if (dor && context->device[device_id].dor[i])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[second]>>>
					(GPU(convnet)->device[device_id].forwards[i] + ((dual_device + 1) * second_device_id + device_id) * batch * layer->net.full_connect.count, context->device[device_id].dor[i]);
				// finished (second_device_id, device_id), available on device_id with stream second
				cudaEventRecord(context->device[device_id].model_joint[second], context->device[device_id].model_stream[second]);
			}
		}
		// wait for the copy to finish
		for (device_id = 0; device_id < dual_device + 1; device_id++)
		{
			int first = (convnet->count - 1 - count) & dual_device;
			int second = (convnet->count - count) & dual_device;
			cudaSetDevice(device_id);
			int other_device_id = (device_id + 1) & dual_device;
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[other_device_id].model_joint[first], 0);
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[device_id].model_joint[second], 0);
		}
	} else {
		for (i = 0; i < convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->device[0].layers + i;
			_cwc_convnet_layer_forward_propagate(layer, 0, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, dor, i == 0 ? context->device[0].input : GPU(convnet)->device[0].forwards[i - 1], GPU(convnet)->device[0].forwards[i], GPU(convnet)->device[0].denoms[i], GPU(convnet)->device[0].unit, context);
		}
	}
}

#ifdef HAVE_GSL

__global__ static void _cwc_kern_relu_backward_propagate(const int batch,
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
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	assert(out_partition == 1); // this cannot handle partition
	float* chm = scratch;
	float* cha = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch;
	float* cbw = scratch + layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels * batch + out_rows * out_cols * layer->net.convolutional.count * batch;
	float alpha = 1, beta = 0;
	int count = layer->net.convolutional.rows * layer->net.convolutional.cols * layer->net.convolutional.count * layer->input.matrix.channels;
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
		cublasSgemv(handle, CUBLAS_OP_N, count, out_rows * batch_group_count, &alpha, cbw, count, unit, 1, &beta, configuration->w, 1); \
	} while (0)
	// special casing for image
	cwc_vary_4_a(x, 1, 2, 3, 4, cwc_vary_4_b, y, 1, 2, 3, 4, cwc_vary_5_c, layer->net.convolutional.rows, 3, 5, 7, 9, 11, vary_block);
#undef vary_block
	assert(cudaGetLastError() == cudaSuccess);
	return 0;
}

static void _cwc_convnet_convolutional_backward_propagate_coefficient_rows(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	static int vary_x[] = { 1, 2, 3, 4 };
	static int vary_y[] = { 1, 2, 3, 4 };
	static int vary_z[] = { 1 };
	CWC_IMPLEMENT_VARY_STUB(VARY(layer)->convolutional.backward.coefficient, vary_x, vary_y, vary_z, _cwc_convnet_convolutional_backward_propagate_coefficient_rows_vary, layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
}

static int _cwc_convnet_convolutional_backward_propagate_coefficient_default_vary(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, ccv_convnet_layer_t* configuration, float* scratch, float* unit, const cudaStream_t& stream, const cublasHandle_t& handle,
		int x, int y, int z)
{
	int out_rows, out_cols, out_partition;
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
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
		cublasSgemv(handle, CUBLAS_OP_N, count, batch_group_count, &alpha, cbw, count, unit, 1, &beta, configuration->w, 1); \
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
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
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
	int out_rows, out_cols, out_partition;
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	// it turns out that first apply relu would save us a lot of computation because no need to low both out and out_grad any more
	_cwc_kern_relu_backward_propagate
	<<<dim3(out_cols, out_rows, layer->net.convolutional.count), batch, 0, stream>>>
	(batch, n, a, out_rows, out_cols, layer->net.convolutional.count);
	assert(cudaGetLastError() == cudaSuccess);
	float alpha = 1, beta = 0;
	if (_cwc_convnet_layer_use_rows(layer))
		_cwc_convnet_convolutional_backward_propagate_coefficient_rows(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
	else
		_cwc_convnet_convolutional_backward_propagate_coefficient_default(layer, batch, a, n, m, b, configuration, scratch, unit, stream, handle);
	// compute the bias directly using gemv routine
	cublasSgemv(handle, CUBLAS_OP_T, out_rows * out_cols * batch, layer->net.convolutional.count, &alpha, a, out_rows * out_cols * batch, unit, 1, &beta, configuration->bias, 1);
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

static void _cwc_convnet_rnorm_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* denoms, float* b, const cudaStream_t& stream)
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

static void _cwc_convnet_max_pool_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, const cudaStream_t& stream)
{
	int out_rows, out_cols, out_partition;
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
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
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
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

static void _cwc_convnet_full_connect_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, float* batch_unit, ccv_convnet_layer_t* configuration, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols, out_partition;
	_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	out_cols = batch;
	rows = layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels;
	// apply relu for full connect layer, not that this requires both n and a, and for the last full connect layer, we re-used the forwards, thus, it required the last full connect layer to not have relu enabled
	if (layer->net.full_connect.relu)
		_cwc_kern_relu_backward_propagate
		<<<dim3(1, out_rows, 1), batch, 0, stream>>>
		(batch, n, a, out_rows, 1, 1);
	float alpha = 1;
	float beta = 0;
	// propagate bias
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, out_rows, batch, &alpha, batch_unit, 1, a, batch, &beta, configuration->bias, 1);
	// propagate error
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch, rows, out_rows, &alpha, a, batch, layer->w, rows, &beta, b, batch);
	// propagate weights
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, out_rows, batch, &alpha, m, batch, a, batch, &beta, configuration->w, rows);
}

__global__ static void _cwc_kern_softmax_with_logistic_loss(const int batch, const int count, float* a, int* c)
{
	int i;
	const int thidx = blockIdx.x * blockDim.x + threadIdx.x;
	float max_val = a[thidx];
	for (i = 1; i < count; i++)
	{
		float prod = a[i * batch + thidx];
		if (prod > max_val)
			max_val = prod;
	}
	float val = 0;
	for (i = 0; i < count; i++)
	{
		float prod = a[i * batch + thidx];
		val += (prod = expf(prod - max_val));
		a[i * batch + thidx] = prod;
	}
	val = 1.0 / val;
	for (i = 0; i < count; i++)
		a[i * batch + thidx] = (i == c[thidx]) - (a[i * batch + thidx] * val);
}

static void _cwc_convnet_softmax_with_logistic_loss(int batch, int count, float* a, int* c, const cudaStream_t& stream)
{
	dim3 num_blocks(ccv_max(1, batch / 64));
	dim3 threads_per_block(ccv_min(batch, 64));
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_softmax_with_logistic_loss
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(batch, count, a, c);
}

__global__ static void _cwc_kern_tests_return(const int batch, const int count, float* a, int* c)
{
	int i;
	const int thidx = blockIdx.x * blockDim.x + threadIdx.x;
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
	dim3 num_blocks(ccv_max(1, batch / 64));
	dim3 threads_per_block(ccv_min(batch, 64));
	assert(threads_per_block.x <= 1024);
	_cwc_kern_tests_return
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

static void _cwc_convnet_net_sgd(ccv_convnet_t* convnet, int device_id, int momentum_read, int batch, ccv_convnet_layer_train_param_t* layer_params, cwc_convnet_context_t* context)
{
	int i, out_rows, out_cols, out_partition;
	dim3 threads_per_block(64);
	assert(threads_per_block.x <= 1024);
	dim3 num_blocks_for_coeff;
	dim3 num_blocks_for_bias;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
		ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
		ccv_convnet_layer_t* momentum = GPU(convnet)->device[device_id].momentums + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
				num_blocks_for_coeff = (layer->wnum + 63) / 64;
				num_blocks_for_bias = (layer->net.convolutional.count + 63) / 64;
				if (momentum_read)
				{
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				} else {
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				// assume coeff and bias in the same continuous memory region
				num_blocks_for_coeff = (layer->wnum + 63) / 64;
				num_blocks_for_bias = (layer->net.full_connect.count + 63) / 64;
				if (momentum_read)
				{
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<1>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.full_connect.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				} else {
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<0>
					<<<num_blocks_for_bias, threads_per_block, 0, context->device[device_id].data_stream>>>
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

static void _cwc_convnet_batch_formation(gsl_rng* rng, ccv_array_t* categorizeds, ccv_dense_matrix_t* mean_activity, ccv_dense_matrix_t* eigenvectors, ccv_dense_matrix_t* eigenvalues, float color_gain, int* idx, ccv_size_t dim, int rows, int cols, int channels, int category_count, int symmetric, int batch, int offset, int size, float* b, int* c)
{
	int i, k, x;
	assert(size <= batch);
	float* channel_gains = (float*)alloca(sizeof(float) * channels);
	memset(channel_gains, 0, sizeof(float) * channels);
	for (i = 0; i < size; i++)
	{
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, idx ? idx[offset + i] : offset + i);
		assert(categorized->c < category_count && categorized->c >= 0); // now only accept classes listed
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
	printf("\n");
}

static void _cwc_convnet_dor_mean_net(ccv_convnet_t* convnet, int device_id, ccv_convnet_layer_train_param_t* layer_params, const cublasHandle_t& handle)
{
	int i;
	for (i = 0; i < convnet->count; i++)
		if (layer_params[i].dor > 0)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
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

static void _cwc_convnet_dor_mean_net_undo(ccv_convnet_t* convnet, int device_id, ccv_convnet_layer_train_param_t* layer_params, const cublasHandle_t& handle)
{
	int i;
	for (i = 0; i < convnet->count; i++)
		if (layer_params[i].dor > 0)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
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

static void _cwc_convnet_dor_formation(ccv_convnet_t* convnet, int device_id, int batch, gsl_rng* rng, ccv_convnet_layer_train_param_t* layer_params, cwc_convnet_context_t* context)
{
	int i, j;
	for (i = 0; i < convnet->count; i++)
		if (context->host[device_id].dor[i])
		{
			assert(context->device[device_id].dor[i]);
			assert(layer_params[i].dor > 0);
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
			int out_rows, out_cols, out_partition;
			_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
			assert(layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT);
			int count = layer->type == CCV_CONVNET_FULL_CONNECT ? layer->net.full_connect.count : out_rows * out_cols * layer->net.convolutional.count;
			for (j = 0; j < batch * count; j++)
				context->host[device_id].dor[i][j] = (gsl_rng_uniform(rng) >= layer_params[i].dor) ? 1.0 : 0.0;
			cudaMemcpyAsync(context->device[device_id].dor[i], context->host[device_id].dor[i], sizeof(float) * count * batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
			assert(cudaGetLastError() == cudaSuccess);
		}
}

static void _cwc_convnet_layer_backward_propagate(ccv_convnet_layer_t* layer, int device_id, int k, int rows, int cols, int batch, float* a, float* n, float* m, float* denoms, float* b, float* batch_unit, float* scratch, ccv_convnet_layer_t* configuration, cwc_convnet_context_t* context)
{
	switch (layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			if (context->device[device_id].dor[k])
			{
				int out_rows, out_cols, out_partition;
				_ccv_convnet_layer_derive_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
				_cwc_kern_mute_neuron
				<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device[device_id].data_stream>>>
				(a, context->device[device_id].dor[k]);
			}
			_cwc_convnet_convolutional_backward_propagate(layer, batch, a, n, m, b, configuration, scratch, batch_unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
			assert(cudaGetLastError() == cudaSuccess);
			break;
		case CCV_CONVNET_FULL_CONNECT:
			if (context->device[device_id].dor[k])
				_cwc_kern_mute_neuron
				<<<layer->net.full_connect.count, batch, 0, context->device[device_id].data_stream>>>
				(a, context->device[device_id].dor[k]);
			_cwc_convnet_full_connect_backward_propagate(layer, batch, a, n, m, b, batch_unit, configuration, context->device[device_id].data_stream, context->device[device_id].data_cublas);
			assert(cudaGetLastError() == cudaSuccess);
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			_cwc_convnet_rnorm_backward_propagate(layer, batch, a, n, m, denoms, b, context->device[device_id].data_stream);
			assert(cudaGetLastError() == cudaSuccess);
			break;
		case CCV_CONVNET_MAX_POOL:
			_cwc_convnet_max_pool_backward_propagate(layer, batch, a, n, m, b, context->device[device_id].data_stream);
			assert(cudaGetLastError() == cudaSuccess);
			break;
		case CCV_CONVNET_AVERAGE_POOL:
			_cwc_convnet_average_pool_backward_propagate(layer, batch, a, b, context->device[device_id].data_stream);
			assert(cudaGetLastError() == cudaSuccess);
			break;
	}
}

static void _cwc_convnet_backward_propagate_error(ccv_convnet_t* convnet, int dual_device, int batch, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	if (dual_device)
	{
		int device_id;
		int count = _cwc_convnet_first_full_connect(convnet);
		for (device_id = 0; device_id < dual_device + 1; device_id++)
		{
			cudaSetDevice(device_id);
			cudaEventRecord(context->device[device_id].data_joint, context->device[device_id].data_stream);
			cudaStreamWaitEvent(context->device[device_id].model_stream[0], context->device[device_id].data_joint, 0);
			cudaStreamWaitEvent(context->device[device_id].model_stream[1], context->device[device_id].data_joint, 0);
		}
		for (i = convnet->count - 1; i >= count; i--)
		{
			int first = (i - count) & dual_device;
			int second = (i - count - 1) & dual_device;
			assert(i > 0);
			for (device_id = 0; device_id < dual_device + 1; device_id++)
			{
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				assert(layer->type == CCV_CONVNET_FULL_CONNECT);
				ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
				int first_device_id = (device_id + first) & dual_device;
				int second_device_id = (device_id + second) & dual_device;
				float* a = (i == convnet->count - 1) ? GPU(convnet)->device[device_id].forwards[i] + second_device_id * batch * layer->net.full_connect.count : GPU(convnet)->device[device_id].backwards[i + 1];
				if (context->device[device_id].dor[i])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device[device_id].data_stream>>>
					(a, context->device[device_id].dor[i]);
				_cwc_convnet_full_connect_backward_propagate(layer, batch,
						a,
						GPU(convnet)->device[device_id].forwards[i],
						GPU(convnet)->device[device_id].forwards[i - 1],
						GPU(convnet)->device[device_id].backwards[i],
						GPU(convnet)->device[device_id].unit, configuration, context->device[device_id].model_stream[first], context->device[device_id].model_cublas[first]);
			}
		}
		for (device_id = 0; device_id < dual_device + 1; device_id++)
		{
			cudaSetDevice(device_id);
			int other_device_id = (device_id + 1) & dual_device;
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[other_device_id].model_joint[0], 0);
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[device_id].model_joint[1], 0);
			for (i = count - 1; i >= 0; i--)
			{
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
				float* a = (i == convnet->count - 1) ? GPU(convnet)->device[device_id].forwards[i] : GPU(convnet)->device[device_id].backwards[i + 1];
				float* m = (i > 0) ? GPU(convnet)->device[device_id].forwards[i - 1] : context->device[device_id].input;
				_cwc_convnet_layer_backward_propagate(layer, device_id, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, a, GPU(convnet)->device[device_id].forwards[i], m, GPU(convnet)->device[device_id].denoms[i], GPU(convnet)->device[device_id].backwards[i], GPU(convnet)->device[device_id].unit, GPU(convnet)->device[device_id].scratch, configuration, context);
			}
		}
	} else {
		for (i = convnet->count - 1; i >= 0; i--)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->device[0].layers + i;
			ccv_convnet_layer_t* configuration = GPU(convnet)->device[0].configurations + i;
			float* a = (i == convnet->count - 1) ? GPU(convnet)->device[0].forwards[i] : GPU(convnet)->device[0].backwards[i + 1];
			float* m = (i > 0) ? GPU(convnet)->device[0].forwards[i - 1] : context->device[0].input;
			_cwc_convnet_layer_backward_propagate(layer, 0, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, a, GPU(convnet)->device[0].forwards[i], m, GPU(convnet)->device[0].denoms[i], GPU(convnet)->device[0].backwards[i], GPU(convnet)->device[0].unit, GPU(convnet)->device[0].scratch, configuration, context);
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
	int i, device_id;
	for (device_id = 0; device_id < GPU(z->convnet)->dual_device + 1; device_id++)
	{
		cudaSetDevice(device_id);
		for (i = 0; i < convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = GPU(z->convnet)->device[device_id].layers + i;
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
				for (device_id = 0; device_id < GPU(z->convnet)->dual_device + 1; device_id++)
				{
					ccv_convnet_layer_t* layer = GPU(z->convnet)->device[device_id].layers + sqlite3_column_int(momentum_data_stmt, 0);
					ccv_convnet_layer_t* momentum = GPU(z->convnet)->device[device_id].momentums + sqlite3_column_int(momentum_data_stmt, 0);
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

static void _cwc_convnet_host_synchronize(ccv_convnet_t* convnet, int device_id)
{
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
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
	// the master state kept in device id == 0
	_cwc_convnet_host_synchronize(z->convnet, 0);
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
			ccv_convnet_layer_t* layer = GPU(z->convnet)->device[0].layers + i;
			ccv_convnet_layer_t* momentum = GPU(z->convnet)->device[0].momentums + i;
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
	assert(batch == 0);
}

__global__ static void _cwc_kern_neuron_scan(float* a, float* b,
		const int rows, const int cols,
		const int out_rows, const int out_cols, const int channels, const int batch)
{
	assert(gridDim.x == cols);
	assert(gridDim.y == rows);
	assert(gridDim.z == channels);
	assert(blockDim.x == batch);
	assert(out_rows > rows);
	assert(out_cols > cols);
	b += (blockIdx.z * rows * cols + blockIdx.y * cols + blockIdx.x) * batch * 5;
	a += (blockIdx.z * out_rows * out_cols + blockIdx.y * out_cols + blockIdx.x) * batch;
	const int thidx = threadIdx.x;
	b[thidx] = a[thidx]; // top left
	b += batch;
	float* c = a + (out_cols - cols) * batch; // top right
	b[thidx] = c[thidx];
	b += batch;
	c = a + (((out_rows - rows) / 2) * out_cols + (out_cols - cols) / 2) * batch; // center
	b[thidx] = c[thidx];
	b += batch;
	c = a + (out_rows - rows) * out_cols * batch; // bottom left
	b[thidx] = c[thidx];
	b += batch;
	c = a + ((out_rows - rows) * out_cols + (out_cols - cols)) * batch; // bottom right
	b[thidx] = c[thidx];
}

__global__ static void _cwc_kern_softmax(float* a, const int batch, const int count)
{
	int i;
	const int thidx = threadIdx.x;
	float max_val = a[thidx];
	for (i = 1; i < count; i++)
	{
		float v = a[i * batch + thidx];
		if (v > max_val)
			max_val = v;
	}
	float val = 0;
	for (i = 0; i < count; i++)
	{
		float v = a[i * batch + thidx];
		val += (v = expf(v - max_val));
		a[i * batch + thidx] = v;
	}
	val = 1.0 / val;
	for (i = 0; i < count; i++)
		a[i * batch + thidx] *= val;
}

template <int vary>
__global__ static void _cwc_kern_classify(float* a, int* c, float* b, const int batch, const int count, const int tops)
{
	int i, j;
	assert(blockDim.x == batch);
	const int thidx = threadIdx.x;
	for (i = 0; i < count; i++)
		#pragma unroll
		for (j = 1; j < vary; j++)
			a[i * batch * vary + thidx] += a[(i * vary + j) * batch + thidx];
	#pragma unroll
	for (i = 0; i < tops; i++)
	{
		float max_val = -1;
		int max_idx = -1;
		for (j = 0; j < count; j++)
		{
			float v = a[j * batch * vary + thidx];
			if (v >= 0 && v > max_val)
				max_val = v, max_idx = j;
		}
		assert(max_idx >= 0);
		a[max_idx * batch * vary + thidx] = -1;
		c[thidx] = max_idx;
		b[thidx] = max_val;
		c += batch;
		b += batch;
	}
}

void cwc_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int symmetric, ccv_array_t** ranks, int tops, int batch)
{
	assert(symmetric == 1); // only works with symmetric as well
	assert(batch == 32); // I haven't figured out to do this for any batch size
	// classify step uses only device 0
	_cwc_convnet_alloc_reserved_for_classify(convnet, tops, batch);
	int i, j, k;
	int rows = convnet->input.height, cols = convnet->input.width, channels = convnet->channels;
	cwc_convnet_context_t* default_context = GPU(convnet)->contexts;
	float* c = default_context->host[0].input;
	for (i = 0; i < batch; i++)
	{
		assert(a[i]->rows == rows || a[i]->cols == cols);
		assert(a[i]->rows >= rows && a[i]->cols >= cols);
		// top / left
		ccv_dense_matrix_t* b = 0;
		ccv_slice(a[i], (ccv_matrix_t**)&b, CCV_32F, 0, 0, rows, cols);
		ccv_subtract(b, convnet->mean_activity, (ccv_matrix_t**)&b, 0);
		for (k = 0; k < channels; k++)
			for (j = 0; j < rows * cols; j++)
				c[(k * rows * cols + j) * batch * 6 + i] = b->data.f32[j * channels + k];
		ccv_flip(b, &b, 0, CCV_FLIP_X);
		for (k = 0; k < channels; k++)
			for (j = 0; j < rows * cols; j++)
				c[(k * rows * cols + j) * batch * 6 + batch + i] = b->data.f32[j * channels + k];
		ccv_matrix_free(b);
		// center
		b = 0;
		ccv_slice(a[i], (ccv_matrix_t**)&b, CCV_32F, (a[i]->rows - rows) / 2, (a[i]->cols - cols) / 2, rows, cols);
		ccv_subtract(b, convnet->mean_activity, (ccv_matrix_t**)&b, 0);
		for (k = 0; k < channels; k++)
			for (j = 0; j < rows * cols; j++)
				c[(k * rows * cols + j) * batch * 6 + 2 * batch + i] = b->data.f32[j * channels + k];
		ccv_flip(b, &b, 0, CCV_FLIP_X);
		for (k = 0; k < channels; k++)
			for (j = 0; j < rows * cols; j++)
				c[(k * rows * cols + j) * batch * 6 + 3 * batch + i] = b->data.f32[j * channels + k];
		ccv_matrix_free(b);
		// bottom / right
		b = 0;
		ccv_slice(a[i], (ccv_matrix_t**)&b, CCV_32F, a[i]->rows - rows, a[i]->cols - cols, rows, cols);
		ccv_subtract(b, convnet->mean_activity, (ccv_matrix_t**)&b, 0);
		for (k = 0; k < channels; k++)
			for (j = 0; j < rows * cols; j++)
				c[(k * rows * cols + j) * batch * 6 + 4 * batch + i] = b->data.f32[j * channels + k];
		ccv_flip(b, &b, 0, CCV_FLIP_X);
		for (k = 0; k < channels; k++)
			for (j = 0; j < rows * cols; j++)
				c[(k * rows * cols + j) * batch * 6 + 5 * batch + i] = b->data.f32[j * channels + k];
		ccv_matrix_free(b);
	}
	cudaMemcpyAsync(default_context->device[0].input, default_context->host[0].input, sizeof(float) * rows * cols * channels * batch * 6, cudaMemcpyHostToDevice, default_context->device[0].data_stream);
	int scan = _cwc_convnet_find_scan(convnet, 0);
	assert(scan >= 0 && scan < convnet->count);
	int out_rows, out_cols, out_partition;
	for (i = 0; i < scan + 1; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[0].layers + i;
		_ccv_convnet_layer_derive_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
		_cwc_convnet_layer_forward_propagate(layer, 0, i, rows, cols, batch * 6, 0, i == 0 ? default_context->device[0].input : GPU(convnet)->device[0].forwards[i - 1], GPU(convnet)->device[0].forwards[i], GPU(convnet)->device[0].denoms[i], GPU(convnet)->device[0].unit, default_context);
		rows = out_rows, cols = out_cols;
	}
	// copy data to scans
	dim3 num_blocks = dim3(GPU(convnet)->device[0].layers[scan + 1].input.matrix.cols, GPU(convnet)->device[0].layers[scan + 1].input.matrix.rows, GPU(convnet)->device[0].layers[scan + 1].input.matrix.channels);
	_cwc_kern_neuron_scan
		<<<num_blocks, batch * 6, 0, default_context->device[0].data_stream>>>
		(GPU(convnet)->device[0].forwards[scan], GPU(convnet)->device[0].scans[scan], GPU(convnet)->device[0].layers[scan + 1].input.matrix.rows, GPU(convnet)->device[0].layers[scan + 1].input.matrix.cols, rows, cols, GPU(convnet)->device[0].layers[scan + 1].input.matrix.channels, batch * 6);
	for (i = scan + 1; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[0].layers + i;
		_cwc_convnet_layer_forward_propagate(layer, 0, i, layer->input.matrix.rows, layer->input.matrix.cols, batch * 30, 0, i == scan + 1 ? GPU(convnet)->device[0].scans[i - 1] : GPU(convnet)->device[0].forwards[i - 1], GPU(convnet)->device[0].forwards[i], GPU(convnet)->device[0].denoms[i], GPU(convnet)->device[0].unit, default_context);
	}
	// doing softmax for the last layer
	int category_count = convnet->layers[convnet->count - 1].net.full_connect.count;
	_cwc_kern_softmax
		<<<1, batch * 30, 0, default_context->device[0].data_stream>>>
		(GPU(convnet)->device[0].forwards[convnet->count - 1], batch * 30, category_count);
	// collect classify results
	_cwc_kern_classify
		<30>
		<<<1, batch, 0, default_context->device[0].data_stream>>>
		(GPU(convnet)->device[0].forwards[convnet->count - 1], default_context->device[0].c, default_context->device[0].out, batch, category_count, tops);
	cudaMemcpyAsync(default_context->host[0].c, default_context->device[0].c, sizeof(int) * batch * tops, cudaMemcpyDeviceToHost, default_context->device[0].data_stream);
	cudaMemcpyAsync(default_context->host[0].out, default_context->device[0].out, sizeof(float) * batch * tops, cudaMemcpyDeviceToHost, default_context->device[0].data_stream);
	// wait for the classify to finish
	cudaStreamSynchronize(default_context->device[0].data_stream);
	// collect result to ccv_array_t
	for (i = 0; i < batch; i++)
	{
		ranks[i] = ccv_array_new(sizeof(ccv_classification_t), tops, 0);
		for (j = 0; j < tops; j++)
		{
			ccv_classification_t classification = {
				.id = default_context->host[0].c[j * batch + i],
				.confidence = default_context->host[0].out[j * batch + i] / 30,
			};
			ccv_array_push(ranks[i], &classification);
		}
	}
}

void cwc_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, const char* filename, ccv_convnet_train_param_t params)
{
#ifdef HAVE_GSL
	assert(params.mini_batch % BATCH_PER_BLOCK == 0);
	int device_id, device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (params.dual_device && device_count < 2)
		params.dual_device = 0;
	assert(device_count > 0);
	// enable peer access
	if (params.dual_device)
		for (device_id = 0; device_id < params.dual_device + 1; device_id++)
		{
			int other_device_id = (device_id + 1) & params.dual_device;
			cudaSetDevice(device_id);
			cudaDeviceEnablePeerAccess(other_device_id, 0);
		}
	_cwc_convnet_alloc_reserved_both(convnet, params.mini_batch, params.dual_device, params.layer_params);
	int i, j, k;
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	int dual_batch = params.mini_batch * (params.dual_device + 1);
	int aligned_padding = categorizeds->rnum % dual_batch;
	int aligned_rnum = categorizeds->rnum - aligned_padding;
	int aligned_batches = categorizeds->rnum / dual_batch;
	int* idx = (int*)ccmalloc(sizeof(int) * (categorizeds->rnum + aligned_padding));
	for (i = 0; i < categorizeds->rnum; i++)
		idx[i] = i;
	params.iterations = ccv_min(params.iterations, aligned_batches);
	gsl_ran_shuffle(rng, idx, categorizeds->rnum, sizeof(int));
	// the last layer has to be full connect, thus we can use it as softmax layer
	assert(convnet->layers[convnet->count - 1].type == CCV_CONVNET_FULL_CONNECT);
	int category_count = convnet->layers[convnet->count - 1].net.full_connect.count;
	struct {
		int* host[2];
		int* device[2];
	} test_returns[2];
	test_returns[0].host[0] = test_returns[0].host[1] =
		test_returns[1].host[0] = test_returns[1].host[1] = 0;
	test_returns[0].device[0] = test_returns[0].device[1] =
		test_returns[1].device[0] = test_returns[1].device[1] = 0;
	cudaEvent_t start[2], stop[2], iteration[2];
	for (device_id = 0; device_id < params.dual_device + 1; device_id++)
	{
		cudaSetDevice(device_id);
		for (i = 0; i < 2; i++)
		{
			cudaMallocHost(&test_returns[i].host[device_id], sizeof(int) * params.mini_batch);
			assert(test_returns[i].host[device_id]);
			cudaMalloc(&test_returns[i].device[device_id], sizeof(int) * params.mini_batch);
			assert(test_returns[i].device[device_id]);
		}
		cudaEventCreate(&start[device_id]);
		cudaEventCreate(&stop[device_id]);
		cudaEventCreate(&iteration[device_id]);
	}
	cwc_convnet_supervised_train_function_state_t z;
	z.idx = idx;
	z.inum = categorizeds->rnum;
	z.convnet = convnet;
	z.eigenvectors = 0;
	z.eigenvalues = 0;
	z.line_no = 0;
	int miss;
	float elapsed_time[2] = {0};
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
			// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in updating
			// undo the mean network for further training
			for (device_id = 0; device_id < params.dual_device + 1; device_id++)
			{
				cudaSetDevice(device_id);
				cudaEventRecord(start[device_id], 0);
				_cwc_convnet_dor_mean_net_undo(z.convnet, device_id, params.layer_params, GPU(z.convnet)->contexts[(z.i + 1) % 2].device[device_id].data_cublas);
			}
			miss = 0;
			// run updates
			for (i = z.i; i < ccv_min(z.i + params.iterations, aligned_batches); i++)
			{
				cwc_convnet_context_t* context = GPU(z.convnet)->contexts + (i % 2);
				for (device_id = 0; device_id < params.dual_device + 1; device_id++)
				{
					cudaSetDevice(device_id);
					_cwc_convnet_batch_formation(rng, categorizeds, z.convnet->mean_activity, z.eigenvectors, z.eigenvalues, params.color_gain, z.idx, z.convnet->input, z.convnet->rows, z.convnet->cols, z.convnet->channels, category_count, params.symmetric, dual_batch, i * dual_batch + device_id * params.mini_batch, params.mini_batch, context->host[device_id].input, context->host[device_id].c);
					cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * z.convnet->rows * z.convnet->cols * z.convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
					assert(cudaGetLastError() == cudaSuccess);
					cudaMemcpyAsync(context->device[device_id].c, context->host[device_id].c, sizeof(int) * params.mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
					assert(cudaGetLastError() == cudaSuccess);
					_cwc_convnet_dor_formation(z.convnet, device_id, params.mini_batch, rng, params.layer_params, context);
					assert(cudaGetLastError() == cudaSuccess);
					// sync with the other stream core so that we can compute on the single true layer parameters
					if (i > z.i)
						cudaEventRecord(stop[device_id], GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].data_stream);
				}
				for (device_id = 0; device_id < params.dual_device + 1; device_id++)
				{
					cudaSetDevice(device_id);
					cudaStreamSynchronize(GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].data_stream);
				}
				assert(cudaGetLastError() == cudaSuccess);
				if (i > z.i) // we have another result, pull these
				{
					for (device_id = 0; device_id < params.dual_device + 1; device_id++)
					{
						cudaSetDevice(device_id);
						int* c = GPU(z.convnet)->contexts[(i + 1) % 2].host[device_id].c;
						for (k = 0; k < params.mini_batch; k++)
							if (c[k] != test_returns[(i + 1) % 2].host[device_id][k])
								++miss;
						cudaEventElapsedTime(&elapsed_time[device_id], iteration[device_id], stop[device_id]);
					}
					FLUSH(" - at epoch %03d / %d => stochastic gradient descent with miss rate %.2f%% at %d / %d (%.3f sec)", z.t + 1, params.max_epoch, miss * 100.0f /((i - z.i) * dual_batch), i + 1, aligned_batches, ccv_max(elapsed_time[0], elapsed_time[1]) / 1000);
				}
				for (device_id = 0; device_id < params.dual_device + 1; device_id++)
				{
					cudaSetDevice(device_id);
					cudaEventRecord(iteration[device_id], context->device[device_id].data_stream);
				}
				_cwc_convnet_encode_impl(z.convnet, params.dual_device, params.mini_batch, 1, context);
				assert(cudaGetLastError() == cudaSuccess);
				// compute miss rate on training data
				int count = _cwc_convnet_first_full_connect(z.convnet);
				for (device_id = 0; device_id < params.dual_device + 1; device_id++)
				{
					cudaSetDevice(device_id);
					int first = (z.convnet->count - count) & params.dual_device;
					int first_device_id = (device_id + first) & params.dual_device;
					_cwc_convnet_tests_return(params.mini_batch, category_count, GPU(z.convnet)->device[device_id].forwards[z.convnet->count - 1] + first_device_id * params.mini_batch * category_count, test_returns[i % 2].device[device_id], context->device[device_id].data_stream);
					assert(cudaGetLastError() == cudaSuccess);
					cudaMemcpyAsync(test_returns[i % 2].host[device_id], test_returns[i % 2].device[device_id], sizeof(int) * params.mini_batch, cudaMemcpyDeviceToHost, context->device[device_id].data_stream);
					assert(cudaGetLastError() == cudaSuccess);
					// do the logistic loss
					_cwc_convnet_softmax_with_logistic_loss(params.mini_batch, category_count, GPU(z.convnet)->device[device_id].forwards[z.convnet->count - 1] + first_device_id * params.mini_batch * category_count, context->device[device_id].c, context->device[device_id].data_stream);
					assert(cudaGetLastError() == cudaSuccess);
				}
				// do backward propagate
				_cwc_convnet_backward_propagate_error(z.convnet, params.dual_device, params.mini_batch, context);
				assert(cudaGetLastError() == cudaSuccess);
				for (device_id = 0; device_id < params.dual_device + 1; device_id++)
				{
					cudaSetDevice(device_id);
					_cwc_convnet_net_sgd(z.convnet, device_id, z.t > 0 || i > 0, params.mini_batch, params.layer_params, context);
				}
				assert(cudaGetLastError() == cudaSuccess);
			}
			for (device_id = 0; device_id < params.dual_device + 1; device_id++)
			{
				cudaSetDevice(device_id);
				cudaDeviceSynchronize(); // synchronize at this point
				// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in testing
				_cwc_convnet_dor_mean_net(z.convnet, device_id, params.layer_params, GPU(z.convnet)->contexts[1].device[device_id].data_cublas);
			}
			// run tests
			miss = 0;
			// run it on one device for now
			device_id = 0;
			cudaSetDevice(device_id);
			for (i = j = 0; i < tests->rnum; i += params.mini_batch, j++)
			{
				cwc_convnet_context_t* context = GPU(z.convnet)->contexts + (j % 2);
				_cwc_convnet_batch_formation(0, tests, z.convnet->mean_activity, 0, 0, 0, 0, z.convnet->input, z.convnet->rows, z.convnet->cols, z.convnet->channels, category_count, params.symmetric, params.mini_batch, i, ccv_min(params.mini_batch, tests->rnum - i), context->host[device_id].input, 0);
				cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * z.convnet->rows * z.convnet->cols * z.convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				if (j > 0)
					cudaEventRecord(stop[device_id], GPU(z.convnet)->contexts[(j + 1) % 2].device[device_id].data_stream);
				// sync with the other stream core so that we can compute on the single true layer parameters
				cudaStreamSynchronize(GPU(z.convnet)->contexts[(j + 1) % 2].device[device_id].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				if (j > 0) // we have another result, pull these
				{
					for (k = 0; k < params.mini_batch; k++)
					{
						ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, k + i - params.mini_batch);
						if (test->c != test_returns[(j + 1) % 2].host[device_id][k])
							++miss;
					}
					cudaEventElapsedTime(&elapsed_time[device_id], iteration[device_id], stop[device_id]);
					FLUSH(" - at epoch %03d / %d => with miss rate %.2f%% at %d / %d (%.3f sec)", z.t + 1, params.max_epoch, miss * 100.0f / i, j + 1, (tests->rnum + params.mini_batch - 1) / params.mini_batch, elapsed_time[device_id] / 1000);
				}
				cudaEventRecord(iteration[device_id], context->device[device_id].data_stream);
				_cwc_convnet_encode_impl(z.convnet, 0, params.mini_batch, 0, context);
				assert(cudaGetLastError() == cudaSuccess);
				_cwc_convnet_tests_return(params.mini_batch, category_count, GPU(z.convnet)->device[device_id].forwards[z.convnet->count - 1], test_returns[j % 2].device[device_id], context->device[device_id].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				cudaMemcpyAsync(test_returns[j % 2].host[device_id], test_returns[j % 2].device[device_id], sizeof(int) * params.mini_batch, cudaMemcpyDeviceToHost, context->device[device_id].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
			}
			cudaDeviceSynchronize(); // synchronize at this point
			for (i = 0; i <= (tests->rnum - 1) % params.mini_batch; i++)
			{
				ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, i + (tests->rnum - 1) / params.mini_batch * params.mini_batch);
				if (test->c != test_returns[(j + 1) % 2].host[device_id][i])
					++miss;
			}
			cudaEventRecord(stop[device_id], 0);
			cudaEventSynchronize(stop[device_id]);
			elapsed_time[device_id] = 0;
			cudaEventElapsedTime(&elapsed_time[device_id], start[device_id], stop[device_id]);
			FLUSH(" - at epoch %03d / %d (%03d - %d) => with miss rate %.2f%% (%.3f sec)\n", z.t + 1, params.max_epoch, z.i + 1, ccv_min(z.i + params.iterations, aligned_batches), miss * 100.0f / tests->rnum, elapsed_time[device_id] / 1000);
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
	for (device_id = 0; device_id < params.dual_device + 1; device_id++)
	{
		cudaSetDevice(device_id);
		cudaEventDestroy(start[device_id]);
		cudaEventDestroy(iteration[device_id]);
		cudaEventDestroy(stop[device_id]);
		for (i = 0; i < 2; i++)
		{
			cudaFree(test_returns[i].device[device_id]);
			cudaFreeHost(test_returns[i].host[device_id]);
		}
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
		int dual_device = GPU(convnet)->dual_device;
		int i, j, device_id;
		for (device_id = 0; device_id < dual_device + 1; device_id++)
		{
			cudaSetDevice(device_id);
			if (GPU(convnet)->device[device_id].scratch)
				cudaFree(GPU(convnet)->device[device_id].scratch);
			cudaFree(GPU(convnet)->device[device_id].unit);
		}
		for (i = 0; i < 2; i++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + i;
			for (device_id = 0; device_id < dual_device + 1; device_id++)
			{
				cudaSetDevice(device_id);
				if (context->host[device_id].input)
					cudaFreeHost(context->host[device_id].input);
				if (context->device[device_id].input)
					cudaFree(context->device[device_id].input);
				if (context->host[device_id].c)
					cudaFreeHost(context->host[device_id].c);
				if (context->device[device_id].c)
					cudaFree(context->device[device_id].c);
				if (context->host[device_id].out)
					cudaFreeHost(context->host[device_id].out);
				if (context->device[device_id].out)
					cudaFree(context->device[device_id].out);
				if (context->device[device_id].data_joint)
					cudaEventDestroy(context->device[device_id].data_joint);
				if (context->device[device_id].data_cublas)
					cublasDestroy(context->device[device_id].data_cublas);
				if (context->device[device_id].data_stream)
					cudaStreamDestroy(context->device[device_id].data_stream);
				for (j = 0; j < 2; j++)
				{
					if (context->device[device_id].model_joint[j])
						cudaEventDestroy(context->device[device_id].model_joint[j]);
					if (context->device[device_id].model_cublas[j])
						cublasDestroy(context->device[device_id].model_cublas[j]);
					if (context->device[device_id].model_stream[j])
						cudaStreamDestroy(context->device[device_id].model_stream[j]);
				}
			}
		}
		for (i = 0; i < convnet->count; i++)
			for (device_id = 0; device_id < dual_device + 1; device_id++)
			{
				cudaSetDevice(device_id);
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				if (layer->w)
					cudaFree(layer->w);
				if (GPU(convnet)->device[device_id].configurations)
				{
					ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
					if (configuration->w)
						cudaFree(configuration->w);
				}
				if (GPU(convnet)->device[device_id].momentums)
				{
					ccv_convnet_layer_t* momentum = GPU(convnet)->device[device_id].momentums + i;
					if (momentum->w)
						cudaFree(momentum->w);
				}
				if (GPU(convnet)->device[device_id].denoms && GPU(convnet)->device[device_id].denoms[i])
					cudaFree(GPU(convnet)->device[device_id].denoms[i]);
				if (GPU(convnet)->device[device_id].forwards && GPU(convnet)->device[device_id].forwards[i])
					cudaFree(GPU(convnet)->device[device_id].forwards[i]);
				if (GPU(convnet)->device[device_id].scans && GPU(convnet)->device[device_id].scans[i])
					cudaFree(GPU(convnet)->device[device_id].scans[i]);
				for (j = 0; j < 2; j++)
				{
					cwc_convnet_context_t* context = GPU(convnet)->contexts + j;
					if (context->host[device_id].dor && context->host[device_id].dor[i])
						cudaFreeHost(context->host[device_id].dor[i]);
					if (context->device[device_id].dor && context->device[device_id].dor[i])
						cudaFree(context->device[device_id].dor[i]);
				}
			}
		assert(convnet->count > 2);
		// we only allocated two backwards layers (backwards[0] is always 0)
		for (i = 1; i < 3; i++)
			for (device_id = 0; device_id < dual_device + 1; device_id++)
			{
				cudaSetDevice(device_id);
				if (GPU(convnet)->device[device_id].backwards && GPU(convnet)->device[device_id].backwards[i])
					cudaFree(GPU(convnet)->device[device_id].backwards[i]);
			}
		ccfree(convnet->reserved);
		convnet->reserved = 0;
	}
}

#endif
