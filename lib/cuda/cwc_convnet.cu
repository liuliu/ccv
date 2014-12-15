#include <cuda.h>
#include <cublas_v2.h>
#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif
extern "C" {
#include "cwc.h"
#include "cwc_internal.h"
#include "../ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#include "cwc_helper.h"
}
#include "../3rdparty/sqlite3/sqlite3.h"
#include "../inl/ccv_convnet_inl.h"

#define MAX_DEVICE_SUPPORT (4)

// this structure holds intermediate on-device memory representation of convnet

typedef struct {
	// on host
	struct {
		float* input; // input per batch
		int* c; // class
		float* out; // confidence score
		float** dor; // dropout regulator, in this version I generate dor on CPU because it is lightweight and gsl has shuffle method, which is better suited for this (and faster than per-node randomization)
	} host[MAX_DEVICE_SUPPORT];
	// on device
	struct {
		// this is modeled after Alex's "One Weird Trick", there are 3 join points for me: 1). forward pass from data parallelism to model parallelism; 2). compute logistic loss; 3). backward pass from model parallelism to data parallelism;
		cudaStream_t data_stream; // based on above description, we need 3 streams, one stream for data parallelism
		cudaStream_t model_stream[2]; // two streams for model parallelism (to overlap data transfer and computation)
		// based on above description, we need 6 events (3 join points):
		// data_joint: in forward pass, when data parallelism is done, and model parallelism will start;
		// model_joint[0]: in forward pass, the first stream's model parallelism is done;
		// model_joint[1]: in forward pass, the second stream's model parallelism is done;
		cudaEvent_t data_joint;
		cudaEvent_t model_joint[2];
		cublasHandle_t data_cublas; // the same, just cublas handle to stream
		cublasHandle_t model_cublas[2]; // the same, just cublas handle to stream
#ifdef HAVE_CUDNN
		// we only use cudnn for convolution and pooling
		cudnnHandle_t data_cudnn;
#endif
		float* input;
		int* c;
		float* out;
		float** dor;
		// events to synchronize a round
		cudaEvent_t start_timing;
		cudaEvent_t stop_timing;
	} device[MAX_DEVICE_SUPPORT];
} cwc_convnet_context_t;

typedef struct {
	size_t memory_usage;
} cwc_convnet_stats_t;

typedef struct {
	int batch;
	int tops;
	int device_count;
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
		int can_access_peer[MAX_DEVICE_SUPPORT];
	} device[MAX_DEVICE_SUPPORT];
	cwc_convnet_context_t contexts[2];
	cwc_convnet_stats_t stats;
} cwc_convnet_t;

#define GPU(x) ((cwc_convnet_t*)((x)->reserved))

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

#ifdef HAVE_CUDNN
// for now, ccv only uses convolutional part of cuDNN, because for pooling and softmax, libccv's implementation is slightly different (supporting padding for pooling)
static void _cwc_convnet_alloc_cudnn(ccv_convnet_t* convnet, int device_id, int start, int length, int rows, int cols, int batch)
{
	int i;
	ccv_convnet_layer_t* prior_layer = 0;
	int out_rows, out_cols, out_partition;
	for (i = start; i < start + length; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
		ccv_convnet_make_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
		{
			if (prior_layer)
				EXTRA(layer)->input_descriptor = EXTRA(prior_layer)->output_descriptor;
			else {
				cudnnCreateTensor4dDescriptor(&EXTRA(layer)->input_descriptor);
				cudnnSetTensor4dDescriptor(EXTRA(layer)->input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, layer->input.convolutional.count, rows, cols);
			}
			cudnnCreateTensor4dDescriptor(&EXTRA(layer)->output_descriptor);
			cudnnSetTensor4dDescriptor(EXTRA(layer)->output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, layer->net.convolutional.count, out_rows, out_cols);
			cudnnCreateFilterDescriptor(&EXTRA(layer)->filter_descriptor);
			cudnnSetFilterDescriptor(EXTRA(layer)->filter_descriptor);
			prior_layer = layer;
		} else
			prior_layer = 0;
		rows = out_rows, cols = out_cols;
	}
}
#endif

static void _cwc_convnet_alloc_layers(ccv_convnet_t* convnet, int device_id, int device_count)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	assert(device_id < device_count);
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
				_cwc_convnet_reorder_full_connect_weights_onto_device(convnet->layers[i].w + device_id * layers[i].wnum, layers[i].w, layers[i].wnum, layers[i].input.matrix.rows * layers[i].input.matrix.cols, layers[i].input.matrix.channels);
				cudaMemcpy(layers[i].bias, convnet->layers[i].bias + device_id * layers[i].net.full_connect.count, sizeof(float) * layers[i].net.full_connect.count, cudaMemcpyHostToDevice);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				layers[i].w = layers[i].bias = 0;
				break;
		}
}

static void _cwc_convnet_alloc_configurations(ccv_convnet_t* convnet, int device_id, int device_count)
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
				cudaMemset(configurations[i].w, 0, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count);
				assert(configurations[i].w);
				configurations[i].bias = configurations[i].w + layers[i].wnum;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				assert(configurations[i].type == CCV_CONVNET_FULL_CONNECT);
				// allocating for configurations 
				configurations[i].w = 0;
				// need to allocate two because we duplex backprop, therefore, need double-buffer for multiple devices
				cudaMalloc(&configurations[i].w, sizeof(float) * (device_count > 1 ? 2 : 1) * (layers[i].wnum + layers[i].net.full_connect.count));
				cudaMemset(configurations[i].w, 0, sizeof(float) * (device_count > 1 ? 2 : 1) * (layers[i].wnum + layers[i].net.full_connect.count));
				GPU(convnet)->stats.memory_usage += sizeof(float) * (device_count > 1 ? 2 : 1) * (layers[i].wnum + layers[i].net.full_connect.count);
				assert(configurations[i].w);
				configurations[i].bias = configurations[i].w + layers[i].wnum * (device_count > 1 ? 2 : 1);
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
				cudaMemset(momentums[i].w, 0, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
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
				cudaMemset(momentums[i].w, 0, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
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

static void _cwc_convnet_alloc_forwards(ccv_convnet_t* convnet, int device_id, int device_count, int start, int length, int rows, int cols, int batch)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	assert(start >= 0 && start + length <= convnet->count);
	int out_rows, out_cols, out_partition;
	for (i = start; i < start + length; i++)
	{
		ccv_convnet_make_output(layers + i, rows, cols, &out_rows, &out_cols, &out_partition);
		// if the next layer is full connect (model parallelism), the forwards neuron needs to hold all batches rather than 1
		int multi_batch = (i + 1 < start + length && layers[i + 1].type == CCV_CONVNET_FULL_CONNECT) ? batch * device_count : batch;
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				GPU(convnet)->device[device_id].forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * multi_batch);
				GPU(convnet)->stats.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].net.convolutional.count * multi_batch;
				assert(GPU(convnet)->device[device_id].forwards[i]);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				GPU(convnet)->device[device_id].forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * layers[i].net.full_connect.count * batch * device_count * device_count);
				GPU(convnet)->stats.memory_usage += sizeof(float) * layers[i].net.full_connect.count * batch * device_count * device_count;
				assert(GPU(convnet)->device[device_id].forwards[i]);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				assert(i > 0);
				GPU(convnet)->device[device_id].forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * multi_batch);
				GPU(convnet)->stats.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * multi_batch;
				assert(GPU(convnet)->device[device_id].forwards[i]);
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				GPU(convnet)->device[device_id].forwards[i] = 0;
				cudaMalloc(&GPU(convnet)->device[device_id].forwards[i], sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * multi_batch);
				GPU(convnet)->stats.memory_usage += sizeof(float) * out_rows * out_cols * layers[i].input.matrix.channels * multi_batch;
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
		ccv_convnet_make_output(layers + i, rows, cols, &out_rows, &out_cols, &out_partition);
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

static void _cwc_convnet_alloc_backwards(ccv_convnet_t* convnet, int device_id, int device_count, int batch)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	// find the layer with max memory usage, and then allocate two of them, because for backward propagate, no need to preserve the results
	size_t max_memory_usage[2] = {0};
	for (i = 1; i < convnet->count; i++)
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				max_memory_usage[i % 2] = ccv_max(max_memory_usage[i % 2], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(i > 0);
				// for full connect layer, because it uses model parallelism, each layer needs to hold multiple batches
				max_memory_usage[i % 2] = ccv_max(max_memory_usage[i % 2], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch * device_count);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				max_memory_usage[i % 2] = ccv_max(max_memory_usage[i % 2], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(i > 0);
				max_memory_usage[i % 2] = ccv_max(max_memory_usage[i % 2], sizeof(float) * layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch);
				break;
		}
	assert(convnet->count > 2);
	// allocate two layers
	GPU(convnet)->device[device_id].backwards[0] = 0;
	for (i = 1; i < 3; i++)
	{
		GPU(convnet)->device[device_id].backwards[i] = 0;
		cudaMalloc(&GPU(convnet)->device[device_id].backwards[i], max_memory_usage[i % 2]);
		GPU(convnet)->stats.memory_usage += max_memory_usage[i % 2];
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
		ccv_convnet_make_output(layers + i, rows, cols, &out_rows, &out_cols, &out_partition);
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

static void _cwc_convnet_alloc_context(ccv_convnet_t* convnet, int device_id, int context_id, int device_count)
{
	cwc_convnet_context_t* context = GPU(convnet)->contexts + context_id;
	cudaStreamCreate(&context->device[device_id].data_stream);
	cublasCreate(&context->device[device_id].data_cublas);
	cublasSetStream(context->device[device_id].data_cublas, context->device[device_id].data_stream);
	cudaEventCreate(&context->device[device_id].start_timing);
	cudaEventCreate(&context->device[device_id].stop_timing);
	int i;
	if (device_count > 1)
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

static void _cwc_convnet_alloc_scratch(ccv_convnet_t* convnet, int device_id, int device_count, int batch)
{
	int i;
	int out_rows, out_cols, out_partition;
	size_t scratch_space = 6; // for debugging and computing the stats
	ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
	for (i = 0; i < convnet->count; i++)
		if (layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
		{
			int use_rows = cwc_convnet_layer_use_rows(layers + i);
			ccv_convnet_make_output(layers + i, layers[i].input.matrix.rows, layers[i].input.matrix.cols, &out_rows, &out_cols, &out_partition);
			scratch_space = ccv_max(scratch_space, ccv_max(layers[i].wnum, layers[i].net.convolutional.count));
			scratch_space = ccv_max(scratch_space,
					(size_t)out_rows * out_cols * layers[i].net.convolutional.count * batch + // output layer reorder
					layers[i].input.matrix.rows * layers[i].input.matrix.cols * layers[i].input.matrix.channels * batch + // input layer reorder
					layers[i].wnum * (use_rows ? out_rows : 1) * (batch / BATCH_PER_BLOCK)); // unconsolidated weights output
		} else if (layers[i].type == CCV_CONVNET_FULL_CONNECT && device_count > 1)
			// for multiple device only, when we need it to copy over outputs for backprop
			scratch_space = ccv_max(scratch_space, ccv_max(layers[i].input.node.count * batch, layers[i].net.full_connect.count * batch * 2));
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
			ccv_convnet_make_output(layers + i, layers[i].input.matrix.rows, layers[i].input.matrix.cols, &out_rows, &out_cols, &out_partition);
			if (cwc_convnet_layer_use_rows(layers + i))
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
	convnet->reserved = (cwc_convnet_t*)ccmalloc(sizeof(cwc_convnet_t) + sizeof(cwc_convnet_layer_t) * convnet->count + sizeof(ccv_convnet_layer_t) * convnet->count + sizeof(float*) * convnet->count * 3);
	GPU(convnet)->batch = batch;
	GPU(convnet)->tops = tops;
	GPU(convnet)->device_count = 1;
	GPU(convnet)->layer_params = 0;
	GPU(convnet)->stats.memory_usage = 0;
	cwc_convnet_layer_t* layer_extra = (cwc_convnet_layer_t*)(GPU(convnet) + 1);
	memset(layer_extra, 0, sizeof(cwc_convnet_layer_t) * convnet->count);
	GPU(convnet)->device[0].layers = (ccv_convnet_layer_t*)(layer_extra + convnet->count);
	memcpy(GPU(convnet)->device[0].layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	ccv_convnet_layer_t* layers = GPU(convnet)->device[0].layers;
	// point reserved place to layer_vary
	int i;
	for (i = 0; i < convnet->count; i++)
		layers[i].reserved = layer_extra + i;
	// alloc and copy layers
	_cwc_convnet_alloc_layers(convnet, 0, 1);
	GPU(convnet)->device[0].configurations = 0;
	GPU(convnet)->device[0].momentums = 0;
	GPU(convnet)->device[0].scratch = 0;
	_cwc_convnet_make_unit(convnet, 0, batch * 30);
	int scan = _cwc_convnet_find_scan(convnet, 0);
	GPU(convnet)->device[0].forwards = (float**)(GPU(convnet)->device[0].layers + convnet->count);
	// alloc forwards until the scan layer (for initial 6 patches)
	_cwc_convnet_alloc_forwards(convnet, 0, 1, 0, scan + 1, convnet->input.height, convnet->input.width, batch * 6);
	// alloc forwards from scan layer (for scanned 30 patches)
	_cwc_convnet_alloc_forwards(convnet, 0, 1, scan + 1, convnet->count - scan - 1, GPU(convnet)->device[0].layers[scan + 1].input.matrix.rows, GPU(convnet)->device[0].layers[scan + 1].input.matrix.cols, batch * 30);
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
	_cwc_convnet_alloc_context(convnet, 0, 0, 1);
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
static void _cwc_convnet_alloc_reserved_both(ccv_convnet_t* convnet, int batch, int device_count, ccv_convnet_layer_train_param_t* layer_params)
{
	if (GPU(convnet) && (GPU(convnet)->batch != batch || GPU(convnet)->tops != 0 || GPU(convnet)->device_count != device_count || GPU(convnet)->layer_params != layer_params))
		ccv_convnet_compact(convnet);
	else if (GPU(convnet))
		return; // it is allocated properly, no-op
	uint8_t* reserved = (uint8_t*)ccmalloc(sizeof(cwc_convnet_t) + (sizeof(cwc_convnet_layer_t) * convnet->count + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 10) * device_count);
	convnet->reserved = (cwc_convnet_t*)reserved;
	GPU(convnet)->batch = batch;
	GPU(convnet)->tops = 0;
	GPU(convnet)->device_count = device_count;
	GPU(convnet)->layer_params = layer_params;
	GPU(convnet)->stats.memory_usage = 0;
	int i, device_id;
	for (device_id = 0; device_id < device_count; device_id++)
	{
		cudaSetDevice(device_id);
		GPU(convnet)->device[device_id].scans = 0;
		for (i = 0; i < device_count; i++)
			GPU(convnet)->device[device_id].can_access_peer[i] = (i == device_id); // init to it can access itself
		cwc_convnet_layer_t* layer_extra = (cwc_convnet_layer_t*)(reserved + sizeof(cwc_convnet_t) + (sizeof(cwc_convnet_layer_t) * convnet->count + sizeof(ccv_convnet_layer_t) * convnet->count * 3 + sizeof(float*) * convnet->count * 10) * device_id);
		memset(layer_extra, 0, sizeof(cwc_convnet_layer_t) * convnet->count);
		GPU(convnet)->device[device_id].layers = (ccv_convnet_layer_t*)(layer_extra + convnet->count);
		memcpy(GPU(convnet)->device[device_id].layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
		ccv_convnet_layer_t* layers = GPU(convnet)->device[device_id].layers;
		for (i = 0; i < convnet->count; i++)
		{
			// point reserved place to layer_extra
			layers[i].reserved = layer_extra + i;
			// depends on if it is device_count or not, full_connect will use model parallelism, therefore, here we split the model into half
			if (layers[i].type == CCV_CONVNET_FULL_CONNECT)
			{
				assert(convnet->layers[i].net.full_connect.count % device_count == 0);
				layers[i].net.full_connect.count = convnet->layers[i].net.full_connect.count / device_count;
				layers[i].wnum = layers[i].net.full_connect.count * layers[i].input.node.count;
			}
		}
		// hook up configurations (the backprop coefficients)
		GPU(convnet)->device[device_id].configurations = GPU(convnet)->device[device_id].layers + convnet->count;
		memcpy(GPU(convnet)->device[device_id].configurations, layers, sizeof(ccv_convnet_layer_t) * convnet->count);
		// hook up momentums
		GPU(convnet)->device[device_id].momentums = GPU(convnet)->device[device_id].layers + convnet->count * 2;
		memcpy(GPU(convnet)->device[device_id].momentums, layers, sizeof(ccv_convnet_layer_t) * convnet->count);
		// alloc and copy layers
		_cwc_convnet_alloc_layers(convnet, device_id, device_count);
		// alloc scratch space (for backprop on convolutional layer)
		_cwc_convnet_alloc_scratch(convnet, device_id, device_count, batch);
		// alloc and make unit vector
		_cwc_convnet_make_unit(convnet, device_id, batch);
		// alloc & copy configurations (the backprop coefficients)
		_cwc_convnet_alloc_configurations(convnet, device_id, device_count);
		// alloc & copy momentums
		_cwc_convnet_alloc_momentums(convnet, device_id);
		// hook up forwards and alloc forwards
		GPU(convnet)->device[device_id].forwards = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3);
		_cwc_convnet_alloc_forwards(convnet, device_id, device_count, 0, convnet->count, convnet->rows, convnet->cols, batch);
		// hook up denoms and alloc denoms
		GPU(convnet)->device[device_id].denoms = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3) + convnet->count * 2;
		_cwc_convnet_alloc_denoms(convnet, device_id, 0, convnet->count, convnet->rows, convnet->cols, batch);
		// hook up backwards and alloc backwards
		GPU(convnet)->device[device_id].backwards = (float**)(GPU(convnet)->device[device_id].layers + convnet->count * 3) + convnet->count;
		// hook up dor and alloc dor
		_cwc_convnet_alloc_backwards(convnet, device_id, device_count, batch);
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
			_cwc_convnet_alloc_context(convnet, device_id, i, device_count);
		}
	}
}

static void _cwc_convnet_enable_peer_access(ccv_convnet_t* convnet, int device_count)
{
	int device_id, other_device_id;
	for (device_id = 0; device_id < device_count; device_id++)
		for (other_device_id = 0; other_device_id < device_count; other_device_id++)
			if (device_id != other_device_id)
			{
				cudaSetDevice(device_id);
				GPU(convnet)->device[device_id].can_access_peer[other_device_id] = 0;
				assert(cudaSuccess == cudaDeviceCanAccessPeer(&GPU(convnet)->device[device_id].can_access_peer[other_device_id], device_id, other_device_id));
				if (GPU(convnet)->device[device_id].can_access_peer[other_device_id]) // only enable peer access when can access peer
					cudaDeviceEnablePeerAccess(other_device_id, 0);
			} else
				GPU(convnet)->device[device_id].can_access_peer[other_device_id] = 1; // of course
}

// =========================================== KERNEL CODE ===================================================

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
			cwc_convnet_convolutional_forward_propagate(layer, rows, cols, batch, a, b, context->device[device_id].data_stream);
			if (dor && context->device[device_id].dor[k])
			{
				int out_rows, out_cols, out_partition;
				ccv_convnet_make_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
				_cwc_kern_mute_neuron
				<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device[device_id].data_stream>>>
				(b, context->device[device_id].dor[k]);
			}
			break;
		case CCV_CONVNET_FULL_CONNECT:
			assert(k > 0);
			cwc_convnet_full_connect_forward_propagate(layer, batch, a, b, batch_unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
			if (dor && context->device[device_id].dor[k])
				_cwc_kern_mute_neuron
				<<<layer->net.full_connect.count, batch, 0, context->device[device_id].data_stream>>>
				(b, context->device[device_id].dor[k]);
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			assert(k > 0);
			cwc_convnet_rnorm_forward_propagate(layer, rows, cols, batch, a, b, denoms, context->device[device_id].data_stream);
			break;
		case CCV_CONVNET_MAX_POOL:
			assert(k > 0);
			cwc_convnet_max_pool_forward_propagate(layer, rows, cols, batch, a, b, context->device[device_id].data_stream);
			break;
		case CCV_CONVNET_AVERAGE_POOL:
			assert(k > 0);
			cwc_convnet_average_pool_forward_propagate(layer, rows, cols, batch,  a, b, context->device[device_id].data_stream);
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
static void _cwc_convnet_encode_impl(ccv_convnet_t* convnet, int device_count, int batch, int dor, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	if (device_count > 1)
	{
		int j, device_id, other_device_id;
		int count = _cwc_convnet_first_full_connect(convnet);
		for (i = 0; i < count; i++)
		{
			for (device_id = 0; device_id < device_count; device_id++)
			{
				cudaSetDevice(device_id);
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				// adding offset so that follow up computation is easier
				float* a = (i == count - 1) ? GPU(convnet)->device[device_id].forwards[i] + device_id * batch * GPU(convnet)->device[device_id].layers[count].input.node.count : GPU(convnet)->device[device_id].forwards[i];
				_cwc_convnet_layer_forward_propagate(layer, device_id, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, dor, i == 0 ? context->device[device_id].input : GPU(convnet)->device[device_id].forwards[i - 1], a, GPU(convnet)->device[device_id].denoms[i], GPU(convnet)->device[device_id].unit, context);
			}
		}
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			cudaEventRecord(context->device[device_id].data_joint, context->device[device_id].data_stream);
		}
		// the connecting layer from data parallelism to model parallelism
		// this is different because first, I need to copy the full batch from first GPU to the second GPU (and vice versa).
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + count;
			assert(layer->type == CCV_CONVNET_FULL_CONNECT);
			for (i = 0; i < device_count; i++)
			{
				cudaStreamWaitEvent(context->device[device_id].model_stream[i % 2], context->device[i].data_joint, 0);
				if (device_id != i)
					// overlap the memory copy and the gemm computation
					// although it is slightly faster to use source stream, but we sync on the destination stream because otherwise we need to signal an event, and in that case, total cost is higher
					cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].forwards[count - 1] + i * batch * layer->input.node.count, device_id, GPU(convnet)->device[i].forwards[count - 1] + i * batch * layer->input.node.count, i, sizeof(float) * batch * layer->input.node.count, context->device[device_id].model_stream[i % 2]);
				// the last compute issued on this device
				cwc_convnet_full_connect_forward_propagate(layer, batch,
						GPU(convnet)->device[device_id].forwards[count - 1] + i * batch * layer->input.node.count,
						GPU(convnet)->device[device_id].forwards[count] + (device_count * i + device_id) * batch * layer->net.full_connect.count,
						GPU(convnet)->device[device_id].unit,
						context->device[device_id].model_stream[i % 2], context->device[device_id].model_cublas[i % 2]);
				if (dor && context->device[device_id].dor[count])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[i % 2]>>>
					(GPU(convnet)->device[device_id].forwards[count] + (device_count * i + device_id) * batch * layer->net.full_connect.count, context->device[device_id].dor[count]);
				// figure out the free memory to use
			}
		}
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			cudaEventRecord(context->device[device_id].model_joint[0], context->device[device_id].model_stream[0]);
			cudaEventRecord(context->device[device_id].model_joint[1], context->device[device_id].model_stream[1]);
		}
		for (i = count + 1; i < convnet->count; i++)
		{
			for (device_id = 0; device_id < device_count; device_id++)
			{
				cudaSetDevice(device_id);
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				assert(layer->type == CCV_CONVNET_FULL_CONNECT);
				// finishing copy the 1 / 4th so that everyone can proceed
				assert(layer->input.node.count % device_count == 0);
				int input_node_count = layer->input.node.count / device_count;
				for (j = 0; j < device_count; j++)
				{
					// copy data in, and do the computation
					for (other_device_id = 0; other_device_id < device_count; other_device_id++)
						if (other_device_id != device_id)
						{
							cudaStreamWaitEvent(context->device[device_id].model_stream[j % 2], context->device[other_device_id].model_joint[j % 2], 0);
							// wait the previous iteration on this to finish
							cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].forwards[i - 1] + (device_count * j + other_device_id) * batch * input_node_count, device_id, GPU(convnet)->device[other_device_id].forwards[i - 1] + (device_count * j + other_device_id) * batch * input_node_count, other_device_id, sizeof(float) * batch * input_node_count, context->device[device_id].model_stream[j % 2]);
						}
					// first do the computation on the device that we already have full data
					cwc_convnet_full_connect_forward_propagate(layer, batch,
							GPU(convnet)->device[device_id].forwards[i - 1] + j * batch * layer->input.node.count,
							GPU(convnet)->device[device_id].forwards[i] + (device_count * j + device_id) * batch * layer->net.full_connect.count,
							GPU(convnet)->device[device_id].unit, context->device[device_id].model_stream[j % 2], context->device[device_id].model_cublas[j % 2]);
					if (dor && context->device[device_id].dor[i])
						_cwc_kern_mute_neuron
						<<<layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[j % 2]>>>
						(GPU(convnet)->device[device_id].forwards[i] + (device_count * j + device_id) * batch * layer->net.full_connect.count,
						 context->device[device_id].dor[i]);
				}
			}
			// record events after we completed issuing command to avoid race (otherwise we are waiting events we just recorded)
			for (device_id = 0; device_id < device_count; device_id++)
			{
				cudaSetDevice(device_id);
				cudaEventRecord(context->device[device_id].model_joint[0], context->device[device_id].model_stream[0]);
				cudaEventRecord(context->device[device_id].model_joint[1], context->device[device_id].model_stream[1]);
			}
		}
		// copy each batch full result to each device
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			ccv_convnet_layer_t* last_layer = GPU(convnet)->device[device_id].layers + (convnet->count - 1);
			assert(last_layer->type == CCV_CONVNET_FULL_CONNECT);
			int output_node_count = last_layer->net.full_connect.count; // this is already halved.
			// copy data in, and do the computation
			for (other_device_id = 0; other_device_id < device_count; other_device_id++)
				if (other_device_id != device_id)
				{
					cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[other_device_id].model_joint[device_id % 2], 0);
					cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].forwards[convnet->count - 1] + (device_count * device_id + other_device_id) * batch * output_node_count, device_id, GPU(convnet)->device[other_device_id].forwards[convnet->count - 1] + (device_count * device_id + other_device_id) * batch * output_node_count, other_device_id, sizeof(float) * batch * output_node_count, context->device[device_id].data_stream);
				}
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[device_id].model_joint[device_id % 2], 0);
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

__global__ static void _cwc_kern_softmax_with_logistic_loss(const int batch, const int count, float* a, int* c)
{
	const int thidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thidx < batch)
	{
		int i;
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
}

static void _cwc_convnet_softmax_with_logistic_loss(int batch, int count, float* a, int* c, const cudaStream_t& stream)
{
	dim3 num_blocks((batch + 63) / 64);
	dim3 threads_per_block(ccv_min(batch, 64));
	assert(threads_per_block.x <= 1024);
	int shared_memory_size = sizeof(float) * batch;
	_cwc_kern_softmax_with_logistic_loss
	<<<num_blocks, threads_per_block, shared_memory_size, stream>>>
	(batch, count, a, c);
}

__global__ static void _cwc_kern_tests_return(const int batch, const int count, float* a, int* c)
{
	const int thidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thidx < batch)
	{
		int i;
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
}

static void _cwc_convnet_tests_return(int batch, int count, float* a, int* c, const cudaStream_t& stream)
{
	dim3 num_blocks((batch + 63) / 64);
	dim3 threads_per_block(ccv_min(batch, 64));
	assert(threads_per_block.x <= 1024);
	_cwc_kern_tests_return
	<<<num_blocks, threads_per_block, 0, stream>>>
	(batch, count, a, c);
}

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
		float velocity = momentum_rate * momentum[thidx] - decay_and_learn * old_a + learn_rate * grad[thidx];
		a[thidx] = velocity + old_a;
		momentum[thidx] = velocity;
	}
}

static void _cwc_convnet_collect_disk_stats(ccv_convnet_t* convnet)
{
	int i, j;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = convnet->layers + i;
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT)
		{
			int bias_count = layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count;
			float w_asum = 0;
			for (j = 0; j < layer->wnum; j++)
				w_asum += layer->w[j];
			float w_mean = w_asum / layer->wnum;
			float w_variance = 0;
			for (j = 0; j < layer->wnum; j++)
				w_variance += (layer->w[j] - w_mean) * (layer->w[j] - w_mean);
			w_variance = w_variance / layer->wnum;
			float bias_asum = 0;
			for (j = 0; j < bias_count; j++)
				bias_asum += layer->bias[j];
			float bias_mean = bias_asum / bias_count;
			float bias_variance = 0;
			for (j = 0; j < bias_count; j++)
				bias_variance += (layer->bias[j] - bias_mean) * (layer->bias[j] - bias_mean);
			bias_variance = bias_variance / bias_count;
			PRINT(CCV_CLI_VERBOSE, " - %03d * %g %g | %g %g\n", i, w_mean, w_variance, bias_mean, bias_variance);
		}
	}
}

static void _cwc_convnet_collect_runtime_stats(ccv_convnet_t* convnet, cwc_convnet_context_t* context)
{
	int i;
	// collecting stats from device 0
	const int device_id = 0;
	cudaSetDevice(device_id);
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
		ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
		ccv_convnet_layer_t* momentum = GPU(convnet)->device[device_id].momentums + i;
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT)
		{
			int bias_count = layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count;
			float layer_w_nrm2;
			cublasSgemv(context->device[device_id].data_cublas, CUBLAS_OP_T, layer->wnum, 1, &one, layer->w, layer->wnum, GPU(convnet)->device[device_id].unit, 1, &zero, GPU(convnet)->device[device_id].scratch, 1);
			cublasSnrm2(context->device[device_id].data_cublas, layer->wnum, layer->w, 1, &layer_w_nrm2);
			float layer_bias_nrm2;
			cublasSgemv(context->device[device_id].data_cublas, CUBLAS_OP_T, bias_count, 1, &one, layer->bias, bias_count, GPU(convnet)->device[device_id].unit, 1, &zero, GPU(convnet)->device[device_id].scratch + 1, 1);
			cublasSnrm2(context->device[device_id].data_cublas, bias_count, layer->bias, 1, &layer_bias_nrm2);
			float configuration_w_nrm2;
			cublasSgemv(context->device[device_id].data_cublas, CUBLAS_OP_T, layer->wnum, 1, &one, configuration->w, layer->wnum, GPU(convnet)->device[device_id].unit, 1, &zero, GPU(convnet)->device[device_id].scratch + 2, 1);
			cublasSnrm2(context->device[device_id].data_cublas, configuration->wnum, configuration->w, 1, &configuration_w_nrm2);
			float configuration_bias_nrm2;
			cublasSgemv(context->device[device_id].data_cublas, CUBLAS_OP_T, bias_count, 1, &one, configuration->bias, bias_count, GPU(convnet)->device[device_id].unit, 1, &zero, GPU(convnet)->device[device_id].scratch + 3, 1);
			cublasSnrm2(context->device[device_id].data_cublas, bias_count, configuration->bias, 1, &configuration_bias_nrm2);
			float momentum_w_nrm2;
			cublasSgemv(context->device[device_id].data_cublas, CUBLAS_OP_T, layer->wnum, 1, &one, momentum->w, layer->wnum, GPU(convnet)->device[device_id].unit, 1, &zero, GPU(convnet)->device[device_id].scratch + 4, 1);
			cublasSnrm2(context->device[device_id].data_cublas, momentum->wnum, momentum->w, 1, &momentum_w_nrm2);
			float momentum_bias_nrm2;
			cublasSgemv(context->device[device_id].data_cublas, CUBLAS_OP_T, bias_count, 1, &one, momentum->bias, bias_count, GPU(convnet)->device[device_id].unit, 1, &zero, GPU(convnet)->device[device_id].scratch + 5, 1);
			cublasSnrm2(context->device[device_id].data_cublas, bias_count, momentum->bias, 1, &momentum_bias_nrm2);
			float sum[6];
			cudaMemcpyAsync(&sum, GPU(convnet)->device[device_id].scratch, sizeof(float) * 6, cudaMemcpyDeviceToHost, context->device[device_id].data_stream);
			cudaStreamSynchronize(context->device[device_id].data_stream);
			float layer_w_mean = sum[0] / layer->wnum;
			float layer_bias_mean = sum[1] / bias_count;
			float layer_w_variance = (layer_w_nrm2 * layer_w_nrm2 - sum[0] * sum[0] / layer->wnum) / (layer->wnum - 1);
			float layer_bias_variance = (layer_bias_nrm2 * layer_bias_nrm2 - sum[1] * sum[1] / bias_count) / (bias_count - 1);
			float configuration_w_mean = sum[2] / layer->wnum;
			float configuration_bias_mean = sum[3] / bias_count;
			float configuration_w_variance = (configuration_w_nrm2 * configuration_w_nrm2 - sum[2] * sum[2] / layer->wnum) / (layer->wnum - 1);
			float configuration_bias_variance = (configuration_bias_nrm2 * configuration_bias_nrm2 - sum[3] * sum[3] / bias_count) / (bias_count - 1);
			float momentum_w_mean = sum[4] / layer->wnum;
			float momentum_bias_mean = sum[5] / bias_count;
			float momentum_w_variance = (momentum_w_nrm2 * momentum_w_nrm2 - sum[4] * sum[4] / layer->wnum) / (layer->wnum - 1);
			float momentum_bias_variance = (momentum_bias_nrm2 * momentum_bias_nrm2 - sum[5] * sum[5] / bias_count) / (bias_count - 1);
			PRINT(CCV_CLI_VERBOSE, " - %03d * %g %g | %g %g\n - %d # %g %g | %g %g\n - %d @ %g %g | %g %g\n", i, layer_w_mean, layer_w_variance, layer_bias_mean, layer_bias_variance, i, configuration_w_mean, configuration_w_variance, configuration_bias_mean, configuration_bias_variance, i, momentum_w_mean, momentum_w_variance, momentum_bias_mean, momentum_bias_variance);
		}
	}
}

static void _cwc_convnet_reduce_data_parallelism(ccv_convnet_t* convnet, int device_count, cwc_convnet_context_t* context)
{
	int i, j, k, device_id;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* host_layer = convnet->layers + i;
		if (host_layer->type == CCV_CONVNET_FULL_CONNECT)
			for (device_id = 0; device_id < device_count; device_id++)
			{
				cudaSetDevice(device_id);
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
				// it is duplexed, let's compute it back again
				cublasSaxpy(context->device[device_id].data_cublas, layer->wnum, &one, configuration->w + layer->wnum, 1, configuration->w, 1);
				cublasSaxpy(context->device[device_id].data_cublas, layer->net.full_connect.count, &one, configuration->bias + layer->net.full_connect.count, 1, configuration->bias, 1);
			}
	}
	for (i = (device_count >> 1); i > 0; i = (i >> 1))
	{
		assert((device_count % i) == 0);
		k = device_count / i;
		for (device_id = 0; device_id < device_count; device_id += k)
		{
			int other_device_id = device_id + (k >> 1);
			cudaSetDevice(other_device_id);
			cudaEventRecord(context->device[other_device_id].data_joint, context->device[other_device_id].data_stream);
		}
		for (device_id = 0; device_id < device_count; device_id += k)
		{
			cudaSetDevice(device_id);
			int other_device_id = device_id + (k >> 1);
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[other_device_id].data_joint, 0);
			for (j = 0; j < convnet->count; j++)
			{
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + j;
				if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
				{
					ccv_convnet_layer_t* configuration_a = GPU(convnet)->device[device_id].configurations + j;
					ccv_convnet_layer_t* configuration_b = GPU(convnet)->device[other_device_id].configurations + j;
					if (GPU(convnet)->device[device_id].can_access_peer[other_device_id])
					{
						cublasSaxpy(context->device[device_id].data_cublas, layer->wnum, &one, configuration_b->w, 1, configuration_a->w, 1);
						cublasSaxpy(context->device[device_id].data_cublas, layer->net.convolutional.count, &one, configuration_b->bias, 1, configuration_a->bias, 1);
					} else {
						cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].scratch, device_id, configuration_b->w, other_device_id, sizeof(float) * layer->wnum, context->device[device_id].data_stream);
						cublasSaxpy(context->device[device_id].data_cublas, layer->wnum, &one, GPU(convnet)->device[device_id].scratch, 1, configuration_a->w, 1);
						cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].scratch, device_id, configuration_b->bias, other_device_id, sizeof(float) * layer->net.convolutional.count, context->device[device_id].data_stream);
						cublasSaxpy(context->device[device_id].data_cublas, layer->net.convolutional.count, &one, GPU(convnet)->device[device_id].scratch, 1, configuration_a->bias, 1);
					}
				}
			}
		}
	}
	cudaSetDevice(0);
	cudaEventRecord(context->device[0].data_joint, context->device[0].data_stream);
	// other devices need to sync with the 0 device before proceed (otherwise in net_sgd, it will reset configuration->w to 0 on its own device and cause out-of-sync)
	for (device_id = 1; device_id < device_count; device_id++)
	{
		cudaSetDevice(device_id);
		cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[0].data_joint, 0);
	}
}

static void _cwc_convnet_broadcast_data_parallelism(ccv_convnet_t* convnet, int device_count, cwc_convnet_context_t* context)
{
	assert(device_count > 1);
	cudaSetDevice(0);
	int i, device_id;
	for (device_id = 1; device_id < device_count; device_id++)
		for (i = 0; i < convnet->count; i++)
		{
			ccv_convnet_layer_t* layer_a = GPU(convnet)->device[0].layers + i;
			ccv_convnet_layer_t* layer_b = GPU(convnet)->device[device_id].layers + i;
			if (layer_a->type == CCV_CONVNET_CONVOLUTIONAL)
			{
				cudaMemcpyPeerAsync(layer_b->w, device_id, layer_a->w, 0, sizeof(float) * layer_a->wnum, context->device[0].data_stream);
				cudaMemcpyPeerAsync(layer_b->bias, device_id, layer_a->bias, 0, sizeof(float) * layer_a->net.convolutional.count, context->device[0].data_stream);
			}
		}
	cudaEventRecord(context->device[0].data_joint, context->device[0].data_stream);
	// sync on newly updated parameters
	for (device_id = 1; device_id < device_count; device_id++)
	{
		cudaSetDevice(device_id);
		cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[0].data_joint, 0);
	}
}

static void _cwc_convnet_net_sgd(ccv_convnet_t* convnet, int device_count, int batch, ccv_convnet_layer_train_param_t* layer_params, cwc_convnet_context_t* context)
{
	int i, device_id, out_rows, out_cols, out_partition;
	dim3 threads_per_block(64);
	assert(threads_per_block.x <= 1024);
	dim3 num_blocks_for_coeff;
	dim3 num_blocks_for_bias;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* configuration;
		ccv_convnet_layer_t* momentum;
		ccv_convnet_layer_t* layer = convnet->layers + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				// this is data parallelism, assume you reduced to device 0, only do updates on device 0 now
				device_id = 0;
				cudaSetDevice(device_id);
				layer = GPU(convnet)->device[device_id].layers + i;
				configuration = GPU(convnet)->device[device_id].configurations + i;
				momentum = GPU(convnet)->device[device_id].momentums + i;
				ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
				num_blocks_for_coeff = (layer->wnum + 63) / 64;
				num_blocks_for_bias = (layer->net.convolutional.count + 63) / 64;
				_cwc_kern_net_sgd
				<<<num_blocks_for_coeff, threads_per_block, 0, context->device[device_id].data_stream>>>
				(layer->w, configuration->w, momentum->w, layer->wnum,
				 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
				_cwc_kern_net_sgd
				<<<num_blocks_for_bias, threads_per_block, 0, context->device[device_id].data_stream>>>
				(layer->bias, configuration->bias, momentum->bias, layer->net.convolutional.count,
				 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
				for (device_id = 0; device_id < device_count; device_id++)
				{
					cudaSetDevice(device_id);
					layer = GPU(convnet)->device[device_id].layers + i;
					configuration = GPU(convnet)->device[device_id].configurations + i;
					// reset so that we can accumulate again
					cudaMemsetAsync(configuration->w, 0, sizeof(float) * layer->wnum, context->device[device_id].data_stream);
					cudaMemsetAsync(configuration->bias, 0, sizeof(float) * layer->net.convolutional.count, context->device[device_id].data_stream);
				}
				break;
			case CCV_CONVNET_FULL_CONNECT:
				// this is model parallelism, therefore, updates on each device
				for (device_id = 0; device_id < device_count; device_id++)
				{
					cudaSetDevice(device_id);
					layer = GPU(convnet)->device[device_id].layers + i;
					configuration = GPU(convnet)->device[device_id].configurations + i;
					momentum = GPU(convnet)->device[device_id].momentums + i;
					num_blocks_for_coeff = (layer->wnum + 63) / 64;
					num_blocks_for_bias = (layer->net.full_connect.count + 63) / 64;
					_cwc_kern_net_sgd
					<<<num_blocks_for_coeff, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->w, configuration->w, momentum->w, layer->wnum,
					 layer_params[i].w.learn_rate / batch, layer_params[i].w.momentum, layer_params[i].w.decay * layer_params[i].w.learn_rate);
					_cwc_kern_net_sgd
					<<<num_blocks_for_bias, threads_per_block, 0, context->device[device_id].data_stream>>>
					(layer->bias, configuration->bias, momentum->bias, layer->net.full_connect.count,
					 layer_params[i].bias.learn_rate / batch, layer_params[i].bias.momentum, layer_params[i].bias.decay * layer_params[i].bias.learn_rate);
					// reset so that we can accumulate again
					cudaMemsetAsync(configuration->w, 0, sizeof(float) * (device_count > 1 ? 2 : 1) * layer->wnum, context->device[device_id].data_stream);
					cudaMemsetAsync(configuration->bias, 0, sizeof(float) * (device_count > 1 ? 2 : 1) * layer->net.full_connect.count, context->device[device_id].data_stream);
				}
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				break;
		}
	}
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
			ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
			assert(layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT);
			int count = layer->type == CCV_CONVNET_FULL_CONNECT ? layer->net.full_connect.count : out_rows * out_cols * layer->net.convolutional.count;
			for (j = 0; j < batch * count; j++)
				context->host[device_id].dor[i][j] = (gsl_rng_uniform(rng) >= layer_params[i].dor) ? 1.0 : 0.0;
			cudaMemcpyAsync(context->device[device_id].dor[i], context->host[device_id].dor[i], sizeof(float) * count * batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
			ASSERT_NO_CUDA_ERROR();
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
				ccv_convnet_make_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
				_cwc_kern_mute_neuron
				<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device[device_id].data_stream>>>
				(a, context->device[device_id].dor[k]);
			}
			cwc_convnet_convolutional_backward_propagate(layer, batch, a, n, m, b, configuration, scratch, batch_unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
			ASSERT_NO_CUDA_ERROR();
			break;
		case CCV_CONVNET_FULL_CONNECT:
			if (context->device[device_id].dor[k])
				_cwc_kern_mute_neuron
				<<<layer->net.full_connect.count, batch, 0, context->device[device_id].data_stream>>>
				(a, context->device[device_id].dor[k]);
			cwc_convnet_full_connect_backward_propagate(layer, batch, a, n, m, b, batch_unit, configuration->w, configuration->bias, context->device[device_id].data_stream, context->device[device_id].data_cublas);
			ASSERT_NO_CUDA_ERROR();
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			cwc_convnet_rnorm_backward_propagate(layer, batch, a, n, m, denoms, b, context->device[device_id].data_stream);
			ASSERT_NO_CUDA_ERROR();
			break;
		case CCV_CONVNET_MAX_POOL:
			cwc_convnet_max_pool_backward_propagate(layer, batch, a, n, m, b, context->device[device_id].data_stream);
			ASSERT_NO_CUDA_ERROR();
			break;
		case CCV_CONVNET_AVERAGE_POOL:
			cwc_convnet_average_pool_backward_propagate(layer, batch, a, b, context->device[device_id].data_stream);
			ASSERT_NO_CUDA_ERROR();
			break;
	}
}

static void _cwc_convnet_backward_propagate_error(ccv_convnet_t* convnet, int device_count, int batch, cwc_convnet_context_t* context)
{
	assert(batch % 16 == 0);
	int i;
	if (device_count > 1)
	{
		int j, device_id, other_device_id;
		int count = _cwc_convnet_first_full_connect(convnet);
		// wait whatever on device 0 to finish (softmax and loss)
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			cudaEventRecord(context->device[device_id].data_joint, context->device[device_id].data_stream);
		}
		// for the rest of the devices, we first need to copy over from the first device, and then do the computation
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			ccv_convnet_layer_t* last_layer = GPU(convnet)->device[device_id].layers + convnet->count - 1;
			assert(last_layer->type == CCV_CONVNET_FULL_CONNECT);
			ccv_convnet_layer_t* last_configuration = GPU(convnet)->device[device_id].configurations + convnet->count - 1;
			for (i = 0; i < device_count; i++)
			{
				// copy over all remaining blocks to current device
				cudaStreamWaitEvent(context->device[device_id].model_stream[i % 2], context->device[i].data_joint, 0);
				// if current device is not current batch, need to copy it over first
				if (device_id != i)
					cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].forwards[convnet->count - 1] + (device_count * i + device_id) * batch * last_layer->net.full_connect.count, device_id, GPU(convnet)->device[i].forwards[convnet->count - 1] + (device_count * i + device_id) * batch * last_layer->net.full_connect.count, i, sizeof(float) * batch * last_layer->net.full_connect.count, context->device[device_id].model_stream[i % 2]);
				if (context->device[device_id].dor[convnet->count - 1])
					_cwc_kern_mute_neuron
					<<<last_layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[i % 2]>>>
					(GPU(convnet)->device[device_id].forwards[convnet->count - 1] + (device_count * i + device_id) * batch * last_layer->net.full_connect.count, context->device[device_id].dor[convnet->count - 1]);
				cwc_convnet_full_connect_backward_propagate(last_layer, batch,
						GPU(convnet)->device[device_id].forwards[convnet->count - 1] + (device_count * i + device_id) * batch * last_layer->net.full_connect.count,
						0, // this is unused for this layer, otherwise we screwed, in that case, 0 will crash, and we know it
						GPU(convnet)->device[device_id].forwards[convnet->count - 2] + i * batch * last_layer->input.node.count,
						GPU(convnet)->device[device_id].backwards[convnet->count - 1] + i * batch * last_layer->input.node.count,
						GPU(convnet)->device[device_id].unit, last_configuration->w + (i % 2) * last_layer->wnum, last_configuration->bias + (i % 2) * last_layer->net.full_connect.count, context->device[device_id].model_stream[i % 2], context->device[device_id].model_cublas[i % 2]);
			}
		}
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			cudaEventRecord(context->device[device_id].model_joint[0], context->device[device_id].model_stream[0]);
			cudaEventRecord(context->device[device_id].model_joint[1], context->device[device_id].model_stream[1]);
		}
		for (i = convnet->count - 2; i >= count; i--)
		{
			assert(i > 0);
			for (device_id = 0; device_id < device_count; device_id++)
			{
				cudaSetDevice(device_id);
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				assert(layer->type == CCV_CONVNET_FULL_CONNECT);
				ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
				for (j = 0; j < device_count; j++)
				{
					for (other_device_id = 0; other_device_id < device_count; other_device_id++)
						if (other_device_id != device_id)
						{
							cudaStreamWaitEvent(context->device[device_id].model_stream[j % 2], context->device[other_device_id].model_joint[j % 2], 0);
							// make sure we enabled peer to peer direct access, doing sum direct from the other device is faster than copy over and sum from my profiling
							if (GPU(convnet)->device[device_id].can_access_peer[other_device_id])
								cublasSaxpy(context->device[device_id].model_cublas[j % 2], batch * layer->net.full_connect.count, &one, GPU(convnet)->device[other_device_id].backwards[i + 1] + (device_count * j + device_id) * batch * layer->net.full_connect.count, 1, GPU(convnet)->device[device_id].backwards[i + 1] + (device_count * j + device_id) * batch * layer->net.full_connect.count, 1);
							else {
								cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].scratch + (j % 2) * batch * layer->net.full_connect.count, device_id, GPU(convnet)->device[other_device_id].backwards[i + 1] + (device_count * j + device_id) * batch * layer->net.full_connect.count, other_device_id, sizeof(float) * batch * layer->net.full_connect.count, context->device[device_id].model_stream[j % 2]);
								cublasSaxpy(context->device[device_id].model_cublas[j % 2], batch * layer->net.full_connect.count, &one, GPU(convnet)->device[device_id].scratch + (j % 2) * batch * layer->net.full_connect.count, 1, GPU(convnet)->device[device_id].backwards[i + 1] + (device_count * j + device_id) * batch * layer->net.full_connect.count, 1);
							}
						}
					if (context->device[device_id].dor[i])
						_cwc_kern_mute_neuron
						<<<layer->net.full_connect.count, batch, 0, context->device[device_id].model_stream[j % 2]>>>
						(GPU(convnet)->device[device_id].backwards[i + 1] + (device_count * j + device_id) * batch * layer->net.full_connect.count, context->device[device_id].dor[i]);
					cwc_convnet_full_connect_backward_propagate(layer, batch,
							GPU(convnet)->device[device_id].backwards[i + 1] + (device_count * j + device_id) * batch * layer->net.full_connect.count,
							GPU(convnet)->device[device_id].forwards[i] + (device_count * j + device_id) * batch * layer->net.full_connect.count,
							GPU(convnet)->device[device_id].forwards[i - 1] + j * batch * layer->input.node.count,
							GPU(convnet)->device[device_id].backwards[i] + j * batch * layer->input.node.count,
							GPU(convnet)->device[device_id].unit, configuration->w + (j % 2) * layer->wnum, configuration->bias + (j % 2) * layer->net.full_connect.count, context->device[device_id].model_stream[j % 2], context->device[device_id].model_cublas[j % 2]);
				}
			}
			for (device_id = 0; device_id < device_count; device_id++)
			{
				cudaSetDevice(device_id);
				cudaEventRecord(context->device[device_id].model_joint[0], context->device[device_id].model_stream[0]);
				cudaEventRecord(context->device[device_id].model_joint[1], context->device[device_id].model_stream[1]);
			}
		}
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[device_id].model_joint[0], 0);
			cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[device_id].model_joint[1], 0);
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + count;
			for (other_device_id = 0; other_device_id < device_count; other_device_id++)
				if (other_device_id != device_id)
				{
					// waiting other device to finish, and then copy
					cudaStreamWaitEvent(context->device[device_id].data_stream, context->device[other_device_id].model_joint[device_id % 2], 0);
					// make sure we enabled peer to peer direct access, doing sum direct from the other device is faster than copy over and sum from my profiling
					if (GPU(convnet)->device[device_id].can_access_peer[other_device_id])
						cublasSaxpy(context->device[device_id].data_cublas, batch * layer->input.node.count, &one, GPU(convnet)->device[other_device_id].backwards[count] + device_id * batch * layer->input.node.count, 1, GPU(convnet)->device[device_id].backwards[count] + device_id * batch * layer->input.node.count, 1);
					else {
						cudaMemcpyPeerAsync(GPU(convnet)->device[device_id].scratch, device_id, GPU(convnet)->device[other_device_id].backwards[count] + device_id * batch * layer->input.node.count, other_device_id, sizeof(float) * batch * layer->input.node.count, context->device[device_id].data_stream);
						cublasSaxpy(context->device[device_id].data_cublas, batch * layer->input.node.count, &one, GPU(convnet)->device[device_id].scratch, 1, GPU(convnet)->device[device_id].backwards[count] + device_id * batch * layer->input.node.count, 1);
					}
				}
		}
		for (i = count - 1; i >= 0; i--)
		{
			assert(i < convnet->count - 1);
			for (device_id = 0; device_id < device_count; device_id++)
			{
				cudaSetDevice(device_id);
				ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
				ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
				// to make the code in _cwc_convnet_encode_impl easier, the actual data in this layer (count - 1) is layout at device_id * batch * count offset
				float* a = (i == count - 1) ? GPU(convnet)->device[device_id].backwards[i + 1] + device_id * batch * GPU(convnet)->device[device_id].layers[count].input.node.count : GPU(convnet)->device[device_id].backwards[i + 1];
				float* n = (i == count - 1) ? GPU(convnet)->device[device_id].forwards[i] + device_id * batch * GPU(convnet)->device[device_id].layers[count].input.node.count : GPU(convnet)->device[device_id].forwards[i];
				float* m = (i > 0) ? GPU(convnet)->device[device_id].forwards[i - 1] : context->device[device_id].input;
				_cwc_convnet_layer_backward_propagate(layer, device_id, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, a, n, m, GPU(convnet)->device[device_id].denoms[i], GPU(convnet)->device[device_id].backwards[i], GPU(convnet)->device[device_id].unit, GPU(convnet)->device[device_id].scratch, configuration, context);
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
	for (device_id = 0; device_id < GPU(z->convnet)->device_count; device_id++)
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
					ASSERT_NO_CUDA_ERROR();
					break;
				case CCV_CONVNET_FULL_CONNECT:
					_cwc_convnet_reorder_full_connect_weights_onto_device(host_layer->w + device_id * layer->wnum, layer->w, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
					cudaMemcpy(layer->bias, host_layer->bias + device_id * layer->net.full_connect.count, sizeof(float) * layer->net.full_connect.count, cudaMemcpyHostToDevice);
					memcpy(z_layer->w, host_layer->w, sizeof(float) * (z_layer->wnum + z_layer->net.full_connect.count));
					ASSERT_NO_CUDA_ERROR();
					break;
			}
		}
	}
	if (CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
		_cwc_convnet_collect_disk_stats(z->convnet);
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
				ccv_convnet_layer_t* host_layer = z->convnet->layers + sqlite3_column_int(momentum_data_stmt, 0);
				int wnum = sqlite3_column_bytes(momentum_data_stmt, 1) / sizeof(float);
				int bnum = sqlite3_column_bytes(momentum_data_stmt, 2) / sizeof(float);
				if (wnum != host_layer->wnum)
					continue;
				const void* w = sqlite3_column_blob(momentum_data_stmt, 1);
				const void* bias = sqlite3_column_blob(momentum_data_stmt, 2);
				for (device_id = 0; device_id < GPU(z->convnet)->device_count; device_id++)
				{
					ccv_convnet_layer_t* layer = GPU(z->convnet)->device[device_id].layers + sqlite3_column_int(momentum_data_stmt, 0);
					ccv_convnet_layer_t* momentum = GPU(z->convnet)->device[device_id].momentums + sqlite3_column_int(momentum_data_stmt, 0);
					switch (layer->type)
					{
						case CCV_CONVNET_CONVOLUTIONAL:
							if (bnum != layer->net.convolutional.count)
								continue;
							if (device_id == 0)
							{
								// only copy for device 0's momentum
								_cwc_convnet_reorder_convolutional_weights_onto_device((float*)w, momentum->w, layer->wnum, layer->net.convolutional.count, layer->net.convolutional.channels, layer->input.matrix.partition);
								cudaMemcpy(momentum->bias, bias, sizeof(float) * layer->net.convolutional.count, cudaMemcpyHostToDevice);
							}
							break;
						case CCV_CONVNET_FULL_CONNECT:
							if (bnum != layer->net.full_connect.count)
								continue;
							_cwc_convnet_reorder_full_connect_weights_onto_device((float*)w + device_id * layer->wnum, momentum->w, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
							cudaMemcpy(momentum->bias, (float*)bias + device_id * layer->net.full_connect.count, sizeof(float) * layer->net.full_connect.count, cudaMemcpyHostToDevice);
							break;
					}
				}
			}
			sqlite3_finalize(momentum_data_stmt);
		}
		sqlite3_stmt* conv_vary_params_stmt = 0;
		const char conv_vary_params_qs[] =
			"SELECT layer, fx, fy, fz, bcx, bcy, bcz, bgx, bgy, bgz FROM conv_vary_params;";
		if (SQLITE_OK == sqlite3_prepare_v2(db, conv_vary_params_qs, sizeof(conv_vary_params_qs), &conv_vary_params_stmt, 0))
		{
			while(sqlite3_step(conv_vary_params_stmt) == SQLITE_ROW)
			{
				for (device_id = 0; device_id < GPU(z->convnet)->device_count; device_id++)
				{
					ccv_convnet_layer_t* layer = GPU(z->convnet)->device[device_id].layers + sqlite3_column_int(conv_vary_params_stmt, 0);
					assert(layer->type == CCV_CONVNET_CONVOLUTIONAL);
					EXTRA(layer)->vary.convolutional.forward.x = sqlite3_column_int(conv_vary_params_stmt, 1);
					EXTRA(layer)->vary.convolutional.forward.y = sqlite3_column_int(conv_vary_params_stmt, 2);
					EXTRA(layer)->vary.convolutional.forward.z = sqlite3_column_int(conv_vary_params_stmt, 3);
					EXTRA(layer)->vary.convolutional.backward.coefficient.x = sqlite3_column_int(conv_vary_params_stmt, 4);
					EXTRA(layer)->vary.convolutional.backward.coefficient.y = sqlite3_column_int(conv_vary_params_stmt, 5);
					EXTRA(layer)->vary.convolutional.backward.coefficient.z = sqlite3_column_int(conv_vary_params_stmt, 6);
					EXTRA(layer)->vary.convolutional.backward.gradient.x = sqlite3_column_int(conv_vary_params_stmt, 7);
					EXTRA(layer)->vary.convolutional.backward.gradient.y = sqlite3_column_int(conv_vary_params_stmt, 8);
					EXTRA(layer)->vary.convolutional.backward.gradient.z = sqlite3_column_int(conv_vary_params_stmt, 9);
				}
			}
			sqlite3_finalize(conv_vary_params_stmt);
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
				ASSERT_NO_CUDA_ERROR();
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_cwc_convnet_reorder_full_connect_weights_onto_host(layer->w, host_layer->w + device_id * layer->wnum, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
				cudaMemcpy(host_layer->bias + device_id * layer->net.full_connect.count, layer->bias, sizeof(float) * layer->net.full_connect.count, cudaMemcpyDeviceToHost);
				ASSERT_NO_CUDA_ERROR();
				break;
		}
	}
}

static void _cwc_convnet_supervised_train_function_state_write(cwc_convnet_supervised_train_function_state_t* z, const char* filename)
{
	int device_id;
	for (device_id = 0; device_id < GPU(z->convnet)->device_count; device_id++)
		_cwc_convnet_host_synchronize(z->convnet, device_id);
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
			"(layer INTEGER PRIMARY KEY ASC, weight BLOB, bias BLOB);"
			"CREATE TABLE IF NOT EXISTS conv_vary_params "
			"(layer INTEGER PRIMARY KEY ASC, fx INTEGER, fy INTEGER, fz INTEGER, bcx INTEGER, bcy INTEGER, bcz INTEGER, bgx INTEGER, bgy INTEGER, bgz INTEGER);";
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
			ccv_convnet_layer_t* host_layer = z->convnet->layers + i;
			// insert momentum data
			if (host_layer->type == CCV_CONVNET_CONVOLUTIONAL || host_layer->type == CCV_CONVNET_FULL_CONNECT)
			{
				sqlite3_bind_int(momentum_data_insert_stmt, 1, i);
				float* w = (float*)ccmalloc(sizeof(float) * (host_layer->wnum + (host_layer->type == CCV_CONVNET_CONVOLUTIONAL ? host_layer->net.convolutional.count : host_layer->net.full_connect.count)));
				float* bias = w + host_layer->wnum;
				for (device_id = 0; device_id < GPU(z->convnet)->device_count; device_id++)
				{
					ccv_convnet_layer_t* layer = GPU(z->convnet)->device[device_id].layers + i;
					ccv_convnet_layer_t* momentum = GPU(z->convnet)->device[device_id].momentums + i;
					switch (layer->type)
					{
						case CCV_CONVNET_CONVOLUTIONAL:
							if (device_id == 0)
							{
								// only save from device 0
								_cwc_convnet_reorder_convolutional_weights_onto_host(momentum->w, w, layer->wnum, layer->net.convolutional.count, layer->net.convolutional.channels, layer->input.matrix.partition);
								cudaMemcpy(bias, momentum->bias, sizeof(float) * layer->net.convolutional.count, cudaMemcpyDeviceToHost);
								ASSERT_NO_CUDA_ERROR();
							}
							break;
						case CCV_CONVNET_FULL_CONNECT:
							_cwc_convnet_reorder_full_connect_weights_onto_host(momentum->w, w + device_id * layer->wnum, layer->wnum, layer->input.matrix.rows * layer->input.matrix.cols, layer->input.matrix.channels);
							cudaMemcpy(bias + device_id * layer->net.full_connect.count, momentum->bias, sizeof(float) * layer->net.full_connect.count, cudaMemcpyDeviceToHost);
							ASSERT_NO_CUDA_ERROR();
							break;
					}
				}
				sqlite3_bind_blob(momentum_data_insert_stmt, 2, w, sizeof(float) * host_layer->wnum, SQLITE_STATIC);
				sqlite3_bind_blob(momentum_data_insert_stmt, 3, bias, sizeof(float) * (host_layer->type == CCV_CONVNET_CONVOLUTIONAL ? host_layer->net.convolutional.count : host_layer->net.full_connect.count), SQLITE_STATIC);
				assert(SQLITE_DONE == sqlite3_step(momentum_data_insert_stmt));
				sqlite3_reset(momentum_data_insert_stmt);
				sqlite3_clear_bindings(momentum_data_insert_stmt);
				ccfree(w);
			}
		}
		sqlite3_finalize(momentum_data_insert_stmt);
		const char conv_vary_params_insert_qs[] =
			"REPLACE INTO conv_vary_params "
			"(layer, fx, fy, fz, bcx, bcy, bcz, bgx, bgy, bgz) VALUES ($layer, $fx, $fy, $fz, $bcx, $bcy, $bcz, $bgx, $bgy, $bgz);";
		sqlite3_stmt* conv_vary_params_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, conv_vary_params_insert_qs, sizeof(conv_vary_params_insert_qs), &conv_vary_params_insert_stmt, 0));
		for (i = 0; i < z->convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = GPU(z->convnet)->device[0].layers + i;
			// insert momentum data
			if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
			{
				sqlite3_bind_int(conv_vary_params_insert_stmt, 1, i);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 2, EXTRA(layer)->vary.convolutional.forward.x);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 3, EXTRA(layer)->vary.convolutional.forward.y);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 4, EXTRA(layer)->vary.convolutional.forward.z);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 5, EXTRA(layer)->vary.convolutional.backward.coefficient.x);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 6, EXTRA(layer)->vary.convolutional.backward.coefficient.y);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 7, EXTRA(layer)->vary.convolutional.backward.coefficient.z);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 8, EXTRA(layer)->vary.convolutional.backward.gradient.x);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 9, EXTRA(layer)->vary.convolutional.backward.gradient.y);
				sqlite3_bind_int(conv_vary_params_insert_stmt, 10, EXTRA(layer)->vary.convolutional.backward.gradient.z);
				assert(SQLITE_DONE == sqlite3_step(conv_vary_params_insert_stmt));
				sqlite3_reset(conv_vary_params_insert_stmt);
				sqlite3_clear_bindings(conv_vary_params_insert_stmt);
			}
		}
		sqlite3_finalize(conv_vary_params_insert_stmt);
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
		ccv_convnet_make_output(layer, rows, cols, &out_rows, &out_cols, &out_partition);
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

// to print out verbose for debugging

static void CUDART_CB _cwc_stream_callback_verbose(cudaStream_t stream,  cudaError_t status, void* data)
{
	PRINT(CCV_CLI_VERBOSE, " -- device with stream %p %s with status %s\n", stream, (char*)data, cudaGetErrorString(status));
}

#define PRINT_VERBOSE_ON_STREAM(stream, text) \
	do { \
		if (CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE)) \
			cudaStreamAddCallback(stream, _cwc_stream_callback_verbose, (void*)text, 0); \
	} while (0)

void cwc_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, const char* filename, ccv_convnet_train_param_t params)
{
#ifdef HAVE_GSL
	assert(params.mini_batch % BATCH_PER_BLOCK == 0);
	int device_id, device_count = 0;
#define PRINT_VERBOSE_ON_CONTEXT(context, text) /* handy method to print something on main context */ \
	do { \
		if (CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE)) \
			for (device_id = 0; device_id < params.device_count; device_id++) \
			{ \
				cudaSetDevice(device_id); \
				PRINT_VERBOSE_ON_STREAM(context->device[device_id].data_stream, text); \
			} \
	} while (0)
	cudaGetDeviceCount(&device_count);
	if (params.device_count > device_count)
		params.device_count = device_count;
	assert(device_count > 0);
	_cwc_convnet_alloc_reserved_both(convnet, params.mini_batch, params.device_count, params.layer_params);
	// enable peer access if requested, it works if not
	if (params.peer_access)
		_cwc_convnet_enable_peer_access(convnet, params.device_count);
	int i, j, k;
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	int multi_batch = params.mini_batch * params.device_count;
	int aligned_padding = categorizeds->rnum % multi_batch;
	int aligned_rnum = categorizeds->rnum - aligned_padding;
	int aligned_batches = categorizeds->rnum / multi_batch;
	int* idx = (int*)ccmalloc(sizeof(int) * (categorizeds->rnum + aligned_padding));
	for (i = 0; i < categorizeds->rnum; i++)
		idx[i] = i;
	params.iterations = ccv_min(params.iterations, aligned_batches);
	gsl_ran_shuffle(rng, idx, categorizeds->rnum, sizeof(int));
	// the last layer has to be full connect, thus we can use it as softmax layer
	assert(convnet->layers[convnet->count - 1].type == CCV_CONVNET_FULL_CONNECT);
	int category_count = convnet->layers[convnet->count - 1].net.full_connect.count;
	struct {
		int* host[MAX_DEVICE_SUPPORT];
		int* device[MAX_DEVICE_SUPPORT];
	} test_returns[2];
	for (device_id = 0; device_id < params.device_count; device_id++)
	{
		test_returns[0].host[device_id] = test_returns[1].host[device_id] = 0;
		test_returns[0].device[device_id] = test_returns[1].device[device_id] = 0;
	}
	cudaEvent_t start[MAX_DEVICE_SUPPORT], stop[MAX_DEVICE_SUPPORT];
	for (device_id = 0; device_id < params.device_count; device_id++)
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
	}
	cwc_convnet_supervised_train_function_state_t z = {0};
	z.idx = idx;
	z.inum = categorizeds->rnum;
	z.convnet = convnet;
	int miss, sgd_count, stats_count;
	float max_elapsed_time;
	float elapsed_time[MAX_DEVICE_SUPPORT] = {0};
	ccv_function_state_begin(_cwc_convnet_supervised_train_function_state_read, z, filename);
	cwc_convnet_mean_formation(categorizeds, z.convnet->input, z.convnet->channels, params.symmetric, &z.convnet->mean_activity);
	ccv_function_state_resume(_cwc_convnet_supervised_train_function_state_write, z, filename);
	if (z.convnet->channels == 3 && params.color_gain > 0) // do this if we want color gain type of data augmentation, and it is RGB color
		cwc_convnet_channel_eigen(categorizeds, z.convnet->mean_activity, z.convnet->input, z.convnet->channels, &z.eigenvectors, &z.eigenvalues);
	ccv_function_state_resume(_cwc_convnet_supervised_train_function_state_write, z, filename);
	for (z.t = 0; z.t < params.max_epoch; z.t++)
	{
		for (z.i = 0; z.i < aligned_batches; z.i += params.iterations)
		{
			sgd_count = stats_count = 0;
			// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in updating
			// undo the mean network for further training
			for (device_id = 0; device_id < params.device_count; device_id++)
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
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					cudaSetDevice(device_id);
					cwc_convnet_batch_formation(rng, categorizeds, z.convnet->mean_activity, z.eigenvectors, z.eigenvalues, params.image_manipulation, params.color_gain, z.idx, z.convnet->input, params.input.min_dim, params.input.max_dim, z.convnet->rows, z.convnet->cols, z.convnet->channels, category_count, params.symmetric, params.mini_batch, i * multi_batch + device_id * params.mini_batch, params.mini_batch, context->host[device_id].input, context->host[device_id].c);
					cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * z.convnet->rows * z.convnet->cols * z.convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
					cudaMemcpyAsync(context->device[device_id].c, context->host[device_id].c, sizeof(int) * params.mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
					_cwc_convnet_dor_formation(z.convnet, device_id, params.mini_batch, rng, params.layer_params, context);
					ASSERT_NO_CUDA_ERROR();
					// sync with the other stream core so that we can compute on the single true layer parameters
					if (i > z.i)
						cudaEventRecord(GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].stop_timing, GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].data_stream);
				}
				PRINT(CCV_CLI_VERBOSE, " -- host prepared a batch on CPU\n");
				PRINT_VERBOSE_ON_CONTEXT(context, "received batch from host");
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					PRINT(CCV_CLI_VERBOSE, " -- host to sync with data stream %p [%d]\n", GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].data_stream, device_id);
					cudaSetDevice(device_id);
					cudaEventSynchronize(GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].stop_timing);
				}
				PRINT(CCV_CLI_VERBOSE, " -- host synced with previous data streams\n");
				ASSERT_NO_CUDA_ERROR();
				if (i > z.i) // we have another result, pull these
				{
					for (device_id = 0; device_id < params.device_count; device_id++)
					{
						cudaSetDevice(device_id);
						int* c = GPU(z.convnet)->contexts[(i + 1) % 2].host[device_id].c;
						for (k = 0; k < params.mini_batch; k++)
							if (c[k] != test_returns[(i + 1) % 2].host[device_id][k])
								++miss;
						cudaEventElapsedTime(&elapsed_time[device_id], GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].start_timing, GPU(z.convnet)->contexts[(i + 1) % 2].device[device_id].stop_timing);
					}
					max_elapsed_time = elapsed_time[0];
					for (device_id = 1; device_id < params.device_count; device_id++)
						max_elapsed_time = ccv_max(max_elapsed_time, elapsed_time[device_id]);
					FLUSH(CCV_CLI_INFO, " - at epoch %03d / %d => stochastic gradient descent with miss rate %.2f%% at %d / %d (%.3f sec)", z.t + 1, params.max_epoch, miss * 100.0f /((i - z.i) * multi_batch), i + 1, aligned_batches, max_elapsed_time / 1000);
					PRINT(CCV_CLI_VERBOSE, "\n"); // create a new line for verbose
				}
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					cudaSetDevice(device_id);
					cudaEventRecord(context->device[device_id].start_timing, context->device[device_id].data_stream);
				}
				PRINT(CCV_CLI_VERBOSE, " -- host enqueued forward propagate\n");
				_cwc_convnet_encode_impl(z.convnet, params.device_count, params.mini_batch, 1, context);
				PRINT_VERBOSE_ON_CONTEXT(context, "dequeued forward propagate");
				ASSERT_NO_CUDA_ERROR();
				// compute miss rate on training data
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					cudaSetDevice(device_id);
					_cwc_convnet_tests_return(params.mini_batch, category_count, GPU(z.convnet)->device[device_id].forwards[z.convnet->count - 1] + device_id * params.mini_batch * category_count, test_returns[i % 2].device[device_id], context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
					cudaMemcpyAsync(test_returns[i % 2].host[device_id], test_returns[i % 2].device[device_id], sizeof(int) * params.mini_batch, cudaMemcpyDeviceToHost, context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
					// do the logistic loss
					_cwc_convnet_softmax_with_logistic_loss(params.mini_batch, category_count, GPU(z.convnet)->device[device_id].forwards[z.convnet->count - 1] + device_id * params.mini_batch * category_count, context->device[device_id].c, context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
				}
				// do backward propagate
				PRINT(CCV_CLI_VERBOSE, " -- host enqueued backward propagate\n");
				_cwc_convnet_backward_propagate_error(z.convnet, params.device_count, params.mini_batch, context);
				PRINT_VERBOSE_ON_CONTEXT(context, "dequeued backward propagate");
				ASSERT_NO_CUDA_ERROR();
				if (++sgd_count == params.sgd_frequency)
				{
					if ((++stats_count % 25) /* only print this out every 25 sgd's */ == 1 && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
						_cwc_convnet_collect_runtime_stats(z.convnet, context);
					// no need to reduce the data parallelism updates to device 0 if we don't have multiple devices
					if (params.device_count > 1)
					{
						PRINT(CCV_CLI_VERBOSE, " -- host enqueued reduce data parallelism\n");
						_cwc_convnet_reduce_data_parallelism(z.convnet, params.device_count, context);
						PRINT_VERBOSE_ON_CONTEXT(context, "dequeued reduce data parallelism");
					}
					PRINT(CCV_CLI_VERBOSE, " -- host enqueued stochastic gradient descent\n");
					_cwc_convnet_net_sgd(z.convnet, params.device_count, multi_batch * sgd_count, params.layer_params, context);
					PRINT_VERBOSE_ON_CONTEXT(context, "dequeued stochastic gradient descent");
					// no need to fan out the data parallelism updates to all devices if we don't have multiple devices
					if (params.device_count > 1)
					{
						PRINT(CCV_CLI_VERBOSE, " -- host enqueued broadcast data parallelism\n");
						_cwc_convnet_broadcast_data_parallelism(z.convnet, params.device_count, context);
						PRINT_VERBOSE_ON_CONTEXT(context, "dequeued broadcast data parallelism");
					}
					sgd_count = 0; // reset the counter
				}
				ASSERT_NO_CUDA_ERROR();
			}
			for (device_id = 0; device_id < params.device_count; device_id++)
			{
				cudaSetDevice(device_id);
				cudaDeviceSynchronize(); // synchronize at this point
				// using context-1's cublas handle because we will wait this handle to finish when the copy to context-0 is required in testing
				_cwc_convnet_dor_mean_net(z.convnet, device_id, params.layer_params, GPU(z.convnet)->contexts[1].device[device_id].data_cublas);
			}
			// run tests
			miss = 0;
			for (i = j = 0; i < tests->rnum; i += multi_batch, j++)
			{
				cwc_convnet_context_t* context = GPU(z.convnet)->contexts + (j % 2);
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					cudaSetDevice(device_id);
					if (i + device_id * params.mini_batch < tests->rnum)
						cwc_convnet_batch_formation(0, tests, z.convnet->mean_activity, 0, 0, 0, 0, 0, z.convnet->input, params.input.min_dim, params.input.max_dim, z.convnet->rows, z.convnet->cols, z.convnet->channels, category_count, params.symmetric, params.mini_batch, i + device_id * params.mini_batch, ccv_min(params.mini_batch, tests->rnum - i - device_id * params.mini_batch), context->host[device_id].input, 0);
					cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * z.convnet->rows * z.convnet->cols * z.convnet->channels * params.mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
					// sync with the other stream core so that we can compute on the single true layer parameters
					cudaEventRecord(GPU(z.convnet)->contexts[(j + 1) % 2].device[device_id].stop_timing, GPU(z.convnet)->contexts[(j + 1) % 2].device[device_id].data_stream);
				}
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					cudaSetDevice(device_id);
					cudaEventSynchronize(GPU(z.convnet)->contexts[(j + 1) % 2].device[device_id].stop_timing);
				}
				ASSERT_NO_CUDA_ERROR();
				if (j > 0) // we have another result, pull these
				{
					for (device_id = 0; device_id < params.device_count; device_id++)
					{
						cudaSetDevice(device_id);
						for (k = 0; k < params.mini_batch; k++)
						{
							ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, k + i - multi_batch + device_id * params.mini_batch);
							if (test->c != test_returns[(j + 1) % 2].host[device_id][k])
								++miss;
						}
						cudaEventElapsedTime(&elapsed_time[device_id], GPU(z.convnet)->contexts[(j + 1) % 2].device[device_id].start_timing, GPU(z.convnet)->contexts[(j + 1) % 2].device[device_id].stop_timing);
					}
					max_elapsed_time = elapsed_time[0];
					for (device_id = 1; device_id < params.device_count; device_id++)
						max_elapsed_time = ccv_max(max_elapsed_time, elapsed_time[device_id]);
					FLUSH(CCV_CLI_INFO, " - at epoch %03d / %d => with miss rate %.2f%% at %d / %d (%.3f sec)", z.t + 1, params.max_epoch, miss * 100.0f / i, j + 1, (tests->rnum + multi_batch - 1) / multi_batch, max_elapsed_time / 1000);
				}
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					cudaSetDevice(device_id);
					cudaEventRecord(context->device[device_id].start_timing, context->device[device_id].data_stream);
				}
				_cwc_convnet_encode_impl(z.convnet, params.device_count, params.mini_batch, 0, context);
				ASSERT_NO_CUDA_ERROR();
				for (device_id = 0; device_id < params.device_count; device_id++)
				{
					cudaSetDevice(device_id);
					_cwc_convnet_tests_return(params.mini_batch, category_count, GPU(z.convnet)->device[device_id].forwards[z.convnet->count - 1] + device_id * params.mini_batch * category_count, test_returns[j % 2].device[device_id], context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
					cudaMemcpyAsync(test_returns[j % 2].host[device_id], test_returns[j % 2].device[device_id], sizeof(int) * params.mini_batch, cudaMemcpyDeviceToHost, context->device[device_id].data_stream);
					ASSERT_NO_CUDA_ERROR();
				}
			}
			for (device_id = 0; device_id < params.device_count; device_id++)
			{
				cudaSetDevice(device_id);
				cudaDeviceSynchronize(); // synchronize at this point
			}
			for (i = 0; i <= (tests->rnum - 1) % multi_batch; i++)
			{
				ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, i + (tests->rnum - 1) / multi_batch * multi_batch);
				device_id = i / params.mini_batch;
				if (test->c != test_returns[(j + 1) % 2].host[device_id][i % params.mini_batch])
					++miss;
			}
			for (device_id = 0; device_id < params.device_count; device_id++)
			{
				cudaSetDevice(device_id);
				cudaEventRecord(stop[device_id], 0);
				cudaEventSynchronize(stop[device_id]);
				elapsed_time[device_id] = 0;
				cudaEventElapsedTime(&elapsed_time[device_id], start[device_id], stop[device_id]);
			}
			max_elapsed_time = elapsed_time[0];
			for (device_id = 1; device_id < params.device_count; device_id++)
				max_elapsed_time = ccv_max(max_elapsed_time, elapsed_time[device_id]);
			FLUSH(CCV_CLI_INFO, " - at epoch %03d / %d (%03d - %d) => with miss rate %.2f%% (%.3f sec)\n", z.t + 1, params.max_epoch, z.i + 1, ccv_min(z.i + params.iterations, aligned_batches), miss * 100.0f / tests->rnum, max_elapsed_time / 1000);
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
	for (device_id = 0; device_id < params.device_count; device_id++)
	{
		cudaSetDevice(device_id);
		cudaEventDestroy(start[device_id]);
		cudaEventDestroy(stop[device_id]);
		for (i = 0; i < 2; i++)
		{
			cudaFree(test_returns[i].device[device_id]);
			cudaFreeHost(test_returns[i].host[device_id]);
		}
	}
	ccfree(z.idx);
	gsl_rng_free(rng);
#undef PRINT_VERBOSE_ON_CONTEXT
#else
	PRINT(CCV_CLI_ERROR, "cwc_convnet_supervised_train requires GSL library support");
	exit(-1);
#endif
}

void cwc_convnet_compact(ccv_convnet_t* convnet)
{
	if (GPU(convnet))
	{
		int device_count = GPU(convnet)->device_count;
		int i, j, device_id;
		for (device_id = 0; device_id < device_count; device_id++)
		{
			cudaSetDevice(device_id);
			if (GPU(convnet)->device[device_id].scratch)
				cudaFree(GPU(convnet)->device[device_id].scratch);
			cudaFree(GPU(convnet)->device[device_id].unit);
		}
		for (i = 0; i < 2; i++)
		{
			cwc_convnet_context_t* context = GPU(convnet)->contexts + i;
			for (device_id = 0; device_id < device_count; device_id++)
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
				if (context->device[device_id].start_timing)
					cudaEventDestroy(context->device[device_id].start_timing);
				if (context->device[device_id].stop_timing)
					cudaEventDestroy(context->device[device_id].stop_timing);
			}
		}
		for (i = 0; i < convnet->count; i++)
			for (device_id = 0; device_id < device_count; device_id++)
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
			for (device_id = 0; device_id < device_count; device_id++)
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
