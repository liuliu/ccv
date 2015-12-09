#undef USE_DISPATCH // nvcc doesn't support libdispatch
extern "C" {
#include "ccv.h"
}
#include <ctype.h>
#include <cudnn.h>
#include "../lib/ccv_convnet.c"

typedef struct {
	cudnnTensorDescriptor_t tensor;
	cudnnFilterDescriptor_t filter;
	cudnnConvolutionDescriptor_t convolution;
	cudnnConvolutionFwdAlgo_t forwards;
} cwc_cudnn_layer_t;

extern "C" void cwc_cudnn_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params)
{
	int batch = params.mini_batch;
	int i, j;
	const int device_id = 0;
	_cwc_convnet_alloc_reserved_both(convnet, batch, 1, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	for (i = 0; i < convnet->rows * convnet->cols * convnet->channels; i++)
		convnet->mean_activity->data.f32[i] = 128;
	cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, 0, ccv_size(225, 225), 225, 225, convnet->rows, convnet->cols, convnet->channels, 1000, 0, batch, 0, batch, context->host[device_id].input, context->host[device_id].c);
	cudaMemcpy(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch, cudaMemcpyHostToDevice);

	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);

	// first convolutional layer forward propagate
	ccv_convnet_layer_t* first_gpu_layer = GPU(convnet)->device[device_id].layers;
	// these are the setups for TITAN, thus, skip the benching phase
	EXTRA(first_gpu_layer)->vary.convolutional.forward.x = 4;
	EXTRA(first_gpu_layer)->vary.convolutional.forward.y = 8;
	EXTRA(first_gpu_layer)->vary.convolutional.forward.z = 32;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_forward_propagate(first_gpu_layer, first_gpu_layer->input.matrix.rows, first_gpu_layer->input.matrix.cols, batch, context->device[device_id].input, GPU(convnet)->device[device_id].forwards[0], context->device[device_id].data_stream);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	float elapsed_time = 0;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	printf("%d %d %d, elapsed time for first convolutional layer fprop: %f milliseconds\n", EXTRA(first_gpu_layer)->vary.convolutional.forward.x, EXTRA(first_gpu_layer)->vary.convolutional.forward.y, EXTRA(first_gpu_layer)->vary.convolutional.forward.z, elapsed_time);
	int first_out_rows, first_out_cols, first_out_partition, first_out_channels = first_gpu_layer->net.convolutional.count;
	ccv_convnet_make_output(first_gpu_layer, first_gpu_layer->input.matrix.rows, first_gpu_layer->input.matrix.cols, &first_out_rows, &first_out_cols, &first_out_partition);
	float* first_out = 0;
	cudaMallocHost(&first_out, sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch);
	cudaMemcpy(first_out, GPU(convnet)->device[device_id].forwards[0], sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate first convolutional layer on GPU\n");

	int x, y, k, c;
	for (i = 0; i < batch; i++)
	{
		printf("doing batch %d of %d\n", i + 1, batch);
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);
		for (x = 0; x < categorized->matrix->rows * categorized->matrix->cols * CCV_GET_CHANNEL(categorized->matrix->type); x++)
			categorized->matrix->data.f32[x] = categorized->matrix->data.f32[x] - 128;

		// first convolutional layer forward propagate
		ccv_convnet_layer_t* first_cpu_layer = convnet->layers;
		_ccv_convnet_convolutional_forward_propagate(first_cpu_layer, categorized->matrix, convnet->acts);
		ccv_dense_matrix_t* a = convnet->acts[0];
		for (y = 0; y < first_out_rows; y++)
			for (x = 0; x < first_out_cols; x++)
				for (k = 0; k < first_out_channels; k++)
				{
					float p = first_out[k * first_out_rows * first_out_cols * batch + (y * first_out_cols + x) * batch + i];
					float q = a->data.f32[y * first_out_cols * first_out_channels + x * first_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv fprop 1: %d %d %d %d: |%f - %f| = %f\n", i, x, y, k, p, q, delta);
				}
	}
}
