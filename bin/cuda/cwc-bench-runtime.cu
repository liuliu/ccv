#undef USE_DISPATCH // nvcc doesn't support libdispatch
extern "C" {
#include "ccv.h"
}
#include <ctype.h>
#define CASE_TESTS // so that we don't include public available methods
#include "../lib/cuda/cwc_convnet.cu"
#include "../lib/ccv_convnet.c"

extern "C" void cwc_bench_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params)
{
	int batch = params.mini_batch;
	int i;
	const int device_id = 0;
	_cwc_convnet_alloc_reserved_both(convnet, batch, 0, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	for (i = 0; i < convnet->rows * convnet->cols * convnet->channels; i++)
		convnet->mean_activity->data.f32[i] = 128;
	_cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, ccv_size(225, 225), convnet->rows, convnet->cols, convnet->channels, 1000, 0, batch, 0, batch, context->host[device_id].input, context->host[device_id].c);
	cudaMemcpy(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch, cudaMemcpyHostToDevice);

	cudaEvent_t overallStart;
	cudaEvent_t overallStop;
	cudaEventCreate(&overallStart);
	cudaEventCreate(&overallStop);
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;
	VARY(GPU(convnet)->device[device_id].layers + 0)->convolutional.forward.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 0)->convolutional.forward.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 0)->convolutional.forward.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.forward.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.forward.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.forward.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.forward.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.forward.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.forward.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.forward.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.forward.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.forward.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.forward.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.forward.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.forward.z = 32;
	cudaEventRecord(overallStart, context->device[device_id].data_stream);
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
		cudaEventRecord(start, context->device[device_id].data_stream);
		_cwc_convnet_layer_forward_propagate(layer, device_id, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, 0, i == 0 ? context->device[device_id].input : GPU(convnet)->device[device_id].forwards[i - 1], GPU(convnet)->device[device_id].forwards[i], GPU(convnet)->device[device_id].denoms[i], GPU(convnet)->device[device_id].unit, context);
		cudaEventRecord(stop, context->device[device_id].data_stream);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time, start, stop);
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
			printf("%d %d %d, elapsed time for layer %d fprop: %f milliseconds\n", VARY(layer)->convolutional.forward.x, VARY(layer)->convolutional.forward.y, VARY(layer)->convolutional.forward.z, i + 1, elapsed_time);
		else
			printf("elapsed time for layer %d fprop: %f milliseconds\n", i + 1, elapsed_time);
	}
	cudaEventRecord(overallStop, context->device[device_id].data_stream);
	cudaEventSynchronize(overallStop);
	cudaEventElapsedTime(&elapsed_time, overallStart, overallStop);
	printf("forward pass %f milliseconds\n", elapsed_time);

	VARY(GPU(convnet)->device[device_id].layers + 0)->convolutional.backward.coefficient.x = 1;
	VARY(GPU(convnet)->device[device_id].layers + 0)->convolutional.backward.coefficient.y = 3;
	VARY(GPU(convnet)->device[device_id].layers + 0)->convolutional.backward.coefficient.z = 1;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.backward.coefficient.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.backward.coefficient.y = 4;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.backward.coefficient.z = 16;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.backward.gradient.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.backward.gradient.y = 6;
	VARY(GPU(convnet)->device[device_id].layers + 3)->convolutional.backward.gradient.z = 24;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.backward.coefficient.x = 8;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.backward.coefficient.y = 3;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.backward.coefficient.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.backward.gradient.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.backward.gradient.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 6)->convolutional.backward.gradient.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.backward.coefficient.x = 8;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.backward.coefficient.y = 3;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.backward.coefficient.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.backward.gradient.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.backward.gradient.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 7)->convolutional.backward.gradient.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.backward.coefficient.x = 8;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.backward.coefficient.y = 4;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.backward.coefficient.z = 32;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.backward.gradient.x = 4;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.backward.gradient.y = 8;
	VARY(GPU(convnet)->device[device_id].layers + 8)->convolutional.backward.gradient.z = 32;
	float* a = 0;
	cudaMalloc(&a, sizeof(float) * 1000 * batch);
	cudaMemcpy(a, GPU(convnet)->device[device_id].forwards[convnet->count - 1], sizeof(float) * 1000 * batch, cudaMemcpyDeviceToDevice);
	cudaEventRecord(overallStart, context->device[device_id].data_stream);
	for (i = convnet->count - 1; i >= 0; i--)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + i;
		ccv_convnet_layer_t* configuration = GPU(convnet)->device[device_id].configurations + i;
		cudaEventRecord(start, context->device[device_id].data_stream);
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				if (context->device[device_id].dor[i])
				{
					int out_rows, out_cols, out_partition;
					_ccv_convnet_layer_derive_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
					_cwc_kern_mute_neuron
					<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device[device_id].data_stream>>>
					(i == convnet->count - 1 ? a : GPU(convnet)->device[device_id].backwards[i + 1], context->device[device_id].dor[i]);
				}
				_cwc_convnet_convolutional_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[device_id].backwards[i + 1], GPU(convnet)->device[device_id].forwards[i], i > 0 ? GPU(convnet)->device[device_id].forwards[i - 1] : context->device[device_id].input, GPU(convnet)->device[device_id].backwards[i], configuration, GPU(convnet)->device[device_id].scratch, GPU(convnet)->device[device_id].unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				if (context->device[device_id].dor[i])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device[device_id].data_stream>>>
					(i == convnet->count - 1 ? a : GPU(convnet)->device[device_id].backwards[i + 1], context->device[device_id].dor[i]);
				_cwc_convnet_full_connect_backward_propagate(layer, batch,  i == convnet->count - 1 ? a : GPU(convnet)->device[device_id].backwards[i + 1], GPU(convnet)->device[device_id].forwards[i], i > 0 ? GPU(convnet)->device[device_id].forwards[i - 1] : context->device[device_id].input, GPU(convnet)->device[device_id].backwards[i], GPU(convnet)->device[device_id].unit, configuration, context->device[device_id].data_stream, context->device[device_id].data_cublas);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				_cwc_convnet_rnorm_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[device_id].backwards[i + 1], GPU(convnet)->device[device_id].forwards[i], i > 0 ? GPU(convnet)->device[device_id].forwards[i - 1] : context->device[device_id].input, GPU(convnet)->device[device_id].denoms[i], GPU(convnet)->device[device_id].backwards[i], context->device[device_id].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_MAX_POOL:
				_cwc_convnet_max_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[device_id].backwards[i + 1], GPU(convnet)->device[device_id].forwards[i], i > 0 ? GPU(convnet)->device[device_id].forwards[i - 1] : context->device[device_id].input, GPU(convnet)->device[device_id].backwards[i], context->device[device_id].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_cwc_convnet_average_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[device_id].backwards[i + 1], GPU(convnet)->device[device_id].backwards[i], context->device[device_id].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
		}
		cudaEventRecord(stop, context->device[device_id].data_stream);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time, start, stop);
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
			printf("%d %d %d, %d %d %d, elapsed time for layer %d bprop: %f milliseconds\n", VARY(layer)->convolutional.backward.coefficient.x, VARY(layer)->convolutional.backward.coefficient.y, VARY(layer)->convolutional.backward.coefficient.z, VARY(layer)->convolutional.backward.gradient.x, VARY(layer)->convolutional.backward.gradient.y, VARY(layer)->convolutional.backward.gradient.z, i + 1, elapsed_time);
		else
			printf("elapsed time for layer %d bprop: %f milliseconds\n", i + 1, elapsed_time);
	}
	cudaEventRecord(overallStop, context->device[device_id].data_stream);
	cudaEventSynchronize(overallStop);
	cudaEventElapsedTime(&elapsed_time, overallStart, overallStop);
	printf("backward pass %f milliseconds\n", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(overallStart);
	cudaEventDestroy(overallStop);
	cudaFree(a);
}
