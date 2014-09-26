#undef USE_DISPATCH // nvcc doesn't support libdispatch
extern "C" {
#include "ccv.h"
}
#include <ctype.h>
#define CASE_TESTS // so that we don't include public available methods
#include "../lib/cuda/cwc_convnet.cu"
#include "../lib/ccv_convnet.c"

extern "C" void cwc_forwards_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params)
{
	int dual_batch = params.mini_batch;
	int category_count = 1000;
	int mini_batch = dual_batch / 2;
	_cwc_convnet_alloc_reserved_both(convnet, mini_batch, 2, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	int i, device_id, other_device_id;
	int conv_layers[] = {0, 3, 6, 7, 8};
	for (device_id = 0; device_id < 2; device_id++)
		for (i = 0; i < 5; i++)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + conv_layers[i];
			EXTRA(layer)->vary.convolutional.forward.x = 4;
			EXTRA(layer)->vary.convolutional.forward.y = 8;
			EXTRA(layer)->vary.convolutional.forward.z = 32;
			if (conv_layers[i] == 3)
			{
				EXTRA(layer)->vary.convolutional.backward.gradient.x = 4;
				EXTRA(layer)->vary.convolutional.backward.gradient.y = 6;
				EXTRA(layer)->vary.convolutional.backward.gradient.z = 24;
				EXTRA(layer)->vary.convolutional.backward.coefficient.x = 6;
				EXTRA(layer)->vary.convolutional.backward.coefficient.y = 4;
				EXTRA(layer)->vary.convolutional.backward.coefficient.z = 24;
			} else if (conv_layers[i] == 0) {
				EXTRA(layer)->vary.convolutional.backward.coefficient.x = 1;
				EXTRA(layer)->vary.convolutional.backward.coefficient.y = 3;
				EXTRA(layer)->vary.convolutional.backward.coefficient.z = 1;
			} else {
				EXTRA(layer)->vary.convolutional.backward.gradient.x = 8;
				EXTRA(layer)->vary.convolutional.backward.gradient.y = 4;
				EXTRA(layer)->vary.convolutional.backward.gradient.z = 32;
				EXTRA(layer)->vary.convolutional.backward.coefficient.x = 8;
				EXTRA(layer)->vary.convolutional.backward.coefficient.y = 4;
				EXTRA(layer)->vary.convolutional.backward.coefficient.z = 32;
			}
		}
	for (device_id = 0; device_id < params.device_count; device_id++)
		for (other_device_id = 0; other_device_id < params.device_count; other_device_id++)
			if (device_id != other_device_id)
			{
				cudaSetDevice(device_id);
				cudaDeviceEnablePeerAccess(other_device_id, 0);
			}
	// doing model parallelism
	for (device_id = 0; device_id < 2; device_id++)
	{
		cudaSetDevice(device_id);
		_cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, convnet->input, convnet->rows, convnet->cols, convnet->channels, category_count, 0, mini_batch, mini_batch * device_id, mini_batch, context->host[device_id].input, context->host[device_id].c);
		cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
		assert(cudaGetLastError() == cudaSuccess);
		cudaMemcpyAsync(context->device[device_id].c, context->host[device_id].c, sizeof(int) * mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
		assert(cudaGetLastError() == cudaSuccess);
	}
	for (device_id = 0; device_id < 2; device_id++)
	{
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
	}
	cudaSetDevice(0);
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, context->device[0].data_stream);
	_cwc_convnet_encode_impl(convnet, 2, mini_batch, 0, context);
	for (device_id = 0; device_id < 2; device_id++)
	{
		cudaSetDevice(device_id);
		// do the logistic loss
		_cwc_convnet_softmax_with_logistic_loss(mini_batch, category_count, GPU(convnet)->device[device_id].forwards[convnet->count - 1] + device_id * mini_batch * category_count, context->device[device_id].c, context->device[device_id].data_stream);
	}
	_cwc_convnet_backward_propagate_error(convnet, 2, mini_batch, context);
	cudaSetDevice(1);
	cudaEventRecord(context->device[1].data_joint, context->device[1].data_stream);
	cudaSetDevice(0);
	cudaStreamWaitEvent(context->device[0].data_stream, context->device[1].data_joint, 0);
	cudaEventRecord(stop, context->device[0].data_stream);
	cudaEventSynchronize(stop);
	assert(cudaGetLastError() == cudaSuccess);
	float elapsed_time = 0;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("dual GPU uses %f ms\n", elapsed_time);
	float *dual_out[2] = {0};
	cudaMallocHost(&dual_out[0], sizeof(float) * dual_batch * category_count);
	cudaMallocHost(&dual_out[1], sizeof(float) * dual_batch * category_count);
	float *back_out[2] = {0};
	ccv_convnet_layer_t* second_layer = convnet->layers + 1;
	int second_count = second_layer->input.matrix.rows * second_layer->input.matrix.cols * second_layer->input.matrix.channels;
	cudaMallocHost(&back_out[0], sizeof(float) * mini_batch * second_count);
	cudaMallocHost(&back_out[1], sizeof(float) * mini_batch * second_count);
	for (device_id = 0; device_id < 2; device_id++)
	{
		cudaSetDevice(device_id);
		cudaMemcpy(dual_out[device_id], GPU(convnet)->device[device_id].forwards[convnet->count - 1], sizeof(float) * dual_batch * category_count, cudaMemcpyDeviceToHost);
		cudaMemcpy(back_out[device_id], GPU(convnet)->device[device_id].backwards[1], sizeof(float) * second_count, cudaMemcpyDeviceToHost);
	}
	ccv_convnet_compact(convnet);
	assert(cudaGetLastError() == cudaSuccess);
	// do it on one device
	device_id = 0;
	cudaSetDevice(device_id);
	_cwc_convnet_alloc_reserved_both(convnet, dual_batch, 1, params.layer_params);
	assert(cudaGetLastError() == cudaSuccess);
	context = GPU(convnet)->contexts;
	for (i = 0; i < 5; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + conv_layers[i];
		EXTRA(layer)->vary.convolutional.forward.x = 4;
		EXTRA(layer)->vary.convolutional.forward.y = 8;
		EXTRA(layer)->vary.convolutional.forward.z = 32;
		if (conv_layers[i] == 3)
		{
			EXTRA(layer)->vary.convolutional.backward.gradient.x = 4;
			EXTRA(layer)->vary.convolutional.backward.gradient.y = 6;
			EXTRA(layer)->vary.convolutional.backward.gradient.z = 24;
			EXTRA(layer)->vary.convolutional.backward.coefficient.x = 6;
			EXTRA(layer)->vary.convolutional.backward.coefficient.y = 4;
			EXTRA(layer)->vary.convolutional.backward.coefficient.z = 24;
		} else if (conv_layers[i] == 0) {
			EXTRA(layer)->vary.convolutional.backward.coefficient.x = 1;
			EXTRA(layer)->vary.convolutional.backward.coefficient.y = 3;
			EXTRA(layer)->vary.convolutional.backward.coefficient.z = 1;
		} else {
			EXTRA(layer)->vary.convolutional.backward.gradient.x = 8;
			EXTRA(layer)->vary.convolutional.backward.gradient.y = 4;
			EXTRA(layer)->vary.convolutional.backward.gradient.z = 32;
			EXTRA(layer)->vary.convolutional.backward.coefficient.x = 8;
			EXTRA(layer)->vary.convolutional.backward.coefficient.y = 4;
			EXTRA(layer)->vary.convolutional.backward.coefficient.z = 32;
		}
	}
	_cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, convnet->input, convnet->rows, convnet->cols, convnet->channels, category_count, 0, dual_batch, 0, dual_batch, context->host[device_id].input, context->host[device_id].c);
	cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * dual_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	cudaMemcpyAsync(context->device[device_id].c, context->host[device_id].c, sizeof(int) * dual_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	cudaDeviceSynchronize();
	cudaEventRecord(start, context->device[0].data_stream);
	_cwc_convnet_encode_impl(convnet, 1, dual_batch, 0, context);
	// do the logistic loss
	_cwc_convnet_softmax_with_logistic_loss(dual_batch, category_count, GPU(convnet)->device[device_id].forwards[convnet->count - 1], context->device[device_id].c, context->device[device_id].data_stream);
	_cwc_convnet_backward_propagate_error(convnet, 1, dual_batch, context);
	cudaEventRecord(stop, context->device[0].data_stream);
	cudaEventSynchronize(stop);
	elapsed_time = 0;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("one GPU uses %f ms\n", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();
	float* out = 0;
	cudaMallocHost(&out, sizeof(float) * dual_batch * category_count);
	cudaMemcpy(out, GPU(convnet)->device[device_id].forwards[convnet->count - 1], sizeof(float) * dual_batch * category_count, cudaMemcpyDeviceToHost);
	float* back = 0;
	cudaMallocHost(&back, sizeof(float) * dual_batch * second_count);
	cudaMemcpy(back, GPU(convnet)->device[device_id].backwards[1], sizeof(float) * dual_batch * second_count, cudaMemcpyDeviceToHost);
	ccv_convnet_free(convnet);
	int j;
	for (i = 0; i < category_count; i++)
	{
		for (j = 0; j < mini_batch; j++)
			if (fabs(out[i * dual_batch + j] - dual_out[0][i * mini_batch + j]) > 1e-6)
				printf("softmax with logistic loss doesn't match: %d %d %g %g %g\n", i, j, out[i * dual_batch + j], dual_out[0][i * mini_batch + j], dual_out[1][i * mini_batch + j]);
		for (j = 0; j < mini_batch; j++)
			if (fabs(out[i * dual_batch + mini_batch + j] - dual_out[1][category_count * mini_batch + i * mini_batch + j]) > 1e-6)
				printf("softmax with logistic loss doesn't match: %d %d %g %g %g\n", i, j + mini_batch, out[i * dual_batch + mini_batch + j], dual_out[0][category_count * mini_batch + i * mini_batch + j], dual_out[1][1000 * mini_batch + i * mini_batch + j]);
	}/*
	for (i = 0; i < second_count; i++)
	{
		for (j = 0; j < mini_batch; j++)
			if (fabs(back[i * dual_batch + j] - back_out[0][i * mini_batch + j]) > 1e-6)
				printf("the last layer of backwards propagated error doesn't match: %d %d %g %g\n", i, j, back[i * dual_batch + j], back_out[0][i * mini_batch + j]);
	}*/
	cudaFreeHost(dual_out[0]);
	cudaFreeHost(dual_out[1]);
	cudaFreeHost(back_out[0]);
	cudaFreeHost(back_out[1]);
	cudaFreeHost(out);
	cudaFreeHost(back);
}
