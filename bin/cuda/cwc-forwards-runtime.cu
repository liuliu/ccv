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
	int mini_batch = dual_batch / 2;
	_cwc_convnet_alloc_reserved_both(convnet, mini_batch, 2, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	int i, device_id;
	int conv_layers[] = {0, 3, 6, 7, 8};
	for (device_id = 0; device_id < 2; device_id++)
		for (i = 0; i < 5; i++)
		{
			ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + conv_layers[i];
			EXTRA(layer)->vary.convolutional.forward.x = 4;
			EXTRA(layer)->vary.convolutional.forward.y = 8;
			EXTRA(layer)->vary.convolutional.forward.z = 32;
		}
	// doing model parallelism
	for (device_id = 0; device_id < 2; device_id++)
	{
		cudaSetDevice(device_id);
		_cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, convnet->input, convnet->rows, convnet->cols, convnet->channels, 1000, 0, mini_batch, mini_batch * device_id, mini_batch, context->host[device_id].input, context->host[device_id].c);
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
	_cwc_convnet_encode_impl(convnet, 2, mini_batch, 0, context);
	float *dual_out[2] = {0};
	cudaMallocHost(&dual_out[0], sizeof(float) * dual_batch * 1000);
	cudaMallocHost(&dual_out[1], sizeof(float) * dual_batch * 1000);
	for (device_id = 0; device_id < 2; device_id++)
	{
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		cudaMemcpy(dual_out[device_id], GPU(convnet)->device[device_id].forwards[convnet->count - 1], sizeof(float) * dual_batch * 1000, cudaMemcpyDeviceToHost);
	}
	ccv_convnet_compact(convnet);
	assert(cudaGetLastError() == cudaSuccess);
	// do it on one device
	device_id = 0;
	cudaSetDevice(device_id);
	mini_batch = dual_batch;
	_cwc_convnet_alloc_reserved_both(convnet, mini_batch, 1, params.layer_params);
	assert(cudaGetLastError() == cudaSuccess);
	context = GPU(convnet)->contexts;
	for (i = 0; i < 5; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[device_id].layers + conv_layers[i];
		EXTRA(layer)->vary.convolutional.forward.x = 4;
		EXTRA(layer)->vary.convolutional.forward.y = 8;
		EXTRA(layer)->vary.convolutional.forward.z = 32;
	}
	_cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, convnet->input, convnet->rows, convnet->cols, convnet->channels, 1000, 0, mini_batch, 0, mini_batch, context->host[device_id].input, context->host[device_id].c);
	cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	cudaMemcpyAsync(context->device[device_id].c, context->host[device_id].c, sizeof(int) * mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	cudaDeviceSynchronize();
	_cwc_convnet_encode_impl(convnet, 1, mini_batch, 0, context);
	cudaDeviceSynchronize();
	float* out = 0;
	cudaMallocHost(&out, sizeof(float) * dual_batch * 1000);
	cudaMemcpy(out, GPU(convnet)->device[device_id].forwards[convnet->count - 1], sizeof(float) * mini_batch * 1000, cudaMemcpyDeviceToHost);
	ccv_convnet_free(convnet);
	int j;
	for (i = 0; i < 1000; i++)
	{
		for (j = 0; j < mini_batch / 2; j++)
			if (fabs(out[i * mini_batch + j] - dual_out[0][i * (mini_batch / 2) + j]) > 1e-3)
				printf("%d %d %f %f %f\n", i, j, out[i * mini_batch + j], dual_out[0][i * (mini_batch / 2) + j], dual_out[1][i * (mini_batch / 2) + j]);
		for (j = 0; j < mini_batch / 2; j++)
			if (fabs(out[i * mini_batch + mini_batch / 2 + j] - dual_out[0][1000 * mini_batch / 2 + i * (mini_batch / 2) + j]) > 1e-3)
				printf("%d %d %f %f %f\n", i, j + mini_batch / 2, out[i * mini_batch + mini_batch / 2 + j], dual_out[0][1000 * mini_batch / 2 + i * (mini_batch / 2) + j], dual_out[1][1000 * mini_batch / 2 + i * (mini_batch / 2) + j]);
	}
	cudaFreeHost(dual_out[0]);
	cudaFreeHost(dual_out[1]);
	cudaFreeHost(out);
}
