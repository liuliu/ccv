#undef USE_DISPATCH // nvcc doesn't support libdispatch
extern "C" {
#include "ccv.h"
}
#include <ctype.h>
#define CASE_TESTS // so that we don't include public available methods
#include "../lib/cuda/cwc_convnet.cu"
#include "../lib/ccv_convnet.c"

static const int DEVICE_COUNT = 4;

extern "C" void cwc_backwards_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params)
{
	int dual_batch = params.mini_batch;
	int category_count = 1000;
	int mini_batch = dual_batch / DEVICE_COUNT;
	params.device_count = DEVICE_COUNT;
	_cwc_convnet_alloc_reserved_both(convnet, mini_batch, DEVICE_COUNT, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	int i, device_id;
	int conv_layers[] = {0, 3, 6, 7, 8};
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
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
	if (params.peer_access)
		_cwc_convnet_enable_peer_access(convnet, params.device_count);
	// doing model parallelism
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
	{
		cudaSetDevice(device_id);
		cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, 0, convnet->input, params.input.min_dim, params.input.max_dim, convnet->rows, convnet->cols, convnet->channels, category_count, 0, mini_batch, mini_batch * device_id, mini_batch, context->host[device_id].input, context->host[device_id].c);
		cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
		assert(cudaGetLastError() == cudaSuccess);
		cudaMemcpyAsync(context->device[device_id].c, context->host[device_id].c, sizeof(int) * mini_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
		assert(cudaGetLastError() == cudaSuccess);
	}
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
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
	_cwc_convnet_encode_impl(convnet, DEVICE_COUNT, mini_batch, 0, context);
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
	{
		cudaSetDevice(device_id);
		// do the logistic loss
		_cwc_convnet_softmax_with_logistic_loss(mini_batch, category_count, GPU(convnet)->device[device_id].forwards[convnet->count - 1] + device_id * mini_batch * category_count, context->device[device_id].c, context->device[device_id].data_stream);
	}
	_cwc_convnet_backward_propagate_error(convnet, DEVICE_COUNT, mini_batch, context);
	_cwc_convnet_reduce_data_parallelism(convnet, DEVICE_COUNT, context);
	for (device_id = 1; device_id < DEVICE_COUNT; device_id++)
	{
		cudaSetDevice(device_id);
		cudaEventRecord(context->device[device_id].data_joint, context->device[device_id].data_stream);
	}
	cudaSetDevice(0);
	for (device_id = 1; device_id < DEVICE_COUNT; device_id++)
		cudaStreamWaitEvent(context->device[0].data_stream, context->device[device_id].data_joint, 0);
	cudaEventRecord(stop, context->device[0].data_stream);
	cudaEventSynchronize(stop);
	assert(cudaGetLastError() == cudaSuccess);
	float elapsed_time = 0;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d GPUs uses %f ms\n", DEVICE_COUNT, elapsed_time);
	float *dual_out[DEVICE_COUNT] = {0};
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
		cudaMallocHost(&dual_out[device_id], sizeof(float) * dual_batch * category_count);
	float *back_out[DEVICE_COUNT] = {0};
	ccv_convnet_layer_t* second_layer = convnet->layers + 1;
	int second_count = second_layer->input.matrix.rows * second_layer->input.matrix.cols * second_layer->input.matrix.channels;
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
		cudaMallocHost(&back_out[device_id], sizeof(float) * mini_batch * second_count);
	ccv_convnet_layer_t* last_layer = GPU(convnet)->device[0].layers + convnet->count - 1;
	float *dual_w[DEVICE_COUNT] = {0};
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
		cudaMallocHost(&dual_w[device_id], sizeof(float) * last_layer->wnum);
	ccv_convnet_layer_t* second_last_layer = GPU(convnet)->device[0].layers + convnet->count - 2;
	float *dual_w_2[DEVICE_COUNT] = {0};
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
		cudaMallocHost(&dual_w_2[device_id], sizeof(float) * second_last_layer->wnum);
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
	{
		cudaSetDevice(device_id);
		cudaMemcpy(dual_out[device_id], GPU(convnet)->device[device_id].forwards[convnet->count - 1], sizeof(float) * dual_batch * category_count, cudaMemcpyDeviceToHost);
		cudaMemcpy(back_out[device_id], GPU(convnet)->device[device_id].backwards[1], sizeof(float) * mini_batch * second_count, cudaMemcpyDeviceToHost);
		cudaMemcpy(dual_w[device_id], GPU(convnet)->device[device_id].configurations[convnet->count - 1].w, sizeof(float) * last_layer->wnum, cudaMemcpyDeviceToHost);
		cudaMemcpy(dual_w_2[device_id], GPU(convnet)->device[device_id].configurations[convnet->count - 2].w, sizeof(float) * second_last_layer->wnum, cudaMemcpyDeviceToHost);
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
	cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, 0, convnet->input, params.input.min_dim, params.input.max_dim, convnet->rows, convnet->cols, convnet->channels, category_count, 0, dual_batch, 0, dual_batch, context->host[device_id].input, context->host[device_id].c);
	cudaMemcpyAsync(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * dual_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	cudaMemcpyAsync(context->device[device_id].c, context->host[device_id].c, sizeof(int) * dual_batch, cudaMemcpyHostToDevice, context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	cudaDeviceSynchronize();
	cudaEventRecord(start, context->device[device_id].data_stream);
	_cwc_convnet_encode_impl(convnet, 1, dual_batch, 0, context);
	// do the logistic loss
	_cwc_convnet_softmax_with_logistic_loss(dual_batch, category_count, GPU(convnet)->device[device_id].forwards[convnet->count - 1], context->device[device_id].c, context->device[device_id].data_stream);
	_cwc_convnet_backward_propagate_error(convnet, 1, dual_batch, context);
	cudaEventRecord(stop, context->device[device_id].data_stream);
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
	float* w = 0;
	int wnum = GPU(convnet)->device[device_id].configurations[convnet->count - 1].wnum;
	cudaMallocHost(&w, sizeof(float) * wnum);
	cudaMemcpy(w, GPU(convnet)->device[device_id].configurations[convnet->count - 1].w, sizeof(float) * wnum, cudaMemcpyDeviceToHost);
	float* w_2 = 0;
	int wnum_2 = GPU(convnet)->device[device_id].configurations[convnet->count - 2].wnum;
	cudaMallocHost(&w_2, sizeof(float) * wnum_2);
	cudaMemcpy(w_2, GPU(convnet)->device[device_id].configurations[convnet->count - 2].w, sizeof(float) * wnum_2, cudaMemcpyDeviceToHost);
	ccv_convnet_free(convnet);
	int j;
	for (i = 0; i < category_count; i++)
	{
		for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
			for (j = 0; j < mini_batch; j++)
			{
				float p = out[i * dual_batch + mini_batch * device_id + j];
				float q = dual_out[device_id][category_count * mini_batch * device_id + i * mini_batch + j];
				float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
				if (delta > 1e-4)
					printf("softmax with logistic loss doesn't match: %d %d %d %g %g\n", device_id, i, j, out[i * dual_batch + mini_batch * device_id + j], dual_out[device_id][category_count * mini_batch * device_id + i * mini_batch + j]);
			}
	}
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
	{
		const int pwnum = wnum / DEVICE_COUNT;
		for (i = 0; i < pwnum; i++)
		{
			float p = w[i + pwnum * device_id];
			float q = dual_w[device_id][i];
			float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1e-3);
			if (delta > 1e-3)
				printf("the weight update on last layer doesn't match: %d %d %g %g\n", device_id, i + pwnum * device_id, w[i + pwnum * device_id], dual_w[device_id][i]);
		}
	}
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
	{
		const int pwnum_2 = wnum_2 / DEVICE_COUNT;
		for (i = 0; i < pwnum_2; i++)
		{
			float p = w_2[i + pwnum_2 * device_id];
			float q = dual_w_2[device_id][i];
			float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1e-3);
			if (delta > 1e-3)
				printf("the weight update on second to last layer doesn't match: %d %d %g %g\n", device_id, i + pwnum_2 * device_id, w_2[i + pwnum_2 * device_id], dual_w_2[device_id][i]);
		}
	}
	for (i = 0; i < second_count; i++)
	{
		for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
			for (j = 0; j < mini_batch; j++)
			{
				float p = back[i * dual_batch + mini_batch * device_id + j];
				float q = back_out[device_id][i * mini_batch + j];
				float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1e-3);
				if (delta > 1e-3)
					printf("the last layer of backwards propagated error doesn't match: %d %d %d %g %g\n", device_id, i, j, back[i * dual_batch + mini_batch * device_id + j], back_out[device_id][i * mini_batch + j]);
			}
	}
	for (device_id = 0; device_id < DEVICE_COUNT; device_id++)
	{
		cudaFreeHost(dual_out[device_id]);
		cudaFreeHost(back_out[device_id]);
		cudaFreeHost(dual_w[device_id]);
	}
	cudaFreeHost(out);
	cudaFreeHost(back);
	cudaFreeHost(w);
}
