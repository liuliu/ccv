#undef USE_DISPATCH // nvcc doesn't support libdispatch
extern "C" {
#include "ccv.h"
}
#include <ctype.h>
#define CASE_TESTS // so that we don't include public available methods
#include "../lib/cuda/cwc_convnet.cu"
#include "../lib/ccv_convnet.c"

extern "C" void cwc_verify_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params)
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

	// second average pool layer forward propagate
	ccv_convnet_layer_t* second_gpu_layer = GPU(convnet)->device[device_id].layers + 1;
	cwc_convnet_average_pool_forward_propagate(second_gpu_layer, second_gpu_layer->input.matrix.rows, second_gpu_layer->input.matrix.cols, batch, GPU(convnet)->device[device_id].forwards[0], GPU(convnet)->device[device_id].forwards[1], context->device[device_id].data_stream);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	int second_out_rows, second_out_cols, second_out_partition, second_out_channels = second_gpu_layer->input.matrix.channels;
	ccv_convnet_make_output(second_gpu_layer, second_gpu_layer->input.matrix.rows, second_gpu_layer->input.matrix.cols, &second_out_rows, &second_out_cols, &second_out_partition);
	float* second_out = 0;
	cudaMallocHost(&second_out, sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch);
	cudaMemcpy(second_out, GPU(convnet)->device[device_id].forwards[1], sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate second average pool layer on GPU\n");

	// third convolutional layer forward propagate
	ccv_convnet_layer_t* third_gpu_layer = GPU(convnet)->device[device_id].layers + 2;
	// these are the setups for TITAN, thus, skip the benching phase
	EXTRA(third_gpu_layer)->vary.convolutional.forward.x = 4;
	EXTRA(third_gpu_layer)->vary.convolutional.forward.y = 8;
	EXTRA(third_gpu_layer)->vary.convolutional.forward.z = 32;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_forward_propagate(third_gpu_layer, third_gpu_layer->input.matrix.rows, third_gpu_layer->input.matrix.cols, batch, GPU(convnet)->device[device_id].forwards[1], GPU(convnet)->device[device_id].forwards[2], context->device[device_id].data_stream);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for third convolutional layer fprop: %f milliseconds\n", EXTRA(third_gpu_layer)->vary.convolutional.forward.x, EXTRA(third_gpu_layer)->vary.convolutional.forward.y, EXTRA(third_gpu_layer)->vary.convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	int third_out_rows, third_out_cols, third_out_partition, third_out_channels = third_gpu_layer->net.convolutional.count;
	ccv_convnet_make_output(third_gpu_layer, third_gpu_layer->input.matrix.rows, third_gpu_layer->input.matrix.cols, &third_out_rows, &third_out_cols, &third_out_partition);
	float* third_out = 0;
	cudaMallocHost(&third_out, sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch);
	cudaMemcpy(third_out, GPU(convnet)->device[device_id].forwards[2], sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate third convolutional layer on GPU\n");

	// forth average pool layer forward propagate
	ccv_convnet_layer_t* forth_gpu_layer = GPU(convnet)->device[device_id].layers + 3;
	cwc_convnet_average_pool_forward_propagate(forth_gpu_layer, forth_gpu_layer->input.matrix.rows, forth_gpu_layer->input.matrix.cols, batch, GPU(convnet)->device[device_id].forwards[2], GPU(convnet)->device[device_id].forwards[3], context->device[device_id].data_stream);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	int forth_out_rows, forth_out_cols, forth_out_partition, forth_out_channels = forth_gpu_layer->input.matrix.channels;
	ccv_convnet_make_output(forth_gpu_layer, forth_gpu_layer->input.matrix.rows, forth_gpu_layer->input.matrix.cols, &forth_out_rows, &forth_out_cols, &forth_out_partition);
	float* forth_out = 0;
	cudaMallocHost(&forth_out, sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch);
	cudaMemcpy(forth_out, GPU(convnet)->device[device_id].forwards[3], sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate forth average pool layer on GPU\n");

	// fifth convolutional layer forward propagate
	ccv_convnet_layer_t* fifth_gpu_layer = GPU(convnet)->device[device_id].layers + 4;
	// these are the setups for TITAN, thus, skip the benching phase
	EXTRA(fifth_gpu_layer)->vary.convolutional.forward.x = 4;
	EXTRA(fifth_gpu_layer)->vary.convolutional.forward.y = 8;
	EXTRA(fifth_gpu_layer)->vary.convolutional.forward.z = 32;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_forward_propagate(fifth_gpu_layer, fifth_gpu_layer->input.matrix.rows, fifth_gpu_layer->input.matrix.cols, batch, GPU(convnet)->device[device_id].forwards[3], GPU(convnet)->device[device_id].forwards[4], context->device[device_id].data_stream);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for fifth convolutional layer fprop: %f milliseconds\n", EXTRA(fifth_gpu_layer)->vary.convolutional.forward.x, EXTRA(fifth_gpu_layer)->vary.convolutional.forward.y, EXTRA(fifth_gpu_layer)->vary.convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	int fifth_out_rows, fifth_out_cols, fifth_out_partition, fifth_out_channels = fifth_gpu_layer->net.convolutional.count;
	ccv_convnet_make_output(fifth_gpu_layer, fifth_gpu_layer->input.matrix.rows, fifth_gpu_layer->input.matrix.cols, &fifth_out_rows, &fifth_out_cols, &fifth_out_partition);
	float* fifth_out = 0;
	cudaMallocHost(&fifth_out, sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch);
	cudaMemcpy(fifth_out, GPU(convnet)->device[device_id].forwards[4], sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate fifth convolutional layer on GPU\n");

	// sixth convolutional layer forward propagate
	ccv_convnet_layer_t* sixth_gpu_layer = GPU(convnet)->device[device_id].layers + 5;
	// these are the setups for TITAN, thus, skip the benching phase
	EXTRA(sixth_gpu_layer)->vary.convolutional.forward.x = 4;
	EXTRA(sixth_gpu_layer)->vary.convolutional.forward.y = 8;
	EXTRA(sixth_gpu_layer)->vary.convolutional.forward.z = 32;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_forward_propagate(sixth_gpu_layer, sixth_gpu_layer->input.matrix.rows, sixth_gpu_layer->input.matrix.cols, batch, GPU(convnet)->device[device_id].forwards[4], GPU(convnet)->device[device_id].forwards[5], context->device[device_id].data_stream);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for sixth convolutional layer fprop: %f milliseconds\n", EXTRA(sixth_gpu_layer)->vary.convolutional.forward.x, EXTRA(sixth_gpu_layer)->vary.convolutional.forward.y, EXTRA(sixth_gpu_layer)->vary.convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	int sixth_out_rows, sixth_out_cols, sixth_out_partition, sixth_out_channels = sixth_gpu_layer->net.convolutional.count;
	ccv_convnet_make_output(sixth_gpu_layer, sixth_gpu_layer->input.matrix.rows, sixth_gpu_layer->input.matrix.cols, &sixth_out_rows, &sixth_out_cols, &sixth_out_partition);
	float* sixth_out = 0;
	cudaMallocHost(&sixth_out, sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch);
	cudaMemcpy(sixth_out, GPU(convnet)->device[device_id].forwards[5], sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate sixth convolutional layer on GPU\n");

	// seventh convolutional layer forward propagate
	ccv_convnet_layer_t* seventh_gpu_layer = GPU(convnet)->device[device_id].layers + 6;
	// these are the setups for TITAN, thus, skip the benching phase
	EXTRA(seventh_gpu_layer)->vary.convolutional.forward.x = 4;
	EXTRA(seventh_gpu_layer)->vary.convolutional.forward.y = 8;
	EXTRA(seventh_gpu_layer)->vary.convolutional.forward.z = 32;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_forward_propagate(seventh_gpu_layer, seventh_gpu_layer->input.matrix.rows, seventh_gpu_layer->input.matrix.cols, batch, GPU(convnet)->device[device_id].forwards[5], GPU(convnet)->device[device_id].forwards[6], context->device[device_id].data_stream);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for seventh convolutional layer fprop: %f milliseconds\n", EXTRA(seventh_gpu_layer)->vary.convolutional.forward.x, EXTRA(seventh_gpu_layer)->vary.convolutional.forward.y, EXTRA(seventh_gpu_layer)->vary.convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	int seventh_out_rows, seventh_out_cols, seventh_out_partition, seventh_out_channels = seventh_gpu_layer->net.convolutional.count;
	ccv_convnet_make_output(seventh_gpu_layer, seventh_gpu_layer->input.matrix.rows, seventh_gpu_layer->input.matrix.cols, &seventh_out_rows, &seventh_out_cols, &seventh_out_partition);
	float* seventh_out = 0;
	cudaMallocHost(&seventh_out, sizeof(float) * seventh_out_rows * seventh_out_cols * seventh_out_channels * batch);
	cudaMemcpy(seventh_out, GPU(convnet)->device[device_id].forwards[6], sizeof(float) * seventh_out_rows * seventh_out_cols * seventh_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate seventh convolutional layer on GPU\n");

	// the last full connect layer forward propagate
	ccv_convnet_layer_t* eleventh_gpu_layer = GPU(convnet)->device[device_id].layers + 10;
	float* eleventh_in = 0;
	cudaMallocHost(&eleventh_in, sizeof(float) * batch * eleventh_gpu_layer->input.node.count);
	for (i = 0; i < batch; i++)
		for (j = 0; j < eleventh_gpu_layer->input.node.count; j++)
			eleventh_in[j * batch + i] = (j - 100 + i) / 200;
	cudaMemcpy(GPU(convnet)->device[device_id].forwards[9], eleventh_in, sizeof(float) * batch * eleventh_gpu_layer->input.node.count, cudaMemcpyHostToDevice);
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_full_connect_forward_propagate(eleventh_gpu_layer, 128, GPU(convnet)->device[device_id].forwards[9], GPU(convnet)->device[device_id].forwards[10], GPU(convnet)->device[device_id].unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("elapsed time for eleventh full connect layer fprop: %f milliseconds\n", elapsed_time);
	float* eleventh_out = 0;
	cudaMallocHost(&eleventh_out, sizeof(float) * batch * eleventh_gpu_layer->net.full_connect.count);
	cudaMemcpy(eleventh_out, GPU(convnet)->device[device_id].forwards[10], sizeof(float) * batch * eleventh_gpu_layer->net.full_connect.count, cudaMemcpyDeviceToHost);
	printf("finished forward propagate eleventh full connect layer on GPU\n");

	// eleventh full connect layer backward propagate
	ccv_convnet_layer_t* eleventh_gpu_configuration = GPU(convnet)->device[device_id].configurations + 10;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_full_connect_backward_propagate(eleventh_gpu_layer, batch, GPU(convnet)->device[device_id].forwards[10], GPU(convnet)->device[device_id].forwards[10], GPU(convnet)->device[device_id].forwards[9], GPU(convnet)->device[device_id].backwards[10], GPU(convnet)->device[device_id].unit, eleventh_gpu_configuration->w, eleventh_gpu_configuration->bias, context->device[device_id].data_stream, context->device[device_id].data_cublas);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("elapsed time for eleventh full connect layer bprop: %f milliseconds\n", elapsed_time);
	float* eleventh_back = 0;
	cudaMallocHost(&eleventh_back, sizeof(float) * eleventh_gpu_layer->input.node.count * batch);
	cudaMemcpy(eleventh_back, GPU(convnet)->device[device_id].backwards[10], sizeof(float) * eleventh_gpu_layer->input.node.count * batch, cudaMemcpyDeviceToHost);
	float* eleventh_grad = 0;
	cudaMallocHost(&eleventh_grad, sizeof(float) * (eleventh_gpu_layer->wnum + eleventh_gpu_layer->net.full_connect.count));
	assert(eleventh_grad);
	cudaMemcpy(eleventh_grad, eleventh_gpu_configuration->w, sizeof(float) * (eleventh_gpu_layer->wnum + eleventh_gpu_layer->net.full_connect.count), cudaMemcpyDeviceToHost);
	printf("finished backward propagate eleventh full connect layer on GPU\n");

	// seventh convolutonal layer backward propagate
	cudaMemcpy(GPU(convnet)->device[device_id].backwards[7], GPU(convnet)->device[device_id].forwards[6], sizeof(float) * seventh_out_rows * seventh_out_cols * seventh_out_channels * batch, cudaMemcpyDeviceToDevice);
	ccv_convnet_layer_t* seventh_gpu_configuration = GPU(convnet)->device[device_id].configurations + 6;
	EXTRA(seventh_gpu_layer)->vary.convolutional.backward.coefficient.x = 8;
	EXTRA(seventh_gpu_layer)->vary.convolutional.backward.coefficient.y = 4;
	EXTRA(seventh_gpu_layer)->vary.convolutional.backward.coefficient.z = 32;
	EXTRA(seventh_gpu_layer)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(seventh_gpu_layer)->vary.convolutional.backward.gradient.y = 8;
	EXTRA(seventh_gpu_layer)->vary.convolutional.backward.gradient.z = 32;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_backward_propagate(seventh_gpu_layer, batch, GPU(convnet)->device[device_id].backwards[7], GPU(convnet)->device[device_id].forwards[6], GPU(convnet)->device[device_id].forwards[5], GPU(convnet)->device[device_id].backwards[6], seventh_gpu_configuration, GPU(convnet)->device[device_id].scratch, GPU(convnet)->device[device_id].unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for seventh convolutional layer bprop: %f milliseconds\n", EXTRA(seventh_gpu_layer)->vary.convolutional.backward.coefficient.x, EXTRA(seventh_gpu_layer)->vary.convolutional.backward.coefficient.y, EXTRA(seventh_gpu_layer)->vary.convolutional.backward.coefficient.z, EXTRA(seventh_gpu_layer)->vary.convolutional.backward.gradient.x, EXTRA(seventh_gpu_layer)->vary.convolutional.backward.gradient.y, EXTRA(seventh_gpu_layer)->vary.convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* seventh_back = 0;
	cudaMallocHost(&seventh_back, sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch);
	cudaMemcpy(seventh_back, GPU(convnet)->device[device_id].backwards[6], sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch, cudaMemcpyDeviceToHost);
	float* seventh_grad = 0;
	cudaMallocHost(&seventh_grad, sizeof(float) * (seventh_gpu_layer->wnum + seventh_gpu_layer->net.convolutional.count));
	assert(seventh_grad);
	cudaMemcpy(seventh_grad, seventh_gpu_configuration->w, sizeof(float) * (seventh_gpu_layer->wnum + seventh_gpu_layer->net.convolutional.count), cudaMemcpyDeviceToHost);
	printf("finished backward propagate seventh convolutional layer on GPU\n");

	// sixth convolutonal layer backward propagate
	ccv_convnet_layer_t* sixth_gpu_configuration = GPU(convnet)->device[device_id].configurations + 5;
	EXTRA(sixth_gpu_layer)->vary.convolutional.backward.coefficient.x = 8;
	EXTRA(sixth_gpu_layer)->vary.convolutional.backward.coefficient.y = 3;
	EXTRA(sixth_gpu_layer)->vary.convolutional.backward.coefficient.z = 32;
	EXTRA(sixth_gpu_layer)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(sixth_gpu_layer)->vary.convolutional.backward.gradient.y = 8;
	EXTRA(sixth_gpu_layer)->vary.convolutional.backward.gradient.z = 32;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_backward_propagate(sixth_gpu_layer, batch, GPU(convnet)->device[device_id].backwards[6], GPU(convnet)->device[device_id].forwards[5], GPU(convnet)->device[device_id].forwards[4], GPU(convnet)->device[device_id].backwards[5], sixth_gpu_configuration, GPU(convnet)->device[device_id].scratch, GPU(convnet)->device[device_id].unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for sixth convolutional layer bprop: %f milliseconds\n", EXTRA(sixth_gpu_layer)->vary.convolutional.backward.coefficient.x, EXTRA(sixth_gpu_layer)->vary.convolutional.backward.coefficient.y, EXTRA(sixth_gpu_layer)->vary.convolutional.backward.coefficient.z, EXTRA(sixth_gpu_layer)->vary.convolutional.backward.gradient.x, EXTRA(sixth_gpu_layer)->vary.convolutional.backward.gradient.y, EXTRA(sixth_gpu_layer)->vary.convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* sixth_back = 0;
	cudaMallocHost(&sixth_back, sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch);
	cudaMemcpy(sixth_back, GPU(convnet)->device[device_id].backwards[5], sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch, cudaMemcpyDeviceToHost);
	float* sixth_grad = 0;
	cudaMallocHost(&sixth_grad, sizeof(float) * (sixth_gpu_layer->wnum + sixth_gpu_layer->net.convolutional.count));
	assert(sixth_grad);
	cudaMemcpy(sixth_grad, sixth_gpu_configuration->w, sizeof(float) * (sixth_gpu_layer->wnum + sixth_gpu_layer->net.convolutional.count), cudaMemcpyDeviceToHost);
	printf("finished backward propagate sixth convolutional layer on GPU\n");

	// fifth convolutonal layer backward propagate
	ccv_convnet_layer_t* fifth_gpu_configuration = GPU(convnet)->device[device_id].configurations + 4;
	EXTRA(fifth_gpu_layer)->vary.convolutional.backward.coefficient.x = 8;
	EXTRA(fifth_gpu_layer)->vary.convolutional.backward.coefficient.y = 3;
	EXTRA(fifth_gpu_layer)->vary.convolutional.backward.coefficient.z = 32;
	EXTRA(fifth_gpu_layer)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(fifth_gpu_layer)->vary.convolutional.backward.gradient.y = 8;
	EXTRA(fifth_gpu_layer)->vary.convolutional.backward.gradient.z = 32;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_backward_propagate(fifth_gpu_layer, batch, GPU(convnet)->device[device_id].backwards[5], GPU(convnet)->device[device_id].forwards[4], GPU(convnet)->device[device_id].forwards[3], GPU(convnet)->device[device_id].backwards[4], fifth_gpu_configuration, GPU(convnet)->device[device_id].scratch, GPU(convnet)->device[device_id].unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for fifth convolutional layer bprop: %f milliseconds\n", EXTRA(fifth_gpu_layer)->vary.convolutional.backward.coefficient.x, EXTRA(fifth_gpu_layer)->vary.convolutional.backward.coefficient.y, EXTRA(fifth_gpu_layer)->vary.convolutional.backward.coefficient.z, EXTRA(fifth_gpu_layer)->vary.convolutional.backward.gradient.x, EXTRA(fifth_gpu_layer)->vary.convolutional.backward.gradient.y, EXTRA(fifth_gpu_layer)->vary.convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* fifth_back = 0;
	cudaMallocHost(&fifth_back, sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch);
	cudaMemcpy(fifth_back, GPU(convnet)->device[device_id].backwards[4], sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch, cudaMemcpyDeviceToHost);
	float* fifth_grad = 0;
	cudaMallocHost(&fifth_grad, sizeof(float) * (fifth_gpu_layer->wnum + fifth_gpu_layer->net.convolutional.count));
	assert(fifth_grad);
	cudaMemcpy(fifth_grad, fifth_gpu_configuration->w, sizeof(float) * (fifth_gpu_layer->wnum + fifth_gpu_layer->net.convolutional.count), cudaMemcpyDeviceToHost);
	printf("finished backward propagate fifth convolutional layer on GPU\n");

	// third convolutonal layer backward propagate
	cudaMemcpy(GPU(convnet)->device[device_id].backwards[3], GPU(convnet)->device[device_id].forwards[2], sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch, cudaMemcpyDeviceToDevice);
	ccv_convnet_layer_t* third_gpu_configuration = GPU(convnet)->device[device_id].configurations + 2;
	EXTRA(third_gpu_layer)->vary.convolutional.backward.coefficient.x = 4;
	EXTRA(third_gpu_layer)->vary.convolutional.backward.coefficient.y = 4;
	EXTRA(third_gpu_layer)->vary.convolutional.backward.coefficient.z = 16;
	EXTRA(third_gpu_layer)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(third_gpu_layer)->vary.convolutional.backward.gradient.y = 6;
	EXTRA(third_gpu_layer)->vary.convolutional.backward.gradient.z = 24;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_backward_propagate(third_gpu_layer, batch, GPU(convnet)->device[device_id].backwards[3], GPU(convnet)->device[device_id].forwards[2], GPU(convnet)->device[device_id].forwards[1], GPU(convnet)->device[device_id].backwards[2], third_gpu_configuration, GPU(convnet)->device[device_id].scratch, GPU(convnet)->device[device_id].unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for third convolutional layer bprop: %f milliseconds\n", EXTRA(third_gpu_layer)->vary.convolutional.backward.coefficient.x, EXTRA(third_gpu_layer)->vary.convolutional.backward.coefficient.y, EXTRA(third_gpu_layer)->vary.convolutional.backward.coefficient.z, EXTRA(third_gpu_layer)->vary.convolutional.backward.gradient.x, EXTRA(third_gpu_layer)->vary.convolutional.backward.gradient.y, EXTRA(third_gpu_layer)->vary.convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* third_back = 0;
	cudaMallocHost(&third_back, sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch);
	cudaMemcpy(third_back, GPU(convnet)->device[device_id].backwards[2], sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch, cudaMemcpyDeviceToHost);
	float* third_grad = 0;
	cudaMallocHost(&third_grad, sizeof(float) * (third_gpu_layer->wnum + third_gpu_layer->net.convolutional.count));
	assert(third_grad);
	cudaMemcpy(third_grad, third_gpu_configuration->w, sizeof(float) * (third_gpu_layer->wnum + third_gpu_layer->net.convolutional.count), cudaMemcpyDeviceToHost);
	printf("finished backward propagate third convolutional layer on GPU\n");

	// second average pool layer backward propagate
	cwc_convnet_average_pool_backward_propagate(second_gpu_layer, batch, GPU(convnet)->device[device_id].backwards[2], GPU(convnet)->device[device_id].backwards[1], context->device[device_id].data_stream);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* second_back = 0;
	cudaMallocHost(&second_back, sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch);
	cudaMemcpy(second_back, GPU(convnet)->device[device_id].backwards[1], sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished backward propagate second average pool layer on GPU\n");

	// first convolutional layer backward propagate
	ccv_convnet_layer_t* first_gpu_configuration = GPU(convnet)->device[device_id].configurations;
	EXTRA(first_gpu_layer)->vary.convolutional.backward.coefficient.x = 1;
	EXTRA(first_gpu_layer)->vary.convolutional.backward.coefficient.y = 3;
	EXTRA(first_gpu_layer)->vary.convolutional.backward.coefficient.z = 1;
	cudaEventRecord(start, context->device[device_id].data_stream);
	cwc_convnet_convolutional_backward_propagate(first_gpu_layer, batch, GPU(convnet)->device[device_id].backwards[1], GPU(convnet)->device[device_id].forwards[0], context->device[device_id].input, GPU(convnet)->device[device_id].backwards[0], first_gpu_configuration, GPU(convnet)->device[device_id].scratch, GPU(convnet)->device[device_id].unit, context->device[device_id].data_stream, context->device[device_id].data_cublas);
	cudaEventRecord(stop, context->device[device_id].data_stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for first convolutional layer bprop: %f milliseconds\n", EXTRA(first_gpu_layer)->vary.convolutional.backward.coefficient.x, EXTRA(first_gpu_layer)->vary.convolutional.backward.coefficient.y, EXTRA(first_gpu_layer)->vary.convolutional.backward.coefficient.z, elapsed_time);
	cudaStreamSynchronize(context->device[device_id].data_stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* first_grad = 0;
	cudaMallocHost(&first_grad, sizeof(float) * (first_gpu_layer->wnum + first_gpu_layer->net.convolutional.count));
	assert(first_grad);
	cudaMemcpy(first_grad, first_gpu_configuration->w, sizeof(float) * (first_gpu_layer->wnum + first_gpu_layer->net.convolutional.count), cudaMemcpyDeviceToHost);
	printf("finished backward propagate first convolutional layer on GPU\n");
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
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
		// second average pool layer forward propagate
		ccv_convnet_layer_t* second_cpu_layer = convnet->layers + 1;
		_ccv_convnet_average_pool_forward_propagate(second_cpu_layer, convnet->acts[0], convnet->acts + 1);
		ccv_dense_matrix_t* b = convnet->acts[1];
		for (y = 0; y < second_out_rows; y++)
			for (x = 0; x < second_out_cols; x++)
				for (k = 0; k < second_out_channels; k++)
				{
					float p = second_out[k * second_out_rows * second_out_cols * batch + (y * second_out_cols + x) * batch + i];
					float q = b->data.f32[y * second_out_cols * second_out_channels + x * second_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("avgpool fprop 2: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// third convolutional layer forward propagate
		ccv_convnet_layer_t* third_cpu_layer = convnet->layers + 2;
		_ccv_convnet_convolutional_forward_propagate(third_cpu_layer, convnet->acts[1], convnet->acts + 2);
		ccv_dense_matrix_t* c = convnet->acts[2];
		for (y = 0; y < third_out_rows; y++)
			for (x = 0; x < third_out_cols; x++)
				for (k = 0; k < third_out_channels; k++)
				{
					float p = third_out[k * third_out_rows * third_out_cols * batch + (y * third_out_cols + x) * batch + i];
					float q = c->data.f32[(y * third_out_cols + x) * third_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv fprop 3: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// forth average pool layer forward propagate
		ccv_convnet_layer_t* forth_cpu_layer = convnet->layers + 3;
		_ccv_convnet_average_pool_forward_propagate(forth_cpu_layer, convnet->acts[2], convnet->acts + 3);
		ccv_dense_matrix_t* d = convnet->acts[3];
		for (y = 0; y < forth_out_rows; y++)
			for (x = 0; x < forth_out_cols; x++)
				for (k = 0; k < forth_out_channels; k++)
				{
					float p = forth_out[k * forth_out_rows * forth_out_cols * batch + (y * forth_out_cols + x) * batch + i];
					float q = d->data.f32[y * forth_out_cols * forth_out_channels + x * forth_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("avgpool fprop 4: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// fifth convolutional layer forward propagate
		ccv_convnet_layer_t* fifth_cpu_layer = convnet->layers + 4;
		_ccv_convnet_convolutional_forward_propagate(fifth_cpu_layer, convnet->acts[3], convnet->acts + 4);
		ccv_dense_matrix_t* e = convnet->acts[4];
		for (y = 0; y < fifth_out_rows; y++)
			for (x = 0; x < fifth_out_cols; x++)
				for (k = 0; k < fifth_out_channels; k++)
				{
					float p = fifth_out[k * fifth_out_rows * fifth_out_cols * batch + (y * fifth_out_cols + x) * batch + i];
					float q = e->data.f32[(y * fifth_out_cols + x) * fifth_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv fprop 5: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// sixth convolutional layer forward propagate
		ccv_convnet_layer_t* sixth_cpu_layer = convnet->layers + 5;
		_ccv_convnet_convolutional_forward_propagate(sixth_cpu_layer, convnet->acts[4], convnet->acts + 5);
		ccv_dense_matrix_t* f = convnet->acts[5];
		for (y = 0; y < sixth_out_rows; y++)
			for (x = 0; x < sixth_out_cols; x++)
				for (k = 0; k < sixth_out_channels; k++)
				{
					float p = sixth_out[k * sixth_out_rows * sixth_out_cols * batch + (y * sixth_out_cols + x) * batch + i];
					float q = f->data.f32[(y * sixth_out_cols + x) * sixth_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv fprop 6: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// seventh convolutional layer forward propagate
		ccv_convnet_layer_t* seventh_cpu_layer = convnet->layers + 6;
		_ccv_convnet_convolutional_forward_propagate(seventh_cpu_layer, convnet->acts[5], convnet->acts + 6);
		ccv_dense_matrix_t* g = convnet->acts[6];
		for (y = 0; y < seventh_out_rows; y++)
			for (x = 0; x < seventh_out_cols; x++)
				for (k = 0; k < seventh_out_channels; k++)
				{
					float p = seventh_out[k * seventh_out_rows * seventh_out_cols * batch + (y * seventh_out_cols + x) * batch + i];
					float q = g->data.f32[(y * seventh_out_cols + x) * seventh_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv fprop 7: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// eleventh full connect layer forward propagate
		ccv_convnet_layer_t* eleventh_cpu_layer = convnet->layers + 10;
		convnet->acts[9] = ccv_dense_matrix_new(eleventh_cpu_layer->input.node.count, 1, CCV_32F | CCV_C1, 0, 0);
		for (k = 0; k < eleventh_cpu_layer->input.node.count; k++)
			convnet->acts[9]->data.f32[k] = eleventh_in[k * batch + i];
		_ccv_convnet_full_connect_forward_propagate(eleventh_cpu_layer, convnet->acts[9], convnet->acts + 10);
		ccv_dense_matrix_t* z = convnet->acts[10];
		for (k = 0; k < eleventh_cpu_layer->net.full_connect.count; k++)
		{
			float p = eleventh_out[k * batch + i];
			float q = z->data.f32[k];
			float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
			if (delta > 1e-4)
				printf("fc fprop 11: %d %d: |%g - %g| = %g\n", i, k, p, q, delta);
		}
		_ccv_convnet_full_connect_backward_propagate(eleventh_cpu_layer, convnet->acts[10], convnet->acts[10], convnet->acts[9], update_params->acts + 9, update_params->layers + 10);
		ccv_matrix_free(convnet->acts[9]);
		ccv_dense_matrix_t* bz = update_params->acts[9];
		for (k = 0; k < eleventh_cpu_layer->input.node.count; k++)
		{
			float p = eleventh_back[k * batch + i];
			float q = bz->data.f32[k];
			float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
			if (delta > 1e-4)
				printf("fc bprop 11: %d %d: |%g - %g| = %g\n", i, k, p, q, delta);
		}

		// seventh convolutional layer backward propagate
		_ccv_convnet_convolutional_backward_propagate(seventh_cpu_layer, convnet->acts[6], convnet->acts[6], convnet->acts[5], update_params->acts + 5, update_params->layers + 6);
		ccv_dense_matrix_t* bg = update_params->acts[5];
		for (y = 0; y < sixth_out_rows; y++)
			for (x = 0; x < sixth_out_cols; x++)
				for (k = 0; k < sixth_out_channels; k++)
				{
					float p = seventh_back[k * sixth_out_rows * sixth_out_cols * batch + (y * sixth_out_cols + x) * batch + i];
					float q = bg->data.f32[(y * sixth_out_cols + x) * sixth_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv bprop 7: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// sixth convolutional layer backward propagate
		_ccv_convnet_convolutional_backward_propagate(sixth_cpu_layer, update_params->acts[5], convnet->acts[5], convnet->acts[4], update_params->acts + 4, update_params->layers + 5);
		ccv_dense_matrix_t* bf = update_params->acts[4];
		for (y = 0; y < fifth_out_rows; y++)
			for (x = 0; x < fifth_out_cols; x++)
				for (k = 0; k < fifth_out_channels; k++)
				{
					float p = sixth_back[k * fifth_out_rows * fifth_out_cols * batch + (y * fifth_out_cols + x) * batch + i];
					float q = bf->data.f32[(y * fifth_out_cols + x) * fifth_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-3)
						printf("conv bprop 6: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// fifth convolutional layer backward propagate
		_ccv_convnet_convolutional_backward_propagate(fifth_cpu_layer, update_params->acts[4], convnet->acts[4], convnet->acts[3], update_params->acts + 3, update_params->layers + 4);
		ccv_dense_matrix_t* be = update_params->acts[3];
		for (y = 0; y < forth_out_rows; y++)
			for (x = 0; x < forth_out_cols; x++)
				for (k = 0; k < forth_out_channels; k++)
				{
					float p = fifth_back[k * forth_out_rows * forth_out_cols * batch + (y * forth_out_cols + x) * batch + i];
					float q = be->data.f32[(y * forth_out_cols + x) * forth_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-2)
						printf("conv bprop 5: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// third convolutional layer backward propagate
		_ccv_convnet_convolutional_backward_propagate(third_cpu_layer, convnet->acts[2], convnet->acts[2], convnet->acts[1], update_params->acts + 1, update_params->layers + 2);
		ccv_dense_matrix_t* bc = update_params->acts[1];
		for (y = 0; y < second_out_rows; y++)
			for (x = 0; x < second_out_cols; x++)
				for (k = 0; k < second_out_channels; k++)
				{
					float p = third_back[k * second_out_rows * second_out_cols * batch + (y * second_out_cols + x) * batch + i];
					float q = bc->data.f32[(y * second_out_cols + x) * second_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv bprop 3: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}
		// second average pool layer backward propagate
		_ccv_convnet_average_pool_backward_propagate(second_cpu_layer, update_params->acts[1], convnet->acts[0], update_params->acts);
		ccv_dense_matrix_t* bb = update_params->acts[0];
		for (y = 0; y < first_out_rows; y++)
			for (x = 0; x < first_out_cols; x++)
				for (k = 0; k < first_out_channels; k++)
				{
					float p = second_back[k * first_out_rows * first_out_cols * batch + (y * first_out_cols + x) * batch + i];
					float q = bb->data.f32[y * first_out_cols * first_out_channels + x * first_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("avgpool bprop 2: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}

		// first convolutional layer backward propagate
		_ccv_convnet_convolutional_backward_propagate(first_cpu_layer, update_params->acts[0], convnet->acts[0], categorized->matrix, 0, update_params->layers);
	}

	ccv_convnet_layer_t* eleventh_cpu_configuration = update_params->layers + 10;
	for (x = 0; x < eleventh_cpu_configuration->net.full_connect.count; x++)
		for (y = 0; y < eleventh_cpu_configuration->input.node.count; y++)
		{
			float p = eleventh_cpu_configuration->w[x * eleventh_cpu_configuration->input.node.count + y];
			float q = eleventh_grad[x * eleventh_cpu_configuration->input.node.count + y];
			float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
			if (delta > 1e-3)
				printf("fc bprop 11: %d %d: |%g - %g| = %g\n", x, y, p, q, delta);
		}
	for (x = 0; x < eleventh_cpu_configuration->net.full_connect.count; x++)
	{
		float p = eleventh_cpu_configuration->bias[x];
		float q = eleventh_grad[eleventh_cpu_configuration->net.full_connect.count * eleventh_cpu_configuration->input.node.count + x];
		float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
		if (delta > 1e-3)
			printf("fc bprop 11 bias: %d: |%g - %g| = %g\n", x, p, q, delta);
	}

	ccv_convnet_layer_t* seventh_cpu_configuration = update_params->layers + 6;
	int seventh_filter_rows = seventh_gpu_layer->net.convolutional.rows;
	int seventh_filter_cols = seventh_gpu_layer->net.convolutional.cols;
	int seventh_filter_count = seventh_gpu_layer->net.convolutional.count;
	int seventh_filter_channels = seventh_gpu_layer->net.convolutional.channels / 2;
	for (y = 0; y < seventh_filter_rows; y++)
		for (x = 0; x < seventh_filter_cols; x++)
			for (k = 0; k < seventh_filter_count; k++)
				for (c = 0; c < seventh_filter_channels; c++)
				{
					float p = seventh_cpu_configuration->w[(y * seventh_filter_cols + x) * seventh_filter_channels + k * seventh_filter_cols * seventh_filter_rows * seventh_filter_channels + c];
					float q = seventh_grad[(y * seventh_filter_cols + x) * seventh_filter_count + k + c * seventh_filter_cols * seventh_filter_rows * seventh_filter_count];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv bprop 7: %d %d %d %d: |%g - %g| = %g\n", x, y, k, c, p, q, delta);
				}
	for (k = 0; k < seventh_filter_count; k++)
	{
		float p = seventh_cpu_configuration->bias[k];
		float q = seventh_grad[seventh_gpu_layer->wnum + k];
		float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
		if (delta > 1e-4)
			printf("conv bprop 7 bias: %d: |%g - %g| = %g\n", k, p, q, delta);
	}

	ccv_convnet_layer_t* sixth_cpu_configuration = update_params->layers + 5;
	int sixth_filter_rows = sixth_gpu_layer->net.convolutional.rows;
	int sixth_filter_cols = sixth_gpu_layer->net.convolutional.cols;
	int sixth_filter_count = sixth_gpu_layer->net.convolutional.count;
	int sixth_filter_channels = sixth_gpu_layer->net.convolutional.channels / 2;
	for (y = 0; y < sixth_filter_rows; y++)
		for (x = 0; x < sixth_filter_cols; x++)
			for (k = 0; k < sixth_filter_count; k++)
				for (c = 0; c < sixth_filter_channels; c++)
				{
					float p = sixth_cpu_configuration->w[(y * sixth_filter_cols + x) * sixth_filter_channels + k * sixth_filter_cols * sixth_filter_rows * sixth_filter_channels + c];
					float q = sixth_grad[(y * sixth_filter_cols + x) * sixth_filter_count + k + c * sixth_filter_cols * sixth_filter_rows * sixth_filter_count];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-3)
						printf("conv bprop 6: %d %d %d %d: |%g - %g| = %g\n", x, y, k, c, p, q, delta);
				}
	for (k = 0; k < sixth_filter_count; k++)
	{
		float p = sixth_cpu_configuration->bias[k];
		float q = sixth_grad[sixth_gpu_layer->wnum + k];
		float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
		if (delta > 1e-4)
			printf("conv bprop 6 bias: %d: |%g - %g| = %g\n", k, p, q, delta);
	}

	ccv_convnet_layer_t* fifth_cpu_configuration = update_params->layers + 4;
	int fifth_filter_rows = fifth_gpu_layer->net.convolutional.rows;
	int fifth_filter_cols = fifth_gpu_layer->net.convolutional.cols;
	int fifth_filter_count = fifth_gpu_layer->net.convolutional.count;
	int fifth_filter_channels = fifth_gpu_layer->net.convolutional.channels;
	for (y = 0; y < fifth_filter_rows; y++)
		for (x = 0; x < fifth_filter_cols; x++)
			for (k = 0; k < fifth_filter_count; k++)
				for (c = 0; c < fifth_filter_channels; c++)
				{
					float p = fifth_cpu_configuration->w[(y * fifth_filter_cols + x) * fifth_filter_channels + k * fifth_filter_cols * fifth_filter_rows * fifth_filter_channels + c];
					float q = fifth_grad[(y * fifth_filter_cols + x) * fifth_filter_count + k + c * fifth_filter_cols * fifth_filter_rows * fifth_filter_count];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-2)
						printf("conv bprop 5: %d %d %d %d: |%g - %g| = %g\n", x, y, k, c, p, q, delta);
				}
	for (k = 0; k < fifth_filter_count; k++)
	{
		float p = fifth_cpu_configuration->bias[k];
		float q = fifth_grad[fifth_gpu_layer->wnum + k];
		float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
		if (delta > 1e-4)
			printf("conv bprop 5 bias: %d: |%g - %g| = %g\n", k, p, q, delta);
	}

	ccv_convnet_layer_t* third_cpu_configuration = update_params->layers + 2;
	int third_filter_rows = third_gpu_layer->net.convolutional.rows;
	int third_filter_cols = third_gpu_layer->net.convolutional.cols;
	int third_filter_count = third_gpu_layer->net.convolutional.count;
	int third_filter_channels = third_gpu_layer->net.convolutional.channels / 2;
	for (y = 0; y < third_filter_rows; y++)
		for (x = 0; x < third_filter_cols; x++)
			for (k = 0; k < third_filter_count; k++)
				for (c = 0; c < third_filter_channels; c++)
				{
					float p = third_cpu_configuration->w[(y * third_filter_cols + x) * third_filter_channels + k * third_filter_cols * third_filter_rows * third_filter_channels + c];
					float q = third_grad[(y * third_filter_cols + x) * third_filter_count + k + c * third_filter_cols * third_filter_rows * third_filter_count];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv bprop 3: %d %d %d %d: |%g - %g| = %g\n", x, y, k, c, p, q, delta);
				}
	for (k = 0; k < third_filter_count; k++)
	{
		float p = third_cpu_configuration->bias[k];
		float q = third_grad[third_gpu_layer->wnum + k];
		float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
		if (delta > 1e-4)
			printf("conv bprop 3 bias: %d: |%g - %g| = %g\n", k, p, q, delta);
	}

	ccv_convnet_layer_t* first_cpu_configuration = update_params->layers;
	int first_filter_rows = first_gpu_layer->net.convolutional.rows;
	int first_filter_cols = first_gpu_layer->net.convolutional.cols;
	int first_filter_count = first_gpu_layer->net.convolutional.count;
	int first_filter_channels = first_gpu_layer->net.convolutional.channels;
	for (y = 0; y < first_filter_rows; y++)
		for (x = 0; x < first_filter_cols; x++)
			for (k = 0; k < first_filter_count; k++)
				for (c = 0; c < first_filter_channels; c++)
				{
					float p = first_cpu_configuration->w[(y * first_filter_cols + x) * first_filter_channels + k * first_filter_cols * first_filter_rows * first_filter_channels + c];
					float q = first_grad[(y * first_filter_cols + x) * first_filter_count + k + c * first_filter_cols * first_filter_rows * first_filter_count];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-3)
						printf("conv bprop 1: %d %d %d %d: |%g - %g| = %g\n", x, y, k, c, p, q, delta);
				}
	for (k = 0; k < first_filter_count; k++)
	{
		float p = first_cpu_configuration->bias[k];
		float q = first_grad[first_gpu_layer->wnum + k];
		float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
		if (delta > 1e-4)
			printf("conv bprop 1 bias: %d: |%g - %g| = %g\n", k, p, q, delta);
	}
	cudaFreeHost(eleventh_in);
}
