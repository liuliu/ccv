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
	_cwc_convnet_alloc_reserved_both(convnet, batch, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	for (i = 0; i < convnet->rows * convnet->cols * convnet->channels; i++)
		convnet->mean_activity->data.f32[i] = 128;
	_cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, ccv_size(225, 225), convnet->rows, convnet->cols, convnet->channels, 1000, 0, batch, 0, batch, context->host.input, context->host.c);
	cudaMemcpy(context->device.input, context->host.input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch, cudaMemcpyHostToDevice);

	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);

	// first convolutional layer forward propagate
	ccv_convnet_layer_t* first_gpu_layer = GPU(convnet)->layers;
	// these are the setups for TITAN, thus, skip the benching phase
	VARY(first_gpu_layer)->convolutional.forward.x = 4;
	VARY(first_gpu_layer)->convolutional.forward.y = 8;
	VARY(first_gpu_layer)->convolutional.forward.z = 32;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_forward_propagate(first_gpu_layer, first_gpu_layer->input.matrix.rows, first_gpu_layer->input.matrix.cols, batch, context->device.input, GPU(convnet)->forwards[0], context->device.stream);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	float elapsed_time = 0;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaStreamSynchronize(context->device.stream);
	printf("%d %d %d, elapsed time for first convolutional layer fprop: %f milliseconds\n", VARY(first_gpu_layer)->convolutional.forward.x, VARY(first_gpu_layer)->convolutional.forward.y, VARY(first_gpu_layer)->convolutional.forward.z, elapsed_time);
	int first_out_rows, first_out_cols, first_out_partition, first_out_channels = first_gpu_layer->net.convolutional.count;
	_ccv_convnet_layer_derive_output(first_gpu_layer, first_gpu_layer->input.matrix.rows, first_gpu_layer->input.matrix.cols, &first_out_rows, &first_out_cols, &first_out_partition);
	float* first_out = 0;
	cudaMallocHost(&first_out, sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch);
	cudaMemcpy(first_out, GPU(convnet)->forwards[0], sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate first convolutional layer on GPU\n");

	// second average pool layer forward propagate
	ccv_convnet_layer_t* second_gpu_layer = GPU(convnet)->layers + 1;
	_cwc_convnet_average_pool_forward_propagate(second_gpu_layer, second_gpu_layer->input.matrix.rows, second_gpu_layer->input.matrix.cols, batch, GPU(convnet)->forwards[0], GPU(convnet)->forwards[1], context->device.stream);
	cudaStreamSynchronize(context->device.stream);
	int second_out_rows, second_out_cols, second_out_partition, second_out_channels = second_gpu_layer->input.matrix.channels;
	_ccv_convnet_layer_derive_output(second_gpu_layer, second_gpu_layer->input.matrix.rows, second_gpu_layer->input.matrix.cols, &second_out_rows, &second_out_cols, &second_out_partition);
	float* second_out = 0;
	cudaMallocHost(&second_out, sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch);
	cudaMemcpy(second_out, GPU(convnet)->forwards[1], sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate second average pool layer on GPU\n");

	// third convolutional layer forward propagate
	ccv_convnet_layer_t* third_gpu_layer = GPU(convnet)->layers + 2;
	// these are the setups for TITAN, thus, skip the benching phase
	VARY(third_gpu_layer)->convolutional.forward.x = 4;
	VARY(third_gpu_layer)->convolutional.forward.y = 8;
	VARY(third_gpu_layer)->convolutional.forward.z = 32;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_forward_propagate(third_gpu_layer, third_gpu_layer->input.matrix.rows, third_gpu_layer->input.matrix.cols, batch, GPU(convnet)->forwards[1], GPU(convnet)->forwards[2], context->device.stream);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for third convolutional layer fprop: %f milliseconds\n", VARY(third_gpu_layer)->convolutional.forward.x, VARY(third_gpu_layer)->convolutional.forward.y, VARY(third_gpu_layer)->convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	int third_out_rows, third_out_cols, third_out_partition, third_out_channels = third_gpu_layer->net.convolutional.count;
	_ccv_convnet_layer_derive_output(third_gpu_layer, third_gpu_layer->input.matrix.rows, third_gpu_layer->input.matrix.cols, &third_out_rows, &third_out_cols, &third_out_partition);
	float* third_out = 0;
	cudaMallocHost(&third_out, sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch);
	cudaMemcpy(third_out, GPU(convnet)->forwards[2], sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate third convolutional layer on GPU\n");

	// forth average pool layer forward propagate
	ccv_convnet_layer_t* forth_gpu_layer = GPU(convnet)->layers + 3;
	_cwc_convnet_average_pool_forward_propagate(forth_gpu_layer, forth_gpu_layer->input.matrix.rows, forth_gpu_layer->input.matrix.cols, batch, GPU(convnet)->forwards[2], GPU(convnet)->forwards[3], context->device.stream);
	cudaStreamSynchronize(context->device.stream);
	int forth_out_rows, forth_out_cols, forth_out_partition, forth_out_channels = forth_gpu_layer->input.matrix.channels;
	_ccv_convnet_layer_derive_output(forth_gpu_layer, forth_gpu_layer->input.matrix.rows, forth_gpu_layer->input.matrix.cols, &forth_out_rows, &forth_out_cols, &forth_out_partition);
	float* forth_out = 0;
	cudaMallocHost(&forth_out, sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch);
	cudaMemcpy(forth_out, GPU(convnet)->forwards[3], sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate forth average pool layer on GPU\n");

	// fifth convolutional layer forward propagate
	ccv_convnet_layer_t* fifth_gpu_layer = GPU(convnet)->layers + 4;
	// these are the setups for TITAN, thus, skip the benching phase
	VARY(fifth_gpu_layer)->convolutional.forward.x = 4;
	VARY(fifth_gpu_layer)->convolutional.forward.y = 8;
	VARY(fifth_gpu_layer)->convolutional.forward.z = 32;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_forward_propagate(fifth_gpu_layer, fifth_gpu_layer->input.matrix.rows, fifth_gpu_layer->input.matrix.cols, batch, GPU(convnet)->forwards[3], GPU(convnet)->forwards[4], context->device.stream);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for fifth convolutional layer fprop: %f milliseconds\n", VARY(fifth_gpu_layer)->convolutional.forward.x, VARY(fifth_gpu_layer)->convolutional.forward.y, VARY(fifth_gpu_layer)->convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	int fifth_out_rows, fifth_out_cols, fifth_out_partition, fifth_out_channels = fifth_gpu_layer->net.convolutional.count;
	_ccv_convnet_layer_derive_output(fifth_gpu_layer, fifth_gpu_layer->input.matrix.rows, fifth_gpu_layer->input.matrix.cols, &fifth_out_rows, &fifth_out_cols, &fifth_out_partition);
	float* fifth_out = 0;
	cudaMallocHost(&fifth_out, sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch);
	cudaMemcpy(fifth_out, GPU(convnet)->forwards[4], sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate fifth convolutional layer on GPU\n");

	// sixth convolutional layer forward propagate
	ccv_convnet_layer_t* sixth_gpu_layer = GPU(convnet)->layers + 5;
	// these are the setups for TITAN, thus, skip the benching phase
	VARY(sixth_gpu_layer)->convolutional.forward.x = 4;
	VARY(sixth_gpu_layer)->convolutional.forward.y = 8;
	VARY(sixth_gpu_layer)->convolutional.forward.z = 32;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_forward_propagate(sixth_gpu_layer, sixth_gpu_layer->input.matrix.rows, sixth_gpu_layer->input.matrix.cols, batch, GPU(convnet)->forwards[4], GPU(convnet)->forwards[5], context->device.stream);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for sixth convolutional layer fprop: %f milliseconds\n", VARY(sixth_gpu_layer)->convolutional.forward.x, VARY(sixth_gpu_layer)->convolutional.forward.y, VARY(sixth_gpu_layer)->convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	int sixth_out_rows, sixth_out_cols, sixth_out_partition, sixth_out_channels = sixth_gpu_layer->net.convolutional.count;
	_ccv_convnet_layer_derive_output(sixth_gpu_layer, sixth_gpu_layer->input.matrix.rows, sixth_gpu_layer->input.matrix.cols, &sixth_out_rows, &sixth_out_cols, &sixth_out_partition);
	float* sixth_out = 0;
	cudaMallocHost(&sixth_out, sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch);
	cudaMemcpy(sixth_out, GPU(convnet)->forwards[5], sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate sixth convolutional layer on GPU\n");

	// seventh convolutional layer forward propagate
	ccv_convnet_layer_t* seventh_gpu_layer = GPU(convnet)->layers + 6;
	// these are the setups for TITAN, thus, skip the benching phase
	VARY(seventh_gpu_layer)->convolutional.forward.x = 4;
	VARY(seventh_gpu_layer)->convolutional.forward.y = 8;
	VARY(seventh_gpu_layer)->convolutional.forward.z = 32;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_forward_propagate(seventh_gpu_layer, seventh_gpu_layer->input.matrix.rows, seventh_gpu_layer->input.matrix.cols, batch, GPU(convnet)->forwards[5], GPU(convnet)->forwards[6], context->device.stream);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for seventh convolutional layer fprop: %f milliseconds\n", VARY(seventh_gpu_layer)->convolutional.forward.x, VARY(seventh_gpu_layer)->convolutional.forward.y, VARY(seventh_gpu_layer)->convolutional.forward.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	int seventh_out_rows, seventh_out_cols, seventh_out_partition, seventh_out_channels = seventh_gpu_layer->net.convolutional.count;
	_ccv_convnet_layer_derive_output(seventh_gpu_layer, seventh_gpu_layer->input.matrix.rows, seventh_gpu_layer->input.matrix.cols, &seventh_out_rows, &seventh_out_cols, &seventh_out_partition);
	float* seventh_out = 0;
	cudaMallocHost(&seventh_out, sizeof(float) * seventh_out_rows * seventh_out_cols * seventh_out_channels * batch);
	cudaMemcpy(seventh_out, GPU(convnet)->forwards[6], sizeof(float) * seventh_out_rows * seventh_out_cols * seventh_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate seventh convolutional layer on GPU\n");

	// seventh convolutonal layer backward propagate
	cudaMemcpy(GPU(convnet)->backwards[7], GPU(convnet)->forwards[6], sizeof(float) * seventh_out_rows * seventh_out_cols * seventh_out_channels * batch, cudaMemcpyDeviceToDevice);
	ccv_convnet_layer_t* seventh_gpu_configuration = GPU(convnet)->configurations + 6;
	VARY(seventh_gpu_layer)->convolutional.backward.coefficient.x = 8;
	VARY(seventh_gpu_layer)->convolutional.backward.coefficient.y = 4;
	VARY(seventh_gpu_layer)->convolutional.backward.coefficient.z = 32;
	VARY(seventh_gpu_layer)->convolutional.backward.gradient.x = 4;
	VARY(seventh_gpu_layer)->convolutional.backward.gradient.y = 8;
	VARY(seventh_gpu_layer)->convolutional.backward.gradient.z = 32;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_backward_propagate(seventh_gpu_layer, batch, GPU(convnet)->backwards[7], GPU(convnet)->forwards[6], GPU(convnet)->forwards[5], GPU(convnet)->backwards[6], seventh_gpu_configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for seventh convolutional layer bprop: %f milliseconds\n", VARY(seventh_gpu_layer)->convolutional.backward.coefficient.x, VARY(seventh_gpu_layer)->convolutional.backward.coefficient.y, VARY(seventh_gpu_layer)->convolutional.backward.coefficient.z, VARY(seventh_gpu_layer)->convolutional.backward.gradient.x, VARY(seventh_gpu_layer)->convolutional.backward.gradient.y, VARY(seventh_gpu_layer)->convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* seventh_back = 0;
	cudaMallocHost(&seventh_back, sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch);
	cudaMemcpy(seventh_back, GPU(convnet)->backwards[6], sizeof(float) * sixth_out_rows * sixth_out_cols * sixth_out_channels * batch, cudaMemcpyDeviceToHost);
	float* seventh_grad = 0;
	cudaMallocHost(&seventh_grad, sizeof(float) * seventh_gpu_layer->wnum);
	assert(seventh_grad);
	cudaMemcpy(seventh_grad, seventh_gpu_configuration->w, sizeof(float) * seventh_gpu_layer->wnum, cudaMemcpyDeviceToHost);
	printf("finished backward propagate seventh convolutional layer on GPU\n");

	// sixth convolutonal layer backward propagate
	ccv_convnet_layer_t* sixth_gpu_configuration = GPU(convnet)->configurations + 5;
	VARY(sixth_gpu_layer)->convolutional.backward.coefficient.x = 8;
	VARY(sixth_gpu_layer)->convolutional.backward.coefficient.y = 3;
	VARY(sixth_gpu_layer)->convolutional.backward.coefficient.z = 32;
	VARY(sixth_gpu_layer)->convolutional.backward.gradient.x = 4;
	VARY(sixth_gpu_layer)->convolutional.backward.gradient.y = 8;
	VARY(sixth_gpu_layer)->convolutional.backward.gradient.z = 32;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_backward_propagate(sixth_gpu_layer, batch, GPU(convnet)->backwards[6], GPU(convnet)->forwards[5], GPU(convnet)->forwards[4], GPU(convnet)->backwards[5], sixth_gpu_configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for sixth convolutional layer bprop: %f milliseconds\n", VARY(sixth_gpu_layer)->convolutional.backward.coefficient.x, VARY(sixth_gpu_layer)->convolutional.backward.coefficient.y, VARY(sixth_gpu_layer)->convolutional.backward.coefficient.z, VARY(sixth_gpu_layer)->convolutional.backward.gradient.x, VARY(sixth_gpu_layer)->convolutional.backward.gradient.y, VARY(sixth_gpu_layer)->convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* sixth_back = 0;
	cudaMallocHost(&sixth_back, sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch);
	cudaMemcpy(sixth_back, GPU(convnet)->backwards[5], sizeof(float) * fifth_out_rows * fifth_out_cols * fifth_out_channels * batch, cudaMemcpyDeviceToHost);
	float* sixth_grad = 0;
	cudaMallocHost(&sixth_grad, sizeof(float) * sixth_gpu_layer->wnum);
	assert(sixth_grad);
	cudaMemcpy(sixth_grad, sixth_gpu_configuration->w, sizeof(float) * sixth_gpu_layer->wnum, cudaMemcpyDeviceToHost);
	printf("finished backward propagate sixth convolutional layer on GPU\n");

	// fifth convolutonal layer backward propagate
	ccv_convnet_layer_t* fifth_gpu_configuration = GPU(convnet)->configurations + 4;
	VARY(fifth_gpu_layer)->convolutional.backward.coefficient.x = 8;
	VARY(fifth_gpu_layer)->convolutional.backward.coefficient.y = 3;
	VARY(fifth_gpu_layer)->convolutional.backward.coefficient.z = 32;
	VARY(fifth_gpu_layer)->convolutional.backward.gradient.x = 4;
	VARY(fifth_gpu_layer)->convolutional.backward.gradient.y = 8;
	VARY(fifth_gpu_layer)->convolutional.backward.gradient.z = 32;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_backward_propagate(fifth_gpu_layer, batch, GPU(convnet)->backwards[5], GPU(convnet)->forwards[4], GPU(convnet)->forwards[3], GPU(convnet)->backwards[4], fifth_gpu_configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for fifth convolutional layer bprop: %f milliseconds\n", VARY(fifth_gpu_layer)->convolutional.backward.coefficient.x, VARY(fifth_gpu_layer)->convolutional.backward.coefficient.y, VARY(fifth_gpu_layer)->convolutional.backward.coefficient.z, VARY(fifth_gpu_layer)->convolutional.backward.gradient.x, VARY(fifth_gpu_layer)->convolutional.backward.gradient.y, VARY(fifth_gpu_layer)->convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* fifth_back = 0;
	cudaMallocHost(&fifth_back, sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch);
	cudaMemcpy(fifth_back, GPU(convnet)->backwards[4], sizeof(float) * forth_out_rows * forth_out_cols * forth_out_channels * batch, cudaMemcpyDeviceToHost);
	float* fifth_grad = 0;
	cudaMallocHost(&fifth_grad, sizeof(float) * fifth_gpu_layer->wnum);
	assert(fifth_grad);
	cudaMemcpy(fifth_grad, fifth_gpu_configuration->w, sizeof(float) * fifth_gpu_layer->wnum, cudaMemcpyDeviceToHost);
	printf("finished backward propagate fifth convolutional layer on GPU\n");

	// third convolutonal layer backward propagate
	cudaMemcpy(GPU(convnet)->backwards[3], GPU(convnet)->forwards[2], sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch, cudaMemcpyDeviceToDevice);
	ccv_convnet_layer_t* third_gpu_configuration = GPU(convnet)->configurations + 2;
	VARY(third_gpu_layer)->convolutional.backward.coefficient.x = 4;
	VARY(third_gpu_layer)->convolutional.backward.coefficient.y = 4;
	VARY(third_gpu_layer)->convolutional.backward.coefficient.z = 16;
	VARY(third_gpu_layer)->convolutional.backward.gradient.x = 4;
	VARY(third_gpu_layer)->convolutional.backward.gradient.y = 6;
	VARY(third_gpu_layer)->convolutional.backward.gradient.z = 24;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_backward_propagate(third_gpu_layer, batch, GPU(convnet)->backwards[3], GPU(convnet)->forwards[2], GPU(convnet)->forwards[1], GPU(convnet)->backwards[2], third_gpu_configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, %d %d %d, elapsed time for third convolutional layer bprop: %f milliseconds\n", VARY(third_gpu_layer)->convolutional.backward.coefficient.x, VARY(third_gpu_layer)->convolutional.backward.coefficient.y, VARY(third_gpu_layer)->convolutional.backward.coefficient.z, VARY(third_gpu_layer)->convolutional.backward.gradient.x, VARY(third_gpu_layer)->convolutional.backward.gradient.y, VARY(third_gpu_layer)->convolutional.backward.gradient.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* third_back = 0;
	cudaMallocHost(&third_back, sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch);
	cudaMemcpy(third_back, GPU(convnet)->backwards[2], sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch, cudaMemcpyDeviceToHost);
	float* third_grad = 0;
	cudaMallocHost(&third_grad, sizeof(float) * third_gpu_layer->wnum);
	assert(third_grad);
	cudaMemcpy(third_grad, third_gpu_configuration->w, sizeof(float) * third_gpu_layer->wnum, cudaMemcpyDeviceToHost);
	printf("finished backward propagate third convolutional layer on GPU\n");

	// second average pool layer backward propagate
	_cwc_convnet_average_pool_backward_propagate(second_gpu_layer, batch, GPU(convnet)->backwards[2], GPU(convnet)->backwards[1], context->device.stream);
	cudaStreamSynchronize(context->device.stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* second_back = 0;
	cudaMallocHost(&second_back, sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch);
	cudaMemcpy(second_back, GPU(convnet)->backwards[1], sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished backward propagate second average pool layer on GPU\n");

	// first convolutional layer backward propagate
	ccv_convnet_layer_t* first_gpu_configuration = GPU(convnet)->configurations;
	VARY(first_gpu_layer)->convolutional.backward.coefficient.x = 1;
	VARY(first_gpu_layer)->convolutional.backward.coefficient.y = 3;
	VARY(first_gpu_layer)->convolutional.backward.coefficient.z = 1;
	cudaEventRecord(start, context->device.stream);
	_cwc_convnet_convolutional_backward_propagate(first_gpu_layer, batch, GPU(convnet)->backwards[1], GPU(convnet)->forwards[0], context->device.input, GPU(convnet)->backwards[0], first_gpu_configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
	cudaEventRecord(stop, context->device.stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("%d %d %d, elapsed time for first convolutional layer bprop: %f milliseconds\n", VARY(first_gpu_layer)->convolutional.backward.coefficient.x, VARY(first_gpu_layer)->convolutional.backward.coefficient.y, VARY(first_gpu_layer)->convolutional.backward.coefficient.z, elapsed_time);
	cudaStreamSynchronize(context->device.stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* first_grad = 0;
	cudaMallocHost(&first_grad, sizeof(float) * first_gpu_layer->wnum);
	assert(first_grad);
	cudaMemcpy(first_grad, first_gpu_configuration->w, sizeof(float) * first_gpu_layer->wnum, cudaMemcpyDeviceToHost);
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
}
