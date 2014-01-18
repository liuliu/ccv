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
	_cwc_convnet_reserve_onto_device(convnet, batch, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	_cwc_convnet_batch_formation(categorizeds, 0, convnet->rows, convnet->cols, convnet->channels, batch, 0, batch, context->host.input, context->host.c);
	cudaMemcpy(context->device.input, context->host.input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch, cudaMemcpyHostToDevice);

	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);

	// first convolutional layer forward propagate
	ccv_convnet_layer_t* first_gpu_layer = GPU(convnet)->layers;
	_cwc_convnet_convolutional_forward_propagate(first_gpu_layer, batch, context->device.input, GPU(convnet)->forwards[0], context->device.stream);
	cudaStreamSynchronize(context->device.stream);
	int first_out_rows, first_out_cols, first_out_channels = first_gpu_layer->net.convolutional.count;
	_cwc_convnet_layer_deduce_output_format(first_gpu_layer, &first_out_rows, &first_out_cols);
	float* first_out = 0;
	cudaMallocHost(&first_out, sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch);
	cudaMemcpy(first_out, GPU(convnet)->forwards[0], sizeof(float) * first_out_rows * first_out_cols * first_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate first convolutional layer on GPU\n");

	// second average pool layer forward propagate
	ccv_convnet_layer_t* second_gpu_layer = GPU(convnet)->layers + 1;
	_cwc_convnet_average_pool_forward_propagate(second_gpu_layer, batch,  GPU(convnet)->forwards[0], GPU(convnet)->forwards[1], context->device.stream);
	cudaStreamSynchronize(context->device.stream);
	int second_out_rows, second_out_cols, second_out_channels = second_gpu_layer->input.matrix.channels;
	_cwc_convnet_layer_deduce_output_format(second_gpu_layer, &second_out_rows, &second_out_cols);
	float* second_out = 0;
	cudaMallocHost(&second_out, sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch);
	cudaMemcpy(second_out, GPU(convnet)->forwards[1], sizeof(float) * second_out_rows * second_out_cols * second_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate second average pool layer on GPU\n");

	// third convolutional layer forward propagate
	ccv_convnet_layer_t* third_gpu_layer = GPU(convnet)->layers + 2;
	_cwc_convnet_convolutional_forward_propagate(third_gpu_layer, batch, GPU(convnet)->forwards[1], GPU(convnet)->forwards[2], context->device.stream);
	cudaStreamSynchronize(context->device.stream);
	int third_out_rows, third_out_cols, third_out_channels = third_gpu_layer->net.convolutional.count;
	_cwc_convnet_layer_deduce_output_format(third_gpu_layer, &third_out_rows, &third_out_cols);
	float* third_out = 0;
	cudaMallocHost(&third_out, sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch);
	cudaMemcpy(third_out, GPU(convnet)->forwards[2], sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch, cudaMemcpyDeviceToHost);
	printf("finished forward propagate third convolutional layer on GPU\n");

	// third convolutonal layer backward propagate
	cudaMemcpy(GPU(convnet)->backwards[3], GPU(convnet)->forwards[2], sizeof(float) * third_out_rows * third_out_cols * third_out_channels * batch, cudaMemcpyDeviceToDevice);
	ccv_convnet_layer_t* third_gpu_configuration = GPU(convnet)->configurations + 2;
	_cwc_convnet_convolutional_backward_propagate(third_gpu_layer, batch, GPU(convnet)->backwards[3], GPU(convnet)->forwards[2], GPU(convnet)->forwards[1], GPU(convnet)->backwards[2], third_gpu_configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
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
	_cwc_convnet_convolutional_backward_propagate(first_gpu_layer, batch, GPU(convnet)->backwards[1], GPU(convnet)->forwards[0], context->device.input, GPU(convnet)->backwards[0], first_gpu_configuration, GPU(convnet)->scratch, GPU(convnet)->unit, context->device.stream, context->device.cublas);
	cudaStreamSynchronize(context->device.stream);
	assert(cudaGetLastError() == cudaSuccess);
	float* first_grad = 0;
	cudaMallocHost(&first_grad, sizeof(float) * first_gpu_layer->wnum);
	assert(first_grad);
	cudaMemcpy(first_grad, first_gpu_configuration->w, sizeof(float) * first_gpu_layer->wnum, cudaMemcpyDeviceToHost);
	printf("finished backward propagate first convolutional layer on GPU\n");

	int i, x, y, k, c;
	for (i = 0; i < batch; i++)
	{
		printf("doing batch %d of %d\n", i + 1, batch);
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);

		// first convolutional layer forward propagate
		ccv_convnet_layer_t* first_cpu_layer = convnet->layers;
		_ccv_convnet_convolutional_forward_propagate(first_cpu_layer, categorized->matrix, 0, convnet->acts);
		ccv_dense_matrix_t* a = convnet->acts[0];
		for (y = 0; y < first_out_rows; y++)
			for (x = 0; x < first_out_cols; x++)
				for (k = 0; k < first_out_channels; k++)
				{
					float p = first_out[k * first_out_rows * first_out_cols * batch + (y * first_out_cols + x) * batch + i];
					float q = a->data.f32[y * first_out_cols * first_out_channels + x * first_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-5)
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
					if (delta > 1e-5)
						printf("avgpool fprop 2: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}

		// third convolutional layer forward propagate
		ccv_convnet_layer_t* third_cpu_layer = convnet->layers + 2;
		_ccv_convnet_convolutional_forward_propagate(third_cpu_layer, convnet->acts[1], 0, convnet->acts + 2);
		ccv_dense_matrix_t* c = convnet->acts[2];
		for (y = 0; y < third_out_rows; y++)
			for (x = 0; x < third_out_cols; x++)
				for (k = 0; k < third_out_channels; k++)
				{
					float p = third_out[k * third_out_rows * third_out_cols * batch + (y * third_out_cols + x) * batch + i];
					float q = c->data.f32[y * third_out_cols * third_out_channels + x * third_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-5)
						printf("conv fprop 3: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}

		// third convolutional layer backward propagate
		_ccv_convnet_convolutional_backward_propagate(third_cpu_layer, convnet->acts[2], convnet->acts[2], 0, convnet->acts[1], update_params->acts + 1, update_params->layers + 2);
		ccv_dense_matrix_t* bc = update_params->acts[1];
		for (y = 0; y < second_out_rows; y++)
			for (x = 0; x < second_out_cols; x++)
				for (k = 0; k < second_out_channels; k++)
				{
					float p = third_back[k * second_out_rows * second_out_cols * batch + (y * second_out_cols + x) * batch + i];
					float q = bc->data.f32[y * second_out_cols * second_out_channels + x * second_out_channels + k];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-5)
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
					if (delta > 1e-5)
						printf("avgpool bprop 2: %d %d %d %d: |%g - %g| = %g\n", i, x, y, k, p, q, delta);
				}

		// first convolutional layer backward propagate
		_ccv_convnet_convolutional_backward_propagate(first_cpu_layer, update_params->acts[0], convnet->acts[0], 0, categorized->matrix, 0, update_params->layers);
	}
	ccv_convnet_layer_t* third_cpu_configuration = update_params->layers + 2;
	int third_filter_rows = third_gpu_layer->net.convolutional.rows;
	int third_filter_cols = third_gpu_layer->net.convolutional.cols;
	int third_filter_count = third_gpu_layer->net.convolutional.count;
	int third_filter_channels = third_gpu_layer->net.convolutional.channels;
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
			for (k = 0; k < 1; k++) // first_filter_count; k++)
				for (c = 0; c < first_filter_channels; c++)
				{
					float p = first_cpu_configuration->w[(y * first_filter_cols + x) * first_filter_channels + k * first_filter_cols * first_filter_rows * first_filter_channels + c];
					float q = first_grad[(y * first_filter_cols + x) * first_filter_count + k + c * first_filter_cols * first_filter_rows * first_filter_count];
					float delta = fabs(p - q) / ccv_max(ccv_max(fabs(p), fabs(q)), 1);
					if (delta > 1e-4)
						printf("conv bprop 1: %d %d %d %d: |%g - %g| = %g\n", x, y, k, c, p, q, delta);
				}
}
