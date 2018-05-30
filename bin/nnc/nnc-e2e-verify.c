#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <inc/ccv_convnet_internal.h>
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static ccv_nnc_graph_t* ccv_nnc_simple_graph(ccv_convnet_t* convnet, ccv_nnc_tensor_t* input, ccv_nnc_tensor_t* output, ccv_nnc_graph_exec_t* source, ccv_nnc_graph_exec_t* dest, ccv_array_t* tensors)
{
	int i;
	// We only create the graph compute to the last fc layer.
	ccv_nnc_graph_t* vgg = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t previous_exec;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = convnet->layers + i;
		int rows, cols, partition;
		ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &rows, &cols, &partition);
		ccv_nnc_tensor_t* tensor = output;
		if (i < convnet->count - 1)
		{
			if (layer->type == CCV_CONVNET_FULL_CONNECT)
				tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(rows * cols * partition), 0);
			else
				tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(rows, cols, (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->input.matrix.channels)), 0);
			ccv_array_push(tensors, &tensor);
		}
		ccv_nnc_graph_exec_t exec = {0};
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
		{
			ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(layer->net.convolutional.count, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.channels), 0);
			memcpy(w->data.f32, layer->w, layer->wnum * sizeof(float));
			ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(layer->net.convolutional.count), 0);
			memcpy(bias->data.f32, layer->bias, layer->net.convolutional.count * sizeof(float));
			ccv_array_push(tensors, &w);
			ccv_array_push(tensors, &bias);
			ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, layer->net.convolutional.count, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.channels);
			ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, input->info, tensor->info);
			cmd = ccv_nnc_cmd_autotune(cmd, 0, hint, 0, TENSOR_LIST(input, w, bias), TENSOR_LIST(tensor), 0);
			exec = ccv_nnc_graph_exec_new(vgg, cmd, hint, TENSOR_LIST(input, w, bias), TENSOR_LIST(tensor));
		} else if (layer->type == CCV_CONVNET_MAX_POOL) {
			ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_MAX_POOL_FORWARD, 0, CMD_GENERIC(layer->net.pool.size, layer->net.pool.size, layer->input.matrix.channels), 0);
			ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, input->info, tensor->info);
			exec = ccv_nnc_graph_exec_new(vgg, cmd, hint, TENSOR_LIST(input), TENSOR_LIST(tensor));
		} else if (layer->type == CCV_CONVNET_FULL_CONNECT) {
			ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(layer->net.full_connect.count, layer->input.node.count), 0);
			memcpy(w->data.f32, layer->w, layer->wnum * sizeof(float));
			ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(layer->net.full_connect.count), 0);
			memcpy(bias->data.f32, layer->bias, layer->net.full_connect.count * sizeof(float));
			ccv_array_push(tensors, &w);
			ccv_array_push(tensors, &bias);
			ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, 0, CMD_GEMM(layer->net.full_connect.count), 0);
			// If the input is not what I expected (array), reshape it.
			if (input->info.dim[0] != ccv_nnc_tensor_count(input->info))
			{
				input = ccv_nnc_tensor_new(input->data.u8, ONE_CPU_TENSOR(ccv_nnc_tensor_count(input->info)), 0);
				ccv_array_push(tensors, &input);
			}
			cmd = ccv_nnc_cmd_autotune(cmd, 0, ccv_nnc_no_hint, 0, TENSOR_LIST(input, w, bias), TENSOR_LIST(tensor), 0);
			exec = ccv_nnc_graph_exec_new(vgg, cmd, ccv_nnc_no_hint, TENSOR_LIST(input, w, bias), TENSOR_LIST(tensor));
		} else {
			assert("unreachable");
		}
		if (i != 0)
			ccv_nnc_graph_exec_concat(vgg, previous_exec, exec);
		previous_exec = exec;
		if (i == 0)
			*source = exec;
		if (i < convnet->count - 1 &&
			(layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT))
		{
			// Create the ReLU layer.
			ccv_nnc_cmd_param_t cmd_params = {};
			ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, 0, cmd_params, 0);
			exec = ccv_nnc_graph_exec_new(vgg, cmd, ccv_nnc_no_hint, TENSOR_LIST(tensor), TENSOR_LIST(tensor));
			ccv_nnc_graph_exec_concat(vgg, previous_exec, exec);
			previous_exec = exec;
		}
		if (i == convnet->count - 1)
			*dest = exec;
		// This is the input of next layer.
		input = tensor;
	}
	return vgg;
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	ccv_convnet_t* convnet = ccv_convnet_read(0, argv[2]);
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	if (image != 0)
	{
		ccv_dense_matrix_t* input = 0;
		ccv_convnet_input_formation(convnet->input, image, &input);
		ccv_matrix_free(image);
		ccv_dense_matrix_t* sliced = 0;
		ccv_slice(input, (ccv_matrix_t**)&sliced, 0, (input->rows - 225) / 2, (input->cols - 225) / 2, 225, 225);
		ccv_matrix_free(input);
		ccv_dense_matrix_t* b = 0;
		unsigned int elapsed_time = get_current_time();
		ccv_convnet_encode(convnet, &sliced, &b, 1);
		printf("ccv_convnet_encode %u ms\n", get_current_time() - elapsed_time);
		ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1000), 0);
		ccv_nnc_graph_exec_t source, dest;
		ccv_array_t* tensors = ccv_array_new(sizeof(ccv_nnc_tensor_t*), 1, 0);
		ccv_nnc_graph_t* graph = ccv_nnc_simple_graph(convnet, (ccv_nnc_tensor_t*)sliced, c, &source, &dest, tensors);
		elapsed_time = get_current_time();
		ccv_nnc_graph_run(graph, 0, 0, &source, 1, &dest, 1);
		printf("ccv_nnc_graph_run %u ms\n", get_current_time() - elapsed_time);
		int i;
		for (i = 0; i < 1000; i++)
			if (fabsf(b->data.f32[i] - c->data.f32[i]) > 1e-4)
				printf("mis-match at %d: %f %f\n", i, b->data.f32[i], c->data.f32[i]);
		ccv_nnc_tensor_free(c);
		ccv_matrix_free(sliced);
		ccv_matrix_free(b);
		ccv_nnc_graph_free(graph);
		for (i = 0; i < tensors->rnum; i++)
			ccv_nnc_tensor_free(*(ccv_nnc_tensor_t**)ccv_array_get(tensors, i));
		ccv_array_free(tensors);
	}
	ccv_convnet_free(convnet);
	return 0;
}
