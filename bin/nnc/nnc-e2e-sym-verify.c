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

static ccv_nnc_symbolic_graph_t* ccv_nnc_simple_symbolic_graph(ccv_convnet_t* convnet, ccv_nnc_tensor_t* input, ccv_nnc_tensor_t* output, ccv_nnc_graph_exec_symbol_t* source_symbol, ccv_nnc_graph_exec_symbol_t* dest_symbol, ccv_nnc_tensor_symbol_t* input_symbol_ref, ccv_nnc_tensor_symbol_t* output_symbol_ref, ccv_nnc_tensor_symbol_t* w_symbols, ccv_nnc_tensor_symbol_t* bias_symbols)
{
	int i;
	// We only create the graph compute to the last fc layer.
	ccv_nnc_symbolic_graph_t* symbolic_vgg = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_param_t input_info = input->info;
	ccv_nnc_tensor_symbol_t input_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, input->info, 0);
	*input_symbol_ref = input_symbol;
	ccv_nnc_tensor_symbol_t output_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, output->info, 0);
	*output_symbol_ref = output_symbol;
	ccv_nnc_graph_exec_symbol_t previous_exec_symbol;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = convnet->layers + i;
		int rows, cols, partition;
		ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &rows, &cols, &partition);
		ccv_nnc_tensor_param_t tensor_info = output->info;
		ccv_nnc_tensor_symbol_t tensor_symbol = output_symbol;
		if (i < convnet->count - 1)
		{
			if (layer->type == CCV_CONVNET_FULL_CONNECT)
				tensor_info = ONE_CPU_TENSOR(rows * cols * partition);
			else
				tensor_info = ONE_CPU_TENSOR(rows, cols, (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->input.matrix.channels));
			tensor_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, tensor_info, 0);
		}
		ccv_nnc_graph_exec_symbol_t exec_symbol = {0};
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
		{
			ccv_nnc_tensor_symbol_t w_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, ONE_CPU_TENSOR(layer->net.convolutional.count, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.channels), 0);
			w_symbols[i] = w_symbol;
			ccv_nnc_tensor_symbol_t bias_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, ONE_CPU_TENSOR(layer->net.convolutional.count), 0);
			bias_symbols[i] = bias_symbol;
			ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, layer->net.convolutional.count, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.channels);
			exec_symbol = ccv_nnc_graph_exec_symbol_new(symbolic_vgg, cmd, TENSOR_SYMBOL_LIST(input_symbol, w_symbol, bias_symbol), TENSOR_SYMBOL_LIST(tensor_symbol), 0);
		} else if (layer->type == CCV_CONVNET_MAX_POOL) {
			ccv_nnc_cmd_t cmd = CMD_MAX_POOL_FORWARD(layer->net.pool.size, layer->net.pool.size);
			exec_symbol = ccv_nnc_graph_exec_symbol_new(symbolic_vgg, cmd, TENSOR_SYMBOL_LIST(input_symbol), TENSOR_SYMBOL_LIST(tensor_symbol), 0);
		} else if (layer->type == CCV_CONVNET_FULL_CONNECT) {
			ccv_nnc_tensor_symbol_t w_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, ONE_CPU_TENSOR(layer->net.full_connect.count, layer->input.node.count), 0);
			w_symbols[i] = w_symbol;
			ccv_nnc_tensor_symbol_t bias_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, ONE_CPU_TENSOR(layer->net.full_connect.count), 0);
			bias_symbols[i] = bias_symbol;
			ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(layer->net.full_connect.count);
			// If the input is not what I expected (array), reshape it.
			if (input_info.dim[0] != ccv_nnc_tensor_count(input_info))
				input_symbol = ccv_nnc_tensor_symbol_alias_new(symbolic_vgg, input_symbol, ccv_nnc_no_ofs, ONE_CPU_TENSOR(ccv_nnc_tensor_count(input_info)).dim, ONE_CPU_TENSOR(ccv_nnc_tensor_count(input_info)), 0);
			exec_symbol = ccv_nnc_graph_exec_symbol_new(symbolic_vgg, cmd, TENSOR_SYMBOL_LIST(input_symbol, w_symbol, bias_symbol), TENSOR_SYMBOL_LIST(tensor_symbol), 0);
		} else {
			assert("unreachable");
		}
		if (i != 0)
			ccv_nnc_graph_exec_symbol_concat(symbolic_vgg, previous_exec_symbol, exec_symbol);
		previous_exec_symbol = exec_symbol;
		if (i == 0)
			*source_symbol = exec_symbol;
		if (i < convnet->count - 1 &&
			(layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT))
		{
			// Create the ReLU layer.
			ccv_nnc_cmd_t cmd = CMD_RELU_FORWARD();
			ccv_nnc_tensor_symbol_t next_symbol = ccv_nnc_tensor_symbol_new(symbolic_vgg, tensor_info, 0);
			exec_symbol = ccv_nnc_graph_exec_symbol_new(symbolic_vgg, cmd, TENSOR_SYMBOL_LIST(tensor_symbol), TENSOR_SYMBOL_LIST(next_symbol), 0);
			ccv_nnc_graph_exec_symbol_concat(symbolic_vgg, previous_exec_symbol, exec_symbol);
			tensor_symbol = next_symbol;
			previous_exec_symbol = exec_symbol;
		}
		if (i == convnet->count - 1)
			*dest_symbol = exec_symbol;
		// This is the input of next layer.
		input_symbol = tensor_symbol;
		input_info = tensor_info;
	}
	return symbolic_vgg;
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
		ccv_nnc_tensor_symbol_t* w_symbols = ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * convnet->count);
		ccv_nnc_tensor_symbol_t* bias_symbols = ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * convnet->count);
		ccv_nnc_graph_exec_symbol_t source_symbol, dest_symbol;
		ccv_nnc_tensor_symbol_t input_symbol, output_symbol;
		ccv_nnc_symbolic_graph_t* graph = ccv_nnc_simple_symbolic_graph(convnet, (ccv_nnc_tensor_t*)sliced, c, &source_symbol, &dest_symbol, &input_symbol, &output_symbol, w_symbols, bias_symbols);
		elapsed_time = get_current_time();
		ccv_nnc_graph_t* run_graph = 0;
		ccv_nnc_tensor_arena_t* tensor_arena = 0;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
		ccv_nnc_symbolic_graph_compile(graph, TENSOR_BIND_MAP(KV(input_symbol, (ccv_nnc_tensor_t*)sliced), KV(output_symbol, c)), 0, 0, GRAPH_EXEC_SYMBOL_LIST(source_symbol), GRAPH_EXEC_SYMBOL_LIST(dest_symbol), &run_graph, &tensor_arena, &graph_exec_arena);
		printf("ccv_nnc_symbolic_graph_compile %u ms\n", get_current_time() - elapsed_time);
		int i;
		for (i = 0; i < convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = convnet->layers + i;
			if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
			{
				ccv_nnc_tensor_t* w = ccv_nnc_tensor_from_symbol(tensor_arena, w_symbols[i]);
				memcpy(w->data.f32, layer->w, layer->wnum * sizeof(float));
				ccv_nnc_tensor_t* bias = ccv_nnc_tensor_from_symbol(tensor_arena, bias_symbols[i]);
				memcpy(bias->data.f32, layer->bias, layer->net.convolutional.count * sizeof(float));
			} else if (layer->type == CCV_CONVNET_FULL_CONNECT) {
				ccv_nnc_tensor_t* w = ccv_nnc_tensor_from_symbol(tensor_arena, w_symbols[i]);
				memcpy(w->data.f32, layer->w, layer->wnum * sizeof(float));
				ccv_nnc_tensor_t* bias = ccv_nnc_tensor_from_symbol(tensor_arena, bias_symbols[i]);
				memcpy(bias->data.f32, layer->bias, layer->net.full_connect.count * sizeof(float));
			}
		}
		FILE* fw0 = fopen("sym-graph.dot", "w+");
		ccv_nnc_symbolic_graph_dot(graph, CCV_NNC_LONG_DOT_GRAPH, fw0);
		fclose(fw0);
		FILE* fw1 = fopen("graph.dot", "w+");
		ccv_nnc_graph_dot(run_graph, CCV_NNC_LONG_DOT_GRAPH, fw1);
		fclose(fw1);
		elapsed_time = get_current_time();
		ccv_nnc_graph_autotune(run_graph, 0, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, source_symbol)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest_symbol)));
		printf("ccv_nnc_graph_autotune %u ms\n", get_current_time() - elapsed_time);
		// Repopulate the weight. Since the weight tensor may be reused, need to repopulate every time run.
		// To avoid this, bind the weights directly or declare as const.
		for (i = 0; i < convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = convnet->layers + i;
			if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
			{
				ccv_nnc_tensor_t* w = ccv_nnc_tensor_from_symbol(tensor_arena, w_symbols[i]);
				memcpy(w->data.f32, layer->w, layer->wnum * sizeof(float));
				ccv_nnc_tensor_t* bias = ccv_nnc_tensor_from_symbol(tensor_arena, bias_symbols[i]);
				memcpy(bias->data.f32, layer->bias, layer->net.convolutional.count * sizeof(float));
			} else if (layer->type == CCV_CONVNET_FULL_CONNECT) {
				ccv_nnc_tensor_t* w = ccv_nnc_tensor_from_symbol(tensor_arena, w_symbols[i]);
				memcpy(w->data.f32, layer->w, layer->wnum * sizeof(float));
				ccv_nnc_tensor_t* bias = ccv_nnc_tensor_from_symbol(tensor_arena, bias_symbols[i]);
				memcpy(bias->data.f32, layer->bias, layer->net.full_connect.count * sizeof(float));
			}
		}
		elapsed_time = get_current_time();
		ccv_nnc_graph_run(run_graph, 0, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, source_symbol)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest_symbol)));
		printf("ccv_nnc_graph_run %u ms\n", get_current_time() - elapsed_time);
		for (i = 0; i < 1000; i++)
			if (fabsf(b->data.f32[i] - c->data.f32[i]) > 1e-4)
				printf("mis-match at %d: %f %f\n", i, b->data.f32[i], c->data.f32[i]);
		ccv_nnc_tensor_free(c);
		ccv_matrix_free(sliced);
		ccv_matrix_free(b);
		ccv_nnc_symbolic_graph_free(graph);
		ccv_nnc_graph_free(run_graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
		ccfree(w_symbols);
		ccfree(bias_symbols);
	}
	ccv_convnet_free(convnet);
	return 0;
}
