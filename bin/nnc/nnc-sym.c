#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>

int main(int argc, char** argv)
{
	ccv_nnc_init();
	ccv_nnc_symbolic_graph_t* graph = ccv_nnc_symbolic_graph_new();
	// Input tensor
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(graph, ONE_CPU_TENSOR(223, 223, 3), "a");
	ccv_nnc_tensor_t* tensor_a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(223, 223, 3), 0);
	// conv1
	ccv_nnc_tensor_symbol_t conv1w = ccv_nnc_tensor_symbol_new(graph, ONE_CPU_TENSOR(64, 7, 7, 3), "conv1w");
	ccv_nnc_tensor_symbol_t conv1b = ccv_nnc_tensor_symbol_new(graph, ONE_CPU_TENSOR(64), "conv1b");
	ccv_nnc_tensor_symbol_t b[9];
	b[0] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "b0");
	ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(graph, CMD_CONVOLUTION_FORWARD(1, 64, 7, 7, 3), TENSOR_SYMBOL_LIST(a, conv1w, conv1b), TENSOR_SYMBOL_LIST(b[0]), "conv1");
	ccv_nnc_graph_exec_symbol_set_hint(graph, conv1, HINT((2, 2), (3, 3)));
	// max1
	b[1] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "b1");
	ccv_nnc_graph_exec_symbol_t max1 = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_MAX_POOL_FORWARD, 0, CMD_GENERIC(2, 2, 64), 0), TENSOR_SYMBOL_LIST(b[0]), TENSOR_SYMBOL_LIST(b[1]), "max1");
	ccv_nnc_graph_exec_symbol_set_hint(graph, max1, HINT((2, 2)));
	ccv_nnc_graph_exec_symbol_concat(graph, conv1, max1);
	b[2] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "b2");
	ccv_nnc_graph_exec_symbol_t relu1 = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0), TENSOR_SYMBOL_LIST(b[1]), TENSOR_SYMBOL_LIST(b[2]), "relu1");
	ccv_nnc_graph_exec_symbol_concat(graph, max1, relu1);
	// conv2(x3)
	b[3] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "b3");
	ccv_nnc_tensor_symbol_t conv2w[3];
	conv2w[0] = ccv_nnc_tensor_symbol_new(graph, ONE_CPU_TENSOR(64, 3, 3, 64), "conv2w");
	ccv_nnc_tensor_symbol_t conv2b[3];
	conv2b[0] = ccv_nnc_tensor_symbol_new(graph, ONE_CPU_TENSOR(64), "conv2b");
	ccv_nnc_graph_exec_symbol_t conv2[3];
	conv2[0] = ccv_nnc_graph_exec_symbol_new(graph, CMD_CONVOLUTION_FORWARD(1, 64, 3, 3, 64), TENSOR_SYMBOL_LIST(b[2], conv2w[0], conv2b[0]), TENSOR_SYMBOL_LIST(b[3]), "conv2");
	ccv_nnc_graph_exec_symbol_concat(graph, relu1, conv2[0]);
	ccv_nnc_graph_exec_symbol_t relu2[3];
	b[4] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "b4");
	relu2[0] = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0), TENSOR_SYMBOL_LIST(b[3]), TENSOR_SYMBOL_LIST(b[4]), "relu2");
	ccv_nnc_graph_exec_symbol_concat(graph, conv2[0], relu2[0]);
	int i;
	for (i = 1; i < 3; i++)
	{
		b[3 + i * 2] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "bn");
		conv2w[i] = ccv_nnc_tensor_symbol_new(graph, ONE_CPU_TENSOR(64, 3, 3, 64), "conv2wn");
		conv2b[i] = ccv_nnc_tensor_symbol_new(graph, ONE_CPU_TENSOR(64), "conv2bn");
		conv2[i] = ccv_nnc_graph_exec_symbol_new(graph, CMD_CONVOLUTION_FORWARD(1, 64, 3, 3, 64), TENSOR_SYMBOL_LIST(b[2 + i * 2], conv2w[i], conv2b[i]), TENSOR_SYMBOL_LIST(b[3 + i * 2]), "conv2n");
		ccv_nnc_graph_exec_symbol_concat(graph, relu2[i - 1], conv2[i]);
		b[4 + i * 2] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "bn");
		relu2[i] = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0), TENSOR_SYMBOL_LIST(b[3 + i * 2]), TENSOR_SYMBOL_LIST(b[4 + i * 2]), "relu2n");
		ccv_nnc_graph_exec_symbol_concat(graph, conv2[i], relu2[i]);
	}
	/*
	ccv_nnc_graph_t* run_graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(graph, TENSOR_SYMBOL_LIST(a), TENSOR_LIST(tensor_a), GRAPH_EXEC_SYMBOL_LIST(conv1), GRAPH_EXEC_SYMBOL_LIST(relu2[2]), &run_graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_graph_free(run_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	*/
	ccv_nnc_symbolic_graph_backward(graph, TENSOR_SYMBOL_LIST(b[8]), TENSOR_SYMBOL_LIST(conv1w), GRAPH_EXEC_SYMBOL_LIST(conv1), GRAPH_EXEC_SYMBOL_LIST(relu2[2]));
	ccv_nnc_symbolic_graph_free(graph);
	ccv_nnc_tensor_free(tensor_a);
	return 0;
}
