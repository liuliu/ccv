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
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(3, 223, 223));
	// conv1
	ccv_nnc_tensor_symbol_t conv1w = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(3, 7, 7, 64));
	ccv_nnc_tensor_symbol_t conv1b = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(64));
	ccv_nnc_tensor_symbol_t b[9];
	b[0] = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
	ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTION_FORWARD, 0, CMD_CONVOLUTION(64, 7, 7), 0), TENSOR_SYMBOL_LIST(a, conv1w, conv1b), TENSOR_SYMBOL_LIST(b[0]));
	ccv_nnc_graph_exec_symbol_set_hint(graph, conv1, HINT((2, 2), (3, 3)));
	// max1
	b[1] = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
	ccv_nnc_graph_exec_symbol_t max1 = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_MAX_POOL_FORWARD, 0, CMD_GENERIC(2, 2), 0), TENSOR_SYMBOL_LIST(b[0]), TENSOR_SYMBOL_LIST(b[1]));
	ccv_nnc_graph_exec_symbol_set_hint(graph, max1, HINT((2, 2)));
	ccv_nnc_graph_exec_symbol_concat(graph, conv1, max1);
	b[2] = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
	ccv_nnc_graph_exec_symbol_t relu1 = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0), TENSOR_SYMBOL_LIST(b[1]), TENSOR_SYMBOL_LIST(b[2]));
	ccv_nnc_graph_exec_symbol_concat(graph, max1, relu1);
	// conv2(x3)
	b[3] = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
	ccv_nnc_tensor_symbol_t conv2w[3];
	conv2w[0] = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(64, 3, 3, 64));
	ccv_nnc_tensor_symbol_t conv2b[3];
	conv2b[0] = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(64));
	ccv_nnc_graph_exec_symbol_t conv2[3];
	conv2[0] = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTION_FORWARD, 0, CMD_CONVOLUTION(64, 3, 3), 0), TENSOR_SYMBOL_LIST(b[2], conv2w[0], conv2b[0]), TENSOR_SYMBOL_LIST(b[3]));
	ccv_nnc_graph_exec_symbol_concat(graph, relu1, conv2[0]);
	ccv_nnc_graph_exec_symbol_t relu2[3];
	b[4] = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
	relu2[0] = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0), TENSOR_SYMBOL_LIST(b[3]), TENSOR_SYMBOL_LIST(b[4]));
	ccv_nnc_graph_exec_symbol_concat(graph, conv2[0], relu2[0]);
	int i;
	for (i = 1; i < 3; i++)
	{
		b[3 + i * 2] = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
		conv2w[i] = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(64, 3, 3, 64));
		conv2b[i] = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(64));
		conv2[i] = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTION_FORWARD, 0, CMD_CONVOLUTION(64, 3, 3), 0), TENSOR_SYMBOL_LIST(b[2 + i * 2], conv2w[i], conv2b[i]), TENSOR_SYMBOL_LIST(b[3 + i * 2]));
		ccv_nnc_graph_exec_symbol_concat(graph, relu2[i - 1], conv2[i]);
		b[4 + i * 2] = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
		relu2[i] = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0), TENSOR_SYMBOL_LIST(b[3 + i * 2]), TENSOR_SYMBOL_LIST(b[4 + i * 2]));
		ccv_nnc_graph_exec_symbol_concat(graph, conv2[i], relu2[i]);
	}
	ccv_nnc_graph_t* run_graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_symbolic_graph_compile(graph, 0, 0, 0, 0, &conv1, 1, &relu2[2], 1, &run_graph, &tensor_arena);
	ccv_nnc_symbolic_graph_free(graph);
	return 0;
}
