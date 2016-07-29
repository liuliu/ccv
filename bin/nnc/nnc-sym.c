#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>

int main(int argc, char** argv)
{
	ccv_nnc_init();
	ccv_nnc_symbolic_graph_t* graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol(graph, ONE_CPU_TENSOR(3, 224, 224));
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol(graph, ccv_nnc_tensor_auto);
	ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol(graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTION_FORWARD, 0, CMD_CONVOLUTION(64, 7, 7), 0), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b));
	ccv_nnc_graph_exec_symbol_set_hint(graph, conv1, HINT((2, 2), (2, 2), (3, 3)));
	ccv_nnc_symbolic_graph_free(graph);
	return 0;
}
