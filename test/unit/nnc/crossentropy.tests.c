#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("compare softmax + categorical crossentropy v.s. softmax crossentropy command")
{
	ccv_nnc_symbolic_graph_t* const graph = ccv_nnc_symbolic_graph_new();
	// batch size = 2, dim = 3.
	const ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_NHWC(2, 3), "a");
	const ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_NHWC(2, 3), "b0");
	ccv_nnc_graph_exec_symbol_new(graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b0), "softmax");
	const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_LABEL(2), "label");
	const ccv_nnc_tensor_symbol_t loss0 = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "loss0");
	ccv_nnc_graph_exec_symbol_new(graph, CMD_CATEGORICAL_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(b0, label), TENSOR_SYMBOL_LIST(loss0), "categorical crossentropy");
	const ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_NHWC(2, 3), "b1");
	ccv_nnc_graph_exec_symbol_new(graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(a, label), TENSOR_SYMBOL_LIST(b1, NO_TENSOR_SYMBOL), "softmax crossentropy");
	ccv_nnc_graph_exec_symbol_autogen(graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(graph, TENSOR_SYMBOL_LIST(loss0), TENSOR_SYMBOL_LIST(a), SYMBOLIC_GRAPH_SOURCES(graph), SYMBOLIC_GRAPH_DESTINATIONS(graph));
	ccv_nnc_symbolic_graph_backward(graph, TENSOR_SYMBOL_LIST(b1), TENSOR_SYMBOL_LIST(a), SYMBOLIC_GRAPH_SOURCES(graph), SYMBOLIC_GRAPH_DESTINATIONS(graph));
	SYMBOLIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_free(graph);
}

#include "case_main.h"
