#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_CASE("compile a simple symbolic graph with flow")
{
	ccv_nnc_init();
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(2, 21, 31), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(4, 21, 31), "b");
	ccv_nnc_cmd_t forw_cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTION_FORWARD, 0, CMD_CONVOLUTION(4, 2, 3, 5), 0);
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(2, 3, 5, 4), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(4), "bias");
	ccv_nnc_graph_exec_symbol_t forw_symbol = ccv_nnc_graph_exec_symbol(symbolic_graph, forw_cmd, TENSOR_SYMBOL_LIST(a, w, bias), TENSOR_SYMBOL_LIST(b), "forw");
	ccv_nnc_tensor_symbol_t m = ccv_nnc_tensor_symbol(symbolic_graph, b.info, "m");
	ccv_nnc_cmd_t softmax_cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_SOFTMAX_FORWARD, 0, ccv_nnc_cmd_auto, 0);
	ccv_nnc_graph_exec_symbol_t softmax_symbol = ccv_nnc_graph_exec_symbol(symbolic_graph, softmax_cmd, TENSOR_SYMBOL_LIST(b), TENSOR_SYMBOL_LIST(m), "softmax");
	ccv_nnc_graph_exec_symbol_flow(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(forw_symbol, softmax_symbol));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, GRAPH_EXEC_SYMBOL_LIST(forw_symbol), GRAPH_EXEC_SYMBOL_LIST(softmax_symbol), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* m_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, m);
	REQUIRE(a_tensor->data.u8 != b_tensor->data.u8, "tensor a and b shouldn't share the memory.");
	REQUIRE(b_tensor->data.u8 == m_tensor->data.u8, "tensor b and m should share the memory because softmax is an inplace op.");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
