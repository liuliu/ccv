#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static int while_5(ccv_nnc_tensor_t* const* const commons, const int common_size, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const void* const data)
{
	return commons[0]->data.i64[0] < 5;
}

TEST_CASE("new tape from a graph")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, while_graph, "for 1..5");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z0, x), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, prod0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_tensor_symbol_t z3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z3");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z1, y), TENSOR_SYMBOL_LIST(z2), "prod1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z2, z0), TENSOR_SYMBOL_LIST(z3), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0);
	// Add noop graph as a sub-graph after autogen otherwise this will be another source node.
	ccv_nnc_symbolic_graph_t* noop_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, noop_graph, "no op");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_while_params(while_graph, TENSOR_SYMBOL_MAP(KV(z1, z0)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(prod0));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_tape_t* tape = ccv_nnc_tensor_tape_new(graph);
	ccv_nnc_tensor_tape_free(tape);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
