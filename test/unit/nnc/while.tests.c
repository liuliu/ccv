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

int while_5(ccv_nnc_tensor_t* const* const specials, const int special_size, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const void* const data)
{
	return specials[0]->data.i64[0] < 5;
}

TEST_CASE("graph for a while loop to compute 0.34 * 1.11 ^ 5")
{
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_t* while_graph = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t prod0 = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST(y, z), TENSOR_LIST(z));
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph, GRAPH_EXEC_LIST(prod0), GRAPH_EXEC_LIST(prod0), GRAPH_EXEC_LIST(prod0), TENSOR_LIST(z), TENSOR_LIST(z), while_5, 0);
	ccv_nnc_graph_exec_t prod1 = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST(x, z), TENSOR_LIST(z));
	ccv_nnc_graph_exec_concat(graph, loop, prod1);
	x->data.f32[0] = 0.34;
	y->data.f32[0] = 1.11;
	z->data.f32[0] = 1;
	ccv_nnc_graph_while_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(prod1));
	ccv_nnc_graph_free(graph);
	REQUIRE_EQ_WITH_TOLERANCE(z->data.f32[0], 0.34 * 1.11 * 1.11 * 1.11 * 1.11 * 1.11, 1e-6, "computed result of 0.34 * 1.11 ^ 5 should be the same");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
}

TEST_CASE("symbolic graph for a while loop to compute x * y ^ 5")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z0), "prod0");
	ccv_nnc_graph_exec_symbol_t prod1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(y, z0), TENSOR_SYMBOL_LIST(z), "prod1");
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, prod0, prod1);
	ccv_nnc_symbolic_graph_while(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(prod1), GRAPH_EXEC_SYMBOL_LIST(prod1), GRAPH_EXEC_SYMBOL_LIST(prod1), while_5, 0, TENSOR_SYMBOL_LIST(z0), TENSOR_SYMBOL_LIST(z), TENSOR_SYMBOL_MAP(KV(z, z0)), "for 1..5");
	FILE* fw = fopen("while.dot", "w+");
	ccv_nnc_symbolic_graph_dot(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH, fw);
	fclose(fw);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

#include "case_main.h"
