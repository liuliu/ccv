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
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_t prod0 = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST(y, z), TENSOR_LIST(z));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod0);
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph, GRAPH_EXEC_LIST(noop), GRAPH_EXEC_LIST(prod0), GRAPH_EXEC_LIST(noop), TENSOR_LIST(z), TENSOR_LIST(z), while_5, 0);
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

TEST_CASE("graph for a while loop by reuse tensor allocations for 0.32 * 2.8 ^ 5")
{
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_graph_t* while_graph = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_numeric_data_t data_x[] = {x->data, z->data};
	ccv_nnc_tensor_multiview_t xx = ccv_nnc_tensor_multiview(x, 2, 2, data_x);
	ccv_numeric_data_t data_z[] = {z->data, x->data};
	ccv_nnc_tensor_multiview_t zz = ccv_nnc_tensor_multiview(z, 2, 2, data_z);
	ccv_nnc_graph_exec_t prod = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST((ccv_nnc_tensor_t*)&xx, y), TENSOR_LIST((ccv_nnc_tensor_t*)&zz));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod);
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph, GRAPH_EXEC_LIST(noop), GRAPH_EXEC_LIST(prod), GRAPH_EXEC_LIST(noop), TENSOR_LIST((ccv_nnc_tensor_t*)&xx), TENSOR_LIST((ccv_nnc_tensor_t*)&zz), while_5, 0);
	x->data.f32[0] = 0.32;
	y->data.f32[0] = 2.8;
	ccv_nnc_graph_while_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(loop));
	REQUIRE_EQ_WITH_TOLERANCE(z->data.f32[0], 0.32 * 2.8 * 2.8 * 2.8 * 2.8 * 2.8, 1e-5, "computed result of 0.32 * 2.8 ^ 5 should be the same");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
}

TEST_CASE("symbolic graph for a while loop to compute x ^ 5 * y")
{
	/*
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_graph_exec_symbol_t for1_5 = ccv_nnc_symbolic_graph_while(symbolic_graph, while_graph, "for 1..5");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(x, z0), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, prod0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_t prod1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z1, y), TENSOR_SYMBOL_LIST(z2), "prod1");
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, for1_5, prod1);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_while_params(while_graph, TENSOR_SYMBOL_MAP(KV(z1, z0)));
	FILE* fw = fopen("while.dot", "w+");
	ccv_nnc_symbolic_graph_dot(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH, fw);
	fclose(fw);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	*/
}

#include "case_main.h"
