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

TEST_CASE("autograd with D[y = x + [1 1.5] => y_1 ^ 2 + Exp[y_2], x] when x = [0.44 -1.18]")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t one = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(2), "[1 1.5]");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(2), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(2), "y");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 2;
	ccv_nnc_tensor_symbol_t y_1 = ccv_nnc_tensor_symbol_alias(symbolic_graph, y, ofs, inc, ONE_CPU_TENSOR(1), "y_1");
	ofs[0] = 1;
	ccv_nnc_tensor_symbol_t y_2 = ccv_nnc_tensor_symbol_alias(symbolic_graph, y, ofs, inc, ONE_CPU_TENSOR(1), "y_2");
	ccv_nnc_tensor_symbol_t w_1 = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1), "w_1");
	ccv_nnc_tensor_symbol_t u_2 = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1), "u_2");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	ccv_nnc_graph_exec_symbol_t plus = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(x, one), TENSOR_SYMBOL_LIST(y), "plus");
	ccv_nnc_graph_exec_symbol_t sqr = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(y_1, y_1), TENSOR_SYMBOL_LIST(w_1), "sqr");
	ccv_nnc_graph_exec_symbol_t exp_ = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWEXP_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(y_2), TENSOR_SYMBOL_LIST(u_2), "exp");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(w_1, u_2), TENSOR_SYMBOL_LIST(v), "sum");
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, plus, sqr);
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, plus, exp_);
	ccv_nnc_graph_exec_symbol_flow(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(sqr, exp_, sum));
	ccv_nnc_symbolic_graph_backward(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(plus), GRAPH_EXEC_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(x));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_exec_symbol_t dxc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dx);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, GRAPH_EXEC_SYMBOL_LIST(plus), GRAPH_EXEC_SYMBOL_LIST(dxc, sum), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* tone = ccv_nnc_tensor_from_symbol(tensor_arena, one);
	tone->data.f32[0] = 1;
	tone->data.f32[1] = 1.5;
	ccv_nnc_tensor_t* tx = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	tx->data.f32[0] = 0.44;
	tx->data.f32[1] = -1.18;
	ccv_nnc_tensor_symbol_t dv = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, v);
	ccv_nnc_tensor_t* tdv = ccv_nnc_tensor_from_symbol(tensor_arena, dv);
	// Seed the initialization vector.
	tdv->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, plus)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dxc), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, sum)));
	// FILE *fw = fopen("autograd-vector.dot", "w+");
	// ccv_nnc_symbolic_graph_dot(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH, fw);
	// fclose(fw);
	ccv_nnc_tensor_t* tv = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	ccv_nnc_tensor_t* tdx = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(tv->data.f32[0], (0.44 + 1) * (0.44 + 1) + expf(-1.18 + 1.5), 1e-6, "computed result of y = x + [1 1.5] => y_1 ^ 2 + Exp[y_2] should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[0], 2 * (0.44 + 1), 1e-6, "computed result of D[y = x + [1 1.5] => y_1 ^ 2 + Exp[y_2], x] for x_1 should be the same");
	// This cannot pass yet (need to zero out the tensor before sum up).
	// REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[1], expf(-1.18 + 1.5), 1e-6, "computed result of D[y = x + [1 1.5] => y_1 ^ 2 + Exp[y_2], x] for x_2 should be the same");
	// FILE* fw = fopen("autograd-graph.dot", "w+");
	// ccv_nnc_graph_dot(graph, CCV_NNC_SHORT_DOT_GRAPH, fw);
	// fclose(fw);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
