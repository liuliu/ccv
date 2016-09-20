#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_CASE("simple autograd with D[x * x + Log[1 / x], x] when x = 0.84")
{
	ccv_nnc_init();
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t one = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1));
	// w = x * x
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1));
	// u = 1 / x
	ccv_nnc_tensor_symbol_t u = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1));
	// v = Log[u]
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1));
	// z = w + v
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol(symbolic_graph, ONE_CPU_TENSOR(1));
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(x, x), TENSOR_SYMBOL_LIST(w));
	ccv_nnc_graph_exec_symbol_t inv = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_EWDIV_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(one, x), TENSOR_SYMBOL_LIST(u));
	ccv_nnc_graph_exec_symbol_t log = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_EWLOG_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(u), TENSOR_SYMBOL_LIST(v));
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol(symbolic_graph, ccv_nnc_cmd(CCV_NNC_COMPUTE_EWSUM_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(w, v), TENSOR_SYMBOL_LIST(z));
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, prod, sum);
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, inv, log);
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, log, sum);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(prod, inv), GRAPH_EXEC_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(z), TENSOR_SYMBOL_LIST(x));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_t dxexec = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* tz = ccv_nnc_tensor_new(0, z.info, 0);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, TENSOR_SYMBOL_LIST(z), TENSOR_LIST(tz), GRAPH_EXEC_SYMBOL_LIST(prod, inv), GRAPH_EXEC_SYMBOL_LIST(dxexec), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* tone = ccv_nnc_tensor_from_symbol(tensor_arena, one);
	tone->data.f32[0] = 1;
	ccv_nnc_tensor_t* tx = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	tx->data.f32[0] = 0.84;
	ccv_nnc_tensor_symbol_t dz = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, z);
	ccv_nnc_tensor_t* tdz = ccv_nnc_tensor_from_symbol(tensor_arena, dz);
	// Seed the initialization vector.
	tdz->data.f32[0] = 1;
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* tdx = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_graph_run(graph, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, prod), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, inv)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dxexec)));
	REQUIRE_EQ_WITH_TOLERANCE(tz->data.f32[0], 0.84 * 0.84 + logf(1.0 / 0.84), 1e-6, "computed result of x * x + Log[1 / x] should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[0], 2 * 0.84 - (1.0 / 0.84), 1e-6, "computed result of D[x * x + Log[1 / x], x] should be the same");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_free(tz);
}

#include "case_main.h"
