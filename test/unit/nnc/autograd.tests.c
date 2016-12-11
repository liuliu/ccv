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

TEST_CASE("simple autograd with D[x * x + Log[1 / x], x] when x = 0.84")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t one = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "1");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	// w = x * x
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "w");
	// u = 1 / x
	ccv_nnc_tensor_symbol_t u = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "u");
	// v = Log[u]
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	// z = w + v
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(x, x), TENSOR_SYMBOL_LIST(w), "prod");
	ccv_nnc_graph_exec_symbol_t inv = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWDIV_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(one, x), TENSOR_SYMBOL_LIST(u), "inv");
	ccv_nnc_graph_exec_symbol_t log = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWLOG_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(u), TENSOR_SYMBOL_LIST(v), "log");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(w, v), TENSOR_SYMBOL_LIST(z), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(prod, inv, log, sum));
	ccv_nnc_symbolic_graph_backward(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(prod, inv), GRAPH_EXEC_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(z), TENSOR_SYMBOL_LIST(x));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_exec_symbol_t dxc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dx);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, GRAPH_EXEC_SYMBOL_LIST(prod, inv), GRAPH_EXEC_SYMBOL_LIST(dxc, sum), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* tone = ccv_nnc_tensor_from_symbol(tensor_arena, one);
	tone->data.f32[0] = 1;
	ccv_nnc_tensor_t* tx = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	tx->data.f32[0] = 0.84;
	ccv_nnc_tensor_symbol_t dz = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, z);
	ccv_nnc_tensor_t* tdz = ccv_nnc_tensor_from_symbol(tensor_arena, dz);
	// Seed the initialization vector.
	tdz->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, prod), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, inv)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dxc), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, sum)));
	ccv_nnc_tensor_t* tz = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	ccv_nnc_tensor_t* tdx = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(tz->data.f32[0], 0.84 * 0.84 + logf(1.0 / 0.84), 1e-6, "computed result of x * x + Log[1 / x] should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[0], 2 * 0.84 - (1.0 / 0.84), 1e-6, "computed result of D[x * x + Log[1 / x], x] should be the same");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("autograd with D[(x - y) * (x + 1), [x, y]] when x = 43.24 and y = 0.38")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t one = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "1");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	// w = x - y
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "w");
	// u = y + 1
	ccv_nnc_tensor_symbol_t u = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "u");
	// v = w * u
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	ccv_nnc_graph_exec_symbol_t minus = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_AXPY_FORWARD, 0, CMD_BLAS(1, -1), 0), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(w), "minus");
	ccv_nnc_graph_exec_symbol_t plus = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_AXPY_FORWARD, 0, CMD_BLAS(1, 1), 0), TENSOR_SYMBOL_LIST(y, one), TENSOR_SYMBOL_LIST(u), "plus");
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(w, u), TENSOR_SYMBOL_LIST(v), "prod");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(minus, plus, prod));
	ccv_nnc_symbolic_graph_backward(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(minus, plus), GRAPH_EXEC_SYMBOL_LIST(prod), TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(x, y));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_exec_symbol_t dxc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dx);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_exec_symbol_t dyc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dy);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, GRAPH_EXEC_SYMBOL_LIST(minus, plus), GRAPH_EXEC_SYMBOL_LIST(dxc, dyc, prod), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* tone = ccv_nnc_tensor_from_symbol(tensor_arena, one);
	tone->data.f32[0] = 1;
	ccv_nnc_tensor_t* tx = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	tx->data.f32[0] = 43.24;
	ccv_nnc_tensor_t* ty = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ty->data.f32[0] = 0.38;
	ccv_nnc_tensor_symbol_t dv = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, v);
	ccv_nnc_tensor_t* tdv = ccv_nnc_tensor_from_symbol(tensor_arena, dv);
	// Seed the initialization vector.
	tdv->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, minus), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, plus)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dxc), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dyc), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, prod)));
	ccv_nnc_tensor_t* tv = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	ccv_nnc_tensor_t* tdx = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* tdy = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	REQUIRE_EQ_WITH_TOLERANCE(tv->data.f32[0], (43.24 - 0.38) * (0.38 + 1), 1e-6, "computed result of (x + y) * (y + 1) should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[0], 0.38 + 1, 1e-6, "computed result of D[(x + y) * (y + 1), x] should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdy->data.f32[0], -2 * 0.38 + 43.24 - 1, 1e-6, "computed result of D[(x + y) * (y + 1), y] should be the same");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("partial autograd with D[y * x + Log[1 / x], y] when x = 0.84 and y = 1.23")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t one = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "1");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	// w = y * x
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "w");
	// u = 1 / x
	ccv_nnc_tensor_symbol_t u = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "u");
	// v = Log[u]
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	// z = w + v
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(y, x), TENSOR_SYMBOL_LIST(w), "prod");
	ccv_nnc_graph_exec_symbol_t inv = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWDIV_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(one, x), TENSOR_SYMBOL_LIST(u), "inv");
	ccv_nnc_graph_exec_symbol_t log = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWLOG_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(u), TENSOR_SYMBOL_LIST(v), "log");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(w, v), TENSOR_SYMBOL_LIST(z), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(prod, inv, log, sum));
	ccv_nnc_symbolic_graph_backward(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(prod, inv), GRAPH_EXEC_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(z), TENSOR_SYMBOL_LIST(y));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_exec_symbol_t dyc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dy);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, GRAPH_EXEC_SYMBOL_LIST(prod, inv), GRAPH_EXEC_SYMBOL_LIST(dyc, sum), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* tone = ccv_nnc_tensor_from_symbol(tensor_arena, one);
	tone->data.f32[0] = 1;
	ccv_nnc_tensor_t* tx = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	tx->data.f32[0] = 0.84;
	ccv_nnc_tensor_t* ty = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ty->data.f32[0] = 1.23;
	ccv_nnc_tensor_symbol_t dz = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, z);
	ccv_nnc_tensor_t* tdz = ccv_nnc_tensor_from_symbol(tensor_arena, dz);
	// Seed the initialization vector.
	tdz->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, prod), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, inv)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dyc), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, sum)));
	ccv_nnc_tensor_t* tz = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	ccv_nnc_tensor_t* tdy = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	REQUIRE_EQ_WITH_TOLERANCE(tz->data.f32[0], 1.23 * 0.84 + logf(1.0 / 0.84), 1e-6, "computed result of y * x + Log[1 / x] should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdy->data.f32[0], 0.84, 1e-6, "computed result of D[y * x + Log[1 / x], y] should be the same");
	// FILE* fw = fopen("autograd_3.dot", "w+");
	// ccv_nnc_symbolic_graph_dot(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH, fw);
	// fclose(fw);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
