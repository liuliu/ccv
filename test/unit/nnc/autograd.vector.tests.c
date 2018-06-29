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

TEST_CASE("autograd with D[y = x + [1 1.5] => x_1 + (y_1 + y_1 ^ 2) + Exp[y_2], x] when x = [0.44 -1.18]")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t one = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "[1 1.5]");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "y");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 2;
	ccv_nnc_tensor_symbol_t x_1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ofs, inc, ONE_CPU_TENSOR(1), "x_1");
	ccv_nnc_tensor_symbol_t y_1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y, ofs, inc, ONE_CPU_TENSOR(1), "y_1");
	ofs[0] = 1;
	ccv_nnc_tensor_symbol_t y_2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y, ofs, inc, ONE_CPU_TENSOR(1), "y_2");
	ccv_nnc_tensor_symbol_t w_1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "w_1");
	ccv_nnc_tensor_symbol_t u_1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "u_1");
	ccv_nnc_tensor_symbol_t u_2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "u_2");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	ccv_nnc_graph_exec_symbol_t plus = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x, one), TENSOR_SYMBOL_LIST(y), "plus");
	ccv_nnc_graph_exec_symbol_t sqr = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(y_1, y_1), TENSOR_SYMBOL_LIST(w_1), "sqr");
	ccv_nnc_graph_exec_symbol_t plus_y = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(w_1, y_1), TENSOR_SYMBOL_LIST(u_1), "plus_y");
	ccv_nnc_graph_exec_symbol_t exp_ = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(y_2), TENSOR_SYMBOL_LIST(u_2), "exp");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x_1, u_1, u_2), TENSOR_SYMBOL_LIST(v), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(plus, sqr, plus_y, exp_, sum), 0);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(x), GRAPH_EXEC_SYMBOL_LIST(plus), GRAPH_EXEC_SYMBOL_LIST(sum));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_exec_symbol_t dxc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dx);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, GRAPH_EXEC_SYMBOL_LIST(plus), GRAPH_EXEC_SYMBOL_LIST(dxc, sum), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
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
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_source(graph_exec_arena)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_destination(graph_exec_arena)));
	ccv_nnc_tensor_t* tv = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	ccv_nnc_tensor_t* tdx = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(tv->data.f32[0], 0.44 + (0.44 + 1 + (0.44 + 1) * (0.44 + 1)) + expf(-1.18 + 1.5), 1e-6, "computed result of y = x + [1 1.5] => x_1 + (y_1 + y_1 ^ 2) + Exp[y_2] should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[0], 2 + 2 * (0.44 + 1), 1e-6, "computed result of D[y = x + [1 1.5] => x_1 + (y_1 + y_1 ^ 2) + Exp[y_2], x] for x_1 should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[1], expf(-1.18 + 1.5), 1e-6, "computed result of D[y = x + [1 1.5] => x_1 + (y_1 + y_1 ^ 2) + Exp[y_2], x] for x_2 should be the same");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("autograd with D[y_1 = Log[x_1], y_2 = x_2 ^ 2 => y_1 ^ 2 + y_1 * y_2, x] when x = [0.38 -2.8]")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "y");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 2;
	ccv_nnc_tensor_symbol_t x_1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ofs, inc, ONE_CPU_TENSOR(1), "x_1");
	ccv_nnc_tensor_symbol_t y_1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y, ofs, inc, ONE_CPU_TENSOR(1), "y_1");
	ofs[0] = 1;
	ccv_nnc_tensor_symbol_t x_2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ofs, inc, ONE_CPU_TENSOR(1), "x_2");
	ccv_nnc_tensor_symbol_t y_2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y, ofs, inc, ONE_CPU_TENSOR(1), "y_2");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "w");
	ccv_nnc_tensor_symbol_t u = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "u");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	ccv_nnc_graph_exec_symbol_t plus = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(x_1), TENSOR_SYMBOL_LIST(y_1), "log");
	ccv_nnc_graph_exec_symbol_t x_1_sqr = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x_2, x_2), TENSOR_SYMBOL_LIST(y_2), "x_1_sqr");
	ccv_nnc_graph_exec_symbol_t y_1_sqr = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(y_1, y_1), TENSOR_SYMBOL_LIST(w), "y_1_sqr");
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(y_1, y_2), TENSOR_SYMBOL_LIST(u), "prod");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(w, u), TENSOR_SYMBOL_LIST(v), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(plus, x_1_sqr, y_1_sqr, prod, sum), 0);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(x), GRAPH_EXEC_SYMBOL_LIST(plus, x_1_sqr), GRAPH_EXEC_SYMBOL_LIST(sum));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_exec_symbol_t dxc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dx);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, GRAPH_EXEC_SYMBOL_LIST(plus, x_1_sqr), GRAPH_EXEC_SYMBOL_LIST(dxc, sum), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* tx = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	tx->data.f32[0] = 0.38;
	tx->data.f32[1] = -2.8;
	ccv_nnc_tensor_symbol_t dv = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, v);
	ccv_nnc_tensor_t* tdv = ccv_nnc_tensor_from_symbol(tensor_arena, dv);
	// Seed the initialization vector.
	tdv->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_source(graph_exec_arena)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_destination(graph_exec_arena)));
	ccv_nnc_tensor_t* tv = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	ccv_nnc_tensor_t* tdx = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(tv->data.f32[0], logf(0.38) * logf(0.38) + logf(0.38) * (-2.8 * -2.8), 1e-6, "computed result of y_1 = Log[x_1], y_2 = x_2 ^ 2 => y_1 ^ 2 + y_1 * y_2 should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[0], 2 * logf(0.38) / 0.38 + (-2.8 * -2.8) / 0.38, 1e-6, "computed result of D[y_1 = Log[x_1], y_2 = x_2 ^ 2 => y_1 ^ 2 + y_1 * y_2, x] for x_1 should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[1], 2 * -2.8 * logf(0.38), 1e-6, "computed result of D[y_1 = Log[x_1], y_2 = x_2 ^ 2 => y_1 ^ 2 + y_1 * y_2, x] for x_2 should be the same");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("autograd with D[y_1 = Log[x_1] => y_1 ^ 2 + y_1, x] when x = [0.21 -13.22]")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "y");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 2;
	ccv_nnc_tensor_symbol_t x_1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ofs, inc, ONE_CPU_TENSOR(1), "x_1");
	ccv_nnc_tensor_symbol_t y_1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y, ofs, inc, ONE_CPU_TENSOR(1), "y_1");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "w");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	ccv_nnc_graph_exec_symbol_t plus = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(x_1), TENSOR_SYMBOL_LIST(y_1), "log");
	ccv_nnc_graph_exec_symbol_t sqr = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(y_1, y_1), TENSOR_SYMBOL_LIST(w), "sqr");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(w, y_1), TENSOR_SYMBOL_LIST(v), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(plus, sqr, sum), 0);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(x), GRAPH_EXEC_SYMBOL_LIST(plus), GRAPH_EXEC_SYMBOL_LIST(sum));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_exec_symbol_t dxc = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dx);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, GRAPH_EXEC_SYMBOL_LIST(plus), GRAPH_EXEC_SYMBOL_LIST(dxc, sum), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* tx = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	tx->data.f32[0] = 0.21;
	tx->data.f32[1] = -13.22;
	ccv_nnc_tensor_symbol_t dv = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, v);
	ccv_nnc_tensor_t* tdv = ccv_nnc_tensor_from_symbol(tensor_arena, dv);
	// Seed the initialization vector.
	tdv->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_source(graph_exec_arena)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_destination(graph_exec_arena)));
	ccv_nnc_tensor_t* tv = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	ccv_nnc_tensor_t* tdx = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(tv->data.f32[0], logf(0.21) * logf(0.21) + logf(0.21), 1e-6, "computed result of y_1 = Log[x_1] => y_1 ^ 2 + y_1 should be the same");
	REQUIRE_EQ_WITH_TOLERANCE(tdx->data.f32[0], 2 * logf(0.21) / 0.21 + 1 / 0.21, 1e-6, "computed result of D[y_1 = Log[x_1] => y_1 ^ 2 + y_1, x] for x_1 should be the same");
	// Note that the value in tdx->data.f32[1] is undefined.
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("autograd with sliced tensors for convolution doesn't require zeros (similar to Inception module)")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t image = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 3), "image");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128, 3, 3, 3), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128), "bias");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "c");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 100;
	inc[1] = 100;
	inc[2] = 128;
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b0");
	ccv_nnc_tensor_symbol_t c0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c0");
	ofs[2] = 64;
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b1");
	ccv_nnc_tensor_symbol_t c1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c1");
	ofs[2] = 0;
	ofs[0] = 50;
	ccv_nnc_tensor_symbol_t b2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b2");
	ccv_nnc_tensor_symbol_t c2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c2");
	ofs[2] = 64;
	ofs[0] = 50;
	ccv_nnc_tensor_symbol_t b3 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b3");
	ccv_nnc_tensor_symbol_t c3 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c3");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 3), TENSOR_SYMBOL_LIST(image, w, bias), TENSOR_SYMBOL_LIST(b), "conv");
	ccv_nnc_graph_exec_symbol_t relu0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b0), TENSOR_SYMBOL_LIST(c0), "relu0");
	ccv_nnc_graph_exec_symbol_t relu1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b1), TENSOR_SYMBOL_LIST(c1), "relu1");
	ccv_nnc_graph_exec_symbol_t relu2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b2), TENSOR_SYMBOL_LIST(c2), "relu2");
	ccv_nnc_graph_exec_symbol_t relu3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b3), TENSOR_SYMBOL_LIST(c3), "relu3");
	ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 1, 128), "d");
	ccv_nnc_graph_exec_symbol_t pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(100, 100), TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(d), "pool");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(conv, relu0, relu1, relu2, relu3, pool), 0);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(d), TENSOR_SYMBOL_LIST(w, bias, b, c), GRAPH_EXEC_SYMBOL_LIST(conv), GRAPH_EXEC_SYMBOL_LIST(pool));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t db = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b);
	ccv_nnc_tensor_symbol_t dc = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, c);
	REQUIRE(!(ccv_nnc_tensor_symbol_flags(symbolic_graph, db) & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS), "The gradient for b doesn't need to be zero init");
	REQUIRE(!(ccv_nnc_tensor_symbol_flags(symbolic_graph, dc) & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS), "The gradient for c doesn't need to be zero init");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("autograd with sliced tensors for convolution require zeros")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t image = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(3, 100, 100, 3), "image");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128, 3, 3, 3), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128), "bias");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "c");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 100;
	inc[1] = 100;
	inc[2] = 128;
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b0");
	ccv_nnc_tensor_symbol_t c0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c0");
	ofs[2] = 64;
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b1");
	ccv_nnc_tensor_symbol_t c1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c1");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 3), TENSOR_SYMBOL_LIST(image, w, bias), TENSOR_SYMBOL_LIST(b), "conv");
	ccv_nnc_graph_exec_symbol_t relu0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b0), TENSOR_SYMBOL_LIST(c0), "relu0");
	ccv_nnc_graph_exec_symbol_t relu1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b1), TENSOR_SYMBOL_LIST(c1), "relu1");
	ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 1, 128), "d");
	ccv_nnc_graph_exec_symbol_t pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(100, 100), TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(d), "pool");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(conv, relu0, relu1, pool), 0);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(d), TENSOR_SYMBOL_LIST(w, bias, b, c), GRAPH_EXEC_SYMBOL_LIST(conv), GRAPH_EXEC_SYMBOL_LIST(pool));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_SHORT_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t db = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b);
	ccv_nnc_tensor_symbol_t dc = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, c);
	REQUIRE((ccv_nnc_tensor_symbol_flags(symbolic_graph, db) & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS), "The gradient for b needs to be zero init");
	REQUIRE(!(ccv_nnc_tensor_symbol_flags(symbolic_graph, dc) & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS), "The gradient for c doesn't need to be zero init");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("autograd with sliced tensors for convolution that are over-subscribed")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t image = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 3), "image");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128, 3, 3, 3), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128), "bias");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "c");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 100;
	inc[1] = 100;
	inc[2] = 128;
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b0");
	ccv_nnc_tensor_symbol_t c0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c0");
	ofs[2] = 32;
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b1");
	ccv_nnc_tensor_symbol_t c1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c1");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 3), TENSOR_SYMBOL_LIST(image, w, bias), TENSOR_SYMBOL_LIST(b), "conv");
	ccv_nnc_graph_exec_symbol_t relu0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b0), TENSOR_SYMBOL_LIST(c0), "relu0");
	ccv_nnc_graph_exec_symbol_t relu1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b1), TENSOR_SYMBOL_LIST(c1), "relu1");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(conv, relu0, relu1), 0);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(w, bias, b), GRAPH_EXEC_SYMBOL_LIST(conv), GRAPH_EXEC_SYMBOL_LIST(relu0, relu1));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_SHORT_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t db = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b);
	ccv_nnc_graph_exec_symbol_t dbx = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, db);
	REQUIRE(ccv_nnc_graph_exec_symbol_cmd(symbolic_graph, dbx).cmd == CCV_NNC_EWSUM_FORWARD, "Since gradient of b is overlapped, it has to be summed up");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("autograd with sliced tensors for convolution that are over-subscribed with no-op")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t image = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 3), "image");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128, 3, 3, 3), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128), "bias");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100, 100, 128), "c");
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int inc[CCV_NNC_MAX_DIM_ALLOC] = {0};
	inc[0] = 100;
	inc[1] = 100;
	inc[2] = 128;
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b0");
	ccv_nnc_tensor_symbol_t c0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c0");
	ofs[2] = 32;
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, b, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "b1");
	ccv_nnc_tensor_symbol_t c1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, c, ofs, inc, ONE_CPU_TENSOR(50, 100, 64), "c1");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 3), TENSOR_SYMBOL_LIST(image, w, bias), TENSOR_SYMBOL_LIST(b), "conv");
	ccv_nnc_graph_exec_symbol_t relu0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b0), TENSOR_SYMBOL_LIST(c0), "relu0");
	ccv_nnc_graph_exec_symbol_t relu1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(b1), TENSOR_SYMBOL_LIST(c1), "relu1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_NOOP(), TENSOR_SYMBOL_LIST(c0, c1), TENSOR_SYMBOL_LIST(c), "noop");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(conv, relu0, relu1, noop), 0);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(w, bias, b), GRAPH_EXEC_SYMBOL_LIST(conv), GRAPH_EXEC_SYMBOL_LIST(noop));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_SHORT_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t db = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b);
	ccv_nnc_graph_exec_symbol_t dbx = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, db);
	REQUIRE(ccv_nnc_graph_exec_symbol_cmd(symbolic_graph, dbx).cmd == CCV_NNC_EWSUM_FORWARD, "Since gradient of b is overlapped, it has to be summed up");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

#include "case_main.h"
