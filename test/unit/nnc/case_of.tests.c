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

static int piecewise_case_of(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	assert(input_size == 1);
	if (inputs[0]->data.f32[0] < 0)
		return 0;
	else if (inputs[0]->data.f32[0] < 1)
		return -1; // Pass through because the computation is essentially x * 1
	else if (inputs[0]->data.f32[0] < 2)
		return 1;
	else if (inputs[0]->data.f32[0] > 1000)
		return 3;
	else
		return 2;
}

static int while_4(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	return inputs[0]->data.i64[0] < 4;
}

TEST_CASE("graph for a piece-wise linear function")
{
	ccv_nnc_graph_t* const graph = ccv_nnc_graph_new();
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_exec_t case_of = ccv_nnc_graph_case_of_new(graph, CCV_NNC_GRAPH_FORWARD, TENSOR_LIST(x), 0, 0, 0, 1);
	ccv_nnc_graph_set_case_of_expr(graph, case_of, piecewise_case_of, 0, 0);
	ccv_nnc_graph_t* graph_0 = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t set_0 = ccv_nnc_graph_exec_new(graph_0, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, TENSOR_LIST(x));
	ccv_nnc_graph_set_sources(graph_0, GRAPH_EXEC_LIST(set_0));
	ccv_nnc_graph_set_destinations(graph_0, GRAPH_EXEC_LIST(set_0));
	ccv_nnc_graph_set_case_of(graph, case_of, graph_0, 0);
	ccv_nnc_graph_t* graph_1 = ccv_nnc_graph_new();
	ccv_nnc_tensor_t* p1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	p1->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* s1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	s1->data.f32[0] = 0.5;
	ccv_nnc_graph_exec_t prod_1 = ccv_nnc_graph_exec_new(graph_1, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, TENSOR_LIST(x, p1), TENSOR_LIST(x));
	ccv_nnc_graph_exec_t sum_1 = ccv_nnc_graph_exec_new(graph_1, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, TENSOR_LIST(x, s1), TENSOR_LIST(x));
	ccv_nnc_graph_exec_concat(graph_1, prod_1, sum_1);
	ccv_nnc_graph_set_sources(graph_1, GRAPH_EXEC_LIST(prod_1));
	ccv_nnc_graph_set_destinations(graph_1, GRAPH_EXEC_LIST(sum_1));
	ccv_nnc_graph_set_case_of(graph, case_of, graph_1, 1);
	ccv_nnc_graph_t* graph_2 = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t set_2 = ccv_nnc_graph_exec_new(graph_2, CMD_SET_FORWARD(1.5), ccv_nnc_no_hint, 0, 0, TENSOR_LIST(x));
	ccv_nnc_graph_set_sources(graph_2, GRAPH_EXEC_LIST(set_2));
	ccv_nnc_graph_set_destinations(graph_2, GRAPH_EXEC_LIST(set_2));
	ccv_nnc_graph_set_case_of(graph, case_of, graph_2, 2);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	x->data.f32[0] = -1;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(case_of), GRAPH_EXEC_LIST(case_of));
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], 0, 1e-5, "in negative region should equal to 0");
	x->data.f32[0] = 0.76;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(case_of), GRAPH_EXEC_LIST(case_of));
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], 0.76, 1e-5, "y = x in (0, 1)");
	x->data.f32[0] = 1.226;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(case_of), GRAPH_EXEC_LIST(case_of));
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], (1.226 - 1) * 0.5 + 1, 1e-5, "y = (x - 1) * 0.5 + 1 in (1, 2)");
	x->data.f32[0] = 2.1;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(case_of), GRAPH_EXEC_LIST(case_of));
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], 1.5, 1e-5, "y = 1.5 if x > 2");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(p1);
	ccv_nnc_tensor_free(s1);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph for piece-wise function")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_graph_exec_symbol_t case_of = ccv_nnc_symbolic_graph_case_of_new(symbolic_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_MAP(KV(x, y)), "piece-wise linear");
	ccv_nnc_symbolic_graph_set_case_of_expr(symbolic_graph, case_of, piecewise_case_of, 0);
	ccv_nnc_symbolic_graph_t* const symbolic_graph_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y0 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, ONE_CPU_TENSOR(1), "y0");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, symbolic_graph_0, 0, TENSOR_SYMBOL_MAP(KV(y0, y)));
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_0, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(y0), "set");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const symbolic_graph_1 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(1), "y1");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, symbolic_graph_1, 1, TENSOR_SYMBOL_MAP(KV(y1, y)));
	ccv_nnc_tensor_symbol_t s1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(1), "s");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_tensor_symbol_t p1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(1), "p");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_1, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, s1), TENSOR_SYMBOL_LIST(z1), "prod");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_1, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, p1), TENSOR_SYMBOL_LIST(y1), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph_1, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const symbolic_graph_2 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph_2, ONE_CPU_TENSOR(1), "y2");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, symbolic_graph_2, 2, TENSOR_SYMBOL_MAP(KV(y2, y)));
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_2, CMD_SET_FORWARD(1.5), 0, 0, TENSOR_SYMBOL_LIST(y2), "set");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph_2, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, &case_of, 1, &case_of, 1, &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* s1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, s1);
	ccv_nnc_tensor_t* p1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, p1);
	s1_tensor->data.f32[0] = 0.5;
	p1_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[0] = -1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], 0, 1e-5, "in negative region should equal to 0");
	x_tensor->data.f32[0] = 0.76;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], 0.76, 1e-5, "y = x in (0, 1)");
	s1_tensor->data.f32[0] = 0.5;
	p1_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[0] = 1.226;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], (1.226 - 1) * 0.5 + 1, 1e-5, "y = (x - 1) * 0.5 + 1 in (1, 2)");
	s1_tensor->data.f32[0] = 0.5;
	p1_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[0] = 2.1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], 1.5, 1e-5, "y = 1.5 if x > 2");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph case..of when reuse intermediate tensors")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "y");
	ccv_nnc_graph_exec_symbol_t case_of = ccv_nnc_symbolic_graph_case_of_new(symbolic_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_MAP(KV(x, y)), "piece-wise linear vector");
	ccv_nnc_symbolic_graph_set_case_of_expr(symbolic_graph, case_of, piecewise_case_of, 0);
	ccv_nnc_symbolic_graph_t* const symbolic_graph_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, ONE_CPU_TENSOR(2), "z0");
	ccv_nnc_tensor_symbol_t u0 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, ONE_CPU_TENSOR(2, 2), "u0");
	ccv_nnc_tensor_symbol_t w0 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, ONE_CPU_TENSOR(2, 2), "w0");
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, ONE_CPU_TENSOR(2), "b0");
	ccv_nnc_tensor_symbol_t c0 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, ONE_CPU_TENSOR(2), "c0");
	ccv_nnc_tensor_symbol_t y0 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, ONE_CPU_TENSOR(2), "y0");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, symbolic_graph_0, 0, TENSOR_SYMBOL_MAP(KV(y0, y)));
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_0, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(x, u0, b0), TENSOR_SYMBOL_LIST(z0), "mmu");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_0, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(z0, w0, c0), TENSOR_SYMBOL_LIST(y0), "mmw");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const symbolic_graph_1 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(2), "z1");
	ccv_nnc_tensor_symbol_t u1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(2, 2), "u1");
	ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(2, 2), "w1");
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(2), "b1");
	ccv_nnc_tensor_symbol_t c1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(2), "c1");
	ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph_1, ONE_CPU_TENSOR(2), "y1");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, symbolic_graph_1, 1, TENSOR_SYMBOL_MAP(KV(y1, y)));
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_1, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(x, u1, b1), TENSOR_SYMBOL_LIST(z1), "mmu1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_1, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(z1, w1, c1), TENSOR_SYMBOL_LIST(y1), "mmw1");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph_1, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, &case_of, 1, &case_of, 1, &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* z0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z0);
	ccv_nnc_tensor_t* z1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z1);
	REQUIRE(z0_tensor->data.f32 == z1_tensor->data.f32, "z0 and z1 should be allocated to the same location");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph case..of with 4 branches")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_graph_exec_symbol_t case_of = ccv_nnc_symbolic_graph_case_of_new(symbolic_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_MAP(KV(x, y)), "4 branches");
	ccv_nnc_symbolic_graph_set_case_of_expr(symbolic_graph, case_of, piecewise_case_of, 0);
	ccv_nnc_symbolic_graph_t* const case_of_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(case_of_0, ONE_CPU_TENSOR(1), "b0");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_of_0, 0, TENSOR_SYMBOL_MAP(KV(b, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_0, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(b), "exp");
	ccv_nnc_graph_exec_symbol_autogen(case_of_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_1 = ccv_nnc_symbolic_graph_new();
	b = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "b1");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_of_1, 1, TENSOR_SYMBOL_MAP(KV(b, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_1, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(b), "log");
	ccv_nnc_graph_exec_symbol_autogen(case_of_1, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_2 = ccv_nnc_symbolic_graph_new();
	b = ccv_nnc_tensor_symbol_new(case_of_2, ONE_CPU_TENSOR(1), "b2");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_of_2, 2, TENSOR_SYMBOL_MAP(KV(b, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_2, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, x), TENSOR_SYMBOL_LIST(b), "1/b");
	ccv_nnc_graph_exec_symbol_autogen(case_of_2, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_3 = ccv_nnc_symbolic_graph_new();
	b = ccv_nnc_tensor_symbol_new(case_of_3, ONE_CPU_TENSOR(1), "b3");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_of_3, 3, TENSOR_SYMBOL_MAP(KV(b, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_3, CMD_SCALAR_MUL_FORWARD(1.1), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(b), "1.1b");
	ccv_nnc_graph_exec_symbol_autogen(case_of_3, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, &case_of, 1, &case_of, 1, &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = -2.2;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], expf(-2.2), 1e-5, "y should be expf(-2.2)");
	x_tensor->data.f32[0] = 1.1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], logf(1.1), 1e-5, "y should be logf(1.1)");
	x_tensor->data.f32[0] = 0.9;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], 0.9, 1e-5, "y should be 0.9");
	x_tensor->data.f32[0] = 2.2;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], 1. / 2.2, 1e-5, "y should be 1 / 2.2");
	x_tensor->data.f32[0] = 1001;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], 1.1 * 1001, 1e-2, "y should be 1.1 * 1001");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);

}

TEST_CASE("symbolic while graph contains a case..of graph and multiply its output with 0.3")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_symbolic_graph_t* const while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 4");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_4, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_graph_exec_symbol_t case_of = ccv_nnc_symbolic_graph_case_of_new(while_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_MAP(KV(x, y)), "piece-wise linear vector");
	ccv_nnc_symbolic_graph_set_case_of_expr(while_graph, case_of, piecewise_case_of, 0);
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, case_of);
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(case_of));
	ccv_nnc_symbolic_graph_t* const case_of_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y0 = ccv_nnc_tensor_symbol_new(case_of_0, ONE_CPU_TENSOR(1), "y0");
	ccv_nnc_symbolic_graph_set_case_of(while_graph, case_of, case_of_0, 0, TENSOR_SYMBOL_MAP(KV(y0, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_0, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(y0), "set");
	ccv_nnc_graph_exec_symbol_autogen(case_of_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_1 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "y1");
	ccv_nnc_symbolic_graph_set_case_of(while_graph, case_of, case_of_1, 1, TENSOR_SYMBOL_MAP(KV(y1, y)));
	ccv_nnc_tensor_symbol_t s1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "s");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_tensor_symbol_t p1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "p");
	ccv_nnc_graph_exec_symbol_new(case_of_1, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, s1), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_new(case_of_1, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, p1), TENSOR_SYMBOL_LIST(y1), "sum");
	ccv_nnc_graph_exec_symbol_autogen(case_of_1, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_2 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(case_of_2, ONE_CPU_TENSOR(1), "y2");
	ccv_nnc_symbolic_graph_set_case_of(while_graph, case_of, case_of_2, 2, TENSOR_SYMBOL_MAP(KV(y2, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_2, CMD_SET_FORWARD(1.5), 0, 0, TENSOR_SYMBOL_LIST(y2), "set");
	ccv_nnc_graph_exec_symbol_autogen(case_of_2, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(y, x)));
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "a");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(a, y), TENSOR_SYMBOL_LIST(z), "prod1");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* s1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, s1);
	s1_tensor->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* p1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, p1);
	p1_tensor->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	a_tensor->data.f32[0] = 0.3;
	x_tensor->data.f32[0] = 1.226;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_EQ_WITH_TOLERANCE((1 + 0.226 * 0.5 * 0.5 * 0.5 * 0.5) * 0.3, z_tensor->data.f32[0], 1e-6, "The piece-wise linear function applied 4 times");
	a_tensor->data.f32[0] = 0.3;
	x_tensor->data.f32[0] = 0.8;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_EQ_WITH_TOLERANCE(0.8 * 0.3, z_tensor->data.f32[0], 1e-6, "The piece-wise linear function applied 4 times");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic while graph contains a case..of graph takes input by multiplying to 0.8 and multiply its output with 0.3")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b");
	ccv_nnc_symbolic_graph_t* const while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 4");
	ccv_nnc_tensor_symbol_t s0 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "s0");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(b, s0), TENSOR_SYMBOL_LIST(x), "prod");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_4, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_graph_exec_symbol_t case_of = ccv_nnc_symbolic_graph_case_of_new(while_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_MAP(KV(x, y)), "piece-wise linear vector");
	ccv_nnc_symbolic_graph_set_case_of_expr(while_graph, case_of, piecewise_case_of, 0);
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, prod);
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(case_of));
	ccv_nnc_symbolic_graph_t* const case_of_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y0 = ccv_nnc_tensor_symbol_new(case_of_0, ONE_CPU_TENSOR(1), "y0");
	ccv_nnc_symbolic_graph_set_case_of(while_graph, case_of, case_of_0, 0, TENSOR_SYMBOL_MAP(KV(y0, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_0, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(y0), "set");
	ccv_nnc_graph_exec_symbol_autogen(case_of_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_1 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "y1");
	ccv_nnc_symbolic_graph_set_case_of(while_graph, case_of, case_of_1, 1, TENSOR_SYMBOL_MAP(KV(y1, y)));
	ccv_nnc_tensor_symbol_t s1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "s");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_tensor_symbol_t p1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "p");
	ccv_nnc_graph_exec_symbol_new(case_of_1, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, s1), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_new(case_of_1, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, p1), TENSOR_SYMBOL_LIST(y1), "sum");
	ccv_nnc_graph_exec_symbol_autogen(case_of_1, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_2 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(case_of_2, ONE_CPU_TENSOR(1), "y2");
	ccv_nnc_symbolic_graph_set_case_of(while_graph, case_of, case_of_2, 2, TENSOR_SYMBOL_MAP(KV(y2, y)));
	ccv_nnc_graph_exec_symbol_new(case_of_2, CMD_SET_FORWARD(1.5), 0, 0, TENSOR_SYMBOL_LIST(y2), "set");
	ccv_nnc_graph_exec_symbol_autogen(case_of_2, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(y, b)));
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "a");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(a, y), TENSOR_SYMBOL_LIST(z), "prod1");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* s0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, s0);
	s0_tensor->data.f32[0] = 0.8;
	ccv_nnc_tensor_t* s1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, s1);
	s1_tensor->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* p1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, p1);
	p1_tensor->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	a_tensor->data.f32[0] = 0.3;
	b_tensor->data.f32[0] = 2.5;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_EQ_WITH_TOLERANCE(1.1 * 0.8 * 0.8 * 0.3, z_tensor->data.f32[0], 1e-6, "The piece-wise linear function applied 4 times");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
