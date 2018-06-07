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

TEST_CASE("simplify graph (x + y) * (x + y)")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z1), "sum1");
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z2), "sum2");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(z1, z2), TENSOR_SYMBOL_LIST(z), "prod");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(symbolic_graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
			CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
			CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		TENSOR_SYMBOL_LIST(z), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 10;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	y_tensor->data.f32[0] = 8;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], (10 + 8) * (10 + 8), 1e-5, "result should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("simplify graph with data transfer")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t a1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "a1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(a1), "sum1");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "z");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, z, DIM_ALLOC(0), DIM_ALLOC(2), ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(a1), TENSOR_SYMBOL_LIST(z1), "dt1");
	ccv_nnc_tensor_symbol_t a2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "a2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_BACKWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(a2), "dt2");
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, z, DIM_ALLOC(1), DIM_ALLOC(2), ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(a2), TENSOR_SYMBOL_LIST(z2), "dt3");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(symbolic_graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
			CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
			CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		TENSOR_SYMBOL_LIST(z), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const z2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z2);
	z2_tensor->data.f32[0] = 1.2;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	y_tensor->data.f32[0] = 8;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], 1.2 + 8, 1e-5, "result should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[1], 1.2, 1e-5, "result should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

static int custom_case_of(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	assert(input_size == 1);
	if (inputs[0]->data.f32[0] < 0)
		return 0;
	else if (inputs[0]->data.f32[0] < 2)
		return 1;
	return -1;
}

TEST_CASE("simplify graph with case..of")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t sum1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(sum1), "sum1");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_tensor_symbol_t prod1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(sum1, z), TENSOR_SYMBOL_LIST(prod1), "prod1");
	ccv_nnc_tensor_symbol_t sum2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(sum2), "sum2");
	ccv_nnc_tensor_symbol_t prod2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(sum2, z), TENSOR_SYMBOL_LIST(prod2), "prod2");
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_exec_symbol_t case_of = ccv_nnc_symbolic_graph_case_of_new(symbolic_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_MAP(KV(prod1, q)), "case..of");
	ccv_nnc_symbolic_graph_t* const case_of_1 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(case_of_1, ONE_CPU_TENSOR(1), "y1");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_of_1, 0, TENSOR_SYMBOL_MAP(KV(y1, q)));
	ccv_nnc_graph_exec_symbol_new(case_of_1, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(sum1, prod1, prod2), TENSOR_SYMBOL_LIST(y1), 0);
	ccv_nnc_graph_exec_symbol_autogen(case_of_1, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_of_2 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(case_of_2, ONE_CPU_TENSOR(1), "y2");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_of_2, 1, TENSOR_SYMBOL_MAP(KV(y2, q)));
	ccv_nnc_graph_exec_symbol_new(case_of_2, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(prod2, sum2), TENSOR_SYMBOL_LIST(y2), 0);
	ccv_nnc_graph_exec_symbol_autogen(case_of_2, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_set_case_of_expr(symbolic_graph, case_of, custom_case_of, 0);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(symbolic_graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
			CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
			CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		TENSOR_SYMBOL_LIST(q), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	x_tensor->data.f32[0] = -2;
	y_tensor->data.f32[0] = 1.1;
	z_tensor->data.f32[0] = 2.2;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* q_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, q);
	REQUIRE_EQ_WITH_TOLERANCE(q_tensor->data.f32[0], (-2 + 1.1) + (-2 + 1.1) * 2.2 + (-2 + 1.1) * 2.2, 1e-5, "q should be equal");
	x_tensor->data.f32[0] = 1.5;
	y_tensor->data.f32[0] = 1.1;
	z_tensor->data.f32[0] = 2.2;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	q_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, q);
	REQUIRE_EQ_WITH_TOLERANCE(q_tensor->data.f32[0], (1.5 + 1.1) * (1.5 + 1.1) * 2.2, 1e-5, "q should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

static int while_5(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	return inputs[0]->data.i64[0] < 5;
}

TEST_CASE("simplify graph with while, variant 1")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* const while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z1), 0);
	ccv_nnc_tensor_symbol_t p = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "p");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(p), 0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, p), TENSOR_SYMBOL_LIST(z2), 0);
	ccv_nnc_graph_exec_symbol_autogen(while_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(sum));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z2, x)));
	ccv_nnc_tensor_symbol_t f = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "f");
	// Use p here so the graph can be simplified by replacing p with z1.
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(p, z2), TENSOR_SYMBOL_LIST(f), 0);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(while_graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
			CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
			CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		TENSOR_SYMBOL_LIST(z2), SYMBOLIC_GRAPH_SOURCES(while_graph), SYMBOLIC_GRAPH_DESTINATIONS(while_graph));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	y_tensor->data.f32[0] = 1.1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const f_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, f);
	int i;
	float r = 0.5;
	for (i = 0; i < 6; i++)
		r = r * 1.1 * 2;
	REQUIRE_EQ_WITH_TOLERANCE(f_tensor->data.f32[0], r + r * 0.5, 1e-5, "should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("simplify graph with while, variant 2")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* const while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z1), 0);
	ccv_nnc_tensor_symbol_t p = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "p");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(p), 0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, p), TENSOR_SYMBOL_LIST(z2), 0);
	ccv_nnc_graph_exec_symbol_autogen(while_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(sum));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z2, x)));
	ccv_nnc_tensor_symbol_t f = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "f");
	// Use z1 here so the branch produces p will be eliminated.
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, z2), TENSOR_SYMBOL_LIST(f), 0);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(while_graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
			CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
			CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		TENSOR_SYMBOL_LIST(z2), SYMBOLIC_GRAPH_SOURCES(while_graph), SYMBOLIC_GRAPH_DESTINATIONS(while_graph));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	y_tensor->data.f32[0] = 1.1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const f_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, f);
	int i;
	float r = 0.5;
	for (i = 0; i < 6; i++)
		r = r * 1.1 * 2;
	REQUIRE_EQ_WITH_TOLERANCE(f_tensor->data.f32[0], r + r * 0.5, 1e-5, "should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("simplify graph with while, variant 3")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* const while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z1), 0);
	ccv_nnc_tensor_symbol_t p = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "p");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(p), 0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, p), TENSOR_SYMBOL_LIST(z2), 0);
	ccv_nnc_graph_exec_symbol_autogen(while_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(sum));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z2, x)));
	ccv_nnc_tensor_symbol_t f = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "f");
	// Use both z1 and p so this graph cannot be simplified (due to current implementation limitation).
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z1, p, z2), TENSOR_SYMBOL_LIST(f), 0);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(while_graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
			CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
			CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		TENSOR_SYMBOL_LIST(z2), SYMBOLIC_GRAPH_SOURCES(while_graph), SYMBOLIC_GRAPH_DESTINATIONS(while_graph));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	y_tensor->data.f32[0] = 1.1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const f_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, f);
	int i;
	float r = 0.5;
	for (i = 0; i < 6; i++)
		r = r * 1.1 * 2;
	REQUIRE_EQ_WITH_TOLERANCE(f_tensor->data.f32[0], r + r * 0.5 + r * 0.5, 1e-5, "should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
