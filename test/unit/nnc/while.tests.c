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

static int while_5(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	return inputs[0]->data.i64[0] < 5;
}

TEST_CASE("graph for a while loop to compute 0.34 * 1.11 ^ 5")
{
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_t* while_graph = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, CMD_NOOP(), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_t prod0 = ccv_nnc_graph_exec_new(while_graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, TENSOR_LIST(y, z), TENSOR_LIST(z));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod0);
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph);
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(prod0));
	ccv_nnc_tensor_t count_tensor = ccv_nnc_tensor_for_while_count(while_graph);
	ccv_nnc_graph_set_while_expr(while_graph, while_5, 0, TENSOR_LIST(&count_tensor), GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_exec_t prod1 = ccv_nnc_graph_exec_new(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, TENSOR_LIST(x, z), TENSOR_LIST(z));
	ccv_nnc_graph_exec_concat(graph, loop, prod1);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	x->data.f32[0] = 0.34;
	y->data.f32[0] = 1.11;
	z->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(prod1));
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
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph);
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, CMD_NOOP(), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_multiview_t xx;
	ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
			x, z
	}, CCV_NNC_MULTIVIEW_K0N, 2, while_graph, &xx);
	ccv_nnc_tensor_multiview_t zz;
	ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
			z, x
	}, CCV_NNC_MULTIVIEW_K0N, 2, while_graph, &zz);
	ccv_nnc_tensor_t zbb = ccv_nnc_tensor(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_synchronize_to_multiview(&zz, &zbb);
	ccv_nnc_graph_exec_t prod = ccv_nnc_graph_exec_new(while_graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, TENSOR_LIST((ccv_nnc_tensor_t*)&xx, y), TENSOR_LIST((ccv_nnc_tensor_t*)&zz));
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(prod));
	ccv_nnc_tensor_t count_tensor = ccv_nnc_tensor_for_while_count(while_graph);
	ccv_nnc_graph_set_while_expr(while_graph, while_5, 0, TENSOR_LIST(&count_tensor), GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	x->data.f32[0] = 0.32;
	y->data.f32[0] = 2.8;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(loop));
	REQUIRE_EQ_WITH_TOLERANCE(z->data.f32[0], 0.32 * 2.8 * 2.8 * 2.8 * 2.8 * 2.8, 1e-5, "computed result of 0.32 * 2.8 ^ 5 should be the same");
	REQUIRE(z->data.f32 == zbb.data.f32, "Two pointers should be the same");
	ccv_nnc_tensor_multiview_free(xx);
	ccv_nnc_tensor_multiview_free(zz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
}

TEST_CASE("while graph add and re-add reuse tensor allocations for 0.47 * 5.5 ^ 5")
{
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_graph_t* while_graph = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph);
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, CMD_NOOP(), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_multiview_t xx;
	ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
			x, z
	}, CCV_NNC_MULTIVIEW_K0N, 2, while_graph, &xx);
	ccv_nnc_tensor_multiview_t zz;
	ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
			z, x
	}, CCV_NNC_MULTIVIEW_K0N, 2, while_graph, &zz);
	ccv_nnc_graph_exec_t prod = ccv_nnc_graph_exec_new(while_graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, TENSOR_LIST((ccv_nnc_tensor_t*)&xx, y), TENSOR_LIST((ccv_nnc_tensor_t*)&zz));
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(prod));
	ccv_nnc_tensor_t count_tensor = ccv_nnc_tensor_for_while_count(while_graph);
	ccv_nnc_graph_set_while_expr(while_graph, while_5, 0, TENSOR_LIST(&count_tensor), GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod);
	ccv_nnc_graph_exec_set_io(while_graph, prod, TENSOR_LIST(x, y), TENSOR_LIST(z));
	ccv_nnc_graph_exec_set_io(while_graph, prod, TENSOR_LIST((ccv_nnc_tensor_t*)&zz, y), TENSOR_LIST((ccv_nnc_tensor_t*)&xx));
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	x->data.f32[0] = 0.32;
	z->data.f32[0] = 0.47;
	y->data.f32[0] = 5.5;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(loop));
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], 0.47 * 5.5 * 5.5 * 5.5 * 5.5 * 5.5, 1e-2, "computed result of 0.47 * 5.5 ^ 5 should be the same");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
}

TEST_CASE("symbolic graph get while graph from symbol")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* const while_graph_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_graph_exec_symbol_t while_0 = ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph_0, "while 0");
	ccv_nnc_symbolic_graph_t* const while_graph_1 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_graph_exec_symbol_t while_1 = ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph_1, "while 1");
	REQUIRE(while_graph_0 == ccv_nnc_symbolic_graph_from_while_symbol(symbolic_graph, while_0), "graph extracted from symbol should be equal");
	REQUIRE(while_graph_1 == ccv_nnc_symbolic_graph_from_while_symbol(symbolic_graph, while_1), "graph extracted from symbol should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("symbolic graph for a while loop to compute x ^ 5 * y")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "for 1..5");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(z0, x), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, prod0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(z1, y), TENSOR_SYMBOL_LIST(z2), "prod1");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	// Test whether we can ignore the noop graph properly in compilation.
	ccv_nnc_symbolic_graph_t* noop_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, noop_graph, "no op");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z1, z0)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(prod0));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z2);
	x_tensor->data.f32[0] = 0.92;
	y_tensor->data.f32[0] = 3.2;
	z0_tensor->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], 0.92 * 0.92 * 0.92 * 0.92 * 0.92 * 3.2, 1e-6, "z should be equal to x ^ 5 * y");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph for a while loop to compute z * x ^ 5 * y + z")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "for 1..5");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(z0, x), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, prod0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_tensor_symbol_t z3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z3");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(z1, y), TENSOR_SYMBOL_LIST(z2), "prod1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(z2, z0), TENSOR_SYMBOL_LIST(z3), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph), z0), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z1, z0)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(prod0));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z0);
	x_tensor->data.f32[0] = 0.92;
	y_tensor->data.f32[0] = 3.2;
	z0_tensor->data.f32[0] = 1.2;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z3);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], 1.2 * 0.92 * 0.92 * 0.92 * 0.92 * 0.92 * 3.2 + 1.2, 1e-6, "z should be equal to z * x ^ 5 * y + z");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph for a while loop to compute x = max(conv(x, w, b), 3x3) 5 times")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5, 5, 4), "x");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "z");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), TENSOR_SYMBOL_LIST(x, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_t avg = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(z), "avg");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, conv);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph), w), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z, x)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(avg));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 5 * 5 * 4; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 4 * 3 * 3 * 4; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (4 * 3 * 3);
	for (i = 0; i < 4; i++)
		b_tensor->data.f32[i] = 3.2;
	ccv_nnc_tensor_t* x1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memcpy(x1->data.f32, x_tensor->data.f32, sizeof(float) * 5 * 5 * 4);
	ccv_nnc_tensor_t* y1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	for (i = 0; i < 5; i++)
	{
		ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(x1, w_tensor, b_tensor), TENSOR_LIST(y1), 0);
		ccv_nnc_cmd_exec(CMD_AVERAGE_POOL_FORWARD(3, 3), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(y1), TENSOR_LIST(x1), 0);
	}
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_MATRIX_EQ(x1, z_tensor, "5x5x4 matrix should be exactly the same");
	ccv_nnc_tensor_free(x1);
	ccv_nnc_tensor_free(y1);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph for a while loop to compute x = conv(x, w, b) 5 times")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5, 5, 4), "x");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_alias_new(while_graph, y, DIM_ALLOC(0, 0, 1), DIM_ALLOC(5, 5, 4), ONE_CPU_TENSOR(5, 5, 3), "z0");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_alias_new(while_graph, y, DIM_ALLOC(4), DIM_ALLOC(10), ONE_CPU_TENSOR(6), "z1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), TENSOR_SYMBOL_LIST(x, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, conv);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(y, x)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(conv));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z0);
	ccv_nnc_tensor_t* z1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z1);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 5 * 5 * 4; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* x1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memcpy(x1->data.f32, x_tensor->data.f32, sizeof(float) * 5 * 5 * 4);
	for (i = 0; i < 4 * 3 * 3 * 4; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (4 * 3 * 3);
	for (i = 0; i < 4; i++)
		b_tensor->data.f32[i] = 0.1;
	for (i = 0; i < 5 * 5 * 4; i++)
		y_tensor->data.f32[i] = 1;
	REQUIRE(y_tensor->data.f32 != x_tensor->data.f32, "y tensor and x tensor should point to different data");
	ccv_nnc_tensor_t* y1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	for (i = 0; i < 5; i++)
	{
		ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(x1, w_tensor, b_tensor), TENSOR_LIST(y1), 0);
		memcpy(x1->data.f32, y1->data.f32, sizeof(float) * 5 * 5 * 4);
	}
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_MATRIX_EQ(y1, y_tensor, "5x5x4 matrix should be exactly the same");
	REQUIRE(z0_tensor->data.f32 == y_tensor->data.f32 + 1, "z0 should point to the same memory region as y, offset by 1");
	// z0 and z1 trigger different code path (z1 will trigger one additional tensor setup).
	REQUIRE(z1_tensor->data.f32 == y_tensor->data.f32 + 4, "z1 should point to the same memory region as y, offset by 4");
	ccv_nnc_tensor_free(x1);
	ccv_nnc_tensor_free(y1);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph for a while loop to compute x = conv(x, w, b) 5 times then z = x0 + x")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5, 5, 4), "x");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), TENSOR_SYMBOL_LIST(x, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, conv);
	ccv_nnc_tensor_symbol_t xz = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x0");
	ccv_nnc_tensor_symbol_t yz = ccv_nnc_tensor_symbol_alias_new(while_graph, y, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "y0");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5 * 5 * 4), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(xz, yz), TENSOR_SYMBOL_LIST(z), "sum");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(y, x)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(conv));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 5 * 5 * 4; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* x0 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	ccv_nnc_tensor_t* x1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memcpy(x0->data.f32, x_tensor->data.f32, sizeof(float) * 5 * 5 * 4);
	memcpy(x1->data.f32, x_tensor->data.f32, sizeof(float) * 5 * 5 * 4);
	for (i = 0; i < 4 * 3 * 3 * 4; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (4 * 3 * 3);
	for (i = 0; i < 4; i++)
		b_tensor->data.f32[i] = 0.1;
	for (i = 0; i < 5 * 5 * 4; i++)
		y_tensor->data.f32[i] = 1;
	REQUIRE(y_tensor->data.f32 != x_tensor->data.f32, "y tensor and x tensor should point to different data");
	ccv_nnc_tensor_t* y1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	for (i = 0; i < 5; i++)
	{
		ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(x1, w_tensor, b_tensor), TENSOR_LIST(y1), 0);
		memcpy(x1->data.f32, y1->data.f32, sizeof(float) * 5 * 5 * 4);
	}
	for (i = 0; i < 5 * 5 * 4; i++)
		x0->data.f32[i] += y1->data.f32[i];
	ccv_nnc_tensor_t z0 = ccv_nnc_tensor(x0->data.f32, ONE_CPU_TENSOR(5 * 5 * 4), 0);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_MATRIX_EQ(&z0, z_tensor, "5x5x4 matrix should be exactly the same");
	ccv_nnc_tensor_free(x0);
	ccv_nnc_tensor_free(x1);
	ccv_nnc_tensor_free(y1);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph for a while loop to compute (x = conv(x, w, b) 5 times, z = x0 + x (x <- z)) 5 times")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* sub_while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_graph_exec_symbol_t sub_while = ccv_nnc_symbolic_graph_while(while_graph, CCV_NNC_GRAPH_FORWARD, sub_while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "x");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(sub_while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(sub_while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_graph_exec_symbol_t sub_noop = ccv_nnc_graph_exec_symbol_new(sub_while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(sub_while_graph, CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), TENSOR_SYMBOL_LIST(x, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_concat(sub_while_graph, sub_noop, conv);
	ccv_nnc_tensor_symbol_t xz = ccv_nnc_tensor_symbol_alias_new(while_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x0");
	ccv_nnc_tensor_symbol_t yz = ccv_nnc_tensor_symbol_alias_new(while_graph, y, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "y0");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "z");
	ccv_nnc_tensor_symbol_t zz = ccv_nnc_tensor_symbol_alias_new(while_graph, z, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "z0");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(xz, yz), TENSOR_SYMBOL_LIST(zz), "sum");
	ccv_nnc_symbolic_graph_set_while_expr(sub_while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(sub_while_graph), ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(sub_noop));
	ccv_nnc_symbolic_graph_set_carry_overs(sub_while_graph, TENSOR_SYMBOL_MAP(KV(y, x)));
	ccv_nnc_symbolic_graph_set_sources(sub_while_graph, GRAPH_EXEC_SYMBOL_LIST(sub_noop));
	ccv_nnc_symbolic_graph_set_destinations(sub_while_graph, GRAPH_EXEC_SYMBOL_LIST(conv));
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, sub_while);
	ccv_nnc_graph_exec_symbol_concat(while_graph, sub_while, sum);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z, x)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(sum));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	int i, j;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 5 * 5 * 4; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* x0 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	ccv_nnc_tensor_t* x1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memcpy(x0->data.f32, x_tensor->data.f32, sizeof(float) * 5 * 5 * 4);
	for (i = 0; i < 4 * 3 * 3 * 4; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (4 * 3 * 3);
	for (i = 0; i < 4; i++)
		b_tensor->data.f32[i] = 0.1;
	for (i = 0; i < 5 * 5 * 4; i++)
		y_tensor->data.f32[i] = 1;
	REQUIRE(y_tensor->data.f32 != x_tensor->data.f32, "y tensor and x tensor should point to different data");
	ccv_nnc_tensor_t* y1 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	for (i = 0; i < 5; i++)
	{
		memcpy(x1->data.f32, x0->data.f32, sizeof(float) * 5 * 5 * 4);
		for (j = 0; j < 5; j++)
		{
			ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(x1, w_tensor, b_tensor), TENSOR_LIST(y1), 0);
			memcpy(x1->data.f32, y1->data.f32, sizeof(float) * 5 * 5 * 4);
		}
		for (j = 0; j < 5 * 5 * 4; j++)
			x0->data.f32[j] += y1->data.f32[j];
	}
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	REQUIRE_MATRIX_EQ(x0, z_tensor, "5x5x4 matrix should be exactly the same");
	ccv_nnc_tensor_free(x0);
	ccv_nnc_tensor_free(x1);
	ccv_nnc_tensor_free(y1);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("symbolic graph for a while loop to compute y = conv(x1, w, b), x2 = x0 + y 5 times, (x0 <- x1, x1 <- x2)")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5, 5, 4), "x1");
	ccv_nnc_tensor_symbol_t x0 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "x0");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), TENSOR_SYMBOL_LIST(x1, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, conv);
	ccv_nnc_tensor_symbol_t x0z = ccv_nnc_tensor_symbol_alias_new(while_graph, x0, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x0");
	ccv_nnc_tensor_symbol_t yz = ccv_nnc_tensor_symbol_alias_new(while_graph, y, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "y");
	ccv_nnc_tensor_symbol_t x2 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "x2");
	ccv_nnc_tensor_symbol_t x2z = ccv_nnc_tensor_symbol_alias_new(while_graph, x2, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x2");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x0z, yz), TENSOR_SYMBOL_LIST(x2z), "sum");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(x2, x1), KV(x1, x0)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(conv));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x0);
	ccv_nnc_tensor_t* x1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x1);
	ccv_nnc_tensor_t* w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	int i, j;
	memset(x0_tensor->data.f32, 0, sizeof(float) * 5 * 5 * 4);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 5 * 5 * 4; i++)
		x1_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 4 * 3 * 3 * 4; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (4 * 3 * 3);
	for (i = 0; i < 4; i++)
		b_tensor->data.f32[i] = 0.1;
	for (i = 0; i < 5 * 5 * 4; i++)
		y_tensor->data.f32[i] = 1;
	ccv_nnc_tensor_t* x0t = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memset(x0t->data.f32, 0, sizeof(float) * 5 * 5 * 4);
	ccv_nnc_tensor_t* x1t = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memcpy(x1t->data.f32, x1_tensor->data.f32, sizeof(float) * 5 * 5 * 4);
	ccv_nnc_tensor_t* x2t = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	for (i = 0; i < 5; i++)
	{
		ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(x1t, w_tensor, b_tensor), TENSOR_LIST(x2t), 0);
		for (j = 0; j < 5 * 5 * 4; j++)
			x2t->data.f32[j] += x0t->data.f32[j];
		memcpy(x0t->data.f32, x1t->data.f32, sizeof(float) * 5 * 5 * 4);
		memcpy(x1t->data.f32, x2t->data.f32, sizeof(float) * 5 * 5 * 4);
	}
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* x2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x2);
	REQUIRE_MATRIX_EQ(x2t, x2_tensor, "5x5x4 matrix should be exactly the same");
	ccv_nnc_tensor_free(x0t);
	ccv_nnc_tensor_free(x1t);
	ccv_nnc_tensor_free(x2t);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("a while loop to compute y = conv(x1, w, b), y1 = 1.2 * y, x3 = x0 + y1 5 times, then z = y + x3 + x2 + x1 + x0, (x0 <- x1, x1 <- x2, x2 <- x3)")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5, 5, 4), "x1");
	ccv_nnc_tensor_symbol_t y0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5 * 5 * 4), "y0");
	ccv_nnc_tensor_symbol_t x0 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "x0");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5 * 5 * 4), "y1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), TENSOR_SYMBOL_LIST(x1, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, conv);
	ccv_nnc_tensor_symbol_t x0z = ccv_nnc_tensor_symbol_alias_new(while_graph, x0, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x0");
	ccv_nnc_tensor_symbol_t yz = ccv_nnc_tensor_symbol_alias_new(while_graph, y, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "y");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(yz, y0), TENSOR_SYMBOL_LIST(y1), "prod");
	ccv_nnc_tensor_symbol_t x2 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "x2");
	ccv_nnc_tensor_symbol_t x2z = ccv_nnc_tensor_symbol_alias_new(while_graph, x2, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x2");
	ccv_nnc_tensor_symbol_t x3 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "x3");
	ccv_nnc_tensor_symbol_t x3z = ccv_nnc_tensor_symbol_alias_new(while_graph, x3, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x2");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(x0z, y1), TENSOR_SYMBOL_LIST(x3z), "sum");
	ccv_nnc_tensor_symbol_t x1s = ccv_nnc_tensor_symbol_resolve(while_graph, x1);
	ccv_nnc_tensor_symbol_t x1z = ccv_nnc_tensor_symbol_alias_new(while_graph, x1s, ccv_nnc_no_ofs, DIM_ALLOC(5 * 5 * 4), ONE_CPU_TENSOR(5 * 5 * 4), "x1");
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(x3, x2), KV(x2, x1), KV(x1, x0)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(conv));
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5 * 5 * 4), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(yz, x0z, x1z, x2z, x3z), TENSOR_SYMBOL_LIST(z), "sum1");
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x0);
	ccv_nnc_tensor_t* x1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x1);
	ccv_nnc_tensor_t* w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* y0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y0);
	int i, j;
	memset(x0_tensor->data.f32, 0, sizeof(float) * 5 * 5 * 4);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 5 * 5 * 4; i++)
		x1_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 4 * 3 * 3 * 4; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (4 * 3 * 3);
	for (i = 0; i < 4; i++)
		b_tensor->data.f32[i] = 0.1;
	for (i = 0; i < 5 * 5 * 4; i++)
		y_tensor->data.f32[i] = 1;
	for (i = 0; i < 5 * 5 * 4; i++)
		y0_tensor->data.f32[i] = 1.2;
	ccv_nnc_tensor_t* yt = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	ccv_nnc_tensor_t* x0t = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memset(x0t->data.f32, 0, sizeof(float) * 5 * 5 * 4);
	ccv_nnc_tensor_t* x1t = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	memcpy(x1t->data.f32, x1_tensor->data.f32, sizeof(float) * 5 * 5 * 4);
	ccv_nnc_tensor_t* x2t = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	ccv_nnc_tensor_t* x3t = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5, 5, 4), 0);
	for (i = 0; i < 5; i++)
	{
		ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(1, 4, 3, 3, 4), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(x1t, w_tensor, b_tensor), TENSOR_LIST(yt), 0);
		for (j = 0; j < 5 * 5 * 4; j++)
			x3t->data.f32[j] = 1.2 * yt->data.f32[j] + x0t->data.f32[j];
		memcpy(x0t->data.f32, x1t->data.f32, sizeof(float) * 5 * 5 * 4);
		memcpy(x1t->data.f32, x2t->data.f32, sizeof(float) * 5 * 5 * 4);
		memcpy(x2t->data.f32, x3t->data.f32, sizeof(float) * 5 * 5 * 4);
	}
	for (j = 0; j < 5 * 5 * 4; j++)
		x3t->data.f32[j] += yt->data.f32[j] + x2t->data.f32[j] + x1t->data.f32[j] + x0t->data.f32[j];
	ccv_nnc_tensor_t zt = ccv_nnc_tensor(x3t->data.f32, ONE_CPU_TENSOR(5 * 5 * 4), 0);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_MATRIX_EQ(&zt, z_tensor, "100 vector should be exactly the same");
	ccv_nnc_tensor_free(yt);
	ccv_nnc_tensor_free(x0t);
	ccv_nnc_tensor_free(x1t);
	ccv_nnc_tensor_free(x2t);
	ccv_nnc_tensor_free(x3t);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
