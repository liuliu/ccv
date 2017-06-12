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

int while_5(ccv_nnc_tensor_t* const* const commons, const int common_size, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const void* const data)
{
	return commons[0]->data.i64[0] < 5;
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
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph);
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(prod0));
	ccv_nnc_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_LIST(noop));
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
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph);
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t xb = ccv_nnc_tensor(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_multiview_t xx;
	ccv_nnc_tensor_multiview(&xb, (ccv_numeric_data_t[]){
			x->data, z->data
	}, CCV_NNC_MULTIVIEW_K02, while_graph, &xx);
	ccv_nnc_tensor_t zb = ccv_nnc_tensor(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_multiview_t zz;
	ccv_nnc_tensor_multiview(&zb, (ccv_numeric_data_t[]){
			z->data, x->data
	}, CCV_NNC_MULTIVIEW_K02, while_graph, &zz);
	ccv_nnc_tensor_t zbb = ccv_nnc_tensor(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_reference_to_multiview(&zz, 0, &zbb);
	ccv_nnc_graph_exec_t prod = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST((ccv_nnc_tensor_t*)&xx, y), TENSOR_LIST((ccv_nnc_tensor_t*)&zz));
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(prod));
	ccv_nnc_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod);
	x->data.f32[0] = 0.32;
	y->data.f32[0] = 2.8;
	ccv_nnc_graph_while_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(loop));
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
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t xd = ccv_nnc_tensor(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_multiview_t xx;
	ccv_nnc_tensor_multiview(&xd, (ccv_numeric_data_t[]){
			x->data, z->data
	}, CCV_NNC_MULTIVIEW_K02, while_graph, &xx);
	ccv_nnc_tensor_t zd = ccv_nnc_tensor(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_multiview_t zz;
	ccv_nnc_tensor_multiview(&zd, (ccv_numeric_data_t[]){
			z->data, x->data
	}, CCV_NNC_MULTIVIEW_K02, while_graph, &zz);
	ccv_nnc_graph_exec_t prod = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST((ccv_nnc_tensor_t*)&xx, y), TENSOR_LIST((ccv_nnc_tensor_t*)&zz));
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(prod));
	ccv_nnc_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod);
	ccv_nnc_graph_exec_set_io(while_graph, prod, TENSOR_LIST(x, y), TENSOR_LIST(z));
	ccv_nnc_graph_exec_set_io(while_graph, prod, TENSOR_LIST((ccv_nnc_tensor_t*)&zz, y), TENSOR_LIST((ccv_nnc_tensor_t*)&xx));
	x->data.f32[0] = 0.32;
	z->data.f32[0] = 0.47;
	y->data.f32[0] = 5.5;
	ccv_nnc_graph_while_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(loop));
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], 0.47 * 5.5 * 5.5 * 5.5 * 5.5 * 5.5, 1e-2, "computed result of 0.47 * 5.5 ^ 5 should be the same");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
}

TEST_CASE("symbolic graph for a while loop to compute x ^ 5 * y")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, while_graph, "for 1..5");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z0, x), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, prod0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z1, y), TENSOR_SYMBOL_LIST(z2), "prod1");
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
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z2);
	x_tensor->data.f32[0] = 0.92;
	y_tensor->data.f32[0] = 3.2;
	z0_tensor->data.f32[0] = 1;
	ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(graph_exec_arena);
	ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(graph_exec_arena);
	ccv_nnc_graph_while_run(graph, 0, 0, &source, 1, &destination, 1);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], 0.92 * 0.92 * 0.92 * 0.92 * 0.92 * 3.2, 1e-6, "z should be equal to x ^ 5 * y");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("symbolic graph for a while loop to compute z * x ^ 5 * y + z")
{
	/*
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
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* z0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z0);
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z3);
	x_tensor->data.f32[0] = 0.92;
	y_tensor->data.f32[0] = 3.2;
	z0_tensor->data.f32[0] = 1.2;
	ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(graph_exec_arena);
	ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(graph_exec_arena);
	ccv_nnc_graph_while_run(graph, 0, 0, &source, 1, &destination, 1);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], 1.2 * 0.92 * 0.92 * 0.92 * 0.92 * 0.92 * 3.2 + 1.2, 1e-6, "z should be equal to z * x ^ 5 * y + z");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
	*/
}

TEST_CASE("symbolic graph for a while loop to compute x = max(conv(x, w, b), 3x3) 5 times")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5, 5, 4), "x");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "z");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_CONVOLUTION_FORWARD(4, 3, 3, 4), TENSOR_SYMBOL_LIST(x, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_t avg = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(z), "avg");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, conv);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_while_params(while_graph, TENSOR_SYMBOL_MAP(KV(z, x)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(avg));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
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
		ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(4, 3, 3, 4), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(x1, w_tensor, b_tensor), TENSOR_LIST(y1), 0);
		ccv_nnc_cmd_exec(CMD_AVERAGE_POOL_FORWARD(3, 3), HINT((1, 1), (1, 1)), 0, TENSOR_LIST(y1), TENSOR_LIST(x1), 0);
	}
	ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(graph_exec_arena);
	ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(graph_exec_arena);
	ccv_nnc_graph_while_run(graph, 0, 0, &source, 1, &destination, 1);
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
	ccv_nnc_symbolic_graph_while(symbolic_graph, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(5, 5, 4), "x");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4, 3, 3, 4), "w");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(4), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(5, 5, 4), "y");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t conv = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_CONVOLUTION_FORWARD(4, 3, 3, 4), TENSOR_SYMBOL_LIST(x, w, b), TENSOR_SYMBOL_LIST(y), "conv");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, conv);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_while_params(while_graph, TENSOR_SYMBOL_MAP(KV(y, x)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(conv));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	int i;
	for (i = 0; i < 5 * 5 * 4; i++)
		x_tensor->data.f32[i] = 0.92;
	for (i = 0; i < 4 * 3 * 3 * 4; i++)
		w_tensor->data.f32[i] = 3.2;
	for (i = 0; i < 4; i++)
		b_tensor->data.f32[i] = 3.2;
	for (i = 0; i < 5 * 5 * 4; i++)
		y_tensor->data.f32[i] = 1;
	ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(graph_exec_arena);
	ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(graph_exec_arena);
	ccv_nnc_graph_while_run(graph, 0, 0, &source, 1, &destination, 1);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
