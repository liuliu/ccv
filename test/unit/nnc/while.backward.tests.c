#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/_ccv_nnc_graph.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static int while_4(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	return inputs[0]->data.i64[0] < 4;
}

TEST_CASE("graph with a while loop to compute back propagation 0.34 * 1.11 ^ 5")
{
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* x0 = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	x->type |= CCV_TAPE_ALLOC;
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	z->type |= CCV_TAPE_ALLOC;
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_t* while_graph = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph);
	ccv_nnc_tensor_multiview_t xx;
	ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
			x0, z, x
	}, CCV_NNC_MULTIVIEW_K1N, 2, while_graph, &xx);
	xx.type |= CCV_TAPE_ALLOC;
	ccv_nnc_tensor_multiview_t zz;
	ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
			z, x
	}, CCV_NNC_MULTIVIEW_K0N, 2, while_graph, &zz);
	zz.type |= CCV_TAPE_ALLOC;
	ccv_nnc_graph_exec_t prod0 = ccv_nnc_graph_exec_new(while_graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, TENSOR_LIST(y, (ccv_nnc_tensor_t*)&xx), TENSOR_LIST((ccv_nnc_tensor_t*)&zz));
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, CMD_NOOP(), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_concat(while_graph, prod0, noop);
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(prod0));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_tensor_t count_tensor = ccv_nnc_tensor_for_while_count(while_graph);
	ccv_nnc_graph_set_while_expr(while_graph, while_4, 0, TENSOR_LIST(&count_tensor), GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_t* while_back_graph = ccv_nnc_graph_new();
	while_back_graph->peer = while_graph;
	ccv_nnc_graph_exec_t back_loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_BACKWARD, while_back_graph);
	ccv_nnc_tensor_t* dx = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_exec_t back_prod0 = ccv_nnc_graph_exec_new(while_back_graph, CMD_EWPROD_BACKWARD(), ccv_nnc_no_hint, TENSOR_LIST(g, y, (ccv_nnc_tensor_t*)&xx, (ccv_nnc_tensor_t*)&zz), TENSOR_LIST(dx, g));
	ccv_nnc_graph_exec_t back_noop = ccv_nnc_graph_exec_new(while_back_graph, CMD_NOOP(), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_concat(while_back_graph, back_noop, back_prod0);
	ccv_nnc_graph_set_sources(while_back_graph, GRAPH_EXEC_LIST(back_noop));
	ccv_nnc_graph_set_destinations(while_back_graph, GRAPH_EXEC_LIST(back_prod0));
	ccv_nnc_graph_set_while_expr(while_back_graph, while_4, 0, 0, 0, GRAPH_EXEC_LIST(back_noop));
	ccv_nnc_graph_exec_concat(graph, loop, back_loop);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	x0->data.f32[0] = 0.34;
	y->data.f32[0] = 1.11;
	g->data.f32[0] = 1;
	ccv_nnc_tensor_tape_t* tape = ccv_nnc_tensor_tape_new();
	ccv_nnc_graph_run(graph, tape, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(back_loop));
	ccv_nnc_graph_free(graph);
	REQUIRE_EQ_WITH_TOLERANCE(g->data.f32[0], 1.11 * 1.11 * 1.11 * 1.11 * 1.11, 1e-6, "back propagation of 0.34 * 1.11 ^ 5 should be 1.11 ^ 5");
	ccv_nnc_tensor_tape_free(tape);
	ccv_nnc_tensor_multiview_free(xx);
	ccv_nnc_tensor_multiview_free(zz);
	ccv_nnc_tensor_free(x0);
	ccv_nnc_tensor_free(dx);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
	ccv_nnc_tensor_free(g);
}

TEST_CASE("symbolic graph with a while loop z = log(x * y) (x <- z) 5 times, then u = v * z, compute y'")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t xy = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "xy");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_tensor_symbol_t u = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "u");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "v");
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while0");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(xy), "prod0");
	ccv_nnc_graph_exec_symbol_t log0 = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(xy), TENSOR_SYMBOL_LIST(z), "log0");
	ccv_nnc_graph_exec_symbol_autogen(while_graph, GRAPH_EXEC_SYMBOL_LIST(prod0, log0), 0);
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_concat(while_graph, log0, noop);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(z, v), TENSOR_SYMBOL_LIST(u), "prod1");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_4, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z, x)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(prod0));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	x_tensor->data.f32[0] = 1;
	y_tensor->data.f32[0] = 3.2;
	v_tensor->data.f32[0] = 0.22;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* u_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, u);
	float z0 = 1, y0 = 3.2;
	int i;
	for (i = 0; i < 5; i++)
		z0 = log(z0 * y0);
	z0 = 0.22 * z0;
	REQUIRE_EQ_WITH_TOLERANCE(u_tensor->data.f32[0], z0, 1e-6, "u should match the for loop result");
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(u), TENSOR_SYMBOL_LIST(y), ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_exec_symbol_t dyx = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dy);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), &dyx, 1, &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	v_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	x_tensor->data.f32[0] = 1;
	y_tensor->data.f32[0] = 3.2;
	v_tensor->data.f32[0] = 0.22;
	ccv_nnc_tensor_t* du_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, u));
	du_tensor->data.f32[0] = 1;
	ccv_nnc_tensor_tape_t* tape = ccv_nnc_tensor_tape_new();
	ccv_nnc_graph_run(graph, tape, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	// Effectively, we are computing:
	// D[Log[Log[Log[Log[Log[x * y] * y] * y] * y] * y] * v, y]
	// From WolframAlpha: http://www.wolframalpha.com/input/?i=D%5BLog%5BLog%5BLog%5BLog%5BLog%5Bx+*+y%5D+*+y%5D+*+y%5D+*+y%5D+*+y%5D+*+v,+y%5D 
	const float dya = 0.22 * (((1 / log(1 * 3.2) + 1) / (log(3.2 * log(1 * 3.2)) * log(3.2 * log(3.2 * log(1 * 3.2)))) + 1 / log(3.2 * log(3.2 * log(1 * 3.2))) + 1) / (3.2 * log(3.2 * log(3.2 * log(3.2 * log(1 * 3.2))))) + 1 / 3.2);
	REQUIRE_EQ_WITH_TOLERANCE(dy_tensor->data.f32[0], dya, 1e-6, "back propagation of this while loop should match WolframAlpha result");
	ccv_nnc_tensor_tape_free(tape);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
