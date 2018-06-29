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
	else
		return 2;
}

TEST_CASE("symbolic graph for piece-wise function y = f(x), compute y'")
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
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), GRAPH_EXEC_SYMBOL_LIST(case_of), GRAPH_EXEC_SYMBOL_LIST(case_of));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	const ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_exec_symbol_t case_of_backward = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dx);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, &case_of, 1, &case_of_backward, 1, &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(graph_exec_arena);
	ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(graph_exec_arena);
	const ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* s1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, s1);
	ccv_nnc_tensor_t* p1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, p1);
	s1_tensor->data.f32[0] = 0.5;
	p1_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[0] = -1;
	dy_tensor->data.f32[0] = 1;
	ccv_nnc_tensor_tape_t* tape = ccv_nnc_tensor_tape_new();
	ccv_nnc_graph_run(graph, tape, 0, &source, 1, &destination, 1);
	ccv_nnc_tensor_t* dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(dx_tensor->data.f32[0], 0, 1e-5, "in negative region should equal to 0");
	s1_tensor->data.f32[0] = 0.5;
	p1_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[0] = 0.76;
	dy_tensor->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, tape, 0, &source, 1, &destination, 1);
	dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(dx_tensor->data.f32[0], 1, 1e-5, "y = x in (0, 1), y' = 1 (gradient passthrough)");
	s1_tensor->data.f32[0] = 0.5;
	p1_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[0] = 1.226;
	dy_tensor->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, tape, 0, &source, 1, &destination, 1);
	dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(dx_tensor->data.f32[0], 0.5, 1e-5, "y = (x - 1) * 0.5 + 1 in (1, 2), y' = 0.5");
	s1_tensor->data.f32[0] = 0.5;
	p1_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[0] = 2.1;
	dy_tensor->data.f32[0] = 1;
	ccv_nnc_graph_run(graph, tape, 0, &source, 1, &destination, 1);
	dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(dx_tensor->data.f32[0], 0, 1e-5, "y = 1.5 if x > 2, y' = 0");
	ccv_nnc_tensor_tape_free(tape);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
