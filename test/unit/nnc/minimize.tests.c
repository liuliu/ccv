#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("solve least square sum with stochastic gradient descent on symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(2, 2), "a");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(2, 2), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(2), "bias");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(2, 2), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(a, w, bias), TENSOR_SYMBOL_LIST(b), "gemm");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(2, 2), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(b, b), TENSOR_SYMBOL_LIST(c), "square");
	ccv_nnc_tensor_symbol_t s = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "s");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0, 1), TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(s), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t updates[1];
	ccv_nnc_tensor_symbol_map_t aux[1];
	ccv_nnc_graph_exec_symbol_t update_execs[1];
	ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0.001, 0.995, 0.9, 0.9), TENSOR_SYMBOL_LIST(s), TENSOR_SYMBOL_LIST(w), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), updates, aux, update_execs);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), update_execs, 1, &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	// Relies on the inplace ops for SGD set on both updated w / bias, and momentum.
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_cmd_exec(CMD_RANDOM_UNIFORM_FORWARD(-0.5, 0.5), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(w_tensor), 0);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias);
	int i;
	for (i = 0; i < 1; i++)
	{
		ccv_nnc_tensor_t* const aux_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, aux[i].source);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(aux_tensor), 0);
	}
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const f_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, s));
	ccv_nnc_graph_exec_t sgd = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, update_execs[0]);
	for (i = 0; i < 1000; i++)
	{
		a_tensor->data.f32[0] = 10;
		a_tensor->data.f32[1] = 1;
		a_tensor->data.f32[2] = 3;
		a_tensor->data.f32[3] = 5;
		f_tensor->data.f32[0] = 1;
		bias_tensor->data.f32[0] = 1;
		bias_tensor->data.f32[1] = -1;
		if (i == 750)
			ccv_nnc_graph_exec_set(graph, sgd, CMD_SGD_FORWARD(0.000001, 0.995, 0.9, 0.9));
		else if (i == 500)
			ccv_nnc_graph_exec_set(graph, sgd, CMD_SGD_FORWARD(0.00001, 0.995, 0.9, 0.9));
		else if (i == 250)
			ccv_nnc_graph_exec_set(graph, sgd, CMD_SGD_FORWARD(0.0001, 0.995, 0.9, 0.9));
		ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
	}
	REQUIRE_EQ_WITH_TOLERANCE(a_tensor->data.f32[0] * w_tensor->data.f32[0] + a_tensor->data.f32[1] * w_tensor->data.f32[1], -1, 1e-3, "converge for vector 1");
	REQUIRE_EQ_WITH_TOLERANCE(a_tensor->data.f32[0] * w_tensor->data.f32[2] + a_tensor->data.f32[1] * w_tensor->data.f32[3], 1, 1e-3, "converge for vector 1");
	REQUIRE_EQ_WITH_TOLERANCE(a_tensor->data.f32[2] * w_tensor->data.f32[0] + a_tensor->data.f32[3] * w_tensor->data.f32[1], -1, 1e-1, "converge for vector 2");
	REQUIRE_EQ_WITH_TOLERANCE(a_tensor->data.f32[2] * w_tensor->data.f32[2] + a_tensor->data.f32[3] * w_tensor->data.f32[3], 1, 1e-1, "converge for vector 2");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("solve least square sum with stochastic gradient descent on dynamic graph")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t w = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(2, 2));
	ccv_nnc_tensor_variable_t aux = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(2, 2));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_VARIABLE_LIST(aux));
	ccv_nnc_dynamic_graph_exec(graph, CMD_RANDOM_UNIFORM_FORWARD(-0.5, 0.5), ccv_nnc_no_hint, 0, 0, 0, TENSOR_VARIABLE_LIST(w));
	int i;
	for (i = 0; i < 1000; i++)
	{
		ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(2, 2));
		ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_variable(graph, a);
		a_tensor->data.f32[0] = 10;
		a_tensor->data.f32[1] = 1;
		a_tensor->data.f32[2] = 3;
		a_tensor->data.f32[3] = 5;
		ccv_nnc_tensor_variable_t bias = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(2));
		ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_variable(graph, bias);
		bias_tensor->data.f32[0] = 1;
		bias_tensor->data.f32[1] = -1;
		ccv_nnc_tensor_variable_t b = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(a, w, bias), TENSOR_VARIABLE_LIST(b));
		ccv_nnc_tensor_variable_t c = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(b, b), TENSOR_VARIABLE_LIST(c));
		ccv_nnc_tensor_variable_t s = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_REDUCE_SUM_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(c), TENSOR_VARIABLE_LIST(s));
		ccv_nnc_dynamic_graph_minimize(graph, CMD_SGD_FORWARD(0.001, 0.995, 0.9, 0.9), TENSOR_VARIABLE_LIST(s), 0, TENSOR_VARIABLE_LIST(w), &aux);
		ccv_nnc_tensor_variable_free(graph, a);
		ccv_nnc_tensor_variable_free(graph, b);
		ccv_nnc_tensor_variable_free(graph, bias);
		ccv_nnc_tensor_variable_free(graph, c);
		ccv_nnc_tensor_variable_free(graph, s);
	}
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_variable(graph, w);
	REQUIRE_EQ_WITH_TOLERANCE(10 * w_tensor->data.f32[0] + 1 * w_tensor->data.f32[1], -1, 1e-3, "converge for vector 1");
	REQUIRE_EQ_WITH_TOLERANCE(10 * w_tensor->data.f32[2] + 1 * w_tensor->data.f32[3], 1, 1e-3, "converge for vector 1");
	REQUIRE_EQ_WITH_TOLERANCE(3 * w_tensor->data.f32[0] + 5 * w_tensor->data.f32[1], -1, 1e-1, "converge for vector 2");
	REQUIRE_EQ_WITH_TOLERANCE(3 * w_tensor->data.f32[2] + 5 * w_tensor->data.f32[3], 1, 1e-1, "converge for vector 2");
	ccv_nnc_dynamic_graph_free(graph);
}

#include "case_main.h"
