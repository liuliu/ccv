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

TEST_CASE("compare softmax + categorical crossentropy v.s. softmax crossentropy command")
{
	ccv_nnc_symbolic_graph_t* const graph = ccv_nnc_symbolic_graph_new();
	// batch size = 2, dim = 3.
	const ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_NHWC(2, 3), "a");
	const ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_NHWC(2, 3), "b0");
	ccv_nnc_graph_exec_symbol_new(graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b0), "softmax");
	const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_LABEL(2), "label");
	const ccv_nnc_tensor_symbol_t loss0 = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "loss0");
	ccv_nnc_graph_exec_symbol_new(graph, CMD_CATEGORICAL_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(b0, label), TENSOR_SYMBOL_LIST(loss0), "categorical crossentropy");
	const ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(graph, CPU_TENSOR_NHWC(2, 3), "b1");
	const ccv_nnc_tensor_symbol_t loss1 = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_auto, "loss1");
	ccv_nnc_graph_exec_symbol_new(graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(a, label), TENSOR_SYMBOL_LIST(loss1, b1), "softmax crossentropy");
	ccv_nnc_graph_exec_symbol_autogen(graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(graph, TENSOR_SYMBOL_LIST(loss0), TENSOR_SYMBOL_LIST(a), SYMBOLIC_GRAPH_SOURCES(graph), SYMBOLIC_GRAPH_DESTINATIONS(graph));
	const ccv_nnc_tensor_symbol_t dloss0 = ccv_nnc_tensor_symbol_for_backward(graph, loss0);
	const ccv_nnc_tensor_symbol_t da0 = ccv_nnc_tensor_symbol_for_backward(graph, a);
	ccv_nnc_symbolic_graph_backward(graph, TENSOR_SYMBOL_LIST(loss1), TENSOR_SYMBOL_LIST(a), SYMBOLIC_GRAPH_SOURCES(graph), SYMBOLIC_GRAPH_DESTINATIONS(graph));
	const ccv_nnc_tensor_symbol_t dloss1 = ccv_nnc_tensor_symbol_for_backward(graph, loss1);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(dloss0, dloss1), "set 1");
	const ccv_nnc_tensor_symbol_t da1 = ccv_nnc_tensor_symbol_for_backward(graph, a);
	ccv_nnc_graph_exec_symbol_autogen(graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* run_graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(graph), SYMBOLIC_GRAPH_DESTINATIONS(graph), &run_graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	a_tensor->data.f32[0] = 10;
	a_tensor->data.f32[1] = -1;
	a_tensor->data.f32[2] = -5;
	a_tensor->data.f32[3] = 12;
	a_tensor->data.f32[4] = 4;
	a_tensor->data.f32[5] = 24;
	ccv_nnc_tensor_t* const label_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, label);
	label_tensor->data.i32[0] = 2;
	label_tensor->data.i32[1] = 1;
	ccv_nnc_graph_run(run_graph, 0, 0, TRAVERSE_FULL);
	ccv_nnc_tensor_t* const da0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, da0);
	ccv_nnc_tensor_t* const da1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, da1);
	REQUIRE_TENSOR_EQ(da0_tensor, da1_tensor, "two tensors from combined op and separate ops should be equal");
	ccv_nnc_graph_free(run_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(graph);
}

#include "case_main.h"
