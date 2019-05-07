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

TEST_CASE("schedule a simple graph for parallel execution")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "x");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "y");
	const ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z), "mul");
	const ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "a");
	const ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "b");
	const ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "sum");
	const ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "d");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(z, c), TENSOR_SYMBOL_LIST(d), "div");
	const ccv_nnc_tensor_symbol_t d0 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "d0");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(d), TENSOR_SYMBOL_LIST(d0), "log");
	const ccv_nnc_tensor_symbol_t d1 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "d1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(d), TENSOR_SYMBOL_LIST(d1), "exp");
	const ccv_nnc_tensor_symbol_t d2 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "d2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(d0, d1), TENSOR_SYMBOL_LIST(d2), "sum1");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph,
		0, 0,
		TENSOR_SYMBOL_LIST(d2),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_graph_static_schedule(graph, CCV_STREAM_CONTEXT_CPU);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 2;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	y_tensor->data.f32[0] = 0.21;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	a_tensor->data.f32[0] = 2.2;
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	b_tensor->data.f32[0] = 3.2;
	ccv_nnc_graph_run(graph, 0, 0, 0, TRAVERSE_FULL);
	ccv_nnc_tensor_t* const d2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, d2);
	const float dv = 2 * 0.21 / (2.2 + 3.2);
	REQUIRE_EQ_WITH_TOLERANCE(d2_tensor->data.f32[0], logf(dv) + expf(dv), 1e-5, "result should be equal");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("schedule symbolic graph to data parallel")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 16, 32, 32, 3), 0);
	const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 8, 5, 5, 3), 0);
	const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 8), 0);
	const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 16, 32, 32, 8), 0);
	const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
	const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 16, 16, 16, 8), 0);
	const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
	const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 8, 5, 5, 8), 0);
	const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 8), 0);
	const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 16, 8, 8, 8), 0);
	const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
	const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 16, 1, 1, 8), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
	const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 16), "label");
	const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 16, 1, 1, 8), "y5");
	const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "loss");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t updated_params[4];
	ccv_nnc_tensor_symbol_t gradients[4];
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9));
	ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
	ccv_nnc_graph_exec_symbol_t updated_execs[4];
	ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
	ccv_nnc_symbolic_graph_data_parallel(symbolic_graph, 2, TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, gradients, 4, CCV_NNC_PARALLEL_REDUCE_OP_SUM, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), updated_execs, 4);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

#include "case_main.h"
