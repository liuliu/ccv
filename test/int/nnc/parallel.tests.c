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

TEST_CASE("schedule symbolic graph to data parallel")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16, 3, 32, 32), 0);
	const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 8, 3, 5, 5), 0);
	const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 8), 0);
	const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16, 8, 32, 32), 0);
	const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
	const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16, 8, 16, 16), 0);
	const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
	const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 8, 8, 5, 5), 0);
	const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 8), 0);
	const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16, 8, 8, 8), 0);
	const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
	const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16, 8, 1, 1), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
	const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16, 1, 1, 1), "label");
	const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16, 8), "y5");
	const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16), "loss");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t updated_params[4];
	ccv_nnc_tensor_symbol_t gradients[4];
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9));
	ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
	ccv_nnc_graph_exec_symbol_t updated_execs[4];
	ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
	ccv_nnc_symbolic_graph_data_parallel(symbolic_graph, 1, TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), gradients, 4, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), updated_execs, 4);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph,
		0, 0,
		updated_params, 4,
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_graph_static_schedule(graph, CCV_STREAM_CONTEXT_GPU);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_run(graph, 0, ccv_nnc_graph_default_stream(graph), 0, TRAVERSE_FULL);
	ccv_nnc_stream_context_wait(ccv_nnc_graph_default_stream(graph));
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
