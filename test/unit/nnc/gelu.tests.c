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

TEST_CASE("compare fast GELU gradient with fine-grained symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x");
	const ccv_nnc_tensor_symbol_t x_sq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x_sq");
	const ccv_nnc_tensor_symbol_t x_cube = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x_cube");
	const ccv_nnc_tensor_symbol_t x_sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x_sum");
	const ccv_nnc_tensor_symbol_t beta_x_sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "beta_x_sum");
	const ccv_nnc_tensor_symbol_t one = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "one");
	const ccv_nnc_tensor_symbol_t tanh = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "tanh");
	const ccv_nnc_tensor_symbol_t tanh_1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "1_tanh");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(one), "one");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(x, x), TENSOR_SYMBOL_LIST(x_sq), "x_sq");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(0.044715), TENSOR_SYMBOL_LIST(x_sq, x), TENSOR_SYMBOL_LIST(x_cube), "x_cube");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, 1), TENSOR_SYMBOL_LIST(x, x_cube), TENSOR_SYMBOL_LIST(x_sum), "x_sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.797884560802865355), TENSOR_SYMBOL_LIST(x_sum), TENSOR_SYMBOL_LIST(beta_x_sum), "beta_x_sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TANH_FORWARD(), TENSOR_SYMBOL_LIST(beta_x_sum), TENSOR_SYMBOL_LIST(tanh), "tanh");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, 1), TENSOR_SYMBOL_LIST(tanh, one), TENSOR_SYMBOL_LIST(tanh_1), "tanh_1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(0.5), TENSOR_SYMBOL_LIST(x, tanh_1), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	const ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(y, dx),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const swish_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t sx = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x");
	ccv_nnc_tensor_symbol_t sy = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(swish_symbolic_graph, CMD_GELU_FORWARD(1), TENSOR_SYMBOL_LIST(sx), TENSOR_SYMBOL_LIST(sy), "gelu");
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(swish_symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(swish_symbolic_graph, TENSOR_SYMBOL_LIST(sy), TENSOR_SYMBOL_LIST(sx), SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t sdx = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sx);
	const ccv_nnc_tensor_symbol_t sdy = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sy);
	ccv_nnc_graph_t* swish_graph = 0;
	ccv_nnc_tensor_arena_t* swish_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* swish_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(swish_symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(sy, sdx),
		SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph),
		&swish_graph, &swish_tensor_arena, &swish_graph_exec_arena);
	ccv_nnc_tensor_t* const sx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sx);
	memcpy(sx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 10);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* const sdy_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdy);
	for (i = 0; i < 10; i++)
		sdy_tensor->data.f32[i] = dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(swish_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const sy_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sy);
	REQUIRE_TENSOR_EQ(y_tensor, sy_tensor, "graph computed result should match swish op result");
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const sdx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdx);
	REQUIRE_TENSOR_EQ(dx_tensor, sdx_tensor, "gradient computed result should match swish op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(swish_symbolic_graph);
	ccv_nnc_tensor_arena_free(swish_tensor_arena);
	ccv_nnc_graph_exec_arena_free(swish_graph_exec_arena);
	ccv_nnc_graph_free(swish_graph);
}

#include "case_main.h"
