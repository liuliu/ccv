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

TEST_CASE("implement layer norm with other symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "sum");
	ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "mean");
	ccv_nnc_tensor_symbol_t whitening = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "whitening");
	ccv_nnc_tensor_symbol_t sqr = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "sqr");
	ccv_nnc_tensor_symbol_t varsum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "varsum");
	ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "var");
	ccv_nnc_tensor_symbol_t logvar = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "logvar");
	ccv_nnc_tensor_symbol_t logvar_2 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "logvar");
	ccv_nnc_tensor_symbol_t std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "std");
	ccv_nnc_tensor_symbol_t inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "inv_std");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(1, 2, 3), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / (4 * 4 * 10)), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(1, 2, 3), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / (4 * 4 * 10)), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(var), TENSOR_SYMBOL_LIST(logvar), "log(var)");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.5), TENSOR_SYMBOL_LIST(logvar), TENSOR_SYMBOL_LIST(logvar_2), "log(var)/2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(logvar_2), TENSOR_SYMBOL_LIST(std), "std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, std), TENSOR_SYMBOL_LIST(inv_std), "1/std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1, 1), TENSOR_SYMBOL_LIST(whitening, inv_std), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 8 * 4 * 4 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const layer_norm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(layer_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(layer_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(layer_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(layer_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(layer_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(layer_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(layer_norm_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(layer_norm_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(layer_norm_symbolic_graph, CMD_LAYER_NORM_FORWARD(0, 1, 2, 3), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(layer_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* layer_norm_graph = 0;
	ccv_nnc_tensor_arena_t* layer_norm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* layer_norm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(layer_norm_symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(layer_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(layer_norm_symbolic_graph), &layer_norm_graph, &layer_norm_tensor_arena, &layer_norm_graph_exec_arena);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(layer_norm_tensor_arena, bx);
	memcpy(bx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 8 * 4 * 4 * 10);
	ccv_nnc_graph_run(graph, 0, 0, 0, TRAVERSE_FULL);
	ccv_nnc_graph_run(layer_norm_graph, 0, 0, 0, TRAVERSE_FULL);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const by_tensor = ccv_nnc_tensor_from_symbol(layer_norm_tensor_arena, by);
	REQUIRE_TENSOR_EQ(y_tensor, by_tensor, "graph computed result should match batch norm op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(layer_norm_symbolic_graph);
	ccv_nnc_tensor_arena_free(layer_norm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(layer_norm_graph_exec_arena);
	ccv_nnc_graph_free(layer_norm_graph);
}

#include "case_main.h"
