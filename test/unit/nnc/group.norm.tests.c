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

TEST_CASE("implement simple group norm (like layer norm) with other symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "sum");
	ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "mean");
	ccv_nnc_tensor_symbol_t whitening = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "whitening");
	ccv_nnc_tensor_symbol_t sqr = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "sqr");
	ccv_nnc_tensor_symbol_t varsum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "varsum");
	ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "var");
	ccv_nnc_tensor_symbol_t logvar = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "logvar");
	ccv_nnc_tensor_symbol_t logvar_2 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "logvar");
	ccv_nnc_tensor_symbol_t std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "std");
	ccv_nnc_tensor_symbol_t inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "inv_std");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(3), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 10), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(3), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 10), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(var), TENSOR_SYMBOL_LIST(logvar), "log(var)");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.5), TENSOR_SYMBOL_LIST(logvar), TENSOR_SYMBOL_LIST(logvar_2), "log(var)/2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(logvar_2), TENSOR_SYMBOL_LIST(std), "std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, std), TENSOR_SYMBOL_LIST(inv_std), "1/std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(whitening, inv_std), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 8 * 4 * 4 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const group_norm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 4, 4, 10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 4, 4, 10), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_GROUP_NORM_FORWARD(3, 1, 0), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(group_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* group_norm_graph = 0;
	ccv_nnc_tensor_arena_t* group_norm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* group_norm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(group_norm_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(group_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(group_norm_symbolic_graph), &group_norm_graph, &group_norm_tensor_arena, &group_norm_graph_exec_arena);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, bx);
	memcpy(bx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 8 * 4 * 4 * 10);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(group_norm_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const by_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, by);
	REQUIRE_TENSOR_EQ(y_tensor, by_tensor, "graph computed result should match batch norm op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(group_norm_symbolic_graph);
	ccv_nnc_tensor_arena_free(group_norm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(group_norm_graph_exec_arena);
	ccv_nnc_graph_free(group_norm_graph);
}

TEST_CASE("compare simple group norm (like layer norm) gradient with other symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "sum");
	ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "mean");
	ccv_nnc_tensor_symbol_t whitening = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "whitening");
	ccv_nnc_tensor_symbol_t sqr = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "sqr");
	ccv_nnc_tensor_symbol_t varsum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "varsum");
	ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "var");
	ccv_nnc_tensor_symbol_t logvar = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "logvar");
	ccv_nnc_tensor_symbol_t logvar_2 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "logvar");
	ccv_nnc_tensor_symbol_t std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "std");
	ccv_nnc_tensor_symbol_t inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "inv_std");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(3), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 10), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(3), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 10), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(var), TENSOR_SYMBOL_LIST(logvar), "log(var)");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.5), TENSOR_SYMBOL_LIST(logvar), TENSOR_SYMBOL_LIST(logvar_2), "log(var)/2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(logvar_2), TENSOR_SYMBOL_LIST(std), "std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, std), TENSOR_SYMBOL_LIST(inv_std), "1/std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(whitening, inv_std), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 8 * 4 * 4 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const group_norm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 10), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 4, 4, 10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 4, 4, 10), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_GROUP_NORM_FORWARD(3, 1, 0), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(group_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(group_norm_symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale, bias), SYMBOLIC_GRAPH_SOURCES(group_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(group_norm_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(group_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(group_norm_symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(group_norm_symbolic_graph, bx);
	ccv_nnc_graph_t* group_norm_graph = 0;
	ccv_nnc_tensor_arena_t* group_norm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* group_norm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(group_norm_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(group_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(group_norm_symbolic_graph), &group_norm_graph, &group_norm_tensor_arena, &group_norm_graph_exec_arena);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, bx);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, dby);
	for (i = 0; i < 8 * 4 * 4 * 10; i++)
		dby_tensor->data.f32[i] = dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	memcpy(bx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 8 * 4 * 4 * 10);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(group_norm_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, dbx);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dbx_tensor->data.f32, 8 * 4 * 4 * 10, 1e-5, "graph computed result should match layer norm op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(group_norm_symbolic_graph);
	ccv_nnc_tensor_arena_free(group_norm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(group_norm_graph_exec_arena);
	ccv_nnc_graph_free(group_norm_graph);
}

TEST_CASE("implement group norm with other symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "x");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "sum");
	ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "mean");
	ccv_nnc_tensor_symbol_t whitening = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "whitening");
	ccv_nnc_tensor_symbol_t sqr = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "sqr");
	ccv_nnc_tensor_symbol_t varsum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "varsum");
	ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "var");
	ccv_nnc_tensor_symbol_t logvar = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "logvar");
	ccv_nnc_tensor_symbol_t logvar_2 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "logvar");
	ccv_nnc_tensor_symbol_t std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "std");
	ccv_nnc_tensor_symbol_t inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "inv_std");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(1), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 4), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(1), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 4), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(var), TENSOR_SYMBOL_LIST(logvar), "log(var)");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.5), TENSOR_SYMBOL_LIST(logvar), TENSOR_SYMBOL_LIST(logvar_2), "log(var)/2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(logvar_2), TENSOR_SYMBOL_LIST(std), "std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, std), TENSOR_SYMBOL_LIST(inv_std), "1/std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(whitening, inv_std), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 8 * 4 * 4 * 5; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const group_norm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 16, 5), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 16, 5), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 16, 5), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 16, 5), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 5), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 5), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 0), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(group_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* group_norm_graph = 0;
	ccv_nnc_tensor_arena_t* group_norm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* group_norm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(group_norm_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(group_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(group_norm_symbolic_graph), &group_norm_graph, &group_norm_tensor_arena, &group_norm_graph_exec_arena);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, bx);
	memcpy(bx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 8 * 16 * 5);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(group_norm_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const by_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, by);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, by_tensor->data.f32, 8 * 16 * 5, 1e-5, "graph computed result should match batch norm op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(group_norm_symbolic_graph);
	ccv_nnc_tensor_arena_free(group_norm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(group_norm_graph_exec_arena);
	ccv_nnc_graph_free(group_norm_graph);
}

TEST_CASE("compare group norm gradient with other symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "x");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "sum");
	ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "mean");
	ccv_nnc_tensor_symbol_t whitening = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "whitening");
	ccv_nnc_tensor_symbol_t sqr = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "sqr");
	ccv_nnc_tensor_symbol_t varsum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "varsum");
	ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "var");
	ccv_nnc_tensor_symbol_t logvar = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "logvar");
	ccv_nnc_tensor_symbol_t logvar_2 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "logvar");
	ccv_nnc_tensor_symbol_t std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "std");
	ccv_nnc_tensor_symbol_t inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 1, 5), "inv_std");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 4, 5), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(1), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 4), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(1), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 4), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(var), TENSOR_SYMBOL_LIST(logvar), "log(var)");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.5), TENSOR_SYMBOL_LIST(logvar), TENSOR_SYMBOL_LIST(logvar_2), "log(var)/2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(logvar_2), TENSOR_SYMBOL_LIST(std), "std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, std), TENSOR_SYMBOL_LIST(inv_std), "1/std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(whitening, inv_std), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 8 * 4 * 4 * 5; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const group_norm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 16, 5), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 16, 5), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 16, 5), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 16, 5), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 5), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(group_norm_symbolic_graph, CPU_TENSOR_NHWC(32F, 8, 4, 5), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(group_norm_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 0), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(group_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(group_norm_symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale, bias), SYMBOLIC_GRAPH_SOURCES(group_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(group_norm_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(group_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(group_norm_symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(group_norm_symbolic_graph, bx);
	ccv_nnc_graph_t* group_norm_graph = 0;
	ccv_nnc_tensor_arena_t* group_norm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* group_norm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(group_norm_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(group_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(group_norm_symbolic_graph), &group_norm_graph, &group_norm_tensor_arena, &group_norm_graph_exec_arena);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, bx);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, dby);
	for (i = 0; i < 8 * 16 * 5; i++)
		dby_tensor->data.f32[i] = dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	memcpy(bx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 8 * 16 * 5);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(group_norm_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(group_norm_tensor_arena, dbx);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dbx_tensor->data.f32, 8 * 16 * 5, 1e-5, "graph computed result should match layer norm op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(group_norm_symbolic_graph);
	ccv_nnc_tensor_arena_free(group_norm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(group_norm_graph_exec_arena);
	ccv_nnc_graph_free(group_norm_graph);
}

#include "case_main.h"
