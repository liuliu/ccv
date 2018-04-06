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

TEST_CASE("implement batch norm with fine-grained symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "sum");
	ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t whitening = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(8, 4, 4, 10), "whitening");
	ccv_nnc_tensor_symbol_t sqr = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(8, 4, 4, 10), "sqr");
	ccv_nnc_tensor_symbol_t varsum = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "varsum");
	ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_t logvar = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "logvar");
	ccv_nnc_tensor_symbol_t logvar_2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "logvar");
	ccv_nnc_tensor_symbol_t std = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "std");
	ccv_nnc_tensor_symbol_t inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "inv_std");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(8, 4, 4, 10), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0, 1, 2), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / (8 * 4 * 4)), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0, 1, 2), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / (8 * 4 * 4)), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
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
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 8 * 4 * 4 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const batch_norm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(8, 4, 4, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(8, 4, 4, 10), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "bias");
	ccv_nnc_tensor_symbol_t bmean = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t bvar = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_t bmean_out = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t bvar_out = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(batch_norm_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(batch_norm_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(batch_norm_symbolic_graph, CMD_BATCH_NORM_FORWARD(0, 0, 0.9, 0, 1, 2), TENSOR_SYMBOL_LIST(bx, scale, bias, bmean, bvar), TENSOR_SYMBOL_LIST(by, bmean_out, bvar_out, saved_mean, saved_inv_std), "batch_norm");
	ccv_nnc_graph_exec_symbol_autogen(batch_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* batch_norm_graph = 0;
	ccv_nnc_tensor_arena_t* batch_norm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* batch_norm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(batch_norm_symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(batch_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(batch_norm_symbolic_graph), &batch_norm_graph, &batch_norm_tensor_arena, &batch_norm_graph_exec_arena);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(batch_norm_tensor_arena, bx);
	memcpy(bx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 8 * 4 * 4 * 10);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_graph_run(batch_norm_graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const by_tensor = ccv_nnc_tensor_from_symbol(batch_norm_tensor_arena, by);
	REQUIRE_TENSOR_EQ(y_tensor, by_tensor, "graph computed result should match batch norm op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(batch_norm_symbolic_graph);
	ccv_nnc_tensor_arena_free(batch_norm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(batch_norm_graph_exec_arena);
	ccv_nnc_graph_free(batch_norm_graph);
}

TEST_CASE("compare batch norm gradient with fine-grained symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "sum");
	ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t whitening = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "whitening");
	ccv_nnc_tensor_symbol_t sqr = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "sqr");
	ccv_nnc_tensor_symbol_t varsum = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "varsum");
	ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_t std = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "std");
	ccv_nnc_tensor_symbol_t inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(10), "inv_std");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0, 1, 2), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / (2 * 2 * 2)), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0, 1, 2), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / (2 * 2 * 2)), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSQRT_FORWARD(), TENSOR_SYMBOL_LIST(var), TENSOR_SYMBOL_LIST(std), "sqrt(var)");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, std), TENSOR_SYMBOL_LIST(inv_std), "1/std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1, 1), TENSOR_SYMBOL_LIST(whitening, inv_std), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const batch_norm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "bias");
	ccv_nnc_tensor_symbol_t bmean = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t bvar = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_set_flags(batch_norm_symbolic_graph, bmean, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_set_flags(batch_norm_symbolic_graph, bvar, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_t bmean_out = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t bvar_out = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(batch_norm_symbolic_graph, ONE_CPU_TENSOR(10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(batch_norm_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(batch_norm_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(batch_norm_symbolic_graph, CMD_BATCH_NORM_FORWARD(0, 0, 0.9, 0, 1, 2), TENSOR_SYMBOL_LIST(bx, scale, bias, bmean, bvar), TENSOR_SYMBOL_LIST(by, bmean_out, bvar_out, saved_mean, saved_inv_std), "batch_norm");
	ccv_nnc_graph_exec_symbol_autogen(batch_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(batch_norm_symbolic_graph, SYMBOLIC_GRAPH_SOURCES(batch_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(batch_norm_symbolic_graph), TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale, bias));
	ccv_nnc_graph_exec_symbol_autogen(batch_norm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(batch_norm_symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(batch_norm_symbolic_graph, bx);
	ccv_nnc_graph_t* batch_norm_graph = 0;
	ccv_nnc_tensor_arena_t* batch_norm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* batch_norm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(batch_norm_symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(batch_norm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(batch_norm_symbolic_graph), &batch_norm_graph, &batch_norm_tensor_arena, &batch_norm_graph_exec_arena);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(batch_norm_tensor_arena, bx);
	memcpy(bx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * 10);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(batch_norm_tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * 10; i++)
		dby_tensor->data.f32[i] = dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_graph_run(batch_norm_graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(batch_norm_tensor_arena, dbx);
	REQUIRE_TENSOR_EQ(dx_tensor, dbx_tensor, "graph computed result should match batch norm op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(batch_norm_symbolic_graph);
	ccv_nnc_tensor_arena_free(batch_norm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(batch_norm_graph_exec_arena);
	ccv_nnc_graph_free(batch_norm_graph);
}

TEST_CASE("compare batch norm with cudnn")
{
	if (!ccv_nnc_cmd_ok(CCV_NNC_BATCH_NORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		!ccv_nnc_cmd_ok(CCV_NNC_BATCH_NORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		!(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)))
		return;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "bias");
	ccv_nnc_tensor_symbol_t bmean = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "mean");
	ccv_nnc_tensor_symbol_t bvar = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "var");
	ccv_nnc_tensor_symbol_set_flags(symbolic_graph, bmean, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_set_flags(symbolic_graph, bvar, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_t bmean_out = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "mean");
	ccv_nnc_tensor_symbol_t bvar_out = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "var");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_BATCH_NORM_FORWARD(1e-4, 0, 0.9, 0, 1, 2), TENSOR_SYMBOL_LIST(bx, scale, bias, bmean, bvar), TENSOR_SYMBOL_LIST(by, bmean_out, bvar_out, saved_mean, saved_inv_std), "batch_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 2, 2, 10), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const by_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, by);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 2, 2, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(by_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "bias");
	ccv_nnc_tensor_symbol_t cmean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t cvar = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_set_flags(cpu_symbolic_graph, cmean, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_set_flags(cpu_symbolic_graph, cvar, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_t cmean_out = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t cvar_out = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(cscale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(cbias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_BATCH_NORM_FORWARD(1e-4, 0, 0.9, 0, 1, 2), TENSOR_SYMBOL_LIST(cx, cscale, cbias, cmean, cvar), TENSOR_SYMBOL_LIST(cy, cmean_out, cvar_out, csaved_mean, csaved_inv_std), "batch_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * 10);
	ccv_nnc_graph_run(cpu_graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const cy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cy);
	REQUIRE_TENSOR_EQ(y_tensor, cy_tensor, "batch norm result from cudnn should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
}

TEST_CASE("compare batch norm gradient with cudnn")
{
	if (!ccv_nnc_cmd_ok(CCV_NNC_BATCH_NORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		!ccv_nnc_cmd_ok(CCV_NNC_BATCH_NORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		!(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)))
		return;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "bias");
	ccv_nnc_tensor_symbol_t bmean = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "mean");
	ccv_nnc_tensor_symbol_t bvar = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "var");
	ccv_nnc_tensor_symbol_set_flags(symbolic_graph, bmean, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_set_flags(symbolic_graph, bvar, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_t bmean_out = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "mean");
	ccv_nnc_tensor_symbol_t bvar_out = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "var");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(00, 10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(scale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(bias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_BATCH_NORM_FORWARD(1e-4, 0, 0.9, 0, 1, 2), TENSOR_SYMBOL_LIST(bx, scale, bias, bmean, bvar), TENSOR_SYMBOL_LIST(by, bmean_out, bvar_out, saved_mean, saved_inv_std), "batch_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale, bias));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 2, 2, 10), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 2, 2, 10), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * 10; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 2, 2, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "bias");
	ccv_nnc_tensor_symbol_t cmean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t cvar = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_set_flags(cpu_symbolic_graph, cmean, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_set_flags(cpu_symbolic_graph, cvar, CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS);
	ccv_nnc_tensor_symbol_t cmean_out = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "mean");
	ccv_nnc_tensor_symbol_t cvar_out = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "var");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, ONE_CPU_TENSOR(10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(cscale), "set_scale");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_SET_FORWARD(0), 0, 0, TENSOR_SYMBOL_LIST(cbias), "set_bias");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_BATCH_NORM_FORWARD(1e-4, 0, 0.9, 0, 1, 2), TENSOR_SYMBOL_LIST(cx, cscale, cbias, cmean, cvar), TENSOR_SYMBOL_LIST(cy, cmean_out, cvar_out, csaved_mean, csaved_inv_std), "batch_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx, cscale, cbias));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * 10);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * 10);
	ccv_nnc_graph_run(cpu_graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);
	REQUIRE_TENSOR_EQ(dx_tensor, dcx_tensor, "batch norm gradient result from cudnn should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
}

#include "case_main.h"
