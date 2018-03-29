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
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1.0 / (8 * 4 * 4)), TENSOR_SYMBOL_LIST(sum), TENSOR_SYMBOL_LIST(mean), "mean");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, mean), TENSOR_SYMBOL_LIST(whitening), "whitening");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(whitening, whitening), TENSOR_SYMBOL_LIST(sqr), "sqr");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0, 1, 2), TENSOR_SYMBOL_LIST(sqr), TENSOR_SYMBOL_LIST(varsum), "varsum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1.0 / (8 * 4 * 4)), TENSOR_SYMBOL_LIST(varsum), TENSOR_SYMBOL_LIST(var), "var");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(var), TENSOR_SYMBOL_LIST(logvar), "log(var)");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(0.5), TENSOR_SYMBOL_LIST(logvar), TENSOR_SYMBOL_LIST(logvar_2), "log(var)/2");
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
	int i, j;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 8 * 4 * 4 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(8, 4, 4, 10), 0);
	memcpy(xt->data.f32, x_tensor->data.f32, sizeof(float) * 8 * 4 * 4 * 10);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	float meanval[10] = {};
	for (i = 0; i < 8 * 4 * 4; i++)
		for (j = 0; j < 10; j++)
			meanval[j] += xt->data.f32[i * 10 + j];
	for (i = 0; i < 10; i++)
		meanval[i] = meanval[i] / (8 * 4 * 4);
	float invstdval[10] = {};
	for (i = 0; i < 8 * 4 * 4; i++)
		for (j = 0; j < 10; j++)
			invstdval[j] += (xt->data.f32[i * 10 + j] - meanval[j]) * (xt->data.f32[i * 10 + j] - meanval[j]);
	for (i = 0; i < 10; i++)
		invstdval[i] = invstdval[i] / (8 * 4 * 4);
	for (i = 0; i < 10; i++)
		invstdval[i] = 1 / sqrtf(invstdval[i]);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(8, 4, 4, 10), 0);
	for (i = 0; i < 8 * 4 * 4; i++)
		for (j = 0; j < 10; j++)
			yt->data.f32[i * 10 + j] = (xt->data.f32[i * 10 + j] - meanval[j]) * invstdval[j];
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_TENSOR_EQ(y_tensor, yt, "graph computed batch norm should be the same");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(xt);
	ccv_nnc_tensor_free(yt);
}

#include "case_main.h"
