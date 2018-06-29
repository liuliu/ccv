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

TEST_CASE("reduce sum for [[1, 2, 3], [4, 5, 6]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		6,
		15
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, ONE_CPU_TENSOR(2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce sum for [[1, 2, 3], [4, 5, 6]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		5, 7, 9
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, ONE_CPU_TENSOR(3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("use reduce for softmax")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100), "x");
	ccv_nnc_tensor_symbol_t max = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "max");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_MAX_FORWARD(0), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(max), "max");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, max), TENSOR_SYMBOL_LIST(y), "neg");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(z), "exp");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0), TENSOR_SYMBOL_LIST(z), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_tensor_symbol_t inv_sum = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "1 / sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, sum), TENSOR_SYMBOL_LIST(inv_sum), "inv sum");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(100), "a");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(z, inv_sum), TENSOR_SYMBOL_LIST(a), "softmax");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t da = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, a);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_tensor_t* const da_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, TENSOR_BIND_MAP(KV(a, a_tensor), KV(da, da_tensor)), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const tx_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 100; i++)
		x_tensor->data.f32[i] = tx_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 100; i++)
		da_tensor->data.f32[i] = 0;
	da_tensor->data.f32[88] = 1;
	ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
	ccv_nnc_tensor_t* const ta_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tx_tensor), TENSOR_LIST(ta_tensor), 0);
	REQUIRE_TENSOR_EQ(a_tensor, ta_tensor, "softmax should match from the graph");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da_tensor, 0, ta_tensor), TENSOR_LIST(tdx_tensor), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_TENSOR_EQ(dx_tensor, tdx_tensor, "softmax backward should match from the graph");
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(tx_tensor);
	ccv_nnc_tensor_free(ta_tensor);
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(da_tensor);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
