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

TEST_CASE("dynamic graph to compute log(19)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, a)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t b = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(a), TENSOR_VARIABLE_LIST(b));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, b)->data.f32[0], logf(19), 1e-5, "log(19) result should be equal.");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph to compute f(x) = x * log(x) + 1.2 * x, f'(x) where x = 19")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 1.2;
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, z), TENSOR_VARIABLE_LIST(f));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 19 * logf(19) + 1.2 * 19, 1e-5, "f(x) = 1.2 * 19 + 19 * log(19)");
	// Do gradient computation multiple times.
	ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, f, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	ccv_nnc_tensor_variable_t dy = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, f, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(dy));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dy)->data.f32[0], 19, 1e-5, "f'(y) = 19");
	ccv_nnc_tensor_variable_t dz = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, f, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(dz));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dz)->data.f32[0], 1, 1e-5, "f'(z) = 1");
	ccv_nnc_dynamic_graph_backward(graph, f, TENSOR_VARIABLE_LIST(y, x), TENSOR_VARIABLE_LIST(dy, dx));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dy)->data.f32[0], 19, 1e-5, "f'(y) = 19");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph with dense net (extensive use of alias)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 4));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 0.472;
	ccv_nnc_tensor_variable_t x1 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_tensor_variable_t w1 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 1));
	ccv_nnc_tensor_from_variable(graph, w1)->data.f32[0] = 0.234;
	ccv_nnc_tensor_variable_t b1 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, b1)->data.f32[0] = 0.1;
	ccv_nnc_tensor_variable_t x11 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 1), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x1, w1, b1), TENSOR_VARIABLE_LIST(x11));
	ccv_nnc_tensor_variable_t x2 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 2));
	ccv_nnc_tensor_variable_t w2 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 2));
	ccv_nnc_tensor_from_variable(graph, w2)->data.f32[0] = 0.374;
	ccv_nnc_tensor_from_variable(graph, w2)->data.f32[1] = 0.886;
	ccv_nnc_tensor_variable_t b2 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, b2)->data.f32[0] = 0.2;
	ccv_nnc_tensor_variable_t x21 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 2), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x2, w2, b2), TENSOR_VARIABLE_LIST(x21));
	ccv_nnc_tensor_variable_t x3 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 3));
	ccv_nnc_tensor_variable_t w3 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 3));
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[0] = 0.484;
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[1] = 0.912;
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[2] = 0.235;
	ccv_nnc_tensor_variable_t b3 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, b3)->data.f32[0] = 0.3;
	ccv_nnc_tensor_variable_t x31 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 3), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x3, w3, b3), TENSOR_VARIABLE_LIST(x31));
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	/*
	ccv_nnc_tensor_variable_t dw1 = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, x, TENSOR_VARIABLE_LIST(w1), TENSOR_VARIABLE_LIST(dw1));
	*/
	ccv_nnc_dynamic_graph_free(graph);
}

#include "case_main.h"
