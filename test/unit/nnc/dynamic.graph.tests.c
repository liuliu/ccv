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
	ccv_nnc_tensor_variable_t b = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
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
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 1.2;
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(y));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, y), TENSOR_VARIABLE_LIST(f));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 19 * logf(19) + 1.2 * 19, 1e-5, "f(x) = 1.2 * 19 + 19 * log(19)");
	ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph, ccv_nnc_tensor_auto);
	ccv_nnc_dynamic_graph_backward(graph, f, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

#include "case_main.h"
