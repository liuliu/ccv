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
	ccv_nnc_dynamic_graph_free(graph);
}

#include "case_main.h"
