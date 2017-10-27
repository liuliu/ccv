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

/*
static int while_5(ccv_nnc_tensor_t* const* const commons, const int common_size, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const void* const data)
{
	return commons[0]->data.i64[0] < 5;
}
*/

TEST_CASE("graph for a while loop to compute 1.11 ^ 5")
{
	/*
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_tensor_t* x = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* y = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_tensor_t* z = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_nnc_graph_t* while_graph = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t noop = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_t prod0 = ccv_nnc_graph_exec_new(while_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST(y, z), TENSOR_LIST(z));
	ccv_nnc_graph_exec_concat(while_graph, noop, prod0);
	ccv_nnc_graph_exec_t loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_FORWARD, while_graph);
	ccv_nnc_graph_set_sources(while_graph, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_set_destinations(while_graph, GRAPH_EXEC_LIST(prod0));
	ccv_nnc_graph_set_while_expr(while_graph, while_5, 0, GRAPH_EXEC_LIST(noop));
	ccv_nnc_graph_t* while_back_graph = ccv_nnc_graph_new();
	ccv_nnc_graph_exec_t back_loop = ccv_nnc_graph_while(graph, CCV_NNC_GRAPH_BACKWARD, while_back_graph);
	ccv_nnc_graph_exec_t back_prod0 = ccv_nnc_graph_exec_new(while_back_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, TENSOR_LIST(y, z), TENSOR_LIST(z));
	ccv_nnc_graph_exec_t back_noop = ccv_nnc_graph_exec_new(while_back_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_concat(while_graph, back_prod0, back_noop);
	ccv_nnc_graph_exec_concat(graph, loop, back_loop);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	x->data.f32[0] = 0.34;
	y->data.f32[0] = 1.11;
	z->data.f32[0] = 1;
	ccv_nnc_graph_while_run(graph, 0, 0, GRAPH_EXEC_LIST(loop), GRAPH_EXEC_LIST(back_loop));
	ccv_nnc_graph_free(graph);
	REQUIRE_EQ_WITH_TOLERANCE(z->data.f32[0], 0.34 * 1.11 * 1.11 * 1.11 * 1.11 * 1.11, 1e-6, "computed result of 0.34 * 1.11 ^ 5 should be the same");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
	*/
}

#include "case_main.h"
