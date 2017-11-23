#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_graph.h"

ccv_nnc_graph_exec_t ccv_nnc_graph_case_new(ccv_nnc_graph_t* const graph, const uint32_t cmd)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	ccv_nnc_graph_exec_t case_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	return case_exec;
}

void ccv_nnc_graph_set_case_expr(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_graph_case_of_f case_of, const void* cases_data)
{
}

void ccv_nnc_graph_set_case_of(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_graph_t* const case_graph, const int case_of)
{
}
