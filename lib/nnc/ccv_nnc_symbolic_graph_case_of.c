#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-4 API
 */

ccv_nnc_graph_exec_symbol_t ccv_nnc_symbolic_graph_case_of_new(ccv_nnc_symbolic_graph_t* const graph, const uint32_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size, const char* const name)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	// Added one more symbol.
	const ccv_nnc_graph_exec_symbol_t symbol = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), inputs, input_size, 0, 0, name);
	// ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	return symbol;
}

void ccv_nnc_symbolic_graph_set_case_of_expr(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec, ccv_nnc_graph_case_of_f case_of, const void* case_of_data)
{
	ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	symbol_info->case_of.expr = case_of;
	symbol_info->case_of.data = case_of_data;
}

void ccv_nnc_symbolic_graph_set_case_of(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec, ccv_nnc_symbolic_graph_t* const case_graph, const int case_of, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size)
{
}
