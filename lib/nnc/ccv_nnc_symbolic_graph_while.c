#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "ccv_nnc_symbolic_graph_internal.h"

/**
 * Level-4 API
 */

ccv_nnc_graph_exec_symbol_t ccv_nnc_symbolic_graph_while(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* conditions, const int condition_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, const ccv_nnc_graph_while_f while_func, const ccv_nnc_tensor_symbol_t* inputs, const int input_size, const ccv_nnc_tensor_symbol_t* outputs, const int output_size, const ccv_nnc_tensor_symbol_map_t* symbol_map, const int symbol_map_size)
{
	ccv_nnc_graph_exec_symbol_t symbol = {
		.d = graph->exec_symbol_info->rnum,
		.graph = graph
	};
#define visitor(node, idx, ...)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
	return symbol;
}
