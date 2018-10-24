#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

#pragma mark - Level-3.5 API

void ccv_nnc_symbolic_graph_data_parallel(ccv_nnc_symbolic_graph_t* const graph, const int parallel, const ccv_nnc_tensor_symbol_t* const scatters, const int scatter_size, const ccv_nnc_tensor_symbol_t* const gathers, const int gather_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	ccv_nnc_graph_visit_t* const visit = ccv_nnc_graph_visit_new(graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
	ccv_nnc_graph_visit_free(visit);
}
