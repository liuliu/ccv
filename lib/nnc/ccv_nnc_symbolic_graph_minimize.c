#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-3.5 API
 */

void ccv_nnc_symbolic_graph_minimize(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_cmd_t minimizer, const ccv_nnc_tensor_symbol_t* const losses, const int loss_size, const ccv_nnc_tensor_symbol_t* const parameters, const int parameter_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, ccv_nnc_tensor_symbol_t* const updated_parameters, ccv_nnc_tensor_symbol_map_t* const saved_parameters, ccv_nnc_graph_exec_symbol_t* const graph_exec_symbols)
{
	// First, compute gradient.
	ccv_nnc_symbolic_graph_backward(graph, losses, loss_size, parameters, parameter_size, sources, source_size, destinations, destination_size);
	int i;
	for (i = 0; i < parameter_size; i++)
	{
	}
}
