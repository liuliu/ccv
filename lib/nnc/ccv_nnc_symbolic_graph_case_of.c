#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-3.5 API
 */

ccv_nnc_graph_exec_symbol_t ccv_nnc_symbolic_graph_case_of_new(ccv_nnc_symbolic_graph_t* const graph, const uint32_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size, const char* const name)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	// A case_if statement must have meaningful outputs / inputs.
	assert(symbol_map_size > 0);
	ccv_nnc_tensor_symbol_t all_inputs[symbol_map_size * 2 + input_size];
	ccv_nnc_tensor_symbol_t* const outputs = all_inputs + (symbol_map_size + input_size);
	int i;
	for (i = 0; i < symbol_map_size; i++)
		all_inputs[i] = symbol_map[i].source, outputs[i] = symbol_map[i].destination;
	for (i = symbol_map_size; i < symbol_map_size + input_size; i++)
		all_inputs[i] = inputs[i - symbol_map_size];
	// Added one more symbol.
	const ccv_nnc_graph_exec_symbol_t symbol = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), all_inputs, symbol_map_size + input_size, outputs, symbol_map_size, name);
	ccv_nnc_tensor_symbol_set_bypasses(graph, symbol_map, symbol_map_size);
	ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	symbol_info->flags |= CCV_NNC_GRAPH_EXEC_CASE_OF;
	// We are still free to add more inputs to this graph, it is OK, we are covered by the argument.offset / size.
	symbol_info->case_of.argument.offset = symbol_map_size;
	symbol_info->case_of.argument.size = input_size;
	return symbol;
}

void ccv_nnc_symbolic_graph_set_case_of_expr(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec, ccv_nnc_graph_case_of_f case_of, const void* case_of_data)
{
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	symbol_info->case_of.expr = case_of;
	symbol_info->case_of.data = case_of_data;
}

void ccv_nnc_symbolic_graph_set_case_of(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol, ccv_nnc_symbolic_graph_t* const case_graph, const int case_of, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size)
{
	assert(symbol.graph == graph);
	assert(symbol.d >= 0);
	assert(symbol.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	assert(symbol_map_size <= symbol_info->output_size);
	assert(symbol_info->flags == CCV_NNC_GRAPH_EXEC_CASE_OF);
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_symbolic_graph_t*), 1, 0);
	ccv_array_push(graph->sub_graphs, &case_graph);
	case_graph->p_idx = graph->sub_graphs->rnum;
	case_graph->exec_idx = symbol.d + 1;
	case_graph->p = graph;
	// If case_of is larger than the inline graph_ref, we need to allocate.
	if (case_of >= sizeof(symbol_info->_inline_graph_ref) / sizeof(symbol_info->_inline_graph_ref[0]))
	{
		if (!symbol_info->_heap_graph_ref)
		{
			symbol_info->_heap_graph_ref = cccalloc(case_of + 1, sizeof(int));
			// Copy from inline data.
			memcpy(symbol_info->_heap_graph_ref, symbol_info->_inline_graph_ref, sizeof(symbol_info->_inline_graph_ref));
			symbol_info->graph_ref_size = case_of + 1;
		} else if (symbol_info->graph_ref_size <= case_of) {
			symbol_info->_heap_graph_ref = ccrealloc(symbol_info->_heap_graph_ref, sizeof(int) * (case_of + 1));
			// Reset the newly allocated ones to 0.
			memset(symbol_info->_heap_graph_ref + symbol_info->graph_ref_size, 0, sizeof(int) * (case_of + 1 - symbol_info->graph_ref_size));
			symbol_info->graph_ref_size = case_of + 1;
		}
	} else
		symbol_info->graph_ref_size = ccv_max(symbol_info->graph_ref_size, case_of + 1);
	// Set the branch with the graph.
	CCV_NNC_GRAPH_REF(symbol_info)[case_of] = graph->sub_graphs->rnum;
	int i;
	for (i = 0; i < symbol_map_size; i++)
		ccv_nnc_tensor_symbol_hookup(case_graph, graph, symbol_map[i].source, symbol_map[i].destination);
}
