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

void _ccv_nnc_while_graph_map_tensor_symbol(const ccv_array_t* const tensor_symbol_info, int idx, ccv_nnc_symbolic_graph_t* const while_graph, ccv_nnc_tensor_symbol_t* const tensor_symbol_map)
{
	ccv_nnc_tensor_symbol_info_t* tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, idx);
	if (tensor_info->alias_ref)
	{
		if (tensor_symbol_map[tensor_info->alias_ref - 1].d == -1)
		{
			ccv_nnc_tensor_symbol_info_t* alias_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_info->alias_ref - 1);
			assert(alias_info->alias_ref == 0);
			tensor_symbol_map[tensor_info->alias_ref - 1] = ccv_nnc_tensor_symbol_new(while_graph, alias_info->info, alias_info->name);
		}
		tensor_symbol_map[idx] = ccv_nnc_tensor_symbol_alias_new(while_graph, tensor_symbol_map[tensor_info->alias_ref - 1], tensor_info->ofs, tensor_info->inc, tensor_info->info, tensor_info->name);
	} else
		tensor_symbol_map[idx] = ccv_nnc_tensor_symbol_new(while_graph, tensor_info->info, tensor_info->name);
}

CCV_WARN_UNUSED(ccv_nnc_graph_exec_symbol_t) _ccv_nnc_while_graph_map_exec_symbol(const ccv_nnc_graph_exec_symbol_info_t* node, const ccv_array_t* tensor_symbol_info, ccv_nnc_symbolic_graph_t* while_graph, ccv_nnc_tensor_symbol_t* tensor_symbol_map, ccv_nnc_tensor_symbol_t* max_input_symbols, ccv_nnc_tensor_symbol_t* max_output_symbols)
{
	int i;
	for (i = 0; i < node->input_size; i++)
	{
		if (tensor_symbol_map[node->inputs[i]].d == -1)
			_ccv_nnc_while_graph_map_tensor_symbol(tensor_symbol_info, node->inputs[i], while_graph, tensor_symbol_map);
		max_input_symbols[i] = tensor_symbol_map[node->inputs[i]];
	}
	for (i = 0; i < node->output_size; i++)
	{
		if (tensor_symbol_map[node->outputs[i]].d == -1)
			_ccv_nnc_while_graph_map_tensor_symbol(tensor_symbol_info, node->outputs[i], while_graph, tensor_symbol_map);
		max_output_symbols[i] = tensor_symbol_map[node->outputs[i]];
	}
	ccv_nnc_graph_exec_symbol_t exec_symbol = ccv_nnc_graph_exec_symbol_new(while_graph, node->cmd, max_input_symbols, node->input_size, max_output_symbols, node->output_size, node->name);
	ccv_nnc_graph_exec_symbol_set_hint(while_graph, exec_symbol, node->hint);
	return exec_symbol;
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_symbolic_graph_while(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const conditions, const int condition_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_graph_while_f while_func, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size)
{
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	while_graph->while_func = while_func;
	ccv_nnc_graph_exec_symbol_t* exec_symbol_map = ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * graph->exec_symbol_info->rnum);
	int i, j, k;
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		exec_symbol_map[i].graph = 0, exec_symbol_map[i].d = -1;
	ccv_nnc_tensor_symbol_t* tensor_symbol_map = ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * graph->tensor_symbol_info->rnum);
	for (i = 0; i < graph->tensor_symbol_info->rnum; i++)
		tensor_symbol_map[i].graph = 0, tensor_symbol_map[i].d = -1;
	int max_input_size = 0, max_output_size = 0;
	// First pass simply mark the node as visited, so that later we will put it into the while graph.
#define visitor(node, idx, ...) do { \
		exec_symbol_map[idx].d = -2; /* Mark this one as selected such that I will create a symbol for this one in while graph. */ \
		max_input_size = ccv_max(max_input_size, node->input_size); \
		max_output_size = ccv_max(max_input_size, node->output_size); \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	ccv_nnc_tensor_symbol_t* max_input_symbols = max_input_size > 0 ? (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * max_input_size) : 0;
	ccv_nnc_tensor_symbol_t* max_output_symbols = max_output_size > 0 ? (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * max_output_size) : 0;
#define visitor(node, idx, ...) do { \
		/* Create tensor symbols in the while graph if needed. */ \
		if (exec_symbol_map[idx].d < 0) \
		{ \
			assert(exec_symbol_map[idx].d == -2); \
			exec_symbol_map[idx] = _ccv_nnc_while_graph_map_exec_symbol(node, graph->tensor_symbol_info, while_graph, tensor_symbol_map, max_input_symbols, max_output_symbols); \
		} \
		if (node->outgoings && node->outgoings->rnum > 0) \
			for (i = 0; i < node->outgoings->rnum; i++) \
			{ \
				const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, i); \
				if (exec_symbol_map[outgoing_idx].d == -2) /* It is on the path to be included into while graph */ \
				{ \
					const ccv_nnc_graph_exec_symbol_info_t* outgoing_node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, outgoing_idx); \
					exec_symbol_map[outgoing_idx] = _ccv_nnc_while_graph_map_exec_symbol(outgoing_node, graph->tensor_symbol_info, while_graph, tensor_symbol_map, max_input_symbols, max_output_symbols); \
					ccv_nnc_graph_exec_symbol_concat(while_graph, exec_symbol_map[idx], exec_symbol_map[outgoing_idx]); \
				} \
			} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Ready, now create the symbol to the while graph.
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_GRAPH_FORWARD, 0, CMD_GENERIC(), 0);
	// Added one more symbol.
	ccv_nnc_graph_exec_symbol_t symbol = ccv_nnc_graph_exec_symbol_new(graph, cmd, inputs, input_size, outputs, output_size, 0);
	for (i = 0; i < graph->exec_symbol_info->rnum - 1; i++)
	{
		ccv_nnc_graph_exec_symbol_info_t* node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		if (exec_symbol_map[i].d >= 0)
		{
			// Mark this node as dead.
			node->dead = 1;
			if (node->outgoings && node->outgoings->rnum > 0)
				for (j = 0; j < node->outgoings->rnum; j++)
				{
					const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, j);
					// If the outgoing node is not in the while graph, we need to connect this as outgoing node for the while graph.
					if (exec_symbol_map[outgoing_idx].d < 0)
					{
						ccv_nnc_graph_exec_symbol_t outgoing_symbol = {
							.d = outgoing_idx,
							.graph = graph
						};
						ccv_nnc_graph_exec_symbol_concat(graph, symbol, outgoing_symbol);
					}
				}
			// Remove all its outgoing nodes.
			ccv_array_free(node->outgoings);
			node->outgoings = 0;
		} else {
			if (node->outgoings && node->outgoings->rnum > 0)
			{
				int flag = 1;
				for (j = 0; j < node->outgoings->rnum; j++)
				{
					const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, j);
					// If the outgoing node is not in the while graph, we need to connect the while graph as the outgoing node.
					if (exec_symbol_map[outgoing_idx].d >= 0)
					{
						// This node need to be replaced or removed, mark it as -1 for the time being.
						*(int*)ccv_array_get(node->outgoings, j) = flag ? symbol.d : -1;
						flag = 0;
					}
				}
				if (!flag) // Go through the array to remove -1s.
				{
					for (j = 0, k = 0; k < node->outgoings->rnum; k++)
					{
						const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, k);
						if (outgoing_idx >= 0)
						{
							*(int*)ccv_array_get(node->outgoings, j) = outgoing_idx;
							++j;
						}
					}
					node->outgoings->rnum = j; // Now only have j items.
				}
			}
		}
	}
	if (max_input_symbols)
		ccfree(max_input_symbols);
	if (max_output_symbols)
		ccfree(max_output_symbols);
	ccfree(exec_symbol_map);
	ccfree(tensor_symbol_map);
	return symbol;
}
