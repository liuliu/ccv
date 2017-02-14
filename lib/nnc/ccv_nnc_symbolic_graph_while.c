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

void _ccv_nnc_while_graph_map_tensor_symbol(const ccv_array_t* const tensor_symbol_info, int idx, ccv_nnc_symbolic_graph_t* const while_graph, ccv_nnc_tensor_symbol_t* const tensor_symbol_map)
{
	const ccv_nnc_tensor_symbol_info_t* const tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, idx);
	ccv_nnc_tensor_symbol_t tensor_symbol;
	if (tensor_info->alias_ref)
	{
		if (tensor_symbol_map[tensor_info->alias_ref - 1].d == -1)
		{
			ccv_nnc_tensor_symbol_info_t* alias_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_info->alias_ref - 1);
			assert(alias_info->alias_ref == 0);
			tensor_symbol_map[tensor_info->alias_ref - 1] = ccv_nnc_tensor_symbol_new(while_graph, alias_info->info, alias_info->name);
		}
		tensor_symbol = ccv_nnc_tensor_symbol_alias_new(while_graph, tensor_symbol_map[tensor_info->alias_ref - 1], tensor_info->ofs, tensor_info->inc, tensor_info->info, tensor_info->name);
	} else
		tensor_symbol = ccv_nnc_tensor_symbol_new(while_graph, tensor_info->info, tensor_info->name);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, tensor_symbol.d);
	symbol_info->p_ref = idx + 1;
	tensor_symbol_map[idx]  = tensor_symbol;
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_symbolic_graph_while(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_symbolic_graph_t* const while_graph, const char* const name)
{
	assert(while_graph->p == 0);
	assert(while_graph->p_idx == 0);
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_GRAPH_FORWARD, 0, CMD_GENERIC(), 0);
	// Added one more symbol.
	ccv_nnc_graph_exec_symbol_t symbol = ccv_nnc_graph_exec_symbol_new(graph, cmd, 0, 0, 0, 0, name);
	// Assigning graph_ref to it.
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_symbolic_graph_t*), 1, 0);
	ccv_array_push(graph->sub_graphs, &while_graph);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	// Note the extra allocation (the ccv_array_t only holds a pointer to ccv_nnc_symbolic_graph_t*).
	// In this way, we can get the while graph and don't have to worry about it will be an invalid pointer once
	// the array expands (another while graph allocated).
	symbol_info->graph_ref = graph->sub_graphs->rnum;
	return symbol;
}

void ccv_nnc_symbolic_graph_set_while_expr(ccv_nnc_symbolic_graph_t* const while_graph, const ccv_nnc_graph_while_f while_expr, const void* const while_data, const ccv_nnc_graph_exec_symbol_t* const cond_evals, const int cond_eval_size)
{
	while_graph->while_expr = while_expr;
	while_graph->while_data = while_data;
	if (cond_eval_size > 0)
	{
		assert(cond_evals);
		while_graph->cond_eval_size = cond_eval_size;
		while_graph->cond_evals = (ccv_nnc_graph_exec_symbol_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * cond_eval_size);
		memcpy(while_graph->cond_evals, cond_evals, sizeof(ccv_nnc_graph_exec_symbol_t) * cond_eval_size);
	}
}

void ccv_nnc_symbolic_graph_set_while_params(ccv_nnc_symbolic_graph_t* const while_graph, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size)
{
	int i;
	for (i = 0; i < symbol_map_size; i++)
	{
		assert(symbol_map[i].source.graph == while_graph);
		assert(symbol_map[i].destination.graph == while_graph);
		const int source_d = symbol_map[i].source.d;
		const int destination_d = symbol_map[i].destination.d;
		assert(source_d < while_graph->tensor_symbol_info->rnum);
		assert(destination_d < while_graph->tensor_symbol_info->rnum);
		ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, destination_d);
		tensor_symbol_info->assign_ref = source_d + 1;
	}
}

ccv_nnc_symbolic_graph_t* ccv_nnc_symbolic_graph_from_while_symbol(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t while_symbol)
{
	assert(graph->sub_graphs);
	assert(while_symbol.graph == graph);
	assert(while_symbol.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, while_symbol.d);
	assert(symbol_info->graph_ref <= graph->sub_graphs->rnum);
	return *(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, symbol_info->graph_ref - 1);
}
