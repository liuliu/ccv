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

ccv_nnc_graph_exec_symbol_t ccv_nnc_symbolic_graph_while(ccv_nnc_symbolic_graph_t* const graph, ccv_nnc_symbolic_graph_t* const while_graph, const char* const name)
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
	while_graph->p_idx = graph->sub_graphs->rnum;
	while_graph->exec_idx = symbol.d + 1;
	while_graph->p = graph;
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

ccv_nnc_tensor_symbol_t ccv_nnc_find_tensor_symbol_from_graph(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol)
{
	if (symbol.graph == graph)
		return symbol;
	const ccv_nnc_symbolic_graph_t* curr_graph = symbol.graph;
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(symbol.graph->tensor_symbol_info, symbol.d);
	assert(symbol.d >= 0 && symbol.d < curr_graph->tensor_symbol_info->rnum);
	while (curr_graph && curr_graph != graph)
		curr_graph = curr_graph->p;
	if (curr_graph)
	{
		curr_graph = symbol.graph;
		ccv_nnc_tensor_symbol_info_t* curr_symbol_info = symbol_info;
		ccv_nnc_tensor_symbol_t curr_symbol = symbol;
		while (curr_graph != graph)
		{
			ccv_nnc_symbolic_graph_t* const p = curr_graph->p;
			// I need to find the symbol, it must exist.
			assert(curr_symbol_info->p_ref);
			// Move on.
			curr_symbol.d = curr_symbol_info->p_ref - 1;
			curr_symbol.graph = p;
			assert(curr_symbol.d >= 0 && curr_symbol.d < p->tensor_symbol_info->rnum);
			curr_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(p->tensor_symbol_info, curr_symbol.d);
			curr_graph = p;
		}
		return curr_symbol;
	}
	// Otherwise, if the symbol is in the parent graph, this is a bit more expensive because I need to keep a trace stack.
	curr_graph = graph;
	ccv_array_t* trace = ccv_array_new(sizeof(int), 0, 0);
	while (curr_graph && curr_graph != symbol.graph)
	{
		const int p_idx = curr_graph->p_idx - 1;
		ccv_array_push(trace, &p_idx);
		curr_graph = curr_graph->p;
	}
	// If it is not in both the parent graph and the sub-graph, the input is invalid.
	assert(curr_graph);
	curr_graph = symbol.graph;
	ccv_nnc_tensor_symbol_info_t* curr_symbol_info = symbol_info;
	ccv_nnc_tensor_symbol_t curr_symbol = symbol;
	// The graph is a sub graph of the symbol passed in.
	int i;
	for (i = trace->rnum - 1; i >= 0; i--)
	{
		const int p_idx = *(int*)ccv_array_get(trace, i);
		assert(p_idx >= 0);
		assert(curr_graph->sub_graphs);
		assert(curr_symbol_info->s_ref);
		assert(p_idx >= 0 && p_idx < curr_symbol_info->s_ref->rnum);
		const int s_idx = *(int*)ccv_array_get(curr_symbol_info->s_ref, p_idx);
		ccv_nnc_symbolic_graph_t* const s = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(curr_graph->sub_graphs, p_idx);
		// I need to find the symbol, it must exist.
		assert(s_idx);
		curr_symbol.d = s_idx - 1;
		curr_symbol.graph = s;
		assert(curr_symbol.d >= 0 && curr_symbol.d < s->tensor_symbol_info->rnum);
		curr_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(s->tensor_symbol_info, curr_symbol.d);
		// Move on.
		curr_graph = s;
	}
	ccv_array_free(trace);
	return curr_symbol;
}

void ccv_nnc_symbolic_graph_set_while_params(ccv_nnc_symbolic_graph_t* const while_graph, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size)
{
	int i;
	for (i = 0; i < symbol_map_size; i++)
	{
		const ccv_nnc_tensor_symbol_t source = ccv_nnc_find_tensor_symbol_from_graph(while_graph, symbol_map[i].source);
		const ccv_nnc_tensor_symbol_t destination = ccv_nnc_find_tensor_symbol_from_graph(while_graph, symbol_map[i].destination);
		assert(source.graph == while_graph);
		assert(destination.graph == while_graph);
		assert(source.d < while_graph->tensor_symbol_info->rnum);
		assert(destination.d < while_graph->tensor_symbol_info->rnum);
		ccv_nnc_tensor_symbol_info_t* destination_tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, destination.d);
		destination_tensor_symbol_info->assign_ref = source.d + 1;
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
