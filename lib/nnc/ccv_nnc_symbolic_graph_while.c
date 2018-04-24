#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-3.5 API
 */

ccv_nnc_graph_exec_symbol_t ccv_nnc_symbolic_graph_while(ccv_nnc_symbolic_graph_t* const graph, const uint32_t cmd, ccv_nnc_symbolic_graph_t* const while_graph, const char* const name)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	assert(while_graph->p == 0);
	assert(while_graph->p_idx == 0);
	// Added one more symbol.
	ccv_nnc_graph_exec_symbol_t symbol = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, name);
	// Assigning graph_ref to it.
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_symbolic_graph_t*), 1, 0);
	ccv_array_push(graph->sub_graphs, &while_graph);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	symbol_info->flags |= CCV_NNC_GRAPH_EXEC_P_WHILE;
	symbol_info->graph_ref_size = 1;
	// Note the extra allocation (the ccv_array_t only holds a pointer to ccv_nnc_symbolic_graph_t*).
	// In this way, we can get the while graph and don't have to worry about it will be an invalid pointer once
	// the array expands (another while graph allocated).
	CCV_NNC_GRAPH_REF(symbol_info)[0] = graph->sub_graphs->rnum;
	while_graph->p_idx = graph->sub_graphs->rnum;
	while_graph->exec_idx = symbol.d + 1;
	while_graph->p = graph;
	return symbol;
}

void ccv_nnc_symbolic_graph_set_while_expr(ccv_nnc_symbolic_graph_t* const while_graph, const ccv_nnc_graph_while_f while_expr, const void* const while_data, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_graph_exec_symbol_t* const breakpoints, const int breakpoint_size)
{
	const int exec_idx = while_graph->exec_idx - 1;
	assert(exec_idx >= 0 && exec_idx < while_graph->p->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* const exec_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(while_graph->p->exec_symbol_info, exec_idx);
	exec_info->p_while.data = while_data;
	exec_info->p_while.expr = while_expr;
	int i;
	if (input_size > 0)
	{
		assert(inputs);
		exec_info->p_while.inputs = (int*)ccmalloc(sizeof(int) * input_size);
		for (i = 0; i < input_size; i++)
			exec_info->p_while.inputs[i] = ccv_nnc_tensor_symbol_map_raw(while_graph, inputs[i]);
		exec_info->p_while.input_size = input_size;
	}
	if (breakpoint_size > 0)
	{
		assert(breakpoints);
		while_graph->breakpoint_size = breakpoint_size;
		while_graph->breakpoints = (ccv_nnc_graph_exec_symbol_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * breakpoint_size);
		memcpy(while_graph->breakpoints, breakpoints, sizeof(ccv_nnc_graph_exec_symbol_t) * breakpoint_size);
	}
}

void ccv_nnc_symbolic_graph_set_carry_overs(ccv_nnc_symbolic_graph_t* const while_graph, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size)
{
	int i;
	for (i = 0; i < symbol_map_size; i++)
	{
		const ccv_nnc_tensor_symbol_t source = ccv_nnc_tensor_symbol_resolve(while_graph, symbol_map[i].source);
		const ccv_nnc_tensor_symbol_t destination = ccv_nnc_tensor_symbol_resolve(while_graph, symbol_map[i].destination);
		assert(source.graph == while_graph);
		assert(destination.graph == while_graph);
		assert(source.d < while_graph->tensor_symbol_info->rnum);
		assert(destination.d < while_graph->tensor_symbol_info->rnum);
		ccv_nnc_tensor_symbol_info_t* source_tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, source.d);
		ccv_nnc_tensor_symbol_info_t* destination_tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, destination.d);
		// Don't support parameterize with alias. The reason is that to support parameterized loop (for SSA), I choose
		// to simply reuse the piece of memory (allocating the same memory region to both, therefore to enable parameter
		// passing). For alias, it is not possible because alias can pointing to the tensors with different sizes, thus,
		// these pointed tensors cannot share the same memory region. The best way for alias to be parameterized is to
		// create a new tensor of the same size, transfer value over, and parameterized on that tensor instead.
		assert(!destination_tensor_symbol_info->alias_ref);
		assert(!source_tensor_symbol_info->alias_ref);
		destination_tensor_symbol_info->assign_ref = source.d + 1;
		source_tensor_symbol_info->r_assign_ref = destination.d + 1;
	}
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_for_while_count(const ccv_nnc_symbolic_graph_t* const while_graph)
{
	return (ccv_nnc_tensor_symbol_t){
		.d = CCV_NNC_WHILE_COUNT_TENSOR_SYMBOL,
		.graph = while_graph
	};
}

ccv_nnc_symbolic_graph_t* ccv_nnc_symbolic_graph_from_while_symbol(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t while_symbol)
{
	assert(graph->sub_graphs);
	assert(while_symbol.graph == graph);
	assert(while_symbol.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, while_symbol.d);
	assert(CCV_NNC_GRAPH_REF(symbol_info)[0] <= graph->sub_graphs->rnum);
	return *(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(symbol_info)[0] - 1);
}
