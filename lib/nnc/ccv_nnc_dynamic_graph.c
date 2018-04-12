#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

struct ccv_nnc_tensor_variable_s {
	int index;
	int alias_ref;
	ccv_array_t* binded_sources; // array of graph_exec_symbol, use this tensor variable as output.
	ccv_array_t* binded_destinations; // array of graph_exec_symbol, use this tensor variable as input.
	ccv_nnc_tensor_symbol_t symbol;
	ccv_nnc_tensor_view_t* tensor_view;
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
};

struct ccv_nnc_dynamic_graph_s {
	ccv_array_t* var; // Array keeps track of all allocated tensor variable.
	ccv_nnc_symbolic_graph_t* symbolic; // Symbolic graph to keep track of computation.
};

ccv_nnc_dynamic_graph_t* ccv_nnc_dynamic_graph_new(void)
{
	ccv_nnc_dynamic_graph_t* graph = ccmalloc(sizeof(ccv_nnc_dynamic_graph_t));
	graph->var = ccv_array_new(sizeof(ccv_nnc_tensor_variable_t), 1, 0);
	graph->symbolic = ccv_nnc_symbolic_graph_new();
	return graph;
}

void ccv_nnc_dynamic_graph_free(ccv_nnc_dynamic_graph_t* const graph)
{
	int i;
	for (i = 0; i < graph->var->rnum; i++)
	{
		ccv_nnc_tensor_variable_t tensor_variable = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, i);
		if (tensor_variable)
			ccv_nnc_tensor_variable_free(graph, tensor_variable);
	}
	ccv_array_free(graph->var);
	ccv_nnc_symbolic_graph_free(graph->symbolic);
	ccfree(graph);
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_param_t info)
{
	ccv_nnc_tensor_variable_t tensor_variable = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	ccv_array_push(graph->var, &tensor_variable);
	tensor_variable->index = graph->var->rnum - 1;
	tensor_variable->alias_ref = 0;
	tensor_variable->symbol = NO_TENSOR_SYMBOL;
	tensor_variable->symbol.info = info; // Piggy-back on the info inside tensor symbol.
	tensor_variable->tensor_view = 0;
	tensor_variable->binded_sources = 0;
	tensor_variable->binded_destinations = 0;
	return tensor_variable;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_alias_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info)
{
	assert(!tensor_variable->alias_ref);
	ccv_nnc_tensor_variable_t variable_alias = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	ccv_array_push(graph->var, &variable_alias);
	variable_alias->index = graph->var->rnum - 1;
	variable_alias->alias_ref = tensor_variable->index + 1;
	variable_alias->symbol = NO_TENSOR_SYMBOL;
	variable_alias->symbol.info = info;
	variable_alias->tensor_view = 0;
	variable_alias->binded_sources = 0;
	variable_alias->binded_destinations = 0;
	memcpy(variable_alias->ofs, ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(variable_alias->inc, inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	return variable_alias;
}

ccv_nnc_tensor_view_t* ccv_nnc_tensor_from_variable(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	if (tensor_variable->tensor_view)
		return tensor_variable->tensor_view;
	if (!tensor_variable->alias_ref)
	{
		tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(0, tensor_variable->symbol.info, 0);
		return tensor_variable->tensor_view;
	}
	const int alias_ref = tensor_variable->alias_ref - 1;
	assert(alias_ref >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, alias_ref);
	assert(!variable_to->alias_ref);
	if (!variable_to->tensor_view)
		variable_to->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(0, variable_to->symbol.info, 0);
	tensor_variable->tensor_view = ccv_nnc_tensor_view_new((ccv_nnc_tensor_t*)variable_to->tensor_view, tensor_variable->symbol.info.dim, tensor_variable->ofs, tensor_variable->inc);
	return 0;
}

static ccv_nnc_tensor_symbol_t _ccv_nnc_tensor_symbol_from_variable(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	if (tensor_variable->symbol.d >= 0)
		return tensor_variable->symbol;
	if (!tensor_variable->alias_ref)
	{
		assert(!tensor_variable->binded_sources || tensor_variable->binded_sources->rnum == 0);
		tensor_variable->symbol = ccv_nnc_tensor_symbol_new(graph->symbolic, tensor_variable->symbol.info, 0);
		return tensor_variable->symbol;
	}
	const int alias_ref = tensor_variable->alias_ref - 1;
	assert(alias_ref >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, alias_ref);
	assert(!variable_to->alias_ref);
	assert(!tensor_variable->binded_sources || tensor_variable->binded_sources->rnum == 0);
	tensor_variable->symbol = ccv_nnc_tensor_symbol_alias_new(graph->symbolic, _ccv_nnc_tensor_symbol_from_variable(graph, variable_to), tensor_variable->ofs, tensor_variable->inc, tensor_variable->symbol.info, 0);
	return tensor_variable->symbol;
}

static void _ccv_nnc_tensor_symbol_update_for_variable(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	assert(tensor_variable->binded_sources->rnum > 0);
	if (!tensor_variable->alias_ref)
	{
		tensor_variable->symbol = ccv_nnc_tensor_symbol_new(graph->symbolic, tensor_variable->symbol.info, 0);
		return;
	}
	const int alias_ref = tensor_variable->alias_ref - 1;
	assert(alias_ref >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, alias_ref);
	assert(!variable_to->alias_ref);
	// Note that we cannot really keep track whether the original symbol is "free" or not because it can be partially free.
	tensor_variable->symbol = ccv_nnc_tensor_symbol_alias_new(graph->symbolic, _ccv_nnc_tensor_symbol_from_variable(graph, variable_to), tensor_variable->ofs, tensor_variable->inc, tensor_variable->symbol.info, 0);
}

int ccv_nnc_dynamic_graph_exec(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size)
{
	int i, j;
	for (i = 0; i < input_size; i++)
		{ assert(inputs[i]->tensor_view); }
	ccv_nnc_tensor_t* input_tensors[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_tensors[i] = (ccv_nnc_tensor_t*)ccv_nnc_tensor_from_variable(graph, inputs[i]);
	ccv_nnc_tensor_t* output_tensors[ccv_max(1, output_size)];
	for (i = 0; i < output_size; i++)
		output_tensors[i] = (ccv_nnc_tensor_t*)ccv_nnc_tensor_from_variable(graph, outputs[i]);
	ccv_nnc_cmd_exec(cmd, hint, flags, input_tensors, input_size, output_tensors, output_size, 0);
	ccv_nnc_tensor_symbol_t input_symbols[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_symbols[i] = _ccv_nnc_tensor_symbol_from_variable(graph, inputs[i]);
	ccv_nnc_tensor_symbol_t output_symbols[ccv_max(1, output_size)];
	for (i = 0; i < output_size; i++)
		// Refresh the symbol if it is binded to an existing exec. Otherwise we cannot keep the SSA guarantee.
		if (outputs[i]->binded_sources && outputs[i]->binded_sources->rnum > 0)
			_ccv_nnc_tensor_symbol_update_for_variable(graph, outputs[i]);
	for (i = 0; i < output_size; i++)
		output_symbols[i] = _ccv_nnc_tensor_symbol_from_variable(graph, outputs[i]);
	ccv_nnc_graph_exec_symbol_t graph_exec = ccv_nnc_graph_exec_symbol_new(graph->symbolic, cmd, input_symbols, input_size, output_symbols, output_size, 0);
	for (i = 0; i < input_size; i++)
	{
		if (!inputs[i]->binded_destinations)
			inputs[i]->binded_destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
		ccv_array_push(inputs[i]->binded_destinations, &graph_exec);
	}
	for (i = 0; i < output_size; i++)
	{
		if (outputs[i]->binded_sources)
		{
			for (j = 0; j < outputs[i]->binded_sources->rnum; j++)
				ccv_nnc_graph_exec_symbol_concat(graph->symbolic, *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(outputs[i]->binded_sources, j), graph_exec);
			ccv_array_clear(outputs[i]->binded_sources);
		} else
			outputs[i]->binded_sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
		ccv_array_push(outputs[i]->binded_sources, &graph_exec);
		if (outputs[i]->binded_destinations)
			ccv_array_clear(outputs[i]->binded_destinations);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_backward(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t f_variable, const ccv_nnc_tensor_variable_t tensor_variable)
{
	// Both f_variable and tensor_variable should be, at least, executed. Otherwise we cannot differentiate.
	assert(f_variable->symbol.d >= 0);
	assert(tensor_variable->symbol.d >= 0);
	assert(f_variable->binded_sources && f_variable->binded_sources->rnum > 0);
	assert(tensor_variable->binded_destinations && tensor_variable->binded_destinations->rnum > 0);
	ccv_nnc_symbolic_graph_backward(graph->symbolic,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(tensor_variable->binded_destinations, 0), tensor_variable->binded_destinations->rnum,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(f_variable->binded_sources, 0), f_variable->binded_sources->rnum,
		&f_variable->symbol, 1, &tensor_variable->symbol, 1);
	ccv_nnc_tensor_variable_t d_variable = ccv_nnc_tensor_variable_new(graph, tensor_variable->tensor_view->info);
	d_variable->symbol = ccv_nnc_tensor_symbol_for_backward(graph->symbolic, tensor_variable->symbol);
	/*
	ccv_nnc_tensor_view_t* tensor_view = ccv_nnc_tensor_from_variable(graph, d_variable);
	ccv_nnc_graph_t* exec_schedule = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(graph->symbolic, binds, bind_size, sources, source_size, destinations, destination_size, &exec_schedule, &tensor_arena, &exec_arena);
	*/
	return d_variable;
}

void ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	const int index = tensor_variable->index;
	if (tensor_variable->tensor_view)
	{
		if (CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view))
			ccv_nnc_tensor_view_free(tensor_variable->tensor_view);
		else
			ccv_nnc_tensor_free((ccv_nnc_tensor_t*)tensor_variable->tensor_view);
	}
	if (tensor_variable->binded_sources)
		ccv_array_free(tensor_variable->binded_sources);
	if (tensor_variable->binded_destinations)
		ccv_array_free(tensor_variable->binded_destinations);
	ccfree(tensor_variable);
	*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, index) = 0;
}
