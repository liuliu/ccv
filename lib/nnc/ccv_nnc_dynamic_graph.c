#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

struct ccv_nnc_tensor_variable_s {
	int index;
	int alias_ref;
	ccv_nnc_tensor_symbol_t symbol;
	ccv_nnc_tensor_view_t* tensor_view;
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
	tensor_variable->symbol = ccv_nnc_tensor_symbol_new(graph->symbolic, info, 0);
	tensor_variable->tensor_view = 0;
	return tensor_variable;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_alias_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info)
{
	assert(!tensor_variable->alias_ref);
	ccv_nnc_tensor_variable_t variable_alias = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	ccv_array_push(graph->var, &variable_alias);
	variable_alias->index = graph->var->rnum - 1;
	variable_alias->alias_ref = tensor_variable->index + 1;
	variable_alias->symbol = ccv_nnc_tensor_symbol_alias_new(graph->symbolic, tensor_variable->symbol, ofs, inc, info, 0);
	variable_alias->tensor_view = 0;
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
	tensor_variable->tensor_view = 0; // ccv_nnc_tensor_view_new((ccv_nnc_tensor_t*)variable_to->tensor_view, tensor_variable->symbol.info.dim, ccv_nnc_tensor_symbol_alias_ofs(graph->symbolic, tensor_variable->symbol), ccv_nnc_tensor_symbol_alias_inc(graph->symbolic, tensor_variable->symbol));
	return 0;
}

int ccv_nnc_dynamic_graph_exec(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size)
{
	int i;
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
		input_symbols[i] = inputs[i]->symbol;
	ccv_nnc_tensor_symbol_t output_symbols[ccv_max(1, output_size)];
	for (i = 0; i < output_size; i++)
		output_symbols[i] = outputs[i]->symbol;
	ccv_nnc_graph_exec_symbol_new(graph->symbolic, cmd, input_symbols, input_size, output_symbols, output_size, 0);
	return CCV_NNC_EXEC_SUCCESS;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_backward(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t f_variable, const ccv_nnc_tensor_variable_t tensor_variable)
{
	return 0;
}

void ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	ccfree(tensor_variable);
}
