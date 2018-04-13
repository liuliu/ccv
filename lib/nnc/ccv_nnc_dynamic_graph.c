#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

struct ccv_nnc_tensor_variable_s {
	int index;
	int alias_ref;
	ccv_nnc_graph_exec_symbol_t binded_source; // array of graph_exec_symbol, use this tensor variable as output.
	ccv_array_t* binded_destinations; // array of graph_exec_symbol, use this tensor variable as input.
	ccv_nnc_tensor_symbol_t symbol;
	ccv_nnc_tensor_view_t* tensor_view;
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
};

struct ccv_nnc_dynamic_graph_s {
	ccv_array_t* var; // Array keeps track of all allocated tensor variable.
	ccv_nnc_symbolic_graph_t* symbolic; // Symbolic graph to keep track of computation.
	ccv_array_t* ws; // array of integers as workspace
};

ccv_nnc_dynamic_graph_t* ccv_nnc_dynamic_graph_new(void)
{
	ccv_nnc_dynamic_graph_t* graph = ccmalloc(sizeof(ccv_nnc_dynamic_graph_t));
	graph->var = ccv_array_new(sizeof(ccv_nnc_tensor_variable_t), 1, 0);
	graph->symbolic = ccv_nnc_symbolic_graph_new();
	graph->ws = ccv_array_new(sizeof(int), 1, 0);
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
	ccv_array_free(graph->ws);
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
	tensor_variable->binded_source.d = -1;
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
	variable_alias->binded_source.d = -1;
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
		assert(tensor_variable->binded_source.d == -1);
		tensor_variable->symbol = ccv_nnc_tensor_symbol_new(graph->symbolic, tensor_variable->symbol.info, 0);
		return tensor_variable->symbol;
	}
	const int alias_ref = tensor_variable->alias_ref - 1;
	assert(alias_ref >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, alias_ref);
	assert(!variable_to->alias_ref);
	assert(tensor_variable->binded_source.d == -1);
	tensor_variable->symbol = ccv_nnc_tensor_symbol_alias_new(graph->symbolic, _ccv_nnc_tensor_symbol_from_variable(graph, variable_to), tensor_variable->ofs, tensor_variable->inc, tensor_variable->symbol.info, 0);
	return tensor_variable->symbol;
}

static ccv_nnc_tensor_variable_t _ccv_nnc_tensor_variable_exchange_new(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable)
{
	struct ccv_nnc_tensor_variable_s x = *tensor_variable;
	ccv_nnc_tensor_variable_t new_variable;
	// Need to handle alias.
	if (x.alias_ref)
		new_variable = ccv_nnc_tensor_variable_alias_new(graph, *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, x.alias_ref - 1), x.ofs, x.inc, x.symbol.info);
	else
		new_variable = ccv_nnc_tensor_variable_new(graph, x.symbol.info);
	*tensor_variable = *new_variable;
	*new_variable = x;
	// The index should be the same though.
	const int index = new_variable->index;
	new_variable->index = tensor_variable->index;
	tensor_variable->index = index;
	return tensor_variable;
}

int ccv_nnc_dynamic_graph_exec(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < input_size; i++)
		{ assert(inputs[i]->tensor_view); }
	ccv_nnc_tensor_t* input_tensors[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_tensors[i] = (ccv_nnc_tensor_t*)ccv_nnc_tensor_from_variable(graph, inputs[i]);
	ccv_nnc_tensor_symbol_t input_symbols[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_symbols[i] = _ccv_nnc_tensor_symbol_from_variable(graph, inputs[i]);
	ccv_nnc_graph_exec_symbol_t input_sources[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_sources[i] = inputs[i]->binded_source;

	// Refresh the symbol if it is binded to an existing exec. Otherwise we cannot keep the SSA guarantee.
	for (i = 0; i < output_size; i++)
		if (outputs[i]->binded_source.d >= 0)
			outputs[i] = _ccv_nnc_tensor_variable_exchange_new(graph, outputs[i]);
	ccv_nnc_tensor_t* output_tensors[ccv_max(1, output_size)];
	for (i = 0; i < output_size; i++)
		output_tensors[i] = (ccv_nnc_tensor_t*)ccv_nnc_tensor_from_variable(graph, outputs[i]);
	ccv_nnc_cmd_exec(cmd, hint, flags, input_tensors, input_size, output_tensors, output_size, 0);
	ccv_nnc_tensor_symbol_t output_symbols[ccv_max(1, output_size)];
	for (i = 0; i < output_size; i++)
		output_symbols[i] = _ccv_nnc_tensor_symbol_from_variable(graph, outputs[i]);
	ccv_nnc_graph_exec_symbol_t graph_exec = ccv_nnc_graph_exec_symbol_new(graph->symbolic, cmd, input_symbols, input_size, output_symbols, output_size, 0);
	for (i = 0; i < input_size; i++)
		if (input_sources[i].d >= 0)
			ccv_nnc_graph_exec_symbol_concat(graph->symbolic, input_sources[i], graph_exec);
	for (i = 0; i < input_size; i++)
	{
		if (!inputs[i]->binded_destinations)
			inputs[i]->binded_destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
		ccv_array_push(inputs[i]->binded_destinations, &graph_exec);
	}
	for (i = 0; i < output_size; i++)
		outputs[i]->binded_source = graph_exec;
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_insert_if_prior_to_any(const ccv_nnc_symbolic_graph_t* const graph, const int d, ccv_array_t* const sources, uint8_t* const visited, int* const buf0, int* const buf1)
{
	if (visited[(d >> 3)] & (1 << (d & 7)))
		return;
	visited[(d >> 3)] |= (1 << (d & 7));
	buf0[0] = d;
	int* buf[2] = {
		buf0, buf1
	};
	int buf_size[2] = {
		1, 0
	};
	int p = 0, q = 1;
	int i, j, k;
	int flag = 0;
	while (buf_size[p] > 0)
	{
		buf_size[q] = 0;
		for (i = 0; i < buf_size[p]; i++)
		{
			const ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, buf[p][i]);
			if (symbol_info->outgoings)
				for (j = 0; j < symbol_info->outgoings->rnum; j++)
				{
					const int outgoing_idx = *(int*)ccv_array_get(symbol_info->outgoings, j);
					for (k = 0; k < sources->rnum; k++)
					{
						ccv_nnc_graph_exec_symbol_t* const source_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, k);
						// If this outgoing idx is one of the source, replace it with d, or delete it.
						if (source_symbol->d == outgoing_idx)
						{
							if (!flag)
							{
								source_symbol->d = d;
								flag = 1;
							} else {
								// Delete this from the list.
								if (k < sources->rnum - 1)
									source_symbol->d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, sources->rnum - 1))->d;
								--sources->rnum;
							}
							break;
						}
					}
					if (visited[(outgoing_idx >> 3)] & (1 << (outgoing_idx & 7)))
						continue;
					visited[(outgoing_idx >> 3)] |= (1 << (outgoing_idx & 7));
					buf[q][buf_size[q]] = outgoing_idx;
					++buf_size[q];
				}
		}
		CCV_SWAP(p, q, i);
	}
	// If this node is not visited, and we cannot find anything in the sources to replace, this is a new top node.
	if (!flag)
	{
		const ccv_nnc_graph_exec_symbol_t source_symbol = {
			.d = d,
			.graph = graph
		};
		ccv_array_push(sources, &source_symbol);
	}
}

void ccv_nnc_dynamic_graph_backward(ccv_nnc_dynamic_graph_t* const dynamic_graph, const ccv_nnc_tensor_variable_t f_variable, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size)
{
	int i, j, k;
	assert(input_size == output_size);
	assert(input_size > 0);
	// Both f_variable and tensor_variable should be, at least, executed. Otherwise we cannot differentiate.
	assert(f_variable->symbol.d >= 0);
	assert(f_variable->binded_source.d >= 0);
	for (i = 0; i < input_size; i++)
	{
		assert(inputs[i]->symbol.d >= 0);
		assert(inputs[i]->binded_destinations && inputs[i]->binded_destinations->rnum > 0);
	}
	// Refresh the symbol if it is not empty, we will use new symbol for the output tensor variables.
	for (i = 0; i < output_size; i++)
	{
		if (ccv_nnc_is_tensor_auto(outputs[i]->symbol.info))
			outputs[i]->symbol.info = inputs[i]->symbol.info;
		if (outputs[i]->symbol.d >= 0)
			outputs[i] = _ccv_nnc_tensor_variable_exchange_new(dynamic_graph, outputs[i]);
	}
	const int exec_symbol_info_size = dynamic_graph->symbolic->exec_symbol_info->rnum;
	ccv_array_t* const sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
	ccv_array_t* const ws = dynamic_graph->ws;
	uint8_t* const visited = (uint8_t*)cccalloc((exec_symbol_info_size + 7) >> 3, sizeof(uint8_t));
	ccv_array_resize(ws, exec_symbol_info_size * 2);
	for (i = 0; i < input_size; i++)
	{
		ccv_array_t* const binded_destinations = inputs[i]->binded_destinations;
		for (j = 0; j < binded_destinations->rnum; j++)
			_ccv_nnc_insert_if_prior_to_any(dynamic_graph->symbolic,
				((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(binded_destinations, j))->d,
				sources, visited,
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
	}
	ccfree(visited);
	ccv_nnc_tensor_symbol_t input_symbols[input_size];
	for (i = 0; i < input_size; i++)
		input_symbols[i] = inputs[i]->symbol;
	ccv_nnc_symbolic_graph_backward(dynamic_graph->symbolic,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
		&f_variable->binded_source, 1,
		&f_variable->symbol, 1, input_symbols, input_size);
	const ccv_nnc_tensor_symbol_t df = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->symbolic, f_variable->symbol);
	// Bind generated tensors.
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), dynamic_graph->var->rnum + 2, 0);
	for (i = 0; i < dynamic_graph->var->rnum; i++)
	{
		ccv_nnc_tensor_variable_t tensor_var = *(ccv_nnc_tensor_variable_t*)ccv_array_get(dynamic_graph->var, i);
		if (tensor_var->tensor_view && tensor_var->symbol.d >= 0)
		{
			ccv_nnc_tensor_bind_t bind = {
				.symbol = tensor_var->symbol,
				.tensor = (ccv_nnc_tensor_t*)tensor_var->tensor_view
			};
			ccv_array_push(tensor_binds, &bind);
		}
	}
	// Compiled graph comes from the df.
	ccv_array_clear(sources);
	ccv_array_t* const outgoings = ((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(dynamic_graph->symbolic->exec_symbol_info, f_variable->binded_source.d))->outgoings;
	if (outgoings)
		for (i = 0; i < outgoings->rnum; i++)
		{
			const int exec_idx = *(int*)ccv_array_get(outgoings, i);
			const ccv_nnc_graph_exec_symbol_info_t* const outgoing_exec_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(dynamic_graph->symbolic->exec_symbol_info, exec_idx);
			for (j = 0; j < outgoing_exec_info->input_size; j++)
				if (df.d == outgoing_exec_info->inputs[j])
				{
					int flag = 0;
					for (k = 0; !flag && k < sources->rnum; k++)
						flag = (exec_idx == ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, k))->d);
					if (!flag)
					{
						const ccv_nnc_graph_exec_symbol_t source = {
							.d = exec_idx,
							.graph = dynamic_graph->symbolic
						};
						ccv_array_push(sources, &source);
					}
					break;
				}
		}
	// Bind dt tensor.
	for (i = 0; i < output_size; i++)
	{
		outputs[i]->symbol = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->symbolic, inputs[i]->symbol);
		ccv_nnc_tensor_view_t* tensor_view = ccv_nnc_tensor_from_variable(dynamic_graph, outputs[i]);
		const ccv_nnc_tensor_bind_t dt_bind = {
			.symbol = outputs[i]->symbol,
			.tensor = (ccv_nnc_tensor_t*)tensor_view
		};
		ccv_array_push(tensor_binds, &dt_bind);
	}
	ccv_nnc_tensor_t* df_tensor = ccv_nnc_tensor_new(0, f_variable->symbol.info, 0);
	// Have an all 1 tensor as input.
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, &df_tensor, 1, 0);
	const ccv_nnc_tensor_bind_t df_bind = {
		.symbol = df,
		.tensor = df_tensor
	};
	ccv_array_push(tensor_binds, &df_bind);
	for (i = 0; i < exec_symbol_info_size; i++)
	{
		// Skip the one for f_variable so make everything still connected somewhat.
		if (i == f_variable->binded_source.d)
			continue;
		ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(dynamic_graph->symbolic->exec_symbol_info, i);
		if (exec_symbol_info->outgoings)
		{
			ccv_array_clear(dynamic_graph->ws);
			for (j = 0; j < exec_symbol_info->outgoings->rnum; j++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(exec_symbol_info->outgoings, j);
				if (outgoing_idx >= exec_symbol_info_size) // It is backward exec
					ccv_array_push(dynamic_graph->ws, &outgoing_idx);
			}
			for (j = 0; j < dynamic_graph->ws->rnum; j++)
				ccv_nnc_graph_exec_symbol_disjoin(dynamic_graph->symbolic, (ccv_nnc_graph_exec_symbol_t){
					.d = i,
					.graph = dynamic_graph->symbolic
				}, (ccv_nnc_graph_exec_symbol_t){
					.d = *(int*)ccv_array_get(dynamic_graph->ws, j),
					.graph = dynamic_graph->symbolic
				});
		}
	}
	ccv_nnc_graph_exec_symbol_t destinations[output_size];
	for (i = 0; i < output_size; i++)
		destinations[i] = ccv_nnc_graph_exec_symbol_for_backward(dynamic_graph->symbolic, outputs[i]->symbol);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(dynamic_graph->symbolic,
		(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
		destinations, output_size,
		&graph, &tensor_arena, &exec_arena);
	ccv_array_free(sources);
	ccv_array_free(tensor_binds);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(exec_arena);
	ccv_nnc_tensor_free(df_tensor);
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
	if (tensor_variable->binded_destinations)
		ccv_array_free(tensor_variable->binded_destinations);
	ccfree(tensor_variable);
	*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->var, index) = 0;
}

void ccv_nnc_dynamic_graph_dot(const ccv_nnc_dynamic_graph_t* const graph, const int flags, FILE* out)
{
	ccv_nnc_symbolic_graph_dot(graph->symbolic, flags, out);
}
