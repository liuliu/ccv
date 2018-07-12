#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"

/**
 * Level-4 API
 */

ccv_nnc_dynamic_graph_t* ccv_nnc_dynamic_graph_new(void)
{
	ccv_nnc_dynamic_graph_t* graph = ccmalloc(sizeof(ccv_nnc_dynamic_graph_t));
	graph->reuse_var = -1;
	graph->vars = ccv_array_new(sizeof(ccv_nnc_tensor_variable_t), 1, 0);
	graph->binds = ccv_array_new(sizeof(ccv_nnc_tensor_variable_graph_bind_t), 1, 0);
	graph->tape = ccv_nnc_symbolic_graph_new();
	graph->ws = 0;
	return graph;
}

static void _ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int zeroing)
{
	const int index = tensor_variable->index;
	if (tensor_variable->tensor_view)
	{
		if (CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view))
			ccv_nnc_tensor_view_free(tensor_variable->tensor_view);
		else
			ccv_nnc_tensor_free((ccv_nnc_tensor_t*)tensor_variable->tensor_view);
	}
	ccfree(tensor_variable);
	if (zeroing)
		*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, index) = 0;
	int i;
	for (i = graph->vars->rnum - 1; i >= 0; i--)
		if (*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, i) != 0)
		{
			graph->vars->rnum = i + 1;
			break;
		}
	if (index < graph->vars->rnum &&
		(index < graph->reuse_var || graph->reuse_var < 0))
		graph->reuse_var = index;
	else if (graph->reuse_var >= graph->vars->rnum)
		graph->reuse_var = -1;
}

static void _ccv_nnc_tensor_variable_graph_bind_free(ccv_nnc_tensor_variable_graph_bind_t* const bind, const int zeroing)
{
	bind->index = CCV_NNC_TENSOR_NO_VARIABLE;
	if (bind->sources)
		ccv_array_free(bind->sources);
	if (bind->destinations)
		ccv_array_free(bind->destinations);
	if (bind->tensor_view)
	{
		if (CCV_IS_TENSOR_VIEW(bind->tensor_view))
			ccv_nnc_tensor_view_free(bind->tensor_view);
		else
			ccv_nnc_tensor_free((ccv_nnc_tensor_t*)bind->tensor_view);
	}
	if (zeroing)
	{
		bind->sources = 0;
		bind->destinations = 0;
		bind->tensor_view = 0;
	}
}

void ccv_nnc_dynamic_graph_free(ccv_nnc_dynamic_graph_t* const graph)
{
	int i;
	for (i = 0; i < graph->vars->rnum; i++)
	{
		ccv_nnc_tensor_variable_t tensor_variable = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, i);
		if (tensor_variable)
			_ccv_nnc_tensor_variable_free(graph, tensor_variable, 0);
	}
	ccv_array_free(graph->vars);
	for (i = 0; i < graph->binds->rnum; i++)
		_ccv_nnc_tensor_variable_graph_bind_free((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, i), 0);
	ccv_array_free(graph->binds);
	ccv_nnc_symbolic_graph_free(graph->tape);
	if (graph->ws)
		ccv_array_free(graph->ws);
	ccfree(graph);
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_new_impl(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_param_t info)
{
	ccv_nnc_tensor_variable_t tensor_variable = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	tensor_variable->alias_ref = 0;
	tensor_variable->info = info;
	tensor_variable->symbol = NO_TENSOR_SYMBOL;
	tensor_variable->tensor_view = 0;
	if (graph->reuse_var >= 0)
	{
		const int reuse_var = graph->reuse_var;
		assert(reuse_var < graph->vars->rnum);
		tensor_variable->index = reuse_var;
		*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, reuse_var) = tensor_variable;
		int i;
		graph->reuse_var = -1;
		for (i = reuse_var + 1; i < graph->vars->rnum && graph->reuse_var < 0; i++)
			if (*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, i) == 0)
				graph->reuse_var = i;
	} else {
		tensor_variable->index = graph->vars->rnum;
		ccv_array_push(graph->vars, &tensor_variable);
	}
	return tensor_variable;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_alias_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info)
{
	assert(!tensor_variable->alias_ref);
	ccv_nnc_tensor_variable_t variable_alias = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	variable_alias->alias_ref = tensor_variable->index + 1;
	variable_alias->info = info;
	variable_alias->symbol = NO_TENSOR_SYMBOL;
	variable_alias->tensor_view = 0;
	memcpy(variable_alias->ofs, ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(variable_alias->inc, inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	if (graph->reuse_var >= 0)
	{
		const int reuse_var = graph->reuse_var;
		assert(reuse_var < graph->vars->rnum);
		variable_alias->index = reuse_var;
		*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, reuse_var) = variable_alias;
		int i;
		graph->reuse_var = -1;
		for (i = reuse_var + 1; i < graph->vars->rnum && graph->reuse_var < 0; i++)
			if (*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, i) == 0)
				graph->reuse_var = i;
	} else {
		variable_alias->index = graph->vars->rnum;
		ccv_array_push(graph->vars, &variable_alias);
	}
	return variable_alias;
}

ccv_nnc_tensor_t* ccv_nnc_tensor_from_variable(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	if (tensor_variable->tensor_view)
	{
		if (tensor_variable->alias_ref)
		{
			const int alias_ref = tensor_variable->alias_ref - 1;
			assert(alias_ref >= 0);
			ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
			ccv_nnc_tensor_view_t* const tv = tensor_variable->tensor_view;
			// Update the tensor_view pointer every time access it, because the underlying variable it alias to have changed.
			tv->data.u8 = variable_to->tensor_view->data.u8 + tv->off;
		}
		return (ccv_nnc_tensor_t*)tensor_variable->tensor_view;
	}
	if (!tensor_variable->alias_ref)
	{
		tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(0, tensor_variable->info, 0);
		return (ccv_nnc_tensor_t*)tensor_variable->tensor_view;
	}
	const int alias_ref = tensor_variable->alias_ref - 1;
	assert(alias_ref >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
	assert(!variable_to->alias_ref);
	if (!variable_to->tensor_view)
		variable_to->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(0, variable_to->info, 0);
	tensor_variable->tensor_view = ccv_nnc_tensor_view_new((ccv_nnc_tensor_t*)variable_to->tensor_view, tensor_variable->info.dim, tensor_variable->ofs, tensor_variable->inc);
	return 0;
}

static void _ccv_nnc_tensor_symbol_extra_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const ccv_nnc_tensor_symbol_t symbol)
{
	if (symbol.d >= graph->binds->rnum)
	{
		const int rnum = graph->binds->rnum;
		ccv_array_resize(graph->binds, symbol.d + 1);
		int i;
		for (i = rnum; i < graph->binds->rnum; i++)
			((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, i))->index = CCV_NNC_TENSOR_NO_VARIABLE;
	}
	ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, symbol.d);
	bind->index = tensor_variable->index;
	if (bind->sources)
		ccv_array_free(bind->sources);
	bind->sources = 0;
	if (bind->destinations)
		ccv_array_free(bind->destinations);
	bind->destinations = 0;
	bind->tensor_view = 0;
}

static ccv_nnc_tensor_symbol_t _ccv_nnc_tensor_symbol_from_variable(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	if (tensor_variable->symbol.d >= 0)
		return tensor_variable->symbol;
	if (!tensor_variable->alias_ref)
	{
		const ccv_nnc_tensor_symbol_t symbol = tensor_variable->symbol = ccv_nnc_tensor_symbol_new(graph->tape, tensor_variable->info, 0);
		_ccv_nnc_tensor_symbol_extra_new(graph, tensor_variable, symbol);
		return symbol;
	}
	const int alias_ref = tensor_variable->alias_ref - 1;
	assert(alias_ref >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
	assert(!variable_to->alias_ref);
	const ccv_nnc_tensor_symbol_t symbol = tensor_variable->symbol = ccv_nnc_tensor_symbol_alias_new(graph->tape, _ccv_nnc_tensor_symbol_from_variable(graph, variable_to), tensor_variable->ofs, tensor_variable->inc, tensor_variable->info, 0);
	_ccv_nnc_tensor_symbol_extra_new(graph, tensor_variable, symbol);
	return symbol;
}

// Return the tensor variable that is old (the provided tensor variable will have a new setting).
ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_exchange_new(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable)
{
	struct ccv_nnc_tensor_variable_s x = *tensor_variable;
	ccv_nnc_tensor_variable_t new_variable;
	// Need to handle alias.
	if (x.alias_ref)
		new_variable = ccv_nnc_tensor_variable_alias_new(graph, *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, x.alias_ref - 1), x.ofs, x.inc, x.info);
	else
		new_variable = ccv_nnc_tensor_variable_new(graph, x.info);
	*tensor_variable = *new_variable;
	*new_variable = x;
	// The index should be the same though.
	const int index = new_variable->index;
	new_variable->index = tensor_variable->index;
	if (new_variable->symbol.d != CCV_NNC_NO_TENSOR_SYMBOL)
	{
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, new_variable->symbol.d);
		bind->index = new_variable->index;
	}
	tensor_variable->index = index;
	return new_variable;
}

int ccv_nnc_dynamic_graph_exec(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size)
{
	int i, j;
	for (i = 0; i < input_size; i++)
		if (inputs[i] && !inputs[i]->alias_ref)
			{ assert(inputs[i]->tensor_view); }
	ccv_nnc_tensor_t* input_tensors[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_tensors[i] = inputs[i] ? ccv_nnc_tensor_from_variable(graph, inputs[i]) : 0;
	ccv_nnc_tensor_symbol_t input_symbols[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_symbols[i] = inputs[i] ? _ccv_nnc_tensor_symbol_from_variable(graph, inputs[i]) : NO_TENSOR_SYMBOL;
	ccv_array_t* input_sources[ccv_max(1, input_size)];
	ccv_array_t* input_alias_sources[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
	{
		input_sources[i] = input_symbols[i].d != CCV_NNC_NO_TENSOR_SYMBOL ? ((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, input_symbols[i].d))->sources : 0;
		if (inputs[i] && inputs[i]->alias_ref)
		{
			const int alias_ref = outputs[i]->alias_ref - 1;
			assert(alias_ref >= 0);
			ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
			input_alias_sources[i] = ((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, variable_to->symbol.d))->sources;
		} else
			input_alias_sources[i] = 0;
	}
	int output_auto = 0;
	for (i = 0; !output_auto && i < output_size; i++)
		output_auto = outputs[i] ? ccv_nnc_is_tensor_auto(outputs[i]->info) : 0;
	// One extra step, infer the parameters for outputs.
	if (output_auto)
	{
		ccv_nnc_tensor_param_t input_params[ccv_max(1, input_size)];
		for (i = 0; i < input_size; i++)
			input_params[i] = inputs[i] ? inputs[i]->info : ccv_nnc_tensor_auto;
		ccv_nnc_tensor_param_t output_params[ccv_max(1, output_size)];
		for (i = 0; i < output_size; i++)
			output_params[i] = outputs[i] ? outputs[i]->info : ccv_nnc_tensor_auto;
		ccv_nnc_hint_tensor_auto(cmd, input_params, input_size, hint, output_params, output_size);
		for (i = 0; i < output_size; i++)
			if (outputs[i])
				outputs[i]->info = output_params[i];
	}
	int freeable_size = 0;
	ccv_nnc_tensor_variable_t freeables[ccv_max(1, output_size)];
	// Refresh the symbol if it is binded to an existing exec. Otherwise we cannot keep the SSA guarantee.
	for (i = 0; i < output_size; i++)
	{
		// First, go over to see whether there is enforce inplace.
		int enforce_idx = -1;
		for (j = 0; enforce_idx < 0 && j < input_size; j++)
			if (inputs[j] && ccv_nnc_cmd_enforce_inplace(cmd, j, i))
				enforce_idx = j;
		if (enforce_idx >= 0)
			{ assert(outputs[i] == inputs[enforce_idx] && outputs[i]->symbol.d != CCV_NNC_NO_TENSOR_SYMBOL); }
		if (outputs[i] && outputs[i]->symbol.d != CCV_NNC_NO_TENSOR_SYMBOL)
		{
			const ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, outputs[i]->symbol.d);
			if (enforce_idx >= 0)
				{ assert(!bind->destinations || bind->destinations->rnum == 0); }
			if (bind->sources && bind->sources->rnum > 0)
			{
				const ccv_nnc_tensor_variable_t old_var = freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(graph, outputs[i]);
				// If this is enforce output, make sure the tensor view is taken by the output.
				if (enforce_idx >= 0)
				{
					outputs[i]->tensor_view = old_var->tensor_view; // Make sure the tensor view is taken over by the output.
					old_var->tensor_view = 0;
				}
			}
		}
	}
	ccv_nnc_tensor_t* output_tensors[ccv_max(1, output_size)];
	for (i = 0; i < output_size; i++)
		output_tensors[i] = outputs[i] ? ccv_nnc_tensor_from_variable(graph, outputs[i]) : 0;
	ccv_nnc_cmd_exec(cmd, hint, flags, input_tensors, input_size, output_tensors, output_size, 0);
	if (input_size > 0) // No need to record the execution if there is no input.
	{
		ccv_nnc_tensor_symbol_t output_symbols[ccv_max(1, output_size)];
		for (i = 0; i < output_size; i++)
			output_symbols[i] = outputs[i] ? _ccv_nnc_tensor_symbol_from_variable(graph, outputs[i]) : NO_TENSOR_SYMBOL;
		ccv_nnc_graph_exec_symbol_t graph_exec = ccv_nnc_graph_exec_symbol_new(graph->tape, cmd, input_symbols, input_size, output_symbols, output_size, 0);
		// This needs to be done before we set the new sources on the outputs.
		for (i = 0; i < input_size; i++)
		{
			if (input_sources[i])
				for (j = 0; j < input_sources[i]->rnum; j++)
					ccv_nnc_graph_exec_symbol_concat(graph->tape, (ccv_nnc_graph_exec_symbol_t){
						.d = *(int*)ccv_array_get(input_sources[i], j),
						.graph = graph->tape
					}, graph_exec);
			if (input_alias_sources[i])
				for (j = 0; j < input_alias_sources[i]->rnum; j++)
					ccv_nnc_graph_exec_symbol_concat(graph->tape, (ccv_nnc_graph_exec_symbol_t){
						.d = *(int*)ccv_array_get(input_alias_sources[i], j),
						.graph = graph->tape
					}, graph_exec);
		}
		for (i = 0; i < input_size; i++)
		{
			if (!inputs[i])
				continue;
			ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, inputs[i]->symbol.d);
			if (!bind->destinations)
				bind->destinations = ccv_array_new(sizeof(int), 1, 0);
			ccv_array_add_unique_int(bind->destinations, graph_exec.d);
		}
		for (i = 0; i < output_size; i++)
		{
			if (!outputs[i])
				continue;
			ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, outputs[i]->symbol.d);
			assert(!bind->sources); // This is a new symbol, therefore, no binded sources associated yet.
			bind->sources = ccv_array_new(sizeof(int), 1, 0);
			ccv_array_add_unique_int(bind->sources, graph_exec.d);
			if (outputs[i]->alias_ref)
			{
					const int alias_ref = outputs[i]->alias_ref - 1;
					assert(alias_ref >= 0);
					ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
					ccv_nnc_tensor_variable_graph_bind_t* const bind_to = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, variable_to->symbol.d);
					if (!bind_to->sources)
						bind_to->sources = ccv_array_new(sizeof(int), 1, 0);
					ccv_array_add_unique_int(bind_to->sources, graph_exec.d);
			}
		}
	}
	// Now, able to free some of the reused outputs.
	for (i = 0; i < freeable_size; i++)
		ccv_nnc_tensor_variable_free(graph, freeables[i]);
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_update_bind_destinations_when_free(ccv_nnc_symbolic_graph_t* const graph, const int freed_symbol_d, ccv_nnc_tensor_variable_graph_bind_t* const bind, const int tensor_index)
{
	int i;
	if (bind->destinations)
	{
		int flag = 0;
		for (i = 0; !flag && i < bind->destinations->rnum; i++)
		{
			const int symbol_d = *(int*)ccv_array_get(bind->destinations, i);
			if (symbol_d == freed_symbol_d)
			{
				if (i < bind->destinations->rnum - 1)
					*(int*)ccv_array_get(bind->destinations, i) = *(int*)ccv_array_get(bind->destinations, bind->destinations->rnum - 1);
				--bind->destinations->rnum;
				flag = 1;
			}
		}
		// This symbol can be freed.
		if (flag && bind->index < 0 &&
			(!bind->sources || bind->sources->rnum == 0) &&
			(!bind->destinations || bind->destinations->rnum == 0))
		{
			_ccv_nnc_tensor_variable_graph_bind_free(bind, 1);
			ccv_nnc_tensor_symbol_free(graph, (ccv_nnc_tensor_symbol_t){
				.d = tensor_index,
				.graph = graph
			});
		}
	}
}

static void _ccv_nnc_update_bind_sources_when_free(ccv_nnc_symbolic_graph_t* const graph, const int freed_symbol_d, ccv_nnc_tensor_variable_graph_bind_t* const bind, const int tensor_index)
{
	int i;
	if (bind->sources)
	{
		int flag = 0;
		for (i = 0; !flag && i < bind->sources->rnum; i++)
		{
			const int symbol_d = *(int*)ccv_array_get(bind->sources, i);
			if (symbol_d == freed_symbol_d)
			{
				if (i < bind->sources->rnum - 1)
					*(int*)ccv_array_get(bind->sources, i) = *(int*)ccv_array_get(bind->sources, bind->sources->rnum - 1);
				--bind->sources->rnum;
				flag = 1;
			}
		}
		// This symbol can be freed.
		if (flag && bind->index < 0 &&
			(!bind->sources || bind->sources->rnum == 0) &&
			(!bind->destinations || bind->destinations->rnum == 0))
		{
			_ccv_nnc_tensor_variable_graph_bind_free(bind, 1);
			ccv_nnc_tensor_symbol_free(graph, (ccv_nnc_tensor_symbol_t){
				.d = tensor_index,
				.graph = graph
			});
		}
	}
}

static void _ccv_nnc_update_bind_sources_destinations_when_free(ccv_nnc_symbolic_graph_t* const graph, const int freed_symbol_d, ccv_array_t* const binds, const int* const inputs, const int input_size, const int* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < input_size; i++)
		if (inputs[i] >= 0 && inputs[i] < binds->rnum)
		{
			ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, inputs[i]);
			_ccv_nnc_update_bind_destinations_when_free(graph, freed_symbol_d, bind, inputs[i]);
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(graph, (ccv_nnc_tensor_symbol_t){
				.d = inputs[i],
				.graph = graph
			});
			if (alias_to.d >= 0 && alias_to.d < binds->rnum)
				_ccv_nnc_update_bind_destinations_when_free(graph, freed_symbol_d, (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, alias_to.d), alias_to.d);
		}
	// Note that this works because there is no overlap of inputs / outputs. (What about alias?).
	for (i = 0; i < output_size; i++)
		if (outputs[i] >= 0 && outputs[i] < binds->rnum)
		{
			ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, outputs[i]);
			_ccv_nnc_update_bind_sources_when_free(graph, freed_symbol_d, bind, outputs[i]);
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(graph, (ccv_nnc_tensor_symbol_t){
				.d = outputs[i],
				.graph = graph
			});
			if (alias_to.d >= 0 && alias_to.d < binds->rnum)
				_ccv_nnc_update_bind_sources_when_free(graph, freed_symbol_d, (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, alias_to.d), alias_to.d);
		}
}

void ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	// If it contains a symbol, this tensor variable is not a free variable. It is either used as input or output.
	if (tensor_variable->symbol.d != CCV_NNC_NO_TENSOR_SYMBOL)
	{
		// If it is not a free variable, when can we free the symbol and the underlying variable?
		// 1. There should be no sources (the command generate this tensor should be freed);
		// 2. The destinations (the commands that uses this tensor) should have no other inputs, or the other inputs has no binded sources as well.
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, tensor_variable->symbol.d);
		// There should be no source associated with it no more.
		int free_symbol = 0;
		if (!bind->sources || bind->sources->rnum == 0)
		{
			int i, j;
			free_symbol = 1; // Assume we can free this symbol.
			if (!graph->ws)
				graph->ws = ccv_array_new(sizeof(int), bind->destinations ? bind->destinations->rnum : 0, 0);
			ccv_array_t* const ws = graph->ws;
			ccv_array_clear(ws);
			if (bind->destinations)
				for (i = 0; i < bind->destinations->rnum; i++)
					ccv_array_add_unique_int(ws, *(int*)ccv_array_get(bind->destinations, i));
			const int ws_init_size = ws->rnum;
			// Go through all the exec symbols use this tensor, to see whether they have inputs that has other sources.
			if (bind->destinations)
				for (i = 0; i < bind->destinations->rnum;)
				{
					const int symbol_d = *(int*)ccv_array_get(bind->destinations, i);
					const ccv_nnc_graph_exec_symbol_t symbol = {
						.d = symbol_d,
						.graph = graph->tape
					};
					const int* inputs; int input_size;
					ccv_nnc_graph_exec_symbol_io(graph->tape, symbol, &inputs, &input_size, 0, 0);
					int flag = 0;
					for (j = 0; !flag && j < input_size; j++)
						if (inputs[j] >= 0 && inputs[j] < graph->binds->rnum && inputs[j] != tensor_variable->symbol.d)
						{
							ccv_nnc_tensor_variable_graph_bind_t* const other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, inputs[j]);
							flag = other_bind->index >= 0 || (other_bind->sources && other_bind->sources->rnum > 0);
						}
					free_symbol = (free_symbol && !flag);
					if (flag)
						++i;
					else {
						// It is safe to remove this exec. Have to remove it proactively otherwise I may mess up the iteration.
						if (i < bind->destinations->rnum - 1)
							*(int*)ccv_array_get(bind->destinations, i) = *(int*)ccv_array_get(bind->destinations, bind->destinations->rnum - 1);
						--bind->destinations->rnum;
						// Go over inputs and remove all references from binded destinations.
						// and go over outputs remove all references from binded sources.
						const int* outputs; int output_size;
						ccv_nnc_graph_exec_symbol_io(graph->tape, symbol, 0, 0, &outputs, &output_size);
						_ccv_nnc_update_bind_sources_destinations_when_free(graph->tape, symbol_d, graph->binds, inputs, input_size, outputs, output_size);
						const int* outgoings; int outgoing_size;
						ccv_nnc_graph_exec_symbol_to(graph->tape, symbol, &outgoings, &outgoing_size);
						for (j = 0; j < outgoing_size; j++)
							ccv_array_add_unique_int(ws, outgoings[j]);
						ccv_nnc_graph_exec_symbol_free(graph->tape, symbol);
					}
				}
			if (free_symbol)
			{
				// Nothing here requires this symbol, should be able to free it.
				_ccv_nnc_tensor_variable_graph_bind_free((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, tensor_variable->symbol.d), 1);
				ccv_nnc_tensor_symbol_free(graph->tape, tensor_variable->symbol);
				// Now, go over the outgoings, if it is removed, add more to it. Note that the ws array can grow while iterating over.
				for (i = ws_init_size; i < ws->rnum; i++)
				{
					const int symbol_d = *(int*)ccv_array_get(ws, i);
					const ccv_nnc_graph_exec_symbol_t symbol = {
						.d = symbol_d,
						.graph = graph->tape
					};
					const int* inputs; int input_size;
					ccv_nnc_graph_exec_symbol_io(graph->tape, symbol, &inputs, &input_size, 0, 0);
					int flag = 0;
					for (j = 0; !flag && j < input_size; j++)
						if (inputs[j] >= 0 && inputs[j] < graph->binds->rnum)
						{
							ccv_nnc_tensor_variable_graph_bind_t* const other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, inputs[j]);
							flag = other_bind->index >= 0 || (other_bind->sources && other_bind->sources->rnum > 0);
						}
					// Went over all the inputs, it turns out no more inputs has other references, safe to remove.
					if (!flag)
					{
						const int* outputs; int output_size;
						ccv_nnc_graph_exec_symbol_io(graph->tape, symbol, 0, 0, &outputs, &output_size);
						_ccv_nnc_update_bind_sources_destinations_when_free(graph->tape, symbol_d, graph->binds, inputs, input_size, outputs, output_size);
						const int* outgoings; int outgoing_size;
						ccv_nnc_graph_exec_symbol_to(graph->tape, symbol, &outgoings, &outgoing_size);
						// It it has outgoings, add that for further inspection.
						for (j = 0; j < outgoing_size; j++)
							ccv_array_add_unique_int(ws, outgoings[j]);
						ccv_nnc_graph_exec_symbol_free(graph->tape, symbol);
					}
				}
			}
		}
		// If this symbol is not freed, move the tensor view to the bind.
		if (!free_symbol)
		{
			bind->index = CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED; // This tensor variable will be freed, but this symbol extra will continue exists.
			bind->tensor_view = tensor_variable->tensor_view; // Transfer the ownership to the bind.
			tensor_variable->tensor_view = 0;
		}
	}
	_ccv_nnc_tensor_variable_free(graph, tensor_variable, 1);
}

void ccv_nnc_dynamic_graph_dot(const ccv_nnc_dynamic_graph_t* const graph, const int flags, FILE* out)
{
	ccv_nnc_symbolic_graph_dot(graph->tape, flags, out);
}
