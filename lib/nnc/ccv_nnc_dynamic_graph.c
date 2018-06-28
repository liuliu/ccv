#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"

/**
 * Level-4 API
 */

struct ccv_nnc_tensor_variable_s {
	int index;
	int alias_ref;
	ccv_nnc_tensor_param_t info;
	ccv_nnc_tensor_symbol_t symbol;
	ccv_nnc_tensor_view_t* tensor_view;
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
};

enum {
	CCV_NNC_TENSOR_NO_VARIABLE = -1,
	CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED = -2,
};

typedef struct { // Extra information kept per tensor symbol along with symbolic graph.
	int index; // The index back into the tensor variable. -1 meant no associated tensor vairable.
	ccv_array_t* sources; // array of graph_exec_symbol, use this tensor symbol as output.
	ccv_array_t* destinations; // array of graph_exec_symbol, use this tensor symbol as input.
	ccv_nnc_tensor_view_t* tensor_view; // Transfer ownership of the tensor view to here.
} ccv_nnc_tensor_variable_graph_bind_t;

struct ccv_nnc_dynamic_graph_s {
	ccv_array_t* vars; // Array keeps track of all allocated tensor variable.
	ccv_array_t* binds; // Array keeps track of extra information for a tensor symbol.
	ccv_nnc_symbolic_graph_t* tape; // Symbolic graph to keep track of computation.
	ccv_array_t* ws; // array of integers as workspace
};

ccv_nnc_dynamic_graph_t* ccv_nnc_dynamic_graph_new(void)
{
	ccv_nnc_dynamic_graph_t* graph = ccmalloc(sizeof(ccv_nnc_dynamic_graph_t));
	graph->vars = ccv_array_new(sizeof(ccv_nnc_tensor_variable_t), 1, 0);
	graph->binds = ccv_array_new(sizeof(ccv_nnc_tensor_variable_graph_bind_t), 1, 0);
	graph->tape = ccv_nnc_symbolic_graph_new();
	graph->ws = 0;
	return graph;
}

static void _ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
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
	*(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, index) = 0;
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
			_ccv_nnc_tensor_variable_free(graph, tensor_variable);
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
	ccv_array_push(graph->vars, &tensor_variable);
	tensor_variable->index = graph->vars->rnum - 1;
	tensor_variable->alias_ref = 0;
	tensor_variable->info = info;
	tensor_variable->symbol = NO_TENSOR_SYMBOL;
	tensor_variable->tensor_view = 0;
	return tensor_variable;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_alias_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info)
{
	assert(!tensor_variable->alias_ref);
	ccv_nnc_tensor_variable_t variable_alias = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	ccv_array_push(graph->vars, &variable_alias);
	variable_alias->index = graph->vars->rnum - 1;
	variable_alias->alias_ref = tensor_variable->index + 1;
	variable_alias->info = info;
	variable_alias->symbol = NO_TENSOR_SYMBOL;
	variable_alias->tensor_view = 0;
	memcpy(variable_alias->ofs, ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(variable_alias->inc, inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
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
static ccv_nnc_tensor_variable_t _ccv_nnc_tensor_variable_exchange_new(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable)
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
		if (outputs[i] && outputs[i]->symbol.d != CCV_NNC_NO_TENSOR_SYMBOL)
		{
			const ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, outputs[i]->symbol.d);
			if (bind->sources && bind->sources->rnum > 0)
				freeables[freeable_size++] = _ccv_nnc_tensor_variable_exchange_new(graph, outputs[i]);
		}
	for (i = 0; i < input_size; i++)
		if (inputs[i])
			for (j = 0; j < output_size; j++)
				// If enforces inplace, use the same tensor_view as the input.
				if (outputs[j] && ccv_nnc_cmd_enforce_inplace(cmd, i, j))
					outputs[j]->tensor_view = inputs[i]->tensor_view;
	ccv_nnc_tensor_t* output_tensors[ccv_max(1, output_size)];
	for (i = 0; i < output_size; i++)
		output_tensors[i] = outputs[i] ? ccv_nnc_tensor_from_variable(graph, outputs[i]) : 0;
	ccv_nnc_cmd_exec(cmd, hint, flags, input_tensors, input_size, output_tensors, output_size, 0);
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
	// Now, able to free some of the reused outputs.
	for (i = 0; i < freeable_size; i++)
		ccv_nnc_tensor_variable_free(graph, freeables[i]);
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_insert_if_prior_to_any(const ccv_nnc_symbolic_graph_t* const graph, const int d, ccv_array_t* const sources, uint32_t* const visited, int* const buf0, int* const buf1)
{
	if (visited[(d >> 5)] & (1u << (d & 31)))
		return;
	visited[(d >> 5)] |= (1u << (d & 31));
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
			const int* outgoings; int outgoing_size;
			ccv_nnc_graph_exec_symbol_to(graph, (ccv_nnc_graph_exec_symbol_t){
				.d = buf[p][i],
				.graph = graph
			}, &outgoings, &outgoing_size);
			for (j = 0; j < outgoing_size; j++)
			{
				const int outgoing_idx = outgoings[j];
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
				if (visited[(outgoing_idx >> 5)] & (1u << (outgoing_idx & 31)))
					continue;
				visited[(outgoing_idx >> 5)] |= (1u << (outgoing_idx & 31));
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

static void _ccv_nnc_remove_if_prior_to_any(const ccv_nnc_symbolic_graph_t* const graph, const int d, ccv_array_t* const destinations, uint32_t* const visited, int* const buf0, int* const buf1)
{
	int i, j, k;
	// If it is already visited, this is the later one, we are good.
	if (visited[(d >> 5)] & (1u << (d & 31)))
		return;
	visited[(d >> 5)] |= (1u << (d & 31));
	buf0[0] = d;
	int* buf[2] = {
		buf0, buf1
	};
	int buf_size[2] = {
		1, 0
	};
	int p = 0, q = 1;
	int flag = 0;
	while (!flag && buf_size[p] > 0)
	{
		buf_size[q] = 0;
		for (i = 0; !flag && i < buf_size[p]; i++)
		{
			const int* outgoings; int outgoing_size;
			ccv_nnc_graph_exec_symbol_to(graph, (ccv_nnc_graph_exec_symbol_t){
				.d = buf[p][i],
				.graph = graph
			}, &outgoings, &outgoing_size);
			for (j = 0; j < outgoing_size; j++)
			{
				const int outgoing_idx = outgoings[j];
				// If this node happens to be visited, do nothing.
				if (visited[(outgoing_idx >> 5)] & (1u << (outgoing_idx & 31)))
					continue;
				for (k = 0; !flag && k < destinations->rnum; k++)
				{
					ccv_nnc_graph_exec_symbol_t* const destination_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, k);
					// If this outgoing idx is one of the destination, delete current node.
					flag = (destination_symbol->d == outgoing_idx);
				}
				visited[(outgoing_idx >> 5)] |= (1u << (outgoing_idx & 31));
				buf[q][buf_size[q]] = outgoing_idx;
				++buf_size[q];
			}
		}
		CCV_SWAP(p, q, i);
	}
	if (flag)
		for (i = 0; i < destinations->rnum; i++)
		{
			ccv_nnc_graph_exec_symbol_t* const destination_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, i);
			// If this outgoing idx is one of the destination, delete current node.
			if (destination_symbol->d == d)
			{
				// Delete this from the list.
				if (i < destinations->rnum - 1)
					destination_symbol->d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, destinations->rnum - 1))->d;
				--destinations->rnum;
				break;
			}
		}
}

void ccv_nnc_dynamic_graph_backward(ccv_nnc_dynamic_graph_t* const dynamic_graph, const ccv_nnc_tensor_variable_t f_variable, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size)
{
	int d, i, j, k;
	assert(input_size == output_size);
	assert(input_size > 0);
	// Both f_variable and tensor_variable should be, at least, executed. Otherwise we cannot differentiate.
	assert(f_variable->symbol.d >= 0);
	const ccv_nnc_tensor_variable_graph_bind_t* const f_symbol_extra = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, f_variable->symbol.d);
	assert(f_symbol_extra->sources && f_symbol_extra->sources->rnum > 0);
	for (i = 0; i < input_size; i++)
	{
		assert(inputs[i]->symbol.d >= 0);
		assert(((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations &&
			((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations->rnum > 0);
	}
	// Refresh the symbol if it is not empty, we will use new symbol for the output tensor variables.
	for (i = 0; i < output_size; i++)
	{
		if (ccv_nnc_is_tensor_auto(outputs[i]->info))
			outputs[i]->info = inputs[i]->info;
		if (outputs[i]->symbol.d >= 0)
			outputs[i] = _ccv_nnc_tensor_variable_exchange_new(dynamic_graph, outputs[i]);
	}
	const int exec_symbol_info_size = ccv_nnc_graph_exec_symbol_count(dynamic_graph->tape);
	ccv_array_t* const sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
	if (!dynamic_graph->ws)
		dynamic_graph->ws = ccv_array_new(sizeof(int), exec_symbol_info_size * 2 + ((exec_symbol_info_size + 31) >> 5), 0);
	ccv_array_t* const ws = dynamic_graph->ws;
	ccv_array_resize(ws, exec_symbol_info_size * 2 + ((exec_symbol_info_size + 31) >> 5));
	// set visited to all 0.
	memset((uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2), 0, sizeof(uint32_t) * ((exec_symbol_info_size + 31) >> 5));
	for (i = 0; i < input_size; i++)
	{
		ccv_array_t* const destinations = ((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations;
		for (j = 0; j < destinations->rnum; j++)
			_ccv_nnc_insert_if_prior_to_any(dynamic_graph->tape,
				*(int*)ccv_array_get(destinations, j),
				sources, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
	}
	ccv_array_t* const destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), f_symbol_extra->sources->rnum, 0);
	ccv_array_resize(destinations, f_symbol_extra->sources->rnum);
	for (i = 0; i < destinations->rnum; i++)
	{
		ccv_nnc_graph_exec_symbol_t* const symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, i);
		symbol->d = *(int*)ccv_array_get(f_symbol_extra->sources, i);
		symbol->graph = dynamic_graph->tape;
	}
	// Go over sources, because destinations will get removed all the time, thus, the index is not accurate.
	if (destinations->rnum > 1)
		for (i = 0 ; i < f_symbol_extra->sources->rnum; i++)
		{
			memset((uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2), 0, sizeof(uint32_t) * ((exec_symbol_info_size + 31) >> 5));
			_ccv_nnc_remove_if_prior_to_any(dynamic_graph->tape,
				*(int*)ccv_array_get(f_symbol_extra->sources, i),
				destinations, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
		}
	ccv_nnc_tensor_symbol_t input_symbols[input_size];
	for (i = 0; i < input_size; i++)
		input_symbols[i] = inputs[i]->symbol;
	ccv_nnc_symbolic_graph_backward(dynamic_graph->tape,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum,
		&f_variable->symbol, 1, input_symbols, input_size);
	const ccv_nnc_tensor_symbol_t df = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, f_variable->symbol);
	// Bind generated tensors.
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), dynamic_graph->vars->rnum + 2, 0);
	for (i = 0; i < dynamic_graph->vars->rnum; i++)
	{
		ccv_nnc_tensor_variable_t var = *(ccv_nnc_tensor_variable_t*)ccv_array_get(dynamic_graph->vars, i);
		if (var && var->tensor_view && var->symbol.d >= 0)
		{
			ccv_nnc_tensor_bind_t bind = {
				.symbol = var->symbol,
				.tensor = (ccv_nnc_tensor_t*)var->tensor_view
			};
			ccv_array_push(tensor_binds, &bind);
		}
	}
	for (i = 0; i < dynamic_graph->binds->rnum; i++)
	{
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, i);
		if (bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED && bind->tensor_view)
		{
			ccv_nnc_tensor_bind_t b = {
				.symbol = {
					.d = i,
					.graph = dynamic_graph->tape,
				},
				.tensor = (ccv_nnc_tensor_t*)bind->tensor_view
			};
			ccv_array_push(tensor_binds, &b);
		}
	}
	// Compiled graph comes from the df.
	ccv_array_clear(sources);
	assert(df.d >= 0);
	for (d = 0; d < destinations->rnum; d++)
	{
		const ccv_nnc_graph_exec_symbol_t* const destination = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, d);
		const int* outgoings; int outgoing_size;
		ccv_nnc_graph_exec_symbol_to(dynamic_graph->tape, *destination, &outgoings, &outgoing_size);
		for (i = 0; i < outgoing_size; i++)
		{
			const int exec_idx = outgoings[i];
			const int* inputs; int input_size;
			ccv_nnc_graph_exec_symbol_io(dynamic_graph->tape, (ccv_nnc_graph_exec_symbol_t){
				.d = exec_idx,
				.graph = dynamic_graph->tape
			}, &inputs, &input_size, 0, 0);
			for (j = 0; j < input_size; j++)
			{
				const int input = inputs[j];
				const int alias_ref = input >= 0 ? ccv_nnc_tensor_symbol_alias_to(dynamic_graph->tape, (ccv_nnc_tensor_symbol_t){
					.d = input,
					.graph = dynamic_graph->tape
				}).d : CCV_NNC_NO_TENSOR_SYMBOL; // This could be CCV_NNC_NO_TENSOR_SYMBOL, which is negative.
				// alias_ref is either exists, or -1.
				if (df.d == input || df.d == alias_ref)
				{
					int flag = 0;
					for (k = 0; !flag && k < sources->rnum; k++)
						flag = (exec_idx == ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, k))->d);
					if (!flag)
					{
						const ccv_nnc_graph_exec_symbol_t source = {
							.d = exec_idx,
							.graph = dynamic_graph->tape
						};
						ccv_array_push(sources, &source);
					}
					break;
				}
			}
		}
	}
	// Bind dt tensor.
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t symbol = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, inputs[i]->symbol);
		ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_from_variable(dynamic_graph, outputs[i]);
		const ccv_nnc_tensor_bind_t dt_bind = {
			.symbol = symbol,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &dt_bind);
	}
	ccv_nnc_graph_exec_symbol_t set_ones = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, CMD_SET_FORWARD(1), 0, 0, &df, 1, 0);
	for (i = 0; i < sources->rnum; i++)
		ccv_nnc_graph_exec_symbol_concat(dynamic_graph->tape, set_ones, *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, i));
	ccv_array_free(sources);
	ccv_array_clear(destinations);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t symbol = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, inputs[i]->symbol);
		const ccv_nnc_graph_exec_symbol_t destination = ccv_nnc_graph_exec_symbol_for_backward(dynamic_graph->tape, symbol);
		ccv_array_push(destinations, &destination);
	}
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(dynamic_graph->tape,
		(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
		0, 0,
		&set_ones, 1,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum,
		&graph, &tensor_arena, &exec_arena);
	ccv_array_free(destinations);
	ccv_array_free(tensor_binds);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(exec_arena);
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
			assert(bind->destinations && bind->destinations->rnum > 0);
			if (!graph->ws)
				graph->ws = ccv_array_new(sizeof(int), bind->destinations->rnum, 0);
			ccv_array_t* const ws = graph->ws;
			// Go through all the exec symbols use this tensor, to see whether they have inputs that has other sources.
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
				free_symbol = (free_symbol || !flag);
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
				for (i = 0; i < ws->rnum; i++)
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
	_ccv_nnc_tensor_variable_free(graph, tensor_variable);
}

void ccv_nnc_dynamic_graph_dot(const ccv_nnc_dynamic_graph_t* const graph, const int flags, FILE* out)
{
	ccv_nnc_symbolic_graph_dot(graph->tape, flags, out);
}
