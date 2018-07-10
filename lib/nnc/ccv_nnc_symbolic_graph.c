#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-3 API
 */

const ccv_nnc_tensor_param_t ccv_nnc_tensor_auto = {};

int ccv_nnc_is_tensor_auto(const ccv_nnc_tensor_param_t params)
{
	return (memcmp(&params, &ccv_nnc_tensor_auto, sizeof(ccv_nnc_tensor_param_t)) == 0);
}

ccv_nnc_symbolic_graph_t* ccv_nnc_symbolic_graph_new(void)
{
	ccv_nnc_symbolic_graph_t* graph = cccalloc(1, sizeof(ccv_nnc_symbolic_graph_t));
	graph->tensor_symbol_info = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_info_t), 5, 0);
	graph->exec_symbol_info = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_info_t), 5, 0);
	graph->reuse.exec = -1;
	graph->reuse.tensor = -1;
	return graph;
}

ccv_nnc_symbolic_graph_t* ccv_nnc_symbolic_graph_dup(const ccv_nnc_symbolic_graph_t* const graph, ccv_nnc_symbolic_graph_subst_f subst)
{
	ccv_nnc_symbolic_graph_t* new_graph = ccmalloc(sizeof(ccv_nnc_symbolic_graph_t));
	memcpy(new_graph, graph, sizeof(ccv_nnc_symbolic_graph_t));
	new_graph->tensor_symbol_info = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_info_t), graph->tensor_symbol_info->rnum, 0);
	new_graph->tensor_symbol_info->rnum = graph->tensor_symbol_info->rnum;
	memcpy(ccv_array_get(new_graph->tensor_symbol_info, 0), ccv_array_get(graph->tensor_symbol_info, 0), sizeof(ccv_nnc_tensor_symbol_info_t) * graph->tensor_symbol_info->rnum);
	int i;
	for (i = 0; i < new_graph->tensor_symbol_info->rnum; i++)
	{
		ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(new_graph->tensor_symbol_info, i);
		if (symbol_info->name)
		{
			char* const name = symbol_info->name;
			const size_t n = strnlen(name, 63) + 1;
			symbol_info->name = (char*)ccmalloc(n);
			// Don't use strndup because this way I can have custom allocator (for ccmalloc).
			strncpy(symbol_info->name, name, n);
		}
		if (symbol_info->s_ref)
		{
			ccv_array_t* const s_ref = symbol_info->s_ref;
			symbol_info->s_ref = ccv_array_new(sizeof(int), s_ref->rnum, 0);
			symbol_info->s_ref->rnum = s_ref->rnum;
			memcpy(ccv_array_get(symbol_info->s_ref, 0), ccv_array_get(s_ref, 0), sizeof(int) * s_ref->rnum);
		}
	}
	new_graph->exec_symbol_info = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_info_t), graph->exec_symbol_info->rnum, 0);
	new_graph->exec_symbol_info->rnum = graph->exec_symbol_info->rnum;
	memcpy(ccv_array_get(new_graph->exec_symbol_info, 0), ccv_array_get(graph->exec_symbol_info, 0), sizeof(ccv_nnc_graph_exec_symbol_info_t) * graph->exec_symbol_info->rnum);
	for (i = 0; i < new_graph->exec_symbol_info->rnum; i++)
	{
		ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(new_graph->exec_symbol_info, i);
		if (symbol_info->name)
		{
			char* const name = symbol_info->name;
			const size_t n = strnlen(name, 63) + 1;
			symbol_info->name = (char*)ccmalloc(n);
			// Don't use strndup because this way I can have custom allocator (for ccmalloc).
			strncpy(symbol_info->name, name, n);
		}
		if (symbol_info->outgoings)
		{
			ccv_array_t* const outgoings = symbol_info->outgoings;
			symbol_info->outgoings = ccv_array_new(sizeof(int), outgoings->rnum, 0);
			symbol_info->outgoings->rnum = outgoings->rnum;
			memcpy(ccv_array_get(symbol_info->outgoings, 0), ccv_array_get(outgoings, 0), sizeof(int) * outgoings->rnum);
		}
		if (symbol_info->inputs)
		{
			int* const inputs = symbol_info->inputs;
			symbol_info->inputs = (int*)ccmalloc(sizeof(int) * (symbol_info->input_size + symbol_info->output_size));
			symbol_info->outputs = symbol_info->inputs + symbol_info->input_size;
			memcpy(symbol_info->inputs, inputs, sizeof(int) * (symbol_info->input_size + symbol_info->output_size));
		}
		if (symbol_info->_heap_graph_ref)
		{
			int* const heap_graph_ref = symbol_info->_heap_graph_ref;
			symbol_info->_heap_graph_ref = (int*)ccmalloc(sizeof(int) * symbol_info->graph_ref_size);
			memcpy(symbol_info->_heap_graph_ref, heap_graph_ref, sizeof(int) * symbol_info->graph_ref_size);
		}
		if ((symbol_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) && symbol_info->input_size > 0)
		{
			int* const inputs = symbol_info->p_while.inputs;
			symbol_info->p_while.inputs = (int*)ccmalloc(sizeof(int) * symbol_info->p_while.input_size);
			memcpy(symbol_info->p_while.inputs, inputs, sizeof(int) * symbol_info->p_while.input_size);
		}
	}
	if (graph->sources)
	{
		new_graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), graph->sources->rnum, 0);
		new_graph->sources->rnum = graph->sources->rnum;
		memcpy(ccv_array_get(new_graph->sources, 0), ccv_array_get(graph->sources, 0), sizeof(ccv_nnc_graph_exec_symbol_t) * graph->sources->rnum);
		for (i = 0; i < new_graph->sources->rnum; i++)
			((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(new_graph->sources, i))->graph = new_graph;
	}
	if (graph->destinations)
	{
		new_graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), graph->destinations->rnum, 0);
		new_graph->destinations->rnum = graph->destinations->rnum;
		memcpy(ccv_array_get(new_graph->destinations, 0), ccv_array_get(graph->destinations, 0), sizeof(ccv_nnc_graph_exec_symbol_t) * graph->destinations->rnum);
		for (i = 0; i < new_graph->destinations->rnum; i++)
			((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(new_graph->destinations, i))->graph = new_graph;
	}
	if (graph->breakpoints)
	{
		new_graph->breakpoints = (ccv_nnc_graph_exec_symbol_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * graph->breakpoint_size);
		memcpy(new_graph->breakpoints, graph->breakpoints, sizeof(ccv_nnc_graph_exec_symbol_t) * graph->breakpoint_size);
		for (i = 0; i < graph->breakpoint_size; i++)
			new_graph->breakpoints[i].graph = new_graph;
	}
	if (graph->backward_tensor_symbols)
	{
		new_graph->backward_tensor_symbols = (int*)ccmalloc(sizeof(int) * (new_graph->backward_tensor_symbol_size + new_graph->backward_exec_symbol_size));
		if (new_graph->backward_tensor_symbol_size > 0)
			memcpy(new_graph->backward_tensor_symbols, graph->backward_tensor_symbols, sizeof(int) * new_graph->backward_tensor_symbol_size);
		new_graph->backward_exec_symbols = new_graph->backward_tensor_symbols + new_graph->backward_tensor_symbol_size;
		if (new_graph->backward_exec_symbol_size > 0)
			memcpy(new_graph->backward_exec_symbols, graph->backward_exec_symbols, sizeof(int) * new_graph->backward_exec_symbol_size);
	}
	if (subst)
	{
		for (i = 0; i < new_graph->exec_symbol_info->rnum; i++)
		{
			ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(new_graph->exec_symbol_info, i);
			if (!CCV_NNC_GRAPH_EXEC_IS_DEAD(symbol_info->flags))
			{
				symbol_info->cmd = subst((ccv_nnc_graph_exec_symbol_t){
					.d = i,
					.graph = graph,
				}, symbol_info->cmd);
				if (symbol_info->cmd.cmd != CCV_NNC_GRAPH_FORWARD && symbol_info->cmd.cmd != CCV_NNC_GRAPH_BACKWARD)
				{
					symbol_info->graph_ref_size = 0;
					if (symbol_info->_heap_graph_ref)
					{
						ccfree(symbol_info->_heap_graph_ref);
						symbol_info->_heap_graph_ref = 0;
					}
				}
			}
		}
	}
	// TODO: See how and if I need to dup sub-graphs. I also need to figure out what's the relationship between this graph
	// and its parent graph (or how can we use the symbol from the graph properly).
	new_graph->sub_graphs = 0;
	return new_graph;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_nnc_tensor_symbol_t symbol = {
		.d = graph->tensor_symbol_info->rnum,
		.graph = graph
	};
	ccv_nnc_tensor_symbol_info_t symbol_info = {
		.info = info,
	};
	if (name)
	{
		size_t n = strnlen(name, 63) + 1;
		symbol_info.name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		strncpy(symbol_info.name, name, n);
	}
	if (graph->reuse.tensor >= 0)
	{
		const int reuse_tensor_d = graph->reuse.tensor;
		assert(reuse_tensor_d < graph->tensor_symbol_info->rnum);
		*(ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, reuse_tensor_d) = symbol_info;
		int i;
		graph->reuse.tensor = -1;
		for (i = reuse_tensor_d + 1; i < graph->tensor_symbol_info->rnum && graph->reuse.tensor < 0; i++)
			if (CCV_NNC_TENSOR_SYMBOL_IS_DEAD(((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, i))->flags))
				graph->reuse.tensor = i;
		symbol.d = reuse_tensor_d;
	} else
		ccv_array_push(graph->tensor_symbol_info, &symbol_info);
	if (graph->hooks.tensor_symbol_new.func)
		graph->hooks.tensor_symbol_new.func(graph->hooks.tensor_symbol_new.context, symbol, info, name);
	return symbol;
}

void* ccv_nnc_tensor_symbol_new_hook(ccv_nnc_symbolic_graph_t* const graph, ccv_nnc_tensor_symbol_new_hook_f hook, void* context)
{
	void* const prev = graph->hooks.tensor_symbol_new.context;
	graph->hooks.tensor_symbol_new.func = hook;
	graph->hooks.tensor_symbol_new.context = context;
	return prev;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_alias_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* const name)
{
	assert(tensor_symbol.graph == graph);
	int d = tensor_symbol.d;
	assert(d >= 0 && d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* info_d = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
	// Find the root tensor that is not an alias.
	while (info_d->alias_ref)
	{
		d = info_d->alias_ref - 1;
		assert(d >= 0 && d < graph->tensor_symbol_info->rnum);
		info_d = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
	}
	ccv_nnc_tensor_symbol_t alias = {
		.d = graph->tensor_symbol_info->rnum,
		.graph = graph
	};
	int i;
	// Alias comes in two shapes: 1). the total tensor count is strictly smaller or equal to, and without ofs; 2). with ofs, and each dimension is strictly smaller or equal to.
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC; i++)
		{ assert(info.dim[i] + ofs[i] <= inc[i]); }
	// If it is not auto, check dimensions.
	if (!ccv_nnc_is_tensor_auto(info_d->info))
		{ assert(ccv_nnc_dimension_count(inc) <= ccv_nnc_tensor_count(info_d->info)); }
	ccv_nnc_tensor_symbol_info_t alias_info = {
		.alias_ref = d + 1,
		.info = info,
	};
	if (name)
	{
		size_t n = strnlen(name, 63) + 1;
		alias_info.name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		strncpy(alias_info.name, name, n);
	}
	memcpy(alias_info.ofs, ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(alias_info.inc, inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	if (graph->reuse.tensor >= 0)
	{
		const int reuse_tensor_d = graph->reuse.tensor;
		*(ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, reuse_tensor_d) = alias_info;
		int i;
		graph->reuse.tensor = -1;
		for (i = reuse_tensor_d + 1; i < graph->tensor_symbol_info->rnum && graph->reuse.tensor < 0; i++)
			if (CCV_NNC_TENSOR_SYMBOL_IS_DEAD(((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, i))->flags))
				graph->reuse.tensor = i;
		alias.d = reuse_tensor_d;
	} else
		ccv_array_push(graph->tensor_symbol_info, &alias_info);
	if (graph->hooks.tensor_symbol_alias_new.func)
		graph->hooks.tensor_symbol_alias_new.func(graph->hooks.tensor_symbol_alias_new.context, alias, tensor_symbol, ofs, inc, info, name);
	return alias;
}

void* ccv_nnc_tensor_symbol_alias_new_hook(ccv_nnc_symbolic_graph_t* const graph, ccv_nnc_tensor_symbol_alias_new_hook_f hook, void* context)
{
	void* const prev = graph->hooks.tensor_symbol_alias_new.context;
	graph->hooks.tensor_symbol_alias_new.func = hook;
	graph->hooks.tensor_symbol_alias_new.context = context;
	return prev;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_alias_to(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor_symbol)
{
	assert(tensor_symbol.graph == graph);
	int d = tensor_symbol.d;
	assert(d >= 0 && d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* info_d = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
	// Find the root tensor that is not an alias.
	while (info_d->alias_ref)
	{
		d = info_d->alias_ref - 1;
		assert(d >= 0 && d < graph->tensor_symbol_info->rnum);
		info_d = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
	}
	if (d != tensor_symbol.d)
		return (ccv_nnc_tensor_symbol_t){
			.d = d,
			.graph = graph
		};
	return (ccv_nnc_tensor_symbol_t){
		.d = CCV_NNC_NO_TENSOR_SYMBOL,
		.graph = 0
	};
}

// Resolve this tensor symbol to the current graph. If cannot find, return no symbol.
ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_resolve(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor_symbol)
{
	if (graph == tensor_symbol.graph)
		return tensor_symbol;
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol.graph->tensor_symbol_info, tensor_symbol.d);
	assert(!symbol_info->alias_ref);
	// Find if the symbol is in the sub-graph.
	const ccv_nnc_symbolic_graph_t* curr_graph = tensor_symbol.graph;
	assert(tensor_symbol.d >= 0 && tensor_symbol.d < curr_graph->tensor_symbol_info->rnum);
	while (curr_graph && curr_graph != graph)
		curr_graph = curr_graph->p;
	if (curr_graph)
	{
		// The graph is a parent of the symbol passed in.
		curr_graph = tensor_symbol.graph;
		ccv_nnc_tensor_symbol_info_t* curr_symbol_info = symbol_info;
		ccv_nnc_tensor_symbol_t curr_symbol = tensor_symbol;
		while (curr_graph != graph)
		{
			ccv_nnc_symbolic_graph_t* const p = curr_graph->p;
			// Cannot find the relevant one in the parent graph, return no symbol.
			if (!curr_symbol_info->p_ref)
				return (ccv_nnc_tensor_symbol_t){
					.d = CCV_NNC_NO_TENSOR_SYMBOL,
					.graph = graph,
				};
			curr_symbol.d = curr_symbol_info->p_ref - 1;
			curr_symbol.graph = p;
			assert(curr_symbol.d >= 0 && curr_symbol.d < p->tensor_symbol_info->rnum);
			curr_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(p->tensor_symbol_info, curr_symbol.d);
			// Move on.
			curr_graph = p;
		}
		return curr_symbol;
	}
	// Otherwise, if the symbol is in the parent graph, this is a bit more expensive because I need to keep a trace stack.
	curr_graph = graph;
	int d;
	for (d = 0; curr_graph && curr_graph != tensor_symbol.graph; d++)
		curr_graph = curr_graph->p;
	curr_graph = graph;
	assert(d > 0);
	int trace[d];
	for (d = 0; curr_graph && curr_graph != tensor_symbol.graph; d++)
	{
		const int p_idx = curr_graph->p_idx - 1;
		trace[d] = p_idx;
		curr_graph = curr_graph->p;
	}
	// If it is not in both the parent graph and the sub-graph, the input is invalid.
	assert(curr_graph);
	curr_graph = tensor_symbol.graph;
	ccv_nnc_tensor_symbol_info_t* curr_symbol_info = symbol_info;
	ccv_nnc_tensor_symbol_t curr_symbol = tensor_symbol;
	// The graph is a sub graph of the symbol passed in.
	int i;
	for (i = d - 1; i >= 0; i--)
	{
		const int p_idx = trace[i];
		assert(p_idx >= 0);
		// Cannot find the relevant one in the sub-graph, return no symbol.
		if (!curr_graph->sub_graphs || !curr_symbol_info->s_ref ||
			curr_symbol_info->s_ref->rnum != curr_graph->sub_graphs->rnum)
				return (ccv_nnc_tensor_symbol_t){
					.d = CCV_NNC_NO_TENSOR_SYMBOL,
					.graph = graph,
				};
		assert(p_idx >= 0 && p_idx < curr_symbol_info->s_ref->rnum);
		const int s_idx = *(int*)ccv_array_get(curr_symbol_info->s_ref, p_idx);
		ccv_nnc_symbolic_graph_t* const s = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(curr_graph->sub_graphs, p_idx);
		curr_symbol.d = s_idx - 1;
		curr_symbol.graph = s;
		assert(curr_symbol.d >= 0 && curr_symbol.d < s->tensor_symbol_info->rnum);
		curr_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(s->tensor_symbol_info, curr_symbol.d);
		// Move on.
		curr_graph = s;
	}
	return curr_symbol;
}

// This method generate tensor symbols and their links along the way when traverse the graph.
enum {
	MAP_TENSOR_USE_AS_INPUT,
	MAP_TENSOR_USE_AS_OUTPUT,
};

static void _ccv_nnc_graph_exec_add_input_if_needed(ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const int d)
{
	int i;
	for (i = 0; i < exec_symbol_info->input_size; i++)
		if (exec_symbol_info->inputs[i] == d)
			return; // No need to continue, this symbol already exists as input.
	// Expand the array.
	if (!exec_symbol_info->input_size && !exec_symbol_info->output_size)
	{
		exec_symbol_info->inputs = (int*)ccmalloc(sizeof(int));
		exec_symbol_info->inputs[0] = d;
		exec_symbol_info->input_size = 1;
		exec_symbol_info->outputs = exec_symbol_info->inputs + 1;
		return;
	}
	exec_symbol_info->inputs = (int*)ccrealloc(exec_symbol_info->inputs, sizeof(int) * (exec_symbol_info->input_size + 1 + exec_symbol_info->output_size));
	exec_symbol_info->outputs = exec_symbol_info->inputs + exec_symbol_info->input_size;
	if (exec_symbol_info->output_size)
		memmove(exec_symbol_info->outputs + 1, exec_symbol_info->outputs, sizeof(int) * exec_symbol_info->output_size); 
	exec_symbol_info->inputs[exec_symbol_info->input_size] = d;
	++exec_symbol_info->input_size;
	exec_symbol_info->outputs = exec_symbol_info->inputs + exec_symbol_info->input_size;
}

static void _ccv_nnc_graph_exec_add_output_if_needed(ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const int d)
{
	int i;
	for (i = 0; i < exec_symbol_info->output_size; i++)
		if (exec_symbol_info->outputs[i] == d)
			return; // No need to continue, this symbol already exists as output.
	// Expand the array.
	if (!exec_symbol_info->input_size && !exec_symbol_info->output_size)
	{
		exec_symbol_info->inputs = (int*)ccmalloc(sizeof(int));
		exec_symbol_info->outputs = exec_symbol_info->inputs;
		exec_symbol_info->outputs[0] = d;
		exec_symbol_info->output_size = 1;
		return;
	}
	exec_symbol_info->inputs = (int*)ccrealloc(exec_symbol_info->inputs, sizeof(int) * (exec_symbol_info->input_size + exec_symbol_info->output_size + 1));
	exec_symbol_info->outputs = exec_symbol_info->inputs + exec_symbol_info->input_size;
	exec_symbol_info->outputs[exec_symbol_info->output_size] = d;
	++exec_symbol_info->output_size;
}

void ccv_nnc_tensor_symbol_set_peer(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor_symbol, const ccv_nnc_tensor_symbol_t peer_tensor_symbol)
{
	assert(tensor_symbol.graph == graph);
	assert(tensor_symbol.d >= 0);
	assert(tensor_symbol.d < graph->tensor_symbol_info->rnum);
	assert(peer_tensor_symbol.graph == graph->peer);
	assert(peer_tensor_symbol.d >= 0);
	assert(peer_tensor_symbol.d < graph->peer->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* const tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_symbol.d);
	tensor_info->peer_ref = peer_tensor_symbol.d + 1;
}

static int _ccv_nnc_symbolic_graph_map_tensor_symbol_no_alias(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol, const int map_use)
{
	assert(graph && symbol.graph);
	assert(symbol.graph != graph);
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(symbol.graph->tensor_symbol_info, symbol.d);
	assert(!symbol_info->alias_ref);
	// Find if the symbol is in the sub-graph.
	const ccv_nnc_symbolic_graph_t* curr_graph = symbol.graph;
	assert(symbol.d >= 0 && symbol.d < curr_graph->tensor_symbol_info->rnum);
	while (curr_graph && curr_graph != graph)
		curr_graph = curr_graph->p;
	if (curr_graph)
	{
		// The graph is a parent of the symbol passed in. For this case, if we are connecting this symbol to an exec as input,
		// that means it must be an output in these sub-graphs. Otherwise, if we are connecting this symbol to an exec as output,
		// it must be an input in these sub-graphs.
		curr_graph = symbol.graph;
		ccv_nnc_tensor_symbol_info_t* curr_symbol_info = symbol_info;
		ccv_nnc_tensor_symbol_t curr_symbol = symbol;
		while (curr_graph != graph)
		{
			ccv_nnc_symbolic_graph_t* const p = curr_graph->p;
			// I need to find the symbol whether it exists or not before creating new one.
			ccv_nnc_tensor_symbol_t new_symbol;
			ccv_nnc_tensor_symbol_info_t* new_symbol_info;
			if (!curr_symbol_info->p_ref)
			{
				new_symbol = ccv_nnc_tensor_symbol_new(p, curr_symbol_info->info, curr_symbol_info->name);
				new_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(p->tensor_symbol_info, new_symbol.d);
				curr_symbol_info->p_ref = new_symbol.d + 1;
				new_symbol_info->s_ref = ccv_array_new(sizeof(int), p->sub_graphs->rnum, 0);
				new_symbol_info->s_ref->rnum = p->sub_graphs->rnum;
				ccv_array_zero(new_symbol_info->s_ref);
				*(int*)ccv_array_get(new_symbol_info->s_ref, curr_graph->p_idx - 1) = curr_symbol.d + 1;
			} else {
				new_symbol.d = curr_symbol_info->p_ref - 1;
				new_symbol.graph = p;
				assert(new_symbol.d >= 0 && new_symbol.d < p->tensor_symbol_info->rnum);
				new_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(p->tensor_symbol_info, new_symbol.d);
			}
			if (curr_graph->exec_idx)
			{
				// This is a sub-graph.
				assert(p);
				assert(curr_graph->exec_idx > 0 && curr_graph->exec_idx <= p->exec_symbol_info->rnum);
				ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(p->exec_symbol_info, curr_graph->exec_idx - 1);
				switch (map_use)
				{
					case MAP_TENSOR_USE_AS_INPUT:
						_ccv_nnc_graph_exec_add_output_if_needed(exec_symbol_info, new_symbol.d);
						break;
					case MAP_TENSOR_USE_AS_OUTPUT:
						_ccv_nnc_graph_exec_add_input_if_needed(exec_symbol_info, new_symbol.d);
						break;
				}
			}
			// Move on.
			curr_symbol = new_symbol;
			curr_symbol_info = new_symbol_info;
			curr_graph = p;
		}
		return curr_symbol.d;
	}
	// Otherwise, if the symbol is in the parent graph, this is a bit more expensive because I need to keep a trace stack.
	curr_graph = graph;
	int d;
	for (d = 0; curr_graph && curr_graph != symbol.graph; d++)
		curr_graph = curr_graph->p;
	curr_graph = graph;
	assert(d > 0);
	int trace[d];
	for (d = 0; curr_graph && curr_graph != symbol.graph; d++)
	{
		const int p_idx = curr_graph->p_idx - 1;
		trace[d] = p_idx;
		curr_graph = curr_graph->p;
	}
	// If it is not in both the parent graph and the sub-graph, the input is invalid.
	assert(curr_graph);
	curr_graph = symbol.graph;
	ccv_nnc_tensor_symbol_info_t* curr_symbol_info = symbol_info;
	ccv_nnc_tensor_symbol_t curr_symbol = symbol;
	// The graph is a sub graph of the symbol passed in. For this case, if we are connecting this symbol to an exec as input,
	// that means it must be an input in these parent graphs. Otherwise, if we are connecting this symbol to an exec as output,
	// it must be an output in these parent graphs.
	int i;
	for (i = d - 1; i >= 0; i--)
	{
		const int p_idx = trace[i];
		assert(p_idx >= 0);
		assert(curr_graph->sub_graphs);
		if (!curr_symbol_info->s_ref)
		{
			curr_symbol_info->s_ref = ccv_array_new(sizeof(int), curr_graph->sub_graphs->rnum, 0);
			curr_symbol_info->s_ref->rnum = curr_graph->sub_graphs->rnum;
			ccv_array_zero(curr_symbol_info->s_ref);
		} else if (curr_symbol_info->s_ref->rnum != curr_graph->sub_graphs->rnum)
			ccv_array_resize(curr_symbol_info->s_ref, curr_graph->sub_graphs->rnum);
		assert(p_idx >= 0 && p_idx < curr_symbol_info->s_ref->rnum);
		const int s_idx = *(int*)ccv_array_get(curr_symbol_info->s_ref, p_idx);
		ccv_nnc_symbolic_graph_t* const s = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(curr_graph->sub_graphs, p_idx);
		ccv_nnc_tensor_symbol_t new_symbol;
		ccv_nnc_tensor_symbol_info_t* new_symbol_info;
		// I need to find the symbol whether it exists or not before creating new one.
		if (!s_idx)
		{
			new_symbol = ccv_nnc_tensor_symbol_new(s, symbol_info->info, symbol_info->name);
			new_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(s->tensor_symbol_info, new_symbol.d);
			new_symbol_info->p_ref = curr_symbol.d + 1;
			*(int*)ccv_array_get(curr_symbol_info->s_ref, p_idx) = new_symbol.d + 1;
		} else {
			new_symbol.d = s_idx - 1;
			new_symbol.graph = s;
			assert(new_symbol.d >= 0 && new_symbol.d < s->tensor_symbol_info->rnum);
			new_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(s->tensor_symbol_info, new_symbol.d);
		}
		if (s->exec_idx)
		{
			assert(s->p); // This is a sub-graph.
			assert(s->exec_idx > 0 && s->exec_idx <= curr_graph->exec_symbol_info->rnum);
			ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(curr_graph->exec_symbol_info, s->exec_idx - 1);
			switch (map_use)
			{
				case MAP_TENSOR_USE_AS_INPUT:
					_ccv_nnc_graph_exec_add_input_if_needed(exec_symbol_info, curr_symbol.d);
					break;
				case MAP_TENSOR_USE_AS_OUTPUT:
					_ccv_nnc_graph_exec_add_output_if_needed(exec_symbol_info, curr_symbol.d);
					break;
			}
		}
		// Move on.
		curr_symbol = new_symbol;
		curr_symbol_info = new_symbol_info;
		curr_graph = s;
	}
	return curr_symbol.d;
}

static int _ccv_nnc_symbolic_graph_map_tensor_symbol(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol, const int map_use)
{
	assert(graph && symbol.graph);
	assert(symbol.graph != graph);
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(symbol.graph->tensor_symbol_info, symbol.d);
	if (!symbol_info->alias_ref)
		return _ccv_nnc_symbolic_graph_map_tensor_symbol_no_alias(graph, symbol, map_use);
	const int d = symbol_info->alias_ref - 1;
	assert(d >= 0 && d < symbol.graph->tensor_symbol_info->rnum);
	const int map_d = _ccv_nnc_symbolic_graph_map_tensor_symbol_no_alias(graph, (ccv_nnc_tensor_symbol_t){
		.graph = symbol.graph,
		.d = d
	}, map_use);
	const ccv_nnc_tensor_symbol_t alias = ccv_nnc_tensor_symbol_alias_new(graph, (ccv_nnc_tensor_symbol_t){
		.graph = graph,
		.d = map_d
	}, symbol_info->ofs, symbol_info->inc, symbol_info->info, symbol_info->name);
	return alias.d;
}

int ccv_nnc_tensor_symbol_map_raw(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol)
{
	if (symbol.d >= 0)
		return symbol.graph != graph ? _ccv_nnc_symbolic_graph_map_tensor_symbol(graph, symbol, MAP_TENSOR_USE_AS_INPUT) : symbol.d;
	if (symbol.graph == graph || symbol.d == CCV_NNC_NO_TENSOR_SYMBOL)
		return symbol.d;
	ccv_nnc_symbolic_graph_t* curr_graph = graph;
	int d;
	for (d = 0; curr_graph && curr_graph != symbol.graph; d++)
		curr_graph = curr_graph->p;
	assert(curr_graph == symbol.graph);
	return CCV_NNC_ENCODE_WHILE_COUNT_SYMBOL(d);
}

void ccv_nnc_tensor_symbol_hookup(ccv_nnc_symbolic_graph_t* const src_graph, ccv_nnc_symbolic_graph_t* const dest_graph, const ccv_nnc_tensor_symbol_t src_tensor_symbol, const ccv_nnc_tensor_symbol_t dest_tensor_symbol)
{
	assert(src_graph->p == dest_graph || dest_graph->p == src_graph);
	assert(src_tensor_symbol.d >= 0);
	assert(dest_tensor_symbol.d >= 0);
	ccv_nnc_tensor_symbol_t tensor_symbol = src_tensor_symbol;
	if (tensor_symbol.graph != src_graph)
		tensor_symbol = (ccv_nnc_tensor_symbol_t){
			.graph = src_graph,
			.d = _ccv_nnc_symbolic_graph_map_tensor_symbol(src_graph, tensor_symbol, MAP_TENSOR_USE_AS_INPUT),
		};
	ccv_nnc_tensor_symbol_t sub_tensor_symbol = dest_tensor_symbol;
	if (sub_tensor_symbol.graph != dest_graph)
		sub_tensor_symbol = (ccv_nnc_tensor_symbol_t){
			.graph = dest_graph,
			.d = _ccv_nnc_symbolic_graph_map_tensor_symbol(dest_graph, sub_tensor_symbol, MAP_TENSOR_USE_AS_OUTPUT),
		};
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_symbolic_graph_t* sub_graph;
	if (src_graph->p == dest_graph)
	{
		graph = dest_graph;
		sub_graph = src_graph;
		// Swap tensor_symbol and sub_tensor_symbol
		ccv_nnc_tensor_symbol_t x;
		CCV_SWAP(tensor_symbol, sub_tensor_symbol, x);
	} else {
		assert(dest_graph->p == src_graph);
		graph = src_graph;
		sub_graph = dest_graph;
	}
	assert(tensor_symbol.d < graph->tensor_symbol_info->rnum);
	assert(sub_tensor_symbol.d < sub_graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* const sub_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(sub_graph->tensor_symbol_info, sub_tensor_symbol.d);
	sub_tensor_info->p_ref = tensor_symbol.d + 1;
	ccv_nnc_tensor_symbol_info_t* const tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_symbol.d);
	if (!tensor_info->s_ref)
	{
		tensor_info->s_ref = ccv_array_new(sizeof(int), graph->sub_graphs->rnum, 0);
		tensor_info->s_ref->rnum = graph->sub_graphs->rnum;
		ccv_array_zero(tensor_info->s_ref);
	} else if (tensor_info->s_ref->rnum != graph->sub_graphs->rnum)
		ccv_array_resize(tensor_info->s_ref, graph->sub_graphs->rnum);
	const int p_idx = sub_graph->p_idx - 1;
	assert(p_idx >= 0 && p_idx < tensor_info->s_ref->rnum);
	const int s_idx = *(int*)ccv_array_get(tensor_info->s_ref, p_idx);
	assert(s_idx == 0); // Otherwise it is assigned before
	*(int*)ccv_array_get(tensor_info->s_ref, p_idx) = sub_tensor_symbol.d + 1;
}

void ccv_nnc_tensor_symbol_set_bypasses(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_map_t* const symbol_map, const int symbol_map_size)
{
	int i;
	for (i = 0; i < symbol_map_size; i++)
	{
		const ccv_nnc_tensor_symbol_t source = ccv_nnc_tensor_symbol_resolve(graph, symbol_map[i].source);
		const ccv_nnc_tensor_symbol_t destination = ccv_nnc_tensor_symbol_resolve(graph, symbol_map[i].destination);
		assert(source.graph == graph);
		assert(destination.graph == graph);
		assert(source.d < graph->tensor_symbol_info->rnum);
		assert(destination.d < graph->tensor_symbol_info->rnum);
		ccv_nnc_tensor_symbol_info_t* source_tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, source.d);
		ccv_nnc_tensor_symbol_info_t* destination_tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, destination.d);
		// Don't support parameterize with alias. The reason is that to support parameterized loop (for SSA), I choose
		// to simply reuse the piece of memory (allocating the same memory region to both, therefore to enable parameter
		// passing). For alias, it is not possible because alias can pointing to the tensors with different sizes, thus,
		// these pointed tensors cannot share the same memory region. The best way for alias to be parameterized is to
		// create a new tensor of the same size, transfer value over, and parameterized on that tensor instead.
		assert(!destination_tensor_symbol_info->alias_ref);
		assert(!source_tensor_symbol_info->alias_ref);
		destination_tensor_symbol_info->bypass_ref = source.d + 1;
		source_tensor_symbol_info->r_bypass_ref = destination.d + 1;
	}
}

int ccv_nnc_tensor_symbol_set(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor, const ccv_nnc_tensor_param_t info)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	symbol_info->info = info;
	// It also need to propagate to assign_ref if needed.
	if (symbol_info->assign_ref)
	{
		ccv_nnc_tensor_symbol_info_t* const assign_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, symbol_info->assign_ref - 1);
		assign_info->info = info;
	}
	return 0;
}

ccv_nnc_tensor_param_t ccv_nnc_tensor_symbol_params(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	return symbol_info->info;
}

int ccv_nnc_tensor_symbol_set_flags(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor, const int flags)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	symbol_info->flags = flags;
	// It also need to propagate to assign_ref if needed.
	if (symbol_info->assign_ref)
	{
		ccv_nnc_tensor_symbol_info_t* const assign_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, symbol_info->assign_ref - 1);
		assign_info->flags = flags;
	}
	return 0;
}

int ccv_nnc_tensor_symbol_flags(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	return symbol_info->flags;
}

void ccv_nnc_tensor_symbol_free(ccv_nnc_symbolic_graph_t* const graph, ccv_nnc_tensor_symbol_t tensor)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	if (symbol_info->s_ref)
	{
		ccv_array_free(symbol_info->s_ref);
		symbol_info->s_ref = 0;
	}
	if (symbol_info->name)
	{
		ccfree(symbol_info->name);
		symbol_info->name = 0;
	}
	symbol_info->flags |= CCV_NNC_TENSOR_SYMBOL_DEAD;
	int i;
	for (i = graph->tensor_symbol_info->rnum - 1; i >= 0; i--)
		if (!CCV_NNC_TENSOR_SYMBOL_IS_DEAD(((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, i))->flags))
		{
			graph->tensor_symbol_info->rnum = i + 1;
			break;
		}
	if (tensor.d < graph->tensor_symbol_info->rnum &&
		(tensor.d < graph->reuse.tensor || graph->reuse.tensor < 0))
		graph->reuse.tensor = tensor.d;
	else if (graph->reuse.tensor >= graph->tensor_symbol_info->rnum)
		graph->reuse.tensor = -1;
}

static void _ccv_nnc_graph_exec_symbol_set_io(ccv_nnc_symbolic_graph_t* const graph, ccv_nnc_graph_exec_symbol_info_t* const exec_info, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	exec_info->input_size = input_size;
	exec_info->output_size = output_size;
	if (input_size > 0 || output_size > 0)
	{
		if (!exec_info->inputs)
			exec_info->inputs = ccmalloc(sizeof(int) * (input_size + output_size));
		else
			exec_info->inputs = ccrealloc(exec_info->inputs, sizeof(int) * (input_size + output_size));
		exec_info->outputs = exec_info->inputs + input_size;
	}
	int i;
	int tensor_memory = 0, tensor_formats = 0, tensor_datatypes = 0;
	for (i = 0; i < input_size; i++)
	{
		const int d = ccv_nnc_tensor_symbol_map_raw(graph, inputs[i]);
		exec_info->inputs[i] = d;
		if (d >= 0)
		{
			const ccv_nnc_tensor_symbol_info_t* const tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
			tensor_memory |= CCV_TENSOR_GET_MEMORY(tensor_info->info.type), tensor_formats |= tensor_info->info.format, tensor_datatypes |= tensor_info->info.datatype;
		}
	}
	for (i = 0; i < output_size; i++)
	{
		const int d = (outputs[i].graph != graph && outputs[i].d >= 0) ? _ccv_nnc_symbolic_graph_map_tensor_symbol(graph, outputs[i], MAP_TENSOR_USE_AS_OUTPUT) : outputs[i].d;
		exec_info->outputs[i] = d;
		if (d >= 0)
		{
			const ccv_nnc_tensor_symbol_info_t* const tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
			tensor_memory |= CCV_TENSOR_GET_MEMORY(tensor_info->info.type), tensor_formats |= tensor_info->info.format, tensor_datatypes |= tensor_info->info.datatype;
		}
	}
	exec_info->cmd.backend = ccv_nnc_cmd_find_backend(exec_info->cmd, tensor_memory, tensor_formats, tensor_datatypes);
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
{
	ccv_nnc_graph_exec_symbol_t symbol = {
		.d = graph->exec_symbol_info->rnum,
		.graph = graph
	};
	ccv_nnc_graph_exec_symbol_info_t symbol_info = {
		.cmd = cmd,
		.hint = ccv_nnc_no_hint,
	};
	if (name)
	{
		size_t n = strnlen(name, 63) + 1;
		symbol_info.name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		strncpy(symbol_info.name, name, n);
	}
	_ccv_nnc_graph_exec_symbol_set_io(graph, &symbol_info, inputs, input_size, outputs, output_size);
	if (graph->reuse.exec >= 0)
	{
		const int reuse_exec_d = graph->reuse.exec;
		assert(reuse_exec_d < graph->exec_symbol_info->rnum);
		*(ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, reuse_exec_d) = symbol_info;
		int i;
		graph->reuse.exec = -1;
		for (i = reuse_exec_d + 1; i < graph->exec_symbol_info->rnum && graph->reuse.exec < 0; i++)
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i))->flags))
				graph->reuse.exec = i;
		symbol.d = reuse_exec_d;
	} else
		ccv_array_push(graph->exec_symbol_info, &symbol_info);
	if (graph->hooks.graph_exec_symbol_new.func)
		graph->hooks.graph_exec_symbol_new.func(graph->hooks.graph_exec_symbol_new.context, symbol, cmd, inputs, input_size, outputs, output_size, name);
	return symbol;
}

void* ccv_nnc_graph_exec_symbol_new_hook(ccv_nnc_symbolic_graph_t* const graph, ccv_nnc_graph_exec_symbol_new_hook_f hook, void* context)
{
	void* const prev = graph->hooks.graph_exec_symbol_new.context;
	graph->hooks.graph_exec_symbol_new.func = hook;
	graph->hooks.graph_exec_symbol_new.context = context;
	return prev;
}

void ccv_nnc_graph_exec_symbol_set_io(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(exec.graph == graph);
	assert(exec.d >= 0);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* const exec_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	_ccv_nnc_graph_exec_symbol_set_io(graph, exec_info, inputs, input_size, outputs, output_size);
}

void ccv_nnc_graph_exec_symbol_set_peer(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec_symbol, const ccv_nnc_graph_exec_symbol_t peer_exec_symbol)
{
	assert(exec_symbol.graph == graph);
	assert(exec_symbol.d >= 0);
	assert(exec_symbol.d < graph->exec_symbol_info->rnum);
	assert(peer_exec_symbol.graph == graph || peer_exec_symbol.graph == graph->peer);
	assert(peer_exec_symbol.d >= 0);
	if (peer_exec_symbol.graph == graph)
		{ assert(peer_exec_symbol.d < graph->exec_symbol_info->rnum); }
	else
		{ assert(peer_exec_symbol.d < graph->peer->exec_symbol_info->rnum); }
	ccv_nnc_graph_exec_symbol_info_t* const exec_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec_symbol.d);
	exec_info->peer_ref = peer_exec_symbol.d + 1;
}

void ccv_nnc_graph_exec_symbol_set(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec, const ccv_nnc_cmd_t cmd)
{
	assert(graph == exec.graph);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	symbol_info->cmd = cmd;
}

ccv_nnc_cmd_t ccv_nnc_graph_exec_symbol_cmd(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec)
{
	assert(graph == exec.graph);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	return symbol_info->cmd;
}

void ccv_nnc_graph_exec_symbol_set_hint(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec, const ccv_nnc_hint_t hint)
{
	assert(graph == exec.graph);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	symbol_info->hint = hint;
}

int ccv_nnc_graph_exec_symbol_concat(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t source, const ccv_nnc_graph_exec_symbol_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_symbol_info->rnum);
	assert(destination.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* src_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, source.d);
	if (!src_symbol_info->outgoings)
		src_symbol_info->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	else {
		int i;
		// Check if this is already connected, if so, skip.
		for (i = 0; i < src_symbol_info->outgoings->rnum; i++)
			if (*(int*)ccv_array_get(src_symbol_info->outgoings, i) == destination.d)
				return -1;
	}
	ccv_array_push(src_symbol_info->outgoings, &destination.d);
	return 0;
}

void ccv_nnc_graph_exec_symbol_io(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol, const int** const inputs, int* const input_size, const int** const outputs, int* const output_size)
{
	assert(graph == symbol.graph);
	assert(symbol.d < graph->exec_symbol_info->rnum);
	const ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	if (inputs)
		*inputs = symbol_info->inputs;
	if (input_size)
		*input_size = symbol_info->input_size;
	if (outputs)
		*outputs = symbol_info->outputs;
	if (output_size)
		*output_size = symbol_info->output_size;
}

void ccv_nnc_graph_exec_symbol_to(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol, const int** const tos, int* const to_size)
{
	assert(graph == symbol.graph);
	assert(symbol.d < graph->exec_symbol_info->rnum);
	assert(tos);
	assert(to_size);
	const ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	if (!symbol_info->outgoings)
	{
		*tos = 0;
		*to_size = 0;
		return;
	}
	*to_size = symbol_info->outgoings->rnum;
	*tos = (int*)ccv_array_get(symbol_info->outgoings, 0);
}

int ccv_nnc_graph_exec_symbol_count(const ccv_nnc_symbolic_graph_t* const graph)
{
	return graph->exec_symbol_info->rnum;
}

static inline void _ccv_nnc_graph_exec_symbol_free(ccv_nnc_graph_exec_symbol_info_t* const symbol_info, const int zeroing)
{
	if (symbol_info->name)
		ccfree(symbol_info->name);
	if (symbol_info->_heap_graph_ref)
		ccfree(symbol_info->_heap_graph_ref);
	ccv_array_t* outgoings = symbol_info->outgoings;
	if (outgoings)
		ccv_array_free(outgoings);
	// We allocate inputs & outputs in continuous fashion, therefore, only need to free the input array.
	if (symbol_info->inputs)
		ccfree(symbol_info->inputs);
	if (symbol_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
		if (symbol_info->p_while.inputs)
			ccfree(symbol_info->p_while.inputs);
	if (zeroing)
	{
		symbol_info->name = 0;
		symbol_info->_heap_graph_ref = 0;
		symbol_info->outgoings = 0;
		symbol_info->inputs = 0;
		symbol_info->input_size = 0;
		symbol_info->outputs = 0;
		symbol_info->output_size = 0;
	}
}

void ccv_nnc_graph_exec_symbol_free(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol)
{
	assert(graph == symbol.graph);
	assert(symbol.d < graph->exec_symbol_info->rnum);
	// If any of the exec symbol have reference to it, has to remove that.
	int i, j, k;
	ccv_nnc_graph_exec_symbol_info_t* const free_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, symbol.d);
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		if (i != symbol.d)
		{
			ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
			if (symbol_info->outgoings)
				for (j = 0; j < symbol_info->outgoings->rnum; j++)
					if (*(int*)ccv_array_get(symbol_info->outgoings, j) == symbol.d)
					{
						if (j < symbol_info->outgoings->rnum - 1)
							*(int*)ccv_array_get(symbol_info->outgoings, j) = *(int*)ccv_array_get(symbol_info->outgoings, symbol_info->outgoings->rnum - 1);
						--symbol_info->outgoings->rnum;
						if (free_symbol_info->outgoings)
							for (k = 0; k < free_symbol_info->outgoings->rnum; k++)
								ccv_array_add_unique_int(symbol_info->outgoings, *(int*)ccv_array_get(free_symbol_info->outgoings, k));
						break;
					}
		}
	// Deallocate any memory for exec symbol.
	_ccv_nnc_graph_exec_symbol_free(free_symbol_info, 1);
	free_symbol_info->flags = CCV_NNC_GRAPH_EXEC_DEAD; // Mark this as dead.
	// If everything from symbol.d to the end of the graph is dead, we can reclaim this memory.
	for (i = graph->exec_symbol_info->rnum - 1; i >= 0; i--)
		if (!CCV_NNC_GRAPH_EXEC_IS_DEAD(((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i))->flags))
		{
			graph->exec_symbol_info->rnum = i + 1;
			break;
		}
	// Loop over sources and destinations to remove this.
	if (graph->sources)
		for (i = 0; i < graph->sources->rnum; i++)
			if (*(int*)ccv_array_get(graph->sources, i) == symbol.d)
			{
				if (i < graph->sources->rnum - 1)
					*(int*)ccv_array_get(graph->sources, i) = *(int*)ccv_array_get(graph->sources, graph->sources->rnum - 1);
				--graph->sources->rnum;
				break;
			}
	if (graph->destinations)
		for (i = 0; i < graph->destinations->rnum; i++)
			if (*(int*)ccv_array_get(graph->destinations, i) == symbol.d)
			{
				if (i < graph->destinations->rnum - 1)
					*(int*)ccv_array_get(graph->destinations, i) = *(int*)ccv_array_get(graph->destinations, graph->destinations->rnum - 1);
				--graph->destinations->rnum;
				break;
			}
	if (symbol.d < graph->exec_symbol_info->rnum &&
		(symbol.d < graph->reuse.exec || graph->reuse.exec < 0))
		graph->reuse.exec = symbol.d;
	else if (graph->reuse.exec >= graph->exec_symbol_info->rnum)
		graph->reuse.exec = -1;
}

int ccv_nnc_graph_exec_symbol_disjoin(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t source, const ccv_nnc_graph_exec_symbol_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_symbol_info->rnum);
	assert(destination.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* src_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, source.d);
	if (!src_symbol_info->outgoings)
		return -1;
	ccv_nnc_graph_exec_symbol_info_t* const dest_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, destination.d);
	int i, j;
	// Check if this is already disjoined, if so, skip.
	for (i = 0; i < src_symbol_info->outgoings->rnum; i++)
		if (*(int*)ccv_array_get(src_symbol_info->outgoings, i) == destination.d)
		{
			if (i < src_symbol_info->outgoings->rnum - 1)
				*(int*)ccv_array_get(src_symbol_info->outgoings, i) = *(int*)ccv_array_get(src_symbol_info->outgoings, src_symbol_info->outgoings->rnum - 1);
			--src_symbol_info->outgoings->rnum;
			if (dest_symbol_info->outgoings)
				for (j = 0; j < dest_symbol_info->outgoings->rnum; j++)
					ccv_array_add_unique_int(src_symbol_info->outgoings, *(int*)ccv_array_get(dest_symbol_info->outgoings, j));
			return 0;
		}
	return -1;
}

#define CCV_NNC_IS_AUTOGEN_ALL_EXECS(x) ((x) & CCV_NNC_AUTOGEN_ALL_EXECS)
#define CCV_NNC_IS_AUTOGEN_SOURCES_AND_DESTINATIONS(x) ((x) & CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS)

int ccv_nnc_over_tensor_symbol_aliases(const ccv_nnc_tensor_symbol_info_t* const tensor_a, const ccv_nnc_tensor_symbol_info_t* const tensor_b)
{
	int i;
	const int* inc = tensor_a->inc;
	// Only can compare if the inc is the same, otherwise, we can only assume it overlaps.
	if (memcmp(inc, tensor_b->inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0)
		return 1;
	const int* ofs = tensor_a->ofs;
	const int* dim = tensor_a->info.dim;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && dim[i] && tensor_b->info.dim[i]; i++)
		if (ccv_min(ofs[i] + dim[i], tensor_b->ofs[i] + tensor_b->info.dim[i]) <= ccv_max(ofs[i], tensor_b->ofs[i]))
			return 0; // Cannot overlap.
	return 1;
}

int ccv_nnc_graph_exec_symbol_autogen(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const execs, const int exec_size, const int flags)
{
	int i, j, x, y;
	for (i = 0; i < exec_size; i++)
		if (execs[i].graph == graph)
		{
			assert(execs[i].d >= 0);
			assert(execs[i].d < graph->exec_symbol_info->rnum);
		}
	if (!CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) && exec_size)
		{ assert(execs); }
	const int exec_total_size = CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) ? graph->exec_symbol_info->rnum : exec_size;
	for (i = 0; i < exec_total_size; i++)
	{
		if (!CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) && execs[i].graph != graph)
			continue;
		int idx = CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) ? i : execs[i].d;
		ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, idx);
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(symbol_info->flags))
			continue;
		// Autogen for sub-graphs.
		if (CCV_NNC_GRAPH_REF(symbol_info)[0])
			ccv_nnc_graph_exec_symbol_autogen(*(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(symbol_info)[0] - 1), execs, exec_size, flags);
	}
	for (i = 0; i < exec_total_size; i++)
	{
		if (!CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) && execs[i].graph != graph)
			continue;
		int a_idx = CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) ? i : execs[i].d;
		ccv_nnc_graph_exec_symbol_info_t* a_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, a_idx);
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(a_symbol_info->flags))
			continue;
		for (j = i + 1; j < exec_total_size; j++)
		{
			if (!CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) && execs[j].graph != graph)
				continue;
			int b_idx = CCV_NNC_IS_AUTOGEN_ALL_EXECS(flags) ? j : execs[j].d;
			// Skip if they are the same.
			if (a_idx == b_idx)
				continue;
			ccv_nnc_graph_exec_symbol_info_t* b_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, b_idx);
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(b_symbol_info->flags))
				continue;
			int b_to_a = 0;
			for (x = 0; x < a_symbol_info->input_size && !b_to_a; x++)
			{
				int a = a_symbol_info->inputs[x];
				if (a < 0)
					continue;
				// Handle alias as well.
				ccv_nnc_tensor_symbol_info_t* a_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, a);
				if (a_tensor_info->alias_ref)
					a = a_tensor_info->alias_ref - 1;
				for (y = 0; y < b_symbol_info->output_size && !b_to_a; y++)
				{
					int b = b_symbol_info->outputs[y];
					if (b < 0)
						continue;
					ccv_nnc_tensor_symbol_info_t* b_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, b);
					if (b_tensor_info->alias_ref)
						b = b_tensor_info->alias_ref - 1;
					if (a == b && // This two have matching inputs and outputs.
						(!a_tensor_info->alias_ref ||
						 !b_tensor_info->alias_ref || // If any of them are not alias, the must overlap, you can concatenate.
						 ccv_nnc_over_tensor_symbol_aliases(a_tensor_info, b_tensor_info))) // Otherwise, we explicitly check whether it overlaps, if it does, concatenate.
						b_to_a = 1;
				}
			}
			if (b_to_a)
			{
				if (execs)
					ccv_nnc_graph_exec_symbol_concat(graph, execs[j], execs[i]);
				else
					ccv_nnc_graph_exec_symbol_concat(graph,
						(ccv_nnc_graph_exec_symbol_t) {
							.d = j,
							.graph = graph
						}, (ccv_nnc_graph_exec_symbol_t) {
							.d = i,
							.graph = graph
						}
					);
			}
			int a_to_b = 0;
			for (x = 0; x < a_symbol_info->output_size && !a_to_b; x++)
			{
				int a = a_symbol_info->outputs[x];
				if (a < 0)
					continue;
				// Handle alias as well.
				ccv_nnc_tensor_symbol_info_t* a_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, a);
				if (a_tensor_info->alias_ref)
					a = a_tensor_info->alias_ref - 1;
				for (y = 0; y < b_symbol_info->input_size && !a_to_b; y++)
				{
					int b = b_symbol_info->inputs[y];
					if (b < 0)
						continue;
					ccv_nnc_tensor_symbol_info_t* b_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, b);
					if (b_tensor_info->alias_ref)
						b = b_tensor_info->alias_ref - 1;
					if (a == b && // This two have matching inputs and outputs.
						(!a_tensor_info->alias_ref ||
						 !b_tensor_info->alias_ref || // If any of them are not alias, the must overlap, you can concatenate.
						 ccv_nnc_over_tensor_symbol_aliases(a_tensor_info, b_tensor_info))) // Otherwise, we explicitly check whether it overlaps, if it does, concatenate.
						a_to_b = 1;
				}
			}
			if (a_to_b)
			{
				if (execs)
					ccv_nnc_graph_exec_symbol_concat(graph, execs[i], execs[j]);
				else
					ccv_nnc_graph_exec_symbol_concat(graph,
						(ccv_nnc_graph_exec_symbol_t) {
							.d = i,
							.graph = graph
						}, (ccv_nnc_graph_exec_symbol_t) {
							.d = j,
							.graph = graph
						}
					);
			}
		}
	}
	// If flag says so, loop over to find sources / destinations too.
	if (CCV_NNC_IS_AUTOGEN_SOURCES_AND_DESTINATIONS(flags))
	{
		uint8_t* flags = (uint8_t*)cccalloc(sizeof(uint8_t), graph->exec_symbol_info->rnum);
		for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		{
			ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(symbol_info->flags))
			{
				flags[i] = 3; // Skip.
				continue;
			}
			if (symbol_info->outgoings && symbol_info->outgoings->rnum)
			{
				flags[i] |= 2;
				for (j = 0; j < symbol_info->outgoings->rnum; j++)
					flags[*(int*)ccv_array_get(symbol_info->outgoings, j)] |= 1;
			}
		}
		if (!graph->sources)
			graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
		else
			ccv_array_clear(graph->sources);
		if (!graph->destinations)
			graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
		else
			ccv_array_clear(graph->destinations);
		for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		{
			if (flags[i] == 3)
				continue;
			ccv_nnc_graph_exec_symbol_t exec = {
				.d = i,
				.graph = graph,
			};
			if (!(flags[i] & 1))
				ccv_array_push(graph->sources, &exec);
			if (!(flags[i] & 2))
				ccv_array_push(graph->destinations, &exec);
		}
		ccfree(flags);
	}
	return 0;
}

ccv_nnc_graph_exec_symbol_t* ccv_nnc_symbolic_graph_sources(const ccv_nnc_symbolic_graph_t* const graph)
{
	return graph->sources ? (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(graph->sources, 0) : 0;
}

void ccv_nnc_symbolic_graph_add_source(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t source)
{
	if (!graph->sources)
		graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	assert(source.graph == graph);
	ccv_array_push(graph->sources, &source);
}

void ccv_nnc_symbolic_graph_set_sources(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size)
{
	if (!graph->sources)
		graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	else
		ccv_array_clear(graph->sources);
	int i;
	for (i = 0; i < source_size; i++)
		ccv_nnc_symbolic_graph_add_source(graph, sources[i]);
}

int ccv_nnc_symbolic_graph_source_size(const ccv_nnc_symbolic_graph_t* const graph)
{
	return graph->sources ? graph->sources->rnum : 0;
}

ccv_nnc_graph_exec_symbol_t* ccv_nnc_symbolic_graph_destinations(const ccv_nnc_symbolic_graph_t* const graph)
{
	return graph->destinations ? (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(graph->destinations, 0) : 0;
}

void ccv_nnc_symbolic_graph_add_destination(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t destination)
{
	if (!graph->destinations)
		graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	assert(destination.graph == graph);
	ccv_array_push(graph->destinations, &destination);
}

void ccv_nnc_symbolic_graph_set_destinations(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	if (!graph->destinations)
		graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	else
		ccv_array_clear(graph->destinations);
	int i;
	for (i = 0; i < destination_size; i++)
		ccv_nnc_symbolic_graph_add_destination(graph, destinations[i]);
}

int ccv_nnc_symbolic_graph_destination_size(const ccv_nnc_symbolic_graph_t* const graph)
{
	return graph->destinations ? graph->destinations->rnum : 0;
}

static void _ccv_nnc_symbolic_graph_dot_exec_symbol(const int index, const ccv_nnc_graph_exec_symbol_info_t* const symbol_info, const int flags, FILE* out)
{
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	if (symbol_info->name)
		fputs(symbol_info->name, out);
	else
		fprintf(out, "node%d", index);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		fputs("|Command: ", out);
		fputs(ccv_nnc_cmd_name(symbol_info->cmd.cmd), out);
		fputc('}', out);
	}
}

static void _ccv_nnc_symbolic_graph_dot_tensor_symbol(const int index, const ccv_nnc_tensor_symbol_info_t* const symbol_info, const ccv_nnc_tensor_symbol_info_t* const alias_info, const int html_like, const int flags, FILE* out)
{
	// if it has an alias pointer, or, it is a long form.
	if ((flags == CCV_NNC_LONG_DOT_GRAPH || alias_info) && !html_like)
		fputc('{', out);
	if (symbol_info->name)
		fputs(symbol_info->name, out);
	else
		fprintf(out, "tensor%d", index);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		int flag = -1;
		if (symbol_info->flags & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS)
			flag = fputs(" (0", out); // Output if it is zero init'ed.
		if (symbol_info->flags & CCV_NNC_TENSOR_SYMBOL_TAPE_VAR)
			if (flag)
				flag = (flag >= 0) ? fputs(",t", out) : fputs(" (t", out); // Output is a tape variable
		if (flag >= 0)
			fputs(")", out);
	}
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		int i;
		if (html_like)
			fprintf(out, "</td><td>%d", symbol_info->info.dim[0]);
		else
			fprintf(out, "|%d", symbol_info->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && symbol_info->info.dim[i]; i++)
			fprintf(out, "x%d", symbol_info->info.dim[i]);
	}
	if (alias_info)
	{
		if (html_like)
			fputs("</td><td border=\"0\">as. ", out);
		else
			fputs("|as. ", out);
		if (alias_info->name)
			fputs(alias_info->name, out);
		else
			fprintf(out, "tensor%d", symbol_info->alias_ref - 1);
		if (flags == CCV_NNC_LONG_DOT_GRAPH)
		{
			int flag = -1;
			if (alias_info->flags & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS)
				flag = fputs(" (0", out); // Output if it is zero init'ed.
			if (alias_info->flags & CCV_NNC_TENSOR_SYMBOL_TAPE_VAR)
				flag = (flag >= 0) ? fputs(",t", out) : fputs(" (t", out); // Output is a tape variable
			if (flag >= 0)
				fputs(")", out);
		}
	}
	if ((flags == CCV_NNC_LONG_DOT_GRAPH || alias_info) && !html_like)
		fputc('}', out);
}

static void _ccv_nnc_symbolic_graph_dot_node(const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const int index, const ccv_array_t* const tensor_symbol_info, const int flags, FILE* out)
{
	fprintf(out, "node%d [shape=record,label=\"", index);
	_ccv_nnc_symbolic_graph_dot_exec_symbol(index, exec_symbol_info, flags, out);
	int i;
	if (exec_symbol_info->input_size > 0)
	{
		fputs("|{Input", out);
		for (i = 0; i < exec_symbol_info->input_size; i++)
		{
			if (exec_symbol_info->inputs[i] >= 0)
			{
				fputc('|', out);
				const ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, exec_symbol_info->inputs[i]);
				const ccv_nnc_tensor_symbol_info_t* const alias_symbol = tensor_symbol->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_symbol->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->inputs[i], tensor_symbol, alias_symbol, 0, flags, out);
			} else
				fputs("|-", out);
		}
		fputc('}', out);
	}
	if (exec_symbol_info->output_size > 0)
	{
		fputs("|{Output", out);
		for (i = 0; i < exec_symbol_info->output_size; i++)
		{
			if (exec_symbol_info->outputs[i] >= 0)
			{
				fputc('|', out);
				const ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, exec_symbol_info->outputs[i]);
				const ccv_nnc_tensor_symbol_info_t* const alias_symbol = tensor_symbol->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_symbol->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->outputs[i], tensor_symbol, alias_symbol, 0, flags, out);
			} else
				fputs("|-", out);
		}
		fputc('}', out);
	}
	fputs("\"];\n", out);
}

static void _ccv_nnc_symbolic_graph_dot_while_label(const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const int index, const ccv_array_t* const tensor_symbol_info, const ccv_nnc_symbolic_graph_t* const while_graph, const int flags, FILE* out)
{
	int i;
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputs("<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"><tr><td colspan=\"3\" border=\"0\"><b>", out);
	else
		fputs("<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"><tr><td colspan=\"2\" border=\"0\"><b>", out);
	if (exec_symbol_info->name)
		fputs(exec_symbol_info->name, out);
	else
		fprintf(out, "while%d", index);
	fputs(" </b>Command: ", out);
	fputs(ccv_nnc_cmd_name(exec_symbol_info->cmd.cmd), out);
	fputs("</td></tr>", out);
	const int p_idx = while_graph->p_idx - 1;
	assert(p_idx >= 0);
	if (exec_symbol_info->input_size > 0)
	{
		fprintf(out, "<tr><td rowspan=\"%d\">Input</td>", exec_symbol_info->input_size);
		for (i = 0; i < exec_symbol_info->input_size; i++)
		{
			if (i > 0)
				fputs("<tr>", out);
			if (exec_symbol_info->inputs[i] >= 0)
			{
				fputs("<td>", out);
				const ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, exec_symbol_info->inputs[i]);
				const ccv_nnc_tensor_symbol_info_t* const alias_symbol = tensor_symbol->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_symbol->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->inputs[i], tensor_symbol, alias_symbol, 1, flags, out);
				fputs("</td><td border=\"0\">=&gt; ", out);
				const int s_idx = *(int*)ccv_array_get(tensor_symbol->s_ref, p_idx) - 1;
				assert(s_idx >= 0);
				const ccv_nnc_tensor_symbol_info_t* const sub_tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, s_idx);
				if (sub_tensor_symbol->name)
					fputs(sub_tensor_symbol->name, out);
				else
					fprintf(out, "tensor%d", s_idx);
				fputs("</td></tr>", out);
			} else {
				if (flags == CCV_NNC_LONG_DOT_GRAPH)
					fputs("<td colspan=\"3\">-</td></tr>", out);
				else
					fputs("<td colspan=\"2\">-</td></tr>", out);
			}
		}
	}
	if (exec_symbol_info->output_size > 0)
	{
		fprintf(out, "<tr><td rowspan=\"%d\">Output</td>", exec_symbol_info->output_size);
		for (i = 0; i < exec_symbol_info->output_size; i++)
		{
			if (i > 0)
				fputs("<tr>", out);
			if (exec_symbol_info->outputs[i] >= 0)
			{
				fputs("<td>", out);
				ccv_nnc_tensor_symbol_info_t* tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, exec_symbol_info->outputs[i]);
				ccv_nnc_tensor_symbol_info_t* alias_symbol = tensor_symbol->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_symbol->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->outputs[i], tensor_symbol, alias_symbol, 1, flags, out);
				fputs("</td><td border=\"0\">=&gt; ", out);
				const int s_idx = *(int*)ccv_array_get(tensor_symbol->s_ref, p_idx) - 1;
				assert(s_idx >= 0);
				const ccv_nnc_tensor_symbol_info_t* const sub_tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, s_idx);
				if (sub_tensor_symbol->name)
					fputs(sub_tensor_symbol->name, out);
				else
					fprintf(out, "tensor%d", s_idx);
				fputs("</td></tr>", out);
			} else {
				if (flags == CCV_NNC_LONG_DOT_GRAPH)
					fputs("<td colspan=\"3\">-</td></tr>", out);
				else
					fputs("<td colspan=\"2\">-</td></tr>", out);
			}
		}
	}
	for (i = 0; i < while_graph->tensor_symbol_info->rnum; i++)
	{
		const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, i);
		if (tensor_symbol_info->assign_ref)
		{
			if (flags == CCV_NNC_LONG_DOT_GRAPH)
				fputs("<tr><td colspan=\"3\" border=\"0\">", out);
			else
				fputs("<tr><td colspan=\"2\" border=\"0\">", out);
			const ccv_nnc_tensor_symbol_info_t* const assign_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(while_graph->tensor_symbol_info, tensor_symbol_info->assign_ref - 1);
			if (assign_symbol_info->name)
				fputs(assign_symbol_info->name, out);
			else
				fprintf(out, "tensor%d", tensor_symbol_info->assign_ref - 1);
			fputs(" -&gt; ", out);
			if (tensor_symbol_info->name)
				fputs(tensor_symbol_info->name, out);
			else
				fprintf(out, "tensor%d", i);
			fputs("</td></tr>", out);
		}
	}
	fputs("</table>", out);
}

static void _ccv_nnc_symbolic_graph_dot_case_of_label(const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const int index, const ccv_array_t* const tensor_symbol_info, const int flags, FILE* out)
{
	int i;
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputs("<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"><tr><td colspan=\"3\" border=\"0\"><b>", out);
	else
		fputs("<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"><tr><td colspan=\"2\" border=\"0\"><b>", out);
	if (exec_symbol_info->name)
		fputs(exec_symbol_info->name, out);
	else
		fprintf(out, "caseof%d", index);
	fputs(" </b>Command: ", out);
	fputs(ccv_nnc_cmd_name(exec_symbol_info->cmd.cmd), out);
	fputs("</td></tr>", out);
	if (exec_symbol_info->input_size > 0)
	{
		fprintf(out, "<tr><td rowspan=\"%d\">Input</td>", ccv_min(exec_symbol_info->input_size, exec_symbol_info->output_size));
		for (i = 0; i < ccv_min(exec_symbol_info->input_size, exec_symbol_info->output_size); i++)
		{
			if (i > 0)
				fputs("<tr>", out);
			if (exec_symbol_info->inputs[i] >= 0)
			{
				fputs("<td>", out);
				const ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, exec_symbol_info->inputs[i]);
				const ccv_nnc_tensor_symbol_info_t* const alias_symbol = tensor_symbol->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_symbol->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->inputs[i], tensor_symbol, alias_symbol, 1, flags, out);
				fputs("</td></tr>", out);
			} else {
				if (flags == CCV_NNC_LONG_DOT_GRAPH)
					fputs("<td colspan=\"2\">-</td></tr>", out);
				else
					fputs("<td colspan=\"1\">-</td></tr>", out);
			}
		}
	}
	if (exec_symbol_info->output_size > 0)
	{
		fprintf(out, "<tr><td rowspan=\"%d\">Output</td>", exec_symbol_info->output_size);
		for (i = 0; i < exec_symbol_info->output_size; i++)
		{
			if (i > 0)
				fputs("<tr>", out);
			if (exec_symbol_info->outputs[i] >= 0)
			{
				fputs("<td>", out);
				ccv_nnc_tensor_symbol_info_t* tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, exec_symbol_info->outputs[i]);
				ccv_nnc_tensor_symbol_info_t* alias_symbol = tensor_symbol->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(tensor_symbol_info, tensor_symbol->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->outputs[i], tensor_symbol, alias_symbol, 1, flags, out);
				fputs("</td></tr>", out);
			} else {
				if (flags == CCV_NNC_LONG_DOT_GRAPH)
					fputs("<td colspan=\"2\">-</td></tr>", out);
				else
					fputs("<td colspan=\"1\">-</td></tr>", out);
			}
		}
	}
	fputs("</table>", out);
}

static void _ccv_nnc_symbolic_graph_dot_sub_graphs(const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const ccv_array_t* const tensor_symbol_info, const ccv_array_t* const sub_graphs, const int flags, FILE* out, int* c)
{
	int i, j, k;
	// Output this node info within this subgraph.
	if (exec_symbol_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
	{
		fprintf(out, "subgraph cluster%d {\nstyle=\"rounded\";\nnode%d [style=invisible];\nlabel=<", *c, *c);
		const ccv_nnc_symbolic_graph_t* const while_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(sub_graphs, CCV_NNC_GRAPH_REF(exec_symbol_info)[0] - 1);
		_ccv_nnc_symbolic_graph_dot_while_label(exec_symbol_info, *c, tensor_symbol_info, while_graph, flags, out);
	} else if (exec_symbol_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF) {
		fprintf(out, "subgraph cluster%d {\nstyle=\"rounded\";\nnode%d [style=invisible];\nlabel=<", *c, *c);
		_ccv_nnc_symbolic_graph_dot_case_of_label(exec_symbol_info, *c, tensor_symbol_info, flags, out);
	}
	fputs(">;\n", out);
	++(*c);
	for (k = 0; k < exec_symbol_info->graph_ref_size; k++)
	{
		if (exec_symbol_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			fprintf(out, "subgraph cluster%d {\nstyle=\"rounded\";\nnode%d [style=invisible];\nlabel=\"\"\n", *c, *c);
			++(*c);
		}
		const ccv_nnc_symbolic_graph_t* const graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(sub_graphs, CCV_NNC_GRAPH_REF(exec_symbol_info)[k] - 1);
		int* node_id = (int*)ccmalloc(sizeof(int) * graph->exec_symbol_info->rnum);
		for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		{
			node_id[i] = *c;
			const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
			// Skip the dead one.
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info->flags))
				continue;
			if (exec_symbol_info->graph_ref_size)
				_ccv_nnc_symbolic_graph_dot_sub_graphs(exec_symbol_info, graph->tensor_symbol_info, graph->sub_graphs, flags, out, c);
			else {
				_ccv_nnc_symbolic_graph_dot_node(exec_symbol_info, *c, graph->tensor_symbol_info, flags, out);
				++(*c);
			}
		}
		// Output connections.
		for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		{
			const ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
			// Skip the dead one.
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info->flags))
				continue;
			if (exec_symbol_info->outgoings)
				for (j = 0; j < exec_symbol_info->outgoings->rnum; j++)
				{
					const int outgoing_idx = *(int*)ccv_array_get(exec_symbol_info->outgoings, j);
					const ccv_nnc_graph_exec_symbol_info_t* const outgoing_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, outgoing_idx);
					// If both are sub-graphs, have both tail and head specified.
					if (CCV_NNC_GRAPH_REF(exec_symbol_info)[0] && CCV_NNC_GRAPH_REF(outgoing_symbol_info)[0])
						fprintf(out, "node%d -> node%d [ltail=cluster%d,lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i], node_id[outgoing_idx]);
					else if (CCV_NNC_GRAPH_REF(exec_symbol_info)[0] && !CCV_NNC_GRAPH_REF(outgoing_symbol_info)[0])
						fprintf(out, "node%d -> node%d [ltail=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i]);
					else if (!CCV_NNC_GRAPH_REF(exec_symbol_info)[0] && CCV_NNC_GRAPH_REF(outgoing_symbol_info)[0])
						fprintf(out, "node%d -> node%d [lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[outgoing_idx]);
					else
						fprintf(out, "node%d -> node%d;\n", node_id[i], node_id[outgoing_idx]);
				}
		}
		fputs("}\n", out);
		ccfree(node_id);
	}
	// Extra subgraph cluster.
	if (exec_symbol_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		fputs("}\n", out);
}

void ccv_nnc_symbolic_graph_dot(const ccv_nnc_symbolic_graph_t* const graph, const int flags, FILE* out)
{
	fputs("digraph G {\ncompound=true;\n", out);
	int i, j;
	int c = 0;
	int* node_id = (int*)ccmalloc(sizeof(int) * graph->exec_symbol_info->rnum);
	// Output styles.
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
	{
		node_id[i] = c;
		const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		// Skip the dead one.
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info->flags))
			continue;
		if (exec_symbol_info->graph_ref_size)
			_ccv_nnc_symbolic_graph_dot_sub_graphs(exec_symbol_info, graph->tensor_symbol_info, graph->sub_graphs, flags, out, &c);
		else {
			_ccv_nnc_symbolic_graph_dot_node(exec_symbol_info, c, graph->tensor_symbol_info, flags, out);
			++c;
		}
	}
	// Output connections.
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
	{
		const ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		// Skip the dead one.
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info->flags))
			continue;
		if (exec_symbol_info->outgoings)
			for (j = 0; j < exec_symbol_info->outgoings->rnum; j++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(exec_symbol_info->outgoings, j);
				const ccv_nnc_graph_exec_symbol_info_t* const outgoing_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, outgoing_idx);
				// If both are sub-graphs, have both tail and head specified.
				if (exec_symbol_info->graph_ref_size && outgoing_symbol_info->graph_ref_size)
					fprintf(out, "node%d -> node%d [ltail=cluster%d,lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i], node_id[outgoing_idx]);
				else if (exec_symbol_info->graph_ref_size && !outgoing_symbol_info->graph_ref_size)
					fprintf(out, "node%d -> node%d [ltail=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i]);
				else if (!exec_symbol_info->graph_ref_size && outgoing_symbol_info->graph_ref_size)
					fprintf(out, "node%d -> node%d [lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[outgoing_idx]);
				else
					fprintf(out, "node%d -> node%d;\n", node_id[i], node_id[outgoing_idx]);
			}
	}
	fputs("}\n", out);
	ccfree(node_id);
}

void ccv_nnc_symbolic_graph_free(ccv_nnc_symbolic_graph_t* const graph)
{
	int i;
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		_ccv_nnc_graph_exec_symbol_free((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i), 0);
	for (i = 0; i < graph->tensor_symbol_info->rnum; i++)
	{
		ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, i);
		if (symbol_info->name)
			ccfree(symbol_info->name);
		if (symbol_info->s_ref)
			ccv_array_free(symbol_info->s_ref);
	}
	if (graph->sub_graphs)
	{
		for (i = 0; i < graph->sub_graphs->rnum; i++)
			ccv_nnc_symbolic_graph_free(*(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, i));
		ccv_array_free(graph->sub_graphs);
	}
	if (graph->sources)
		ccv_array_free(graph->sources);
	if (graph->destinations)
		ccv_array_free(graph->destinations);
	if (graph->breakpoints)
		ccfree(graph->breakpoints);
	ccv_array_free(graph->tensor_symbol_info);
	ccv_array_free(graph->exec_symbol_info);
	if (graph->backward_tensor_symbols)
		ccfree(graph->backward_tensor_symbols);
	ccfree(graph);
}

void ccv_nnc_symbolic_graph_symbol_infer(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_visit_t* const visit, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info)
{
	memcpy(tensor_symbol_info, ccv_array_get(symbolic_graph->tensor_symbol_info, 0), sizeof(ccv_nnc_tensor_symbol_info_t) * symbolic_graph->tensor_symbol_info->rnum);
	memcpy(exec_symbol_info, ccv_array_get(symbolic_graph->exec_symbol_info, 0), sizeof(ccv_nnc_graph_exec_symbol_info_t) * symbolic_graph->exec_symbol_info->rnum);
	int i;
	if (p_tensor_symbol_info)
		for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
			if (tensor_symbol_info[i].p_ref)
			{
				const int p_ref = tensor_symbol_info[i].p_ref - 1;
				assert(p_ref < p_tensor_symbol_info_size);
				tensor_symbol_info[i].info = p_tensor_symbol_info[p_ref].info;
				// I don't need to copy over inc and ofs for alias.
			}
	int max_input_size = 0, max_output_size = 0;
	// Materialize auto hints.
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
	{
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info[i].flags))
			continue;
		max_input_size = ccv_max(max_input_size, exec_symbol_info[i].input_size);
		max_output_size = ccv_max(max_output_size, exec_symbol_info[i].output_size);
		// If there is no hint and we have input and output tensor specified.
		if (ccv_nnc_is_no_hint(exec_symbol_info[i].hint) &&
			exec_symbol_info[i].input_size > 0 && exec_symbol_info[i].inputs[0] >= 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].inputs[0]].info) &&
			exec_symbol_info[i].output_size > 0 && exec_symbol_info[i].outputs[0] >= 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].outputs[0]].info))
			exec_symbol_info[i].hint = ccv_nnc_hint_auto(exec_symbol_info[i].cmd.info, tensor_symbol_info[exec_symbol_info[i].inputs[0]].info, tensor_symbol_info[exec_symbol_info[i].outputs[0]].info);
	}

	ccv_nnc_tensor_param_t input_params[ccv_max(1, max_input_size)];
	ccv_nnc_tensor_param_t output_params[ccv_max(1, max_output_size)];

	// Materialize auto tensors. This need to go with the topological order.
	// TODO: Need to proper handle sub-graphs (thus, run sub-graph to figure out the tensor properties).
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node) {
		if (node->input_size > 0 && node->output_size > 0)
		{
			for (i = 0; i < node->input_size; i++)
				input_params[i] = node->inputs[i] >= 0 ? tensor_symbol_info[node->inputs[i]].info : ccv_nnc_tensor_auto;
			ccv_nnc_hint_tensor_auto(node->cmd, input_params, node->input_size, node->hint, output_params, node->output_size);
			for (i = 0; i < node->output_size; i++)
				/* Only assign the output parameters if the symbol itself is auto. */
				if (node->outputs[i] >= 0 && ccv_nnc_is_tensor_auto(tensor_symbol_info[node->outputs[i]].info))
					tensor_symbol_info[node->outputs[i]].info = output_params[i];
		}
	} ccv_nnc_graph_visit_endfor
	// If still point to any device, assign default device 00 to it.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		if (CCV_TENSOR_GET_DEVICE(tensor_symbol_info[i].info.type) == CCV_COMPUTE_DEVICE_ANY)
			tensor_symbol_info[i].info.type = (~CCV_COMPUTE_DEVICE_ANY & tensor_symbol_info[i].info.type) | CCV_COMPUTE_DEVICE_000;
}
