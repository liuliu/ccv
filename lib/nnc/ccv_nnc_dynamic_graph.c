#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"

#pragma mark - Level-4 API

ccv_nnc_dynamic_graph_t* ccv_nnc_dynamic_graph_new(void)
{
	ccv_nnc_dynamic_graph_t* graph = ccmalloc(sizeof(ccv_nnc_dynamic_graph_t));
	graph->no_grad = 0;
	graph->reuse_var = -1;
	graph->mp_hdr = -1;
	graph->vars = ccv_array_new(sizeof(ccv_nnc_tensor_variable_t), 1, 0);
	graph->binds = ccv_array_new(sizeof(ccv_nnc_tensor_variable_graph_bind_t), 1, 0);
	graph->tape = ccv_nnc_symbolic_graph_new();
	graph->freed = kh_init(dy_dev);
	graph->allocd = kh_init(dy_alloc);
	// These may not be used as frequent, init as needed.
	graph->stateful_execs = 0;
	graph->synced_streams = 0;
	graph->signal_container = 0;
	graph->ws = 0;
	return graph;
}

static void _ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int zeroing)
{
	const int index = tensor_variable->index;
	if (tensor_variable->tensor_view && !CCV_NNC_IS_EXTERN_TENSOR_VIEW(tensor_variable->tensor_view))
	{
		if (CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view))
			ccv_nnc_tensor_view_free(tensor_variable->tensor_view);
		else {
			if (!tensor_variable->alias_ref && // Return this memory to the graph.
				CCV_TENSOR_GET_MEMORY(tensor_variable->tensor_view->info.type) == CCV_TENSOR_GPU_MEMORY)
				ccv_nnc_dynamic_graph_xpu_free(graph, tensor_variable->tensor_view->data.ptr);
			ccv_nnc_tensor_free((ccv_nnc_tensor_t*)tensor_variable->tensor_view);
		}
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

static void _ccv_nnc_tensor_variable_graph_bind_free(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_graph_bind_t* const bind, const int zeroing)
{
	bind->index = CCV_NNC_TENSOR_NO_VARIABLE;
	if (bind->sources)
		ccv_array_free(bind->sources);
	if (bind->destinations)
		ccv_array_free(bind->destinations);
	if (bind->tensor_view && !CCV_NNC_IS_EXTERN_TENSOR_VIEW(bind->tensor_view))
	{
		if (CCV_IS_TENSOR_VIEW(bind->tensor_view))
			ccv_nnc_tensor_view_free(bind->tensor_view);
		else {
			if (!bind->is_alias && // Return this memory to the graph.
				CCV_TENSOR_GET_MEMORY(bind->tensor_view->info.type) == CCV_TENSOR_GPU_MEMORY)
				ccv_nnc_dynamic_graph_xpu_free(graph, bind->tensor_view->data.ptr);
			ccv_nnc_tensor_free((ccv_nnc_tensor_t*)bind->tensor_view);
		}
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
		_ccv_nnc_tensor_variable_graph_bind_free(graph, (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, i), 0);
	ccv_array_free(graph->binds);
	ccv_nnc_symbolic_graph_free(graph->tape);
	if (graph->ws)
		ccv_array_free(graph->ws);
	khiter_t k;
	if (graph->stateful_execs)
	{
		for (k = kh_begin(graph->stateful_execs); k != kh_end(graph->stateful_execs); ++k)
		{
			if (!kh_exist(graph->stateful_execs, k))
				continue;
			ccfree(kh_val(graph->stateful_execs, k));
		}
		kh_destroy(stateful_exec, graph->stateful_execs);
	}
	if (graph->synced_streams)
	{
		for (k = kh_begin(graph->synced_streams); k != kh_end(graph->synced_streams); ++k)
		{
			if (!kh_exist(graph->synced_streams, k))
				continue;
			ccv_nnc_synced_stream_t* const synced_stream = &kh_val(graph->synced_streams, k);
			ccv_nnc_stream_context_free(synced_stream->stream);
			ccv_nnc_stream_signal_free(synced_stream->synced);
		}
		kh_destroy(synced_stream, graph->synced_streams);
	}
	if (graph->signal_container)
		ccv_nnc_signal_container_free(graph->signal_container);
	ccv_nnc_dynamic_graph_xpu_alloc_destroy(graph);
	ccfree(graph);
}

void ccv_nnc_tensor_variable_set(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, ccv_nnc_tensor_t* const tensor)
{
	assert(!tensor_variable->alias_ref);
	if (tensor_variable->tensor_view && !CCV_NNC_IS_EXTERN_TENSOR_VIEW(tensor_variable->tensor_view))
	{
		assert(!CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view));
		ccv_nnc_tensor_free((ccv_nnc_tensor_t*)tensor_variable->tensor_view);
	}
	tensor_variable->info = tensor->info;
	tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)((uintptr_t)tensor | 1);
}

inline static void _ccv_nnc_tensor_variable_init(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable, const ccv_nnc_tensor_param_t info)
{
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
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_new_impl(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_param_t info)
{
	ccv_nnc_tensor_variable_t tensor_variable = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	tensor_variable->type = CCV_NNC_TENSOR_VARIABLE;
	_ccv_nnc_tensor_variable_init(graph, tensor_variable, info);
	return tensor_variable;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_constant_new_impl(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_param_t info)
{
	ccv_nnc_tensor_variable_t tensor_variable = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	tensor_variable->type = CCV_NNC_TENSOR_CONSTANT;
	_ccv_nnc_tensor_variable_init(graph, tensor_variable, info);
	return tensor_variable;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_alias_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info)
{
	assert(!tensor_variable->alias_ref);
	ccv_nnc_tensor_variable_t variable_alias = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	variable_alias->type = tensor_variable->type;
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

ccv_nnc_tensor_t* ccv_nnc_tensor_from_variable(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, ccv_nnc_stream_context_t* const stream_context)
{
	if (tensor_variable->tensor_view)
	{
		if (tensor_variable->alias_ref)
		{
			const int alias_ref = tensor_variable->alias_ref - 1;
			assert(alias_ref >= 0);
			ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
			if (CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view))
			{
				ccv_nnc_tensor_view_t* const tv = tensor_variable->tensor_view;
				// We cannot have an alias with custom set tensor, otherwise the pointer update is invalid.
				assert(!CCV_NNC_IS_EXTERN_TENSOR_VIEW(tv));
				// Update the tensor_view pointer every time access it, because the underlying variable it alias to have changed.
				tv->data.u8 = CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->data.u8 + tv->off;
			} else {
				ccv_nnc_tensor_t* const tv = (ccv_nnc_tensor_t*)tensor_variable->tensor_view;
				// We cannot have an alias with custom set tensor, otherwise the pointer update is invalid.
				assert(!CCV_NNC_IS_EXTERN_TENSOR_VIEW(tv));
				// Update the tensor_view pointer every time access it, because the underlying variable it alias to have changed.
				tv->data.u8 = CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->data.u8;
			}
		}
		return (ccv_nnc_tensor_t*)CCV_NNC_TENSOR_VIEW(tensor_variable->tensor_view);
	}
	if (!tensor_variable->alias_ref)
	{
		void* ptr = 0;
		if (CCV_TENSOR_GET_MEMORY(tensor_variable->info.type) == CCV_TENSOR_GPU_MEMORY)
			ptr = ccv_nnc_dynamic_graph_xpu_alloc(graph, CCV_TENSOR_GET_DEVICE_ID(tensor_variable->info.type), (intptr_t)stream_context, ccv_nnc_tensor_data_size(tensor_variable->info));
		tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(ptr, tensor_variable->info, 0);
		assert(tensor_variable->tensor_view->data.u8);
		return (ccv_nnc_tensor_t*)tensor_variable->tensor_view;
	}
	const int alias_ref = tensor_variable->alias_ref - 1;
	assert(alias_ref >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
	assert(!variable_to->alias_ref);
	if (!variable_to->tensor_view)
	{
		void* ptr = 0;
		if (CCV_TENSOR_GET_MEMORY(tensor_variable->info.type) == CCV_TENSOR_GPU_MEMORY)
			ptr = ccv_nnc_dynamic_graph_xpu_alloc(graph, CCV_TENSOR_GET_DEVICE_ID(tensor_variable->info.type), (intptr_t)stream_context, ccv_nnc_tensor_data_size(tensor_variable->info));
		variable_to->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(ptr, variable_to->info, 0);
		assert(variable_to->tensor_view->data.u8);
	}
	int no_ofs = 1;
	int i;
	for (i = 0; no_ofs && i < CCV_NNC_MAX_DIM_ALLOC; i++)
		no_ofs = (tensor_variable->ofs[i] == 0);
	int no_inc = 1;
	for (i = 0; no_inc && i < CCV_NNC_MAX_DIM_ALLOC; i++)
		no_inc = (tensor_variable->inc[i] == 0);
	if (!no_inc)
		no_inc = (memcmp(tensor_variable->inc, tensor_variable->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0);
	assert(ccv_nnc_tensor_count(tensor_variable->info) <= ccv_nnc_tensor_count(variable_to->info));
	if (no_ofs && no_inc)
		tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->data.u8, tensor_variable->info, 0);
	else
		tensor_variable->tensor_view = ccv_nnc_tensor_view_new((ccv_nnc_tensor_t*)CCV_NNC_TENSOR_VIEW(variable_to->tensor_view), tensor_variable->info.dim, tensor_variable->ofs, no_inc ? tensor_variable->info.dim : tensor_variable->inc);
	return (ccv_nnc_tensor_t*)tensor_variable->tensor_view;
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
	bind->type = tensor_variable->type;
	bind->index = tensor_variable->index;
	bind->is_alias = !!tensor_variable->alias_ref;
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
	int no_inc = 1;
	int i;
	for (i = 0; no_inc && i < CCV_NNC_MAX_DIM_ALLOC; i++)
		no_inc = (tensor_variable->inc[i] == 0);
	const ccv_nnc_tensor_symbol_t symbol = tensor_variable->symbol = ccv_nnc_tensor_symbol_alias_new(graph->tape, _ccv_nnc_tensor_symbol_from_variable(graph, variable_to), tensor_variable->ofs, no_inc ? tensor_variable->info.dim : tensor_variable->inc, tensor_variable->info, 0);
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

void ccv_nnc_dynamic_graph_set_no_grad(ccv_nnc_dynamic_graph_t* const dynamic_graph, const int no_grad)
{
	dynamic_graph->no_grad = no_grad;
}

static ccv_nnc_synced_stream_t _ccv_nnc_dynamic_graph_get_synced_stream(ccv_nnc_dynamic_graph_t* const graph, const int type)
{
	if (!graph->synced_streams)
		graph->synced_streams = kh_init(synced_stream);
	int ret = 0;
	khiter_t k = kh_put(synced_stream, graph->synced_streams, type, &ret);
	assert(ret >= 0);
	ccv_nnc_synced_stream_t* const synced_stream = &kh_val(graph->synced_streams, k);
	// If ret == 0, the key already exist, we can return directly, otherwise, create and return.
	if (ret != 0)
	{
		synced_stream->stream = ccv_nnc_stream_context_new(type);
		synced_stream->synced = ccv_nnc_stream_signal_new(type);
	}
	return *synced_stream;
}

typedef struct {
	ccv_nnc_dynamic_graph_t* graph;
	int stream_type;
} ccv_nnc_dynamic_graph_neighbor_context_discovery_t;

static ccv_nnc_stream_context_t* _ccv_nnc_dynamic_graph_neighbor_context_discovery(const int device_id, void* const context)
{
	ccv_nnc_dynamic_graph_neighbor_context_discovery_t* const discovery = (ccv_nnc_dynamic_graph_neighbor_context_discovery_t*)context;
	int type = discovery->stream_type;
	CCV_STREAM_SET_DEVICE_ID(type, device_id);
	return _ccv_nnc_dynamic_graph_get_synced_stream(discovery->graph, type).stream;
}

void ccv_nnc_dynamic_graph_exec_ret(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size, const int parallel, ccv_nnc_stream_context_t* const stream_context, ccv_nnc_graph_exec_symbol_t* const graph_execs)
{
	int i, j;
	for (i = 0; i < input_size; i++)
		if (inputs[i] && !inputs[i]->alias_ref)
			{ assert(inputs[i]->tensor_view); }
	ccv_nnc_tensor_t* input_tensors[ccv_max(1, input_size)];
	for (i = 0; i < input_size; i++)
		input_tensors[i] = inputs[i] ? ccv_nnc_tensor_from_variable(graph, inputs[i], stream_context) : 0;
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
			const int alias_ref = inputs[i]->alias_ref - 1;
			assert(alias_ref >= 0);
			ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
			input_alias_sources[i] = ((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, variable_to->symbol.d))->sources;
		} else
			input_alias_sources[i] = 0;
	}
	const int parallel_count = ccv_max(1, parallel);
	assert(input_size % parallel_count == 0);
	const int per_input_size = input_size / parallel_count;
	assert(output_size % parallel_count == 0);
	const int per_output_size = output_size / parallel_count;
	int output_auto = 0;
	for (i = 0; !output_auto && i < output_size; i++)
		output_auto = outputs[i] ? ccv_nnc_is_tensor_auto(outputs[i]->info) : 0;
	// One extra step, infer the parameters for outputs.
	if (output_auto)
	{
		ccv_nnc_tensor_param_t input_params[ccv_max(1, per_input_size)];
		ccv_nnc_tensor_param_t output_params[ccv_max(1, per_output_size)];
		for (i = 0; i < parallel_count; i++)
		{
			for (j = 0; j < per_input_size; j++)
				input_params[j] = inputs[j + i * per_input_size] ? inputs[j + i * per_input_size]->info : ccv_nnc_tensor_auto;
			for (j = 0; j < per_output_size; j++)
				output_params[j] = outputs[j + i * per_output_size] ? outputs[j + i * per_output_size]->info : ccv_nnc_tensor_auto;
			ccv_nnc_hint_tensor_auto(cmd, input_params, per_input_size, hint, output_params, per_output_size);
			for (j = 0; j < per_output_size; j++)
				if (outputs[j + i * per_output_size])
					outputs[j + i * per_output_size]->info = output_params[j];
		}
	}
	int freeable_size = 0;
	ccv_nnc_tensor_variable_t freeables[ccv_max(1, output_size)];
	// Refresh the symbol if it is binded to an existing exec. Otherwise we cannot keep the SSA guarantee.
	for (i = 0; i < output_size; i++)
	{
		// First, go over to see whether there is enforce inplace.
		int enforce_idx = -1;
		for (j = 0; enforce_idx < 0 && j < input_size; j++)
			if (inputs[j] && ccv_nnc_cmd_enforce_inplace(cmd, j, input_size, i, output_size))
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
	ccv_nnc_tensor_t* output_tensors[ccv_max(1, per_output_size)];
	if (parallel_count > 1)
	{
		const int max_device_id_size = per_input_size + per_output_size;
		assert(max_device_id_size > 0);
		int device_ids[max_device_id_size];
		ccv_nnc_stream_context_t* streams[parallel_count];
		ccv_nnc_stream_signal_t* signal;
		if (stream_context)
		{
			if (!graph->signal_container)
				graph->signal_container = ccv_nnc_signal_container_new();
			signal = ccv_nnc_emit_signal_from_container(graph->signal_container, stream_context);
		}
		for (i = 0; i < parallel_count; i++)
		{
			int flag = 0;
			for (j = 0; !flag && j < per_input_size; j++)
				if (input_tensors[i * per_input_size + j])
					flag = (CCV_TENSOR_GET_MEMORY(input_tensors[i * per_input_size + j]->info.type) == CCV_TENSOR_GPU_MEMORY);
			for (j = 0; j < per_output_size; j++)
			{
				output_tensors[j] = outputs[j + i * per_output_size] ? ccv_nnc_tensor_from_variable(graph, outputs[j + i * per_output_size], stream_context) : 0;
				if (output_tensors[j] && !flag)
					flag = (CCV_TENSOR_GET_MEMORY(output_tensors[j]->info.type) == CCV_TENSOR_GPU_MEMORY);
			}
			const int stream_type = flag ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
			const int tensor_type = flag ? CCV_TENSOR_GPU_MEMORY : CCV_TENSOR_CPU_MEMORY;
			const int device_id_size = ccv_nnc_device_ids_for_io(input_tensors + i * per_input_size, per_input_size, output_tensors, per_output_size, tensor_type, device_ids, max_device_id_size);
			ccv_nnc_synced_stream_t stream_0 = {};
			for (j = 0; j < device_id_size; j++)
			{
				int type = stream_type;
				CCV_STREAM_SET_DEVICE_ID(type, device_ids[j]);
				ccv_nnc_synced_stream_t synced_stream = _ccv_nnc_dynamic_graph_get_synced_stream(graph, type);
				if (!stream_0.stream)
					stream_0 = synced_stream;
			}
			// Wait signal to finish.
			if (stream_context)
			{
				if (stream_0.stream)
					ccv_nnc_stream_context_wait_signal(stream_0.stream, signal);
				else
					ccv_nnc_stream_context_wait(stream_context);
			}
			if (stream_0.stream)
			{
				ccv_nnc_dynamic_graph_neighbor_context_discovery_t discovery = {
					.graph = graph,
					.stream_type = stream_type
				};
				ccv_nnc_stream_context_set_neighbor_discovery(stream_0.stream, _ccv_nnc_dynamic_graph_neighbor_context_discovery, &discovery);
			}
			PRINT(CCV_CLI_INFO, "%s: [%d] -> [%d]\n", ccv_nnc_cmd_name(cmd.cmd), per_input_size, per_output_size);
			int k;
			for (k = 0; k < per_input_size; k++)
			{
				PRINT(CCV_CLI_INFO, "|-> %d. %p (%p:%d)", k + 1, input_tensors[k + i * per_input_size], (input_tensors[k + i * per_input_size] ? input_tensors[k + i * per_input_size]->data.u8 : 0), (input_tensors[k + i * per_input_size] ? CCV_TENSOR_GET_DEVICE_ID(input_tensors[k + i * per_input_size]->info.type) : -1));
				if (input_tensors[k + i * per_input_size] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_INFO))
					ccv_nnc_print_tensor_info(input_tensors[k + i * per_input_size]);
				PRINT(CCV_CLI_INFO, "\n");
			}
			ccv_nnc_cmd_exec(cmd, hint, flags, input_tensors + i * per_input_size, per_input_size, output_tensors, per_output_size, stream_0.stream);
			for (k = 0; k < per_output_size; k++)
			{
				PRINT(CCV_CLI_INFO, "|<- %d. %p (%p:%d)", k + 1, output_tensors[k], (output_tensors[k] ? output_tensors[k]->data.u8 : 0), (output_tensors[k] ? CCV_TENSOR_GET_DEVICE_ID(output_tensors[k]->info.type) : -1));
				if (output_tensors[k] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_INFO))
					ccv_nnc_print_tensor_info(output_tensors[k]);
				PRINT(CCV_CLI_INFO, "\n");
			}
			if (stream_context && stream_0.stream)
			{
				ccv_nnc_stream_context_emit_signal(stream_0.stream, stream_0.synced);
				ccv_nnc_stream_context_wait_signal(stream_context, stream_0.synced);
			}
			streams[i] = stream_0.stream;
		}
		if (!stream_context)
			for (i = 0; i < parallel_count; i++)
				if (streams[i])
					ccv_nnc_stream_context_wait(streams[i]);
	} else {
		for (i = 0; i < per_output_size; i++)
			output_tensors[i] = outputs[i] ? ccv_nnc_tensor_from_variable(graph, outputs[i], stream_context) : 0;
		PRINT(CCV_CLI_INFO, "%s: [%d] -> [%d]\n", ccv_nnc_cmd_name(cmd.cmd), per_input_size, per_output_size);
		for (i = 0; i < per_input_size; i++)
		{
			PRINT(CCV_CLI_INFO, "|-> %d. %p (%p:%d)", i + 1, input_tensors[i], (input_tensors[i] ? input_tensors[i]->data.u8 : 0), (input_tensors[i] ? CCV_TENSOR_GET_DEVICE_ID(input_tensors[i]->info.type) : -1));
			if (input_tensors[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_INFO))
				ccv_nnc_print_tensor_info(input_tensors[i]);
			PRINT(CCV_CLI_INFO, "\n");
		}
		ccv_nnc_cmd_exec(cmd, hint, flags, input_tensors, per_input_size, output_tensors, per_output_size, stream_context);
		for (i = 0; i < per_output_size; i++)
		{
			PRINT(CCV_CLI_INFO, "|<- %d. %p (%p:%d)", i + 1, output_tensors[i], (output_tensors[i] ? output_tensors[i]->data.u8 : 0), (output_tensors[i] ? CCV_TENSOR_GET_DEVICE_ID(output_tensors[i]->info.type) : -1));
			if (output_tensors[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_INFO))
				ccv_nnc_print_tensor_info(output_tensors[i]);
			PRINT(CCV_CLI_INFO, "\n");
		}
	}
	if (input_size > 0 && !graph->no_grad) // No need to record the execution if there is no input or we disabled gradient computation.
	{
		ccv_nnc_tensor_symbol_t output_symbols[ccv_max(1, output_size)];
		for (i = 0; i < output_size; i++)
			if (outputs[i])
			{
				assert(outputs[i]->type != CCV_NNC_TENSOR_CONSTANT);
				output_symbols[i] = _ccv_nnc_tensor_symbol_from_variable(graph, outputs[i]);
			} else
				output_symbols[i] = NO_TENSOR_SYMBOL;
		int t;
		for (t = 0; t < parallel_count; t++)
		{
			ccv_nnc_graph_exec_symbol_t graph_exec = ccv_nnc_graph_exec_symbol_new(graph->tape, cmd, input_symbols + t * per_input_size, per_input_size, output_symbols + t * per_output_size, per_output_size, 0);
			if (graph_execs)
				graph_execs[t] = graph_exec;
			// This needs to be done before we set the new sources on the outputs.
			for (i = 0; i < per_input_size; i++)
			{
				ccv_array_t* const input_source = input_sources[i + t * per_input_size];
				if (input_source)
					for (j = 0; j < input_source->rnum; j++)
						ccv_nnc_graph_exec_symbol_concat(graph->tape, (ccv_nnc_graph_exec_symbol_t){
							.d = *(int*)ccv_array_get(input_source, j),
							.graph = graph->tape
						}, graph_exec);
				ccv_array_t* const input_alias_source = input_alias_sources[i + t * per_input_size];
				if (input_alias_source)
					for (j = 0; j < input_alias_source->rnum; j++)
						ccv_nnc_graph_exec_symbol_concat(graph->tape, (ccv_nnc_graph_exec_symbol_t){
							.d = *(int*)ccv_array_get(input_alias_source, j),
							.graph = graph->tape
						}, graph_exec);
			}
			for (i = 0; i < per_input_size; i++)
			{
				ccv_nnc_tensor_variable_t const input = inputs[i + t * per_input_size];
				if (!input)
					continue;
				ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, input->symbol.d);
				if (!bind->destinations)
					bind->destinations = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_add_unique_int(bind->destinations, graph_exec.d);
			}
			for (i = 0; i < per_output_size; i++)
			{
				ccv_nnc_tensor_variable_t const output = outputs[i + t * per_output_size];
				if (!output)
					continue;
				ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, output->symbol.d);
				assert(!bind->sources); // This is a new symbol, therefore, no binded sources associated yet.
				bind->sources = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_add_unique_int(bind->sources, graph_exec.d);
				if (output->alias_ref)
				{
						const int alias_ref = output->alias_ref - 1;
						assert(alias_ref >= 0);
						ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_ref);
						ccv_nnc_tensor_variable_graph_bind_t* const bind_to = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, variable_to->symbol.d);
						if (!bind_to->sources)
							bind_to->sources = ccv_array_new(sizeof(int), 1, 0);
						ccv_array_add_unique_int(bind_to->sources, graph_exec.d);
				}
			}
		}
	}
	// Now, able to free some of the reused outputs.
	for (i = 0; i < freeable_size; i++)
		ccv_nnc_tensor_variable_free(graph, freeables[i]);
}

int ccv_nnc_dynamic_graph_exec(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size, const int parallel, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_dynamic_graph_exec_ret(graph, cmd, hint, flags, inputs, input_size, outputs, output_size, parallel, stream_context, 0);
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_update_bind_destinations_when_free(ccv_nnc_dynamic_graph_t* const graph, const int freed_symbol_d, ccv_nnc_tensor_variable_graph_bind_t* const bind, const int tensor_index)
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
			_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
			ccv_nnc_tensor_symbol_free(graph->tape, (ccv_nnc_tensor_symbol_t){
				.d = tensor_index,
				.graph = graph->tape
			});
		}
	}
}

static void _ccv_nnc_update_bind_sources_when_free(ccv_nnc_dynamic_graph_t* const graph, const int freed_symbol_d, ccv_nnc_tensor_variable_graph_bind_t* const bind, const int tensor_index)
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
			_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
			ccv_nnc_tensor_symbol_free(graph->tape, (ccv_nnc_tensor_symbol_t){
				.d = tensor_index,
				.graph = graph->tape
			});
		}
	}
}

static void _ccv_nnc_update_bind_sources_destinations_when_free(ccv_nnc_dynamic_graph_t* const graph, const int freed_symbol_d, ccv_array_t* const binds, const int* const inputs, const int input_size, const int* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < input_size; i++)
		if (inputs[i] >= 0 && inputs[i] < binds->rnum)
		{
			ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, inputs[i]);
			_ccv_nnc_update_bind_destinations_when_free(graph, freed_symbol_d, bind, inputs[i]);
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(graph->tape, (ccv_nnc_tensor_symbol_t){
				.d = inputs[i],
				.graph = graph->tape
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
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(graph->tape, (ccv_nnc_tensor_symbol_t){
				.d = outputs[i],
				.graph = graph->tape
			});
			if (alias_to.d >= 0 && alias_to.d < binds->rnum)
				_ccv_nnc_update_bind_sources_when_free(graph, freed_symbol_d, (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, alias_to.d), alias_to.d);
		}
}

static void _ccv_nnc_stateful_exec_free(khash_t(stateful_exec)* const stateful_execs, const ccv_nnc_graph_exec_symbol_t symbol)
{
	if (!stateful_execs)
		return;
	assert(symbol.d >= 0);
	khiter_t k = kh_get(stateful_exec, stateful_execs, symbol.d);
	if (k == kh_end(stateful_execs))
		return;
	ccfree(kh_val(stateful_execs, k));
	kh_del(stateful_exec, stateful_execs, k);
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
							flag = (other_bind->index >= 0 && other_bind->type != CCV_NNC_TENSOR_CONSTANT) || (other_bind->sources && other_bind->sources->rnum > 0);
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
						_ccv_nnc_update_bind_sources_destinations_when_free(graph, symbol_d, graph->binds, inputs, input_size, outputs, output_size);
						const int* outgoings; int outgoing_size;
						ccv_nnc_graph_exec_symbol_to(graph->tape, symbol, &outgoings, &outgoing_size);
						for (j = 0; j < outgoing_size; j++)
							ccv_array_add_unique_int(ws, outgoings[j]);
						_ccv_nnc_stateful_exec_free(graph->stateful_execs, symbol);
						ccv_nnc_graph_exec_symbol_free(graph->tape, symbol);
					}
				}
			if (free_symbol)
			{
				// Nothing here requires this symbol, should be able to free it.
				_ccv_nnc_tensor_variable_graph_bind_free(graph, (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, tensor_variable->symbol.d), 1);
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
							flag = (other_bind->index >= 0 && other_bind->type != CCV_NNC_TENSOR_CONSTANT) || (other_bind->sources && other_bind->sources->rnum > 0);
						}
					// Went over all the inputs, it turns out no more inputs has other references, safe to remove.
					if (!flag)
					{
						const int* outputs; int output_size;
						ccv_nnc_graph_exec_symbol_io(graph->tape, symbol, 0, 0, &outputs, &output_size);
						_ccv_nnc_update_bind_sources_destinations_when_free(graph, symbol_d, graph->binds, inputs, input_size, outputs, output_size);
						const int* outgoings; int outgoing_size;
						ccv_nnc_graph_exec_symbol_to(graph->tape, symbol, &outgoings, &outgoing_size);
						// It it has outgoings, add that for further inspection.
						for (j = 0; j < outgoing_size; j++)
							ccv_array_add_unique_int(ws, outgoings[j]);
						_ccv_nnc_stateful_exec_free(graph->stateful_execs, symbol);
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
