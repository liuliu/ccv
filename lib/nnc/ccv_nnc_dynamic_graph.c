#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"

// MARK - Level-4 API

ccv_nnc_dynamic_graph_t* ccv_nnc_dynamic_graph_new(void)
{
	ccv_nnc_dynamic_graph_t* graph = ccmalloc(sizeof(ccv_nnc_dynamic_graph_t));
	graph->no_grad = 0;
	graph->reuse_var = -1;
	graph->vars = ccv_array_new(sizeof(ccv_nnc_tensor_variable_t), 1, 0);
	graph->binds = ccv_array_new(sizeof(ccv_nnc_tensor_variable_graph_bind_t), 1, 0);
	graph->tape = ccv_nnc_symbolic_graph_new();
	graph->xpu_alloc.mp_hdr = -1;
	graph->xpu_alloc.freed = kh_init(dy_str);
	graph->xpu_alloc.allocd = kh_init(dy_alloc);
	// These may not be used as frequent, init as needed.
	graph->stateful_execs = 0;
	graph->reuse_stateful_exec = -1;
	graph->stream_map = 0;
	graph->ws = 0;
	return graph;
}

static void _ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int zeroing)
{
	const int index = tensor_variable->index;
	if (tensor_variable->tensor_view)
	{
		if (tensor_variable->destructor_hook.func)
			tensor_variable->destructor_hook.func(graph, (ccv_nnc_tensor_t*)tensor_variable->tensor_view, tensor_variable->destructor_hook.context);
		if (!CCV_NNC_IS_EXTERN_TENSOR_VIEW(tensor_variable->tensor_view))
		{
			if (CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view))
				ccv_nnc_tensor_view_free(tensor_variable->tensor_view);
			else {
				if (!tensor_variable->alias_index_ref && // Return this memory to the graph.
					CCV_TENSOR_GET_MEMORY(tensor_variable->tensor_view->info.type) == CCV_TENSOR_GPU_MEMORY)
					ccv_nnc_xpu_free(&graph->xpu_alloc, tensor_variable->tensor_view->data.u8);
				ccv_nnc_tensor_free((ccv_nnc_tensor_t*)tensor_variable->tensor_view);
			}
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
	if (bind->tensor_view)
	{
		if (bind->destructor_hook.func)
			bind->destructor_hook.func(graph, (ccv_nnc_tensor_t*)bind->tensor_view, bind->destructor_hook.context);
		if (!CCV_NNC_IS_EXTERN_TENSOR_VIEW(bind->tensor_view))
		{
			if (CCV_IS_TENSOR_VIEW(bind->tensor_view))
				ccv_nnc_tensor_view_free(bind->tensor_view);
			else {
				if (!bind->alias_ref && // Return this memory to the graph.
					CCV_TENSOR_GET_MEMORY(bind->tensor_view->info.type) == CCV_TENSOR_GPU_MEMORY)
					ccv_nnc_xpu_free(&graph->xpu_alloc, bind->tensor_view->data.u8);
				ccv_nnc_tensor_free((ccv_nnc_tensor_t*)bind->tensor_view);
			}
		}
	}
	if (zeroing)
	{
		bind->sources = 0;
		bind->destinations = 0;
		bind->tensor_view = 0;
		bind->destructor_hook.func = 0;
		bind->destructor_hook.context = 0;
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
	if (graph->stateful_execs)
	{
		for (i = 0; i < graph->stateful_execs->rnum; i++)
		{
			ccv_nnc_stateful_exec_t* const stateful_exec = *(ccv_nnc_stateful_exec_t**)ccv_array_get(graph->stateful_execs, i);
			if (stateful_exec)
				ccfree(stateful_exec);
		}
		ccv_array_free(graph->stateful_execs);
	}
	if (graph->stream_map)
	{
		khiter_t k;
		for (k = kh_begin(graph->stream_map); k != kh_end(graph->stream_map); ++k)
		{
			if (!kh_exist(graph->stream_map, k))
				continue;
			ccv_nnc_stream_context_t* const stream = kh_val(graph->stream_map, k);
			ccv_nnc_stream_context_free(stream);
		}
		kh_destroy(stream_map, graph->stream_map);
	}
	ccv_nnc_xpu_alloc_destroy(&graph->xpu_alloc);
	ccfree(graph);
}

void ccv_nnc_tensor_variable_set(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, ccv_nnc_tensor_t* const tensor)
{
	assert(!tensor_variable->alias_index_ref);
	if (tensor_variable->tensor_view && !CCV_NNC_IS_EXTERN_TENSOR_VIEW(tensor_variable->tensor_view))
	{
		assert(!CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view));
		ccv_nnc_tensor_free((ccv_nnc_tensor_t*)tensor_variable->tensor_view);
	}
	tensor_variable->info = tensor->info;
	tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)((uintptr_t)tensor | 1);
}

void ccv_nnc_tensor_variable_destructor_hook(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, ccv_nnc_tensor_variable_destructor_f func, void* const context)
{
	tensor_variable->destructor_hook.func = func;
	tensor_variable->destructor_hook.context = context;
}

inline static void _ccv_nnc_tensor_variable_init(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable, const ccv_nnc_tensor_param_t info)
{
	tensor_variable->alias_index_ref = 0;
	tensor_variable->alias_off = 0;
	tensor_variable->destructor_hook.func = 0;
	tensor_variable->destructor_hook.context = 0;
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

int ccv_nnc_tensor_variable_is_constant(const ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	return tensor_variable->type == CCV_NNC_TENSOR_CONSTANT;
}

ccv_nnc_tensor_param_t ccv_nnc_tensor_variable_params(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	return tensor_variable->info;
}

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_alias_new(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info)
{
	ccv_nnc_tensor_variable_t variable_alias = ccmalloc(sizeof(struct ccv_nnc_tensor_variable_s));
	variable_alias->type = tensor_variable->type;
	// If the tensor variable is an alias itself, we point directly to its original.
	if (tensor_variable->alias_index_ref)
	{
		variable_alias->alias_index_ref = tensor_variable->alias_index_ref;
		// The tensor variable need to be fully specified if I am doing alias an alias.
		assert(!ccv_nnc_is_tensor_auto(tensor_variable->info));
		int i;
		int no_stride = 1;
		for (i = 0; no_stride && i < CCV_NNC_MAX_DIM_ALLOC; i++)
			no_stride = (tensor_variable->stride[i] == 0);
		int stride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		int* to_stride;
		if (no_stride)
		{
			ccv_nnc_tensor_get_stride(tensor_variable->info.dim, stride_from_dim);
			to_stride = stride_from_dim;
		} else
			to_stride = tensor_variable->stride;
		// If we provide stride, or reshape to a different size, assert the tensor variable itself is contiguous (otherwise we cannot satisfy the reshape requirements).
		const int different_dim = ccv_nnc_tensor_nd(info.dim) != ccv_nnc_tensor_nd(tensor_variable->info.dim);
		if (different_dim || (stride[0] != 0 && memcmp(stride, to_stride, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0))
			{ assert(ccv_nnc_tensor_view_is_contiguous(tensor_variable->info.dim, to_stride)); }
		// Need to compute alias off, that is the alias off of the tensor variable plus its ofs.
		const off_t off = ccv_nnc_tensor_view_offset(tensor_variable->info.datatype, to_stride, tensor_variable->ofs);
		variable_alias->alias_off = tensor_variable->alias_off + off;
		// If we don't provide stride, copy the stride from previous variable.
		if (stride[0] == 0)
		{
			if (different_dim)
				ccv_nnc_tensor_get_stride(info.dim, variable_alias->stride);
			else
				memcpy(variable_alias->stride, to_stride, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
		} else
			memcpy(variable_alias->stride, stride, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	} else {
		variable_alias->alias_index_ref = tensor_variable->index + 1;
		variable_alias->alias_off = 0;
		memcpy(variable_alias->stride, stride, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	}
	variable_alias->info = info;
	variable_alias->symbol = NO_TENSOR_SYMBOL;
	variable_alias->destructor_hook.func = 0;
	variable_alias->destructor_hook.context = 0;
	variable_alias->tensor_view = 0;
	memcpy(variable_alias->ofs, ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
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

int ccv_nnc_tensor_variable_alias_params(const ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, int ofs[CCV_NNC_MAX_DIM_ALLOC], int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	if (!tensor_variable->alias_index_ref)
		return -1;
	if (ofs)
		memcpy(ofs, tensor_variable->ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	if (stride)
		memcpy(stride, tensor_variable->stride, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	return 0;
}

ccv_nnc_tensor_t* ccv_nnc_tensor_from_variable_impl(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, ccv_nnc_stream_context_t* const stream_context)
{
	if (tensor_variable->tensor_view)
	{
		if (tensor_variable->alias_index_ref)
		{
			const int alias_index = tensor_variable->alias_index_ref - 1;
			assert(alias_index >= 0);
			ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_index);
			if (CCV_IS_TENSOR_VIEW(tensor_variable->tensor_view))
			{
				ccv_nnc_tensor_view_t* const tv = tensor_variable->tensor_view;
				// We cannot have an alias with custom set tensor, otherwise the pointer update is invalid.
				assert(!CCV_NNC_IS_EXTERN_TENSOR_VIEW(tv));
				// Update the tensor_view pointer every time access it, because the underlying variable it alias to have changed.
				ccv_nnc_tensor_data(CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->info, CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->data.u8, tv->off + tensor_variable->alias_off, &tv->data, &tv->dataof);
			} else {
				ccv_nnc_tensor_t* const tv = (ccv_nnc_tensor_t*)tensor_variable->tensor_view;
				// We cannot have an alias with custom set tensor, otherwise the pointer update is invalid.
				assert(!CCV_NNC_IS_EXTERN_TENSOR_VIEW(tv));
				// Update the tensor_view pointer every time access it, because the underlying variable it alias to have changed.
				ccv_nnc_tensor_data(CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->info, CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->data.u8, tensor_variable->alias_off, &tv->data, &tv->dataof);
			}
		}
		return (ccv_nnc_tensor_t*)CCV_NNC_TENSOR_VIEW(tensor_variable->tensor_view);
	}
	if (!tensor_variable->alias_index_ref)
	{
		// If we haven't allocated tensor_variable, we cannot allocate them now (because no shape specified), return 0.
		if (ccv_nnc_is_tensor_auto(tensor_variable->info))
			return 0;
		void* ptr = 0;
		if (CCV_TENSOR_GET_MEMORY(tensor_variable->info.type) == CCV_TENSOR_GPU_MEMORY)
			ptr = ccv_nnc_xpu_alloc(&graph->xpu_alloc, CCV_TENSOR_GET_DEVICE_ID(tensor_variable->info.type), stream_context, ccv_nnc_tensor_data_size(tensor_variable->info));
		tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(ptr, tensor_variable->info, 0);
		assert(tensor_variable->tensor_view->data.u8);
		return (ccv_nnc_tensor_t*)tensor_variable->tensor_view;
	}
	const int alias_index = tensor_variable->alias_index_ref - 1;
	assert(alias_index >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_index);
	assert(!variable_to->alias_index_ref);
	if (!variable_to->tensor_view)
	{
		// If we haven't allocated variable_to, we cannot allocate them now (because no shape specified), return 0.
		if (ccv_nnc_is_tensor_auto(variable_to->info))
			return 0;
		void* ptr = 0;
		assert(variable_to->info.type == tensor_variable->info.type);
		if (CCV_TENSOR_GET_MEMORY(variable_to->info.type) == CCV_TENSOR_GPU_MEMORY)
			ptr = ccv_nnc_xpu_alloc(&graph->xpu_alloc, CCV_TENSOR_GET_DEVICE_ID(variable_to->info.type), stream_context, ccv_nnc_tensor_data_size(variable_to->info));
		variable_to->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(ptr, variable_to->info, 0);
		assert(variable_to->tensor_view->data.u8);
	}
	int i;
	int no_ofs = 1;
	for (i = 0; no_ofs && i < CCV_NNC_MAX_DIM_ALLOC; i++)
		no_ofs = (tensor_variable->ofs[i] == 0);
	int no_stride = 1;
	for (i = 0; no_stride && i < CCV_NNC_MAX_DIM_ALLOC; i++)
		no_stride = (tensor_variable->stride[i] == 0);
	int stride_is_packed = no_stride;
	if (!no_stride) // We have stride, now if it is packed.
		stride_is_packed = ccv_nnc_is_tensor_stride_packed(tensor_variable->stride, tensor_variable->info.dim);
	assert(CCV_GET_DATA_TYPE_SIZE(tensor_variable->info.datatype) * ccv_nnc_tensor_count(tensor_variable->info) + tensor_variable->alias_off <= CCV_GET_DATA_TYPE_SIZE(variable_to->info.datatype) * ccv_nnc_tensor_count(variable_to->info));
	// Allowing vector type to be normal tensor, rather than a tensor view. We cannot have any offset though.
	if (no_ofs && !stride_is_packed)
		stride_is_packed = ccv_nnc_tensor_view_is_contiguous(tensor_variable->info.dim, tensor_variable->stride);
	if (no_ofs && stride_is_packed)
		tensor_variable->tensor_view = (ccv_nnc_tensor_view_t*)ccv_nnc_tensor_new(CCV_NNC_TENSOR_VIEW(variable_to->tensor_view)->data.u8, tensor_variable->info, 0);
	else {
		if (no_stride)
			ccv_nnc_tensor_get_stride(tensor_variable->info.dim, tensor_variable->stride);
		tensor_variable->tensor_view = ccv_nnc_tensor_view_new((ccv_nnc_tensor_t*)CCV_NNC_TENSOR_VIEW(variable_to->tensor_view), tensor_variable->info, tensor_variable->ofs, tensor_variable->stride);
	}
	if  (tensor_variable->alias_off)
		ccv_nnc_tensor_data_add(tensor_variable->tensor_view->info, tensor_variable->alias_off, &tensor_variable->tensor_view->data, &tensor_variable->tensor_view->dataof);
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
	if (tensor_variable->alias_index_ref)
	{
		const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(graph->tape, (ccv_nnc_tensor_symbol_t){
			.d = symbol.d,
			.graph = graph->tape
		});
		assert(alias_to.d >= 0 && alias_to.d < graph->binds->rnum);
		bind->alias_ref = alias_to.d + 1;
	} else
		bind->alias_ref = 0;
	if (bind->sources)
		ccv_array_free(bind->sources);
	bind->sources = 0;
	if (bind->destinations)
		ccv_array_free(bind->destinations);
	bind->destinations = 0;
	bind->destructor_hook.func = 0;
	bind->destructor_hook.context = 0;
	bind->tensor_view = 0;
}

static ccv_nnc_tensor_symbol_t _ccv_nnc_tensor_symbol_from_variable(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	if (tensor_variable->symbol.d >= 0)
		return tensor_variable->symbol;
	if (!tensor_variable->alias_index_ref)
	{
		const ccv_nnc_tensor_symbol_t symbol = tensor_variable->symbol = ccv_nnc_tensor_symbol_new(graph->tape, tensor_variable->info, 0);
		_ccv_nnc_tensor_symbol_extra_new(graph, tensor_variable, symbol);
		return symbol;
	}
	const int alias_index = tensor_variable->alias_index_ref - 1;
	assert(alias_index >= 0);
	ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_index);
	assert(!variable_to->alias_index_ref);
	int no_stride = 1;
	int i;
	for (i = 0; no_stride && i < CCV_NNC_MAX_DIM_ALLOC; i++)
		no_stride = (tensor_variable->stride[i] == 0);
	if (no_stride)
		ccv_nnc_tensor_get_stride(tensor_variable->info.dim, tensor_variable->stride);
	const ccv_nnc_tensor_symbol_t symbol = tensor_variable->symbol = ccv_nnc_tensor_symbol_alias_new(graph->tape, _ccv_nnc_tensor_symbol_from_variable(graph, variable_to), tensor_variable->ofs, tensor_variable->stride, tensor_variable->info, 0);
	_ccv_nnc_tensor_symbol_extra_new(graph, tensor_variable, symbol);
	return symbol;
}

// Return the tensor variable that is old (the provided tensor variable will have a new setting).
ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_exchange_new(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable)
{
	struct ccv_nnc_tensor_variable_s x = *tensor_variable;
	ccv_nnc_tensor_variable_t new_variable;
	// Need to handle alias.
	if (x.alias_index_ref)
		new_variable = ccv_nnc_tensor_variable_alias_new(graph, *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, x.alias_index_ref - 1), x.ofs, x.stride, x.info);
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

void ccv_nnc_dynamic_graph_set_max_concurrency(ccv_nnc_dynamic_graph_t* const dynamic_graph, const int max_stream_count)
{
	dynamic_graph->max_stream_count = max_stream_count;
}

int ccv_nnc_dynamic_graph_set_no_grad(ccv_nnc_dynamic_graph_t* const dynamic_graph, const int no_grad)
{
	if (dynamic_graph->no_grad == no_grad)
		return -1;
	dynamic_graph->no_grad = no_grad;
	return 0;
}

static ccv_nnc_stream_context_t* _ccv_nnc_dynamic_graph_get_stream(ccv_nnc_dynamic_graph_t* const graph, const int type)
{
	if (!graph->stream_map)
		graph->stream_map = kh_init(stream_map);
	int ret = 0;
	khiter_t k = kh_put(stream_map, graph->stream_map, type, &ret);
	assert(ret >= 0);
	ccv_nnc_stream_context_t* stream = kh_val(graph->stream_map, k);
	// If ret == 0, the key already exist, we can return directly, otherwise, create and return.
	if (ret != 0)
	{
		stream = ccv_nnc_stream_context_new(type);
		kh_val(graph->stream_map, k) = stream;
	}
	return stream;
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
	return _ccv_nnc_dynamic_graph_get_stream(discovery->graph, type);
}

void ccv_nnc_dynamic_graph_exec_ret(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size, const int parallel, ccv_nnc_stream_context_t* const stream_context, ccv_nnc_graph_exec_symbol_t* const graph_execs)
{
	int i, j;
	for (i = 0; i < input_size; i++)
		if (inputs[i] && !inputs[i]->alias_index_ref)
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
		if (inputs[i] && inputs[i]->alias_index_ref)
		{
			const int alias_index_ref = inputs[i]->alias_index_ref - 1;
			assert(alias_index_ref >= 0);
			ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_index_ref);
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
		// We don't allow or check "allow inplace" yet. That logic will be at odds with backward logic.
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
					outputs[i]->destructor_hook = old_var->destructor_hook;
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
			signal = ccv_nnc_stream_context_emit_signal_new(stream_context);
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
			ccv_nnc_stream_context_t* stream_0 = 0;
			for (j = 0; j < device_id_size; j++)
			{
				int type = stream_type;
				CCV_STREAM_SET_DEVICE_ID(type, device_ids[j]);
				ccv_nnc_stream_context_t* const stream = _ccv_nnc_dynamic_graph_get_stream(graph, type);
				if (!stream_0)
					stream_0 = stream;
			}
			// Wait signal to finish.
			if (stream_context)
			{
				if (stream_0)
					ccv_nnc_stream_context_wait_signal(stream_0, signal);
				else
					ccv_nnc_stream_context_wait(stream_context);
			}
			if (stream_0)
			{
				ccv_nnc_dynamic_graph_neighbor_context_discovery_t discovery = {
					.graph = graph,
					.stream_type = stream_type
				};
				ccv_nnc_stream_context_set_neighbor_discovery(stream_0, _ccv_nnc_dynamic_graph_neighbor_context_discovery, &discovery);
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
			ccv_nnc_cmd_exec(cmd, hint, flags, input_tensors + i * per_input_size, per_input_size, output_tensors, per_output_size, stream_0);
			for (k = 0; k < per_output_size; k++)
			{
				PRINT(CCV_CLI_INFO, "|<- %d. %p (%p:%d)", k + 1, output_tensors[k], (output_tensors[k] ? output_tensors[k]->data.u8 : 0), (output_tensors[k] ? CCV_TENSOR_GET_DEVICE_ID(output_tensors[k]->info.type) : -1));
				if (output_tensors[k] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_INFO))
					ccv_nnc_print_tensor_info(output_tensors[k]);
				PRINT(CCV_CLI_INFO, "\n");
			}
			if (stream_context && stream_0)
			{
				ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_emit_signal_new(stream_0);
				ccv_nnc_stream_context_wait_signal(stream_context, signal);
			}
			streams[i] = stream_0;
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
	int inputs_are_constants = 1;
	for (i = 0; inputs_are_constants && i < input_size; i++)
		if (inputs[i] && inputs[i]->type != CCV_NNC_TENSOR_CONSTANT)
			inputs_are_constants = 0;
	if (input_size > 0 && !inputs_are_constants && !graph->no_grad) // No need to record the execution if there is no input or we disabled gradient computation.
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
				if (!input || input->type == CCV_NNC_TENSOR_CONSTANT)
					continue;
				ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, input_symbols[i + t * per_input_size].d);
				if (!bind->destinations)
					bind->destinations = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_add_unique_int(bind->destinations, graph_exec.d);
				if (input->alias_index_ref)
				{
						const int alias_index = input->alias_index_ref - 1;
						assert(alias_index >= 0);
						ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_index);
						ccv_nnc_tensor_variable_graph_bind_t* const root_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, variable_to->symbol.d);
						if (!root_bind->destinations)
							root_bind->destinations = ccv_array_new(sizeof(int), 1, 0);
						ccv_array_add_unique_int(root_bind->destinations, graph_exec.d);
				}
			}
			for (i = 0; i < per_output_size; i++)
			{
				ccv_nnc_tensor_variable_t const output = outputs[i + t * per_output_size];
				if (!output)
					continue;
				ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, output_symbols[i + t * per_output_size].d);
				assert(!bind->sources); // This is a new symbol, therefore, no binded sources associated yet.
				bind->sources = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_add_unique_int(bind->sources, graph_exec.d);
				if (output->alias_index_ref)
				{
					const int alias_index = output->alias_index_ref - 1;
					assert(alias_index >= 0);
					ccv_nnc_tensor_variable_t variable_to = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, alias_index);
					ccv_nnc_tensor_variable_graph_bind_t* const root_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, variable_to->symbol.d);
					if (!root_bind->sources)
						root_bind->sources = ccv_array_new(sizeof(int), 1, 0);
					ccv_array_add_unique_int(root_bind->sources, graph_exec.d);
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

static int _ccv_nnc_tensor_variable_is_only_output(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_graph_bind_t* bind, const int symbol_d)
{
	if (bind->alias_ref)
		bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, bind->alias_ref - 1);
	if (!bind->sources || bind->sources->rnum == 0)
		return 1;
	int i;
	for (i = 0; i < bind->sources->rnum; i++)
	{
		const int exec_symbol_d = *(int*)ccv_array_get(bind->sources, i);
		const ccv_nnc_graph_exec_symbol_t exec_symbol = {
			.d = exec_symbol_d,
			.graph = graph->tape
		};
		const int* outputs; int output_size;
		ccv_nnc_graph_exec_symbol_io(graph->tape, exec_symbol, 0, 0, &outputs, &output_size);
		int j;
		for (j = 0; j < output_size; j++)
			if (outputs[j] >= 0 && outputs[j] != symbol_d) // If output is me, it is the only output.
			{
				assert(outputs[j] < graph->binds->rnum);
				const ccv_nnc_tensor_variable_graph_bind_t* other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, outputs[j]);
				// This is in use and is it not a constant symbol.
				if (other_bind->index >= 0 && other_bind->type != CCV_NNC_TENSOR_CONSTANT)
					return 0;
				if (other_bind->alias_ref) // If this is alias, use its original's destinations.
					other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, other_bind->alias_ref - 1);
				if (other_bind->destinations && other_bind->destinations->rnum > 0)
					return 0;
			}
	}
	return 1;
}

static void _ccv_nnc_update_bind_destinations_when_free(ccv_nnc_dynamic_graph_t* const graph, const int freed_exec_symbol_d, ccv_array_t* const binds, ccv_nnc_tensor_variable_graph_bind_t* const bind, const int tensor_index, ccv_array_t* const ws)
{
	int i;
	if (bind->destinations)
	{
		int flag = 0;
		for (i = 0; !flag && i < bind->destinations->rnum; i++)
		{
			const int exec_symbol_d = *(int*)ccv_array_get(bind->destinations, i);
			if (exec_symbol_d == freed_exec_symbol_d)
			{
				if (i < bind->destinations->rnum - 1)
					*(int*)ccv_array_get(bind->destinations, i) = *(int*)ccv_array_get(bind->destinations, bind->destinations->rnum - 1);
				--bind->destinations->rnum;
				flag = 1;
			}
		}
		// This symbol can be freed.
		if (flag && bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED)
		{
			ccv_nnc_tensor_variable_graph_bind_t* root_bind = bind;
			if (bind->alias_ref)
			{
				root_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, bind->alias_ref - 1);
				if (root_bind->index == CCV_NNC_TENSOR_NO_VARIABLE)
					root_bind = bind;
			}
			// If the alias_ref is not freed, we cannot free this, unless it is very clear there is no reference to this any more.
			// It is possible because exec will be freed already, thus, it is safe to remove this alias out.
			if (root_bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED &&
				((!root_bind->sources || root_bind->sources->rnum == 0) || _ccv_nnc_tensor_variable_is_only_output(graph, bind, tensor_index)) &&
				root_bind->destinations->rnum == 0)
			{
				if (root_bind->sources)
					for (i = 0; i < root_bind->sources->rnum; i++)
						ccv_array_add_unique_int(ws, *(int*)ccv_array_get(root_bind->sources, i));
				_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
				ccv_nnc_tensor_symbol_free(graph->tape, (ccv_nnc_tensor_symbol_t){
					.d = tensor_index,
					.graph = graph->tape
				});
			} else if (bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED && // Handle the case the bind is already freed, and it doesn't have any sources or destinations.
				bind->alias_ref && (!bind->sources || bind->sources->rnum == 0) && (!bind->destinations || bind->destinations->rnum == 0)) {
				_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
				ccv_nnc_tensor_symbol_free(graph->tape, (ccv_nnc_tensor_symbol_t){
					.d = tensor_index,
					.graph = graph->tape
				});
			}
		}
	}
}

static void _ccv_nnc_update_bind_sources_when_free(ccv_nnc_dynamic_graph_t* const graph, const int freed_exec_symbol_d, ccv_array_t* const binds, ccv_nnc_tensor_variable_graph_bind_t* const bind, const int tensor_index, ccv_array_t* const ws)
{
	int i;
	if (bind->sources)
	{
		int flag = 0;
		for (i = 0; !flag && i < bind->sources->rnum; i++)
		{
			const int exec_symbol_d = *(int*)ccv_array_get(bind->sources, i);
			if (exec_symbol_d == freed_exec_symbol_d)
			{
				if (i < bind->sources->rnum - 1)
					*(int*)ccv_array_get(bind->sources, i) = *(int*)ccv_array_get(bind->sources, bind->sources->rnum - 1);
				--bind->sources->rnum;
				flag = 1;
			}
		}
		if (flag && !bind->alias_ref && bind->index >= 0 && bind->type == CCV_NNC_TENSOR_CONSTANT && // If it is detached (constant but previously has sources). Now can check again.
			(bind->sources->rnum == 0 || _ccv_nnc_tensor_variable_is_only_output(graph, bind, tensor_index)) &&
			(!bind->destinations || bind->destinations->rnum == 0))
		{
			// If this is constant, set it to be no symbol again.
			ccv_nnc_tensor_variable_t tv = *(ccv_nnc_tensor_variable_t*)ccv_array_get(graph->vars, bind->index);
			tv->symbol = NO_TENSOR_SYMBOL;
			_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
			ccv_nnc_tensor_symbol_free(graph->tape, (ccv_nnc_tensor_symbol_t){
				.d = tensor_index,
				.graph = graph->tape
			});
		} else if (flag && bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED) {
			// This symbol can be freed.
			ccv_nnc_tensor_variable_graph_bind_t* root_bind = bind;
			if (bind->alias_ref)
			{
				root_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, bind->alias_ref - 1);
				if (root_bind->index == CCV_NNC_TENSOR_NO_VARIABLE)
					root_bind = bind;
			}
			// If the alias_ref is not freed, we cannot free this, unless it is very clear there is no reference to this any more.
			// It is possible because exec will be freed already, thus, it is safe to remove this alias out.
			if (root_bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED &&
				(root_bind->sources->rnum == 0 || _ccv_nnc_tensor_variable_is_only_output(graph, bind, tensor_index)) &&
				(!root_bind->destinations || root_bind->destinations->rnum == 0))
			{
				for (i = 0; i < root_bind->sources->rnum; i++)
					ccv_array_add_unique_int(ws, *(int*)ccv_array_get(root_bind->sources, i));
				_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
				ccv_nnc_tensor_symbol_free(graph->tape, (ccv_nnc_tensor_symbol_t){
					.d = tensor_index,
					.graph = graph->tape
				});
			} else if (bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED && // Handle the case the bind is already freed, and it doesn't have any sources or destinations.
				bind->alias_ref && (!bind->sources || bind->sources->rnum == 0) && (!bind->destinations || bind->destinations->rnum == 0)) {
				_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
				ccv_nnc_tensor_symbol_free(graph->tape, (ccv_nnc_tensor_symbol_t){
					.d = tensor_index,
					.graph = graph->tape
				});
			}
		}
	}
}

static void _ccv_nnc_update_bind_sources_destinations_when_free(ccv_nnc_dynamic_graph_t* const graph, const int freed_exec_symbol_d, ccv_array_t* const binds, const int* const inputs, const int input_size, const int* const outputs, const int output_size, ccv_array_t* const ws)
{
	int i;
	for (i = 0; i < input_size; i++)
		if (inputs[i] >= 0 && inputs[i] < binds->rnum)
		{
			ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, inputs[i]);
			if (bind->index == CCV_NNC_TENSOR_NO_VARIABLE)
				continue;
			if (bind->alias_ref)
			{
				const int alias_to = bind->alias_ref - 1;
				ccv_nnc_tensor_variable_graph_bind_t* const root_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, alias_to);
				if (root_bind && root_bind->index != CCV_NNC_TENSOR_NO_VARIABLE)
					_ccv_nnc_update_bind_destinations_when_free(graph, freed_exec_symbol_d, binds, root_bind, alias_to, ws);
			}
			_ccv_nnc_update_bind_destinations_when_free(graph, freed_exec_symbol_d, binds, bind, inputs[i], ws);
		}
	// Note that this works because there is no overlap of inputs / outputs. (What about alias?).
	for (i = 0; i < output_size; i++)
		if (outputs[i] >= 0 && outputs[i] < binds->rnum)
		{
			ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, outputs[i]);
			if (bind->index == CCV_NNC_TENSOR_NO_VARIABLE)
				continue;
			if (bind->alias_ref)
			{
				const int alias_to = bind->alias_ref - 1;
				ccv_nnc_tensor_variable_graph_bind_t* const root_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(binds, alias_to);
				if (root_bind && root_bind->index != CCV_NNC_TENSOR_NO_VARIABLE)
					_ccv_nnc_update_bind_sources_when_free(graph, freed_exec_symbol_d, binds, root_bind, alias_to, ws);
			}
			_ccv_nnc_update_bind_sources_when_free(graph, freed_exec_symbol_d, binds, bind, outputs[i], ws);
		}
}

static void _ccv_nnc_stateful_exec_free_if_possible(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol)
{
	if (!graph->stateful_execs)
		return;
	assert(symbol.d >= 0);
	ccv_array_t* const stateful_execs = graph->stateful_execs;
	ccv_nnc_cmd_t cmd = ccv_nnc_graph_exec_symbol_cmd(graph->tape, symbol);
	ccv_nnc_stateful_exec_t* const stateful_exec = (ccv_nnc_stateful_exec_t*)cmd.data;
	if (!stateful_exec)
		return;
	// If there is no backward, no need to apply gradients.
	// Otherwise, if we applied gradients, we can free it as well.
	// We don't free this stateful exec because apply gradients doesn't require any variables alive.
	if (!stateful_exec->did_backward_but_not_apply_gradients)
	{
		const int index = stateful_exec->index;
		ccfree(stateful_exec);
		if (index < graph->reuse_stateful_exec || graph->reuse_stateful_exec < 0)
			graph->reuse_stateful_exec = index;
		*(ccv_nnc_stateful_exec_t**)ccv_array_get(stateful_execs, index) = 0;
	} else
		stateful_exec->should_free = 1;
}

static int _ccv_nnc_tensor_bind_trace_forward_to_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable, ccv_nnc_tensor_variable_graph_bind_t* const bind, ccv_nnc_tensor_variable_graph_bind_t* const root_bind, int* const ws_start, const int assuming_no_source) // assuming_no_source means we are going to remove sources if possible, thus, it is irrelevant.
{
	int can_free_symbol = 0;
	const int sources_and_is_only_output = (root_bind->sources && root_bind->sources->rnum > 0) && _ccv_nnc_tensor_variable_is_only_output(graph, bind, tensor_variable->symbol.d);
	if (!root_bind->sources || root_bind->sources->rnum == 0 || sources_and_is_only_output || assuming_no_source)
	{
		int i, j;
		can_free_symbol = 1; // Assume we can free this symbol.
		if (!graph->ws)
			graph->ws = ccv_array_new(sizeof(int), root_bind->destinations ? root_bind->destinations->rnum : 0, 0);
		ccv_array_t* const ws = graph->ws;
		ccv_array_clear(ws);
		if (root_bind->destinations)
			for (i = 0; i < root_bind->destinations->rnum; i++)
				ccv_array_add_unique_int(ws, *(int*)ccv_array_get(root_bind->destinations, i));
		const int ws_init_size = ws->rnum;
		*ws_start = ws_init_size;
		// Add all sources from root_bind, in case it has been freed (during update bind sources / destinations when free.
		if (root_bind->sources)
			for (i = 0; i < root_bind->sources->rnum; i++)
				ccv_array_add_unique_int(ws, *(int*)ccv_array_get(root_bind->sources, i));
		// If we cannot loop over any exec symbols (this is not in use). It is simple to determine whether we want
		// to free it or not: if this is an alias and the origin is not freed, we cannot free this symbol.
		if (ws_init_size == 0)
			can_free_symbol = (!bind->alias_ref || root_bind->index < 0);
		// Go through all the exec symbols use this tensor, to see whether they have inputs that has other sources.
		for (i = 0; i < ws_init_size; i++)
		{
			const int exec_symbol_d = *(int*)ccv_array_get(ws, i);
			const ccv_nnc_graph_exec_symbol_t symbol = {
				.d = exec_symbol_d,
				.graph = graph->tape
			};
			const int* inputs; int input_size;
			const int* outputs; int output_size;
			ccv_nnc_graph_exec_symbol_io(graph->tape, symbol, &inputs, &input_size, &outputs, &output_size);
			int flag = 0; // flag denotes whether there are cases to keep this exec symbol.
			if (!root_bind->sources || root_bind->sources->rnum == 0 || assuming_no_source)
			{
				// If there is no sources, check if other sources can depend on this exec, if they do, we cannot free this.
				for (j = 0; !flag && j < input_size; j++)
					if (inputs[j] >= 0 && inputs[j] < graph->binds->rnum && inputs[j] != tensor_variable->symbol.d)
					{
						ccv_nnc_tensor_variable_graph_bind_t* other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, inputs[j]);
						if (other_bind->index >= 0 && other_bind->type != CCV_NNC_TENSOR_CONSTANT)
							flag = 1;
						else {
							if (other_bind->alias_ref) // If this is alias, use its original's destinations.
								other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, other_bind->alias_ref - 1);
							flag = (other_bind->type != CCV_NNC_TENSOR_CONSTANT) && (other_bind->sources && other_bind->sources->rnum > 0); // Constant should have no source, or it is detached.
						}
					}
			} else {
				// If there are sources, check whether we have outputs or not. If we do, we cannot free this.
				for (j = 0; !flag && j < output_size; j++)
					if (outputs[j] >= 0 && outputs[j] < graph->binds->rnum)
					{
						ccv_nnc_tensor_variable_graph_bind_t* other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, outputs[j]);
						if (other_bind->index >= 0 && other_bind->type != CCV_NNC_TENSOR_CONSTANT)
							flag = 1;
						else {
							if (other_bind->alias_ref) // If this is alias, use its original's destinations.
								other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, other_bind->alias_ref - 1);
							flag = (other_bind->destinations && other_bind->destinations->rnum > 0);
						}
					}
			}
			// This exec can be freed if there is no input required or there is no output required.
			can_free_symbol = (can_free_symbol && !flag);
			if (!flag)
			{
				// Go over inputs and remove all references from binded destinations.
				// and go over outputs remove all references from binded sources.
				_ccv_nnc_update_bind_sources_destinations_when_free(graph, exec_symbol_d, graph->binds, inputs, input_size, outputs, output_size, ws);
				const int* outgoings; int outgoing_size;
				ccv_nnc_graph_exec_symbol_to(graph->tape, symbol, &outgoings, &outgoing_size);
				for (j = 0; j < outgoing_size; j++)
					ccv_array_add_unique_int(ws, outgoings[j]);
				_ccv_nnc_stateful_exec_free_if_possible(graph, symbol);
				ccv_nnc_graph_exec_symbol_free(graph->tape, symbol);
			}
		}
	}
	return can_free_symbol;
}

static void _ccv_nnc_tensor_bind_trace_backward_to_free(ccv_nnc_dynamic_graph_t* const graph, ccv_array_t* const ws, const int ws_start)
{
	int i, j;
	// Now, go over the outgoings, if it is removed, add more to it. Note that the ws array can grow while iterating over.
	for (i = ws_start; i < ws->rnum; i++)
	{
		const int exec_symbol_d = *(int*)ccv_array_get(ws, i);
		const ccv_nnc_graph_exec_symbol_t symbol = {
			.d = exec_symbol_d,
			.graph = graph->tape
		};
		const int* inputs; int input_size;
		const int* outputs; int output_size;
		ccv_nnc_graph_exec_symbol_io(graph->tape, symbol, &inputs, &input_size, &outputs, &output_size);
		int flag = 0;
		for (j = 0; !flag && j < input_size; j++)
			if (inputs[j] >= 0 && inputs[j] < graph->binds->rnum)
			{
				ccv_nnc_tensor_variable_graph_bind_t* other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, inputs[j]);
				if (other_bind->index >= 0 && other_bind->type != CCV_NNC_TENSOR_CONSTANT)
					flag = 1;
				else {
					if (other_bind->alias_ref) // If this is alias, use its original's destinations.
						other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, other_bind->alias_ref - 1);
					flag = (other_bind->type != CCV_NNC_TENSOR_CONSTANT) && (other_bind->sources && other_bind->sources->rnum > 0);
				}
			}
		if (flag) // If any inputs make free this destination impossible. Check whether all its outputs are done.
		{
			int output_flag = 0;
			for (j = 0; !output_flag && j < output_size; j++)
				if (outputs[j] >= 0 && outputs[j] < graph->binds->rnum)
				{
					ccv_nnc_tensor_variable_graph_bind_t* other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, outputs[j]);
					if (other_bind->index >= 0 && other_bind->type != CCV_NNC_TENSOR_CONSTANT)
						output_flag = 1;
					else {
						if (other_bind->alias_ref) // If this is alias, use its original's destinations.
							other_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, other_bind->alias_ref - 1);
						output_flag = (other_bind->destinations && other_bind->destinations->rnum > 0);
					}
				}
			if (!output_flag) // If no output is used (used means it has a tensor variable, or it has a destination).
				flag = 0;
		}
		// Went over all the inputs, it turns out no more inputs has other references, safe to remove.
		if (!flag)
		{
			_ccv_nnc_update_bind_sources_destinations_when_free(graph, exec_symbol_d, graph->binds, inputs, input_size, outputs, output_size, ws);
			const int* outgoings; int outgoing_size;
			ccv_nnc_graph_exec_symbol_to(graph->tape, symbol, &outgoings, &outgoing_size);
			// It it has outgoings, add that for further inspection.
			for (j = 0; j < outgoing_size; j++)
				ccv_array_add_unique_int(ws, outgoings[j]);
			_ccv_nnc_stateful_exec_free_if_possible(graph, symbol);
			ccv_nnc_graph_exec_symbol_free(graph->tape, symbol);
		}
	}
}

void ccv_nnc_tensor_variable_free(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	// If it contains a symbol, this tensor variable is not a free variable. It is either used as input or output.
	if (tensor_variable->symbol.d != CCV_NNC_NO_TENSOR_SYMBOL)
	{
		// If it is not a free variable, when can we free the symbol and the underlying variable?
		// 1. There should be no sources (the command generate this tensor should be freed) or the output of these sources is only the current one;
		// 2. The destinations (the commands that uses this tensor) should have no other inputs, or the other inputs has no binded sources as well.
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, tensor_variable->symbol.d);
		// There should be no source associated with it no more.
		// I am free if no exec symbol is producing me or the symbol producing me can only producing me (thus, it is not required to
		// compute gradient because I am the only variable it can compute gradient for).
		ccv_nnc_tensor_variable_graph_bind_t* root_bind = bind;
		if (bind->alias_ref)
		{
			const int alias_to = bind->alias_ref - 1;
			root_bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, alias_to);
		}
		int ws_start;
		const int can_free_symbol = _ccv_nnc_tensor_bind_trace_forward_to_free(graph, tensor_variable, bind, root_bind, &ws_start, 0);
		if (can_free_symbol)
		{
			_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
			ccv_nnc_tensor_symbol_free(graph->tape, tensor_variable->symbol);
			_ccv_nnc_tensor_bind_trace_backward_to_free(graph, graph->ws, ws_start);
		} else { // If this symbol is not freed, move the tensor view to the bind.
			// If current bind is an alias, and it doesn't have any sources or destinations. We cannot find this alias
			// through any exec. This is not only safe to delete, but has to be deleted. We don't need to handle this
			// if free_symbol is true, because when that happens, root_bind will be deleted, and we will clean up the
			// alias in that process.
			if (bind->alias_ref && (!bind->sources || bind->sources->rnum == 0) && (!bind->destinations || bind->destinations->rnum == 0))
			{
				_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
				ccv_nnc_tensor_symbol_free(graph->tape, tensor_variable->symbol);
			} else {
				bind->index = CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED; // This tensor variable will be freed, but this symbol extra will continue exists.
				bind->destructor_hook.func = tensor_variable->destructor_hook.func; // Transfer the destructor callback.
				bind->destructor_hook.context = tensor_variable->destructor_hook.context; // Transfer the destructor callback context.
				bind->tensor_view = tensor_variable->tensor_view; // Transfer the ownership to the bind.
				tensor_variable->tensor_view = 0;
			}
		}
	}
	_ccv_nnc_tensor_variable_free(graph, tensor_variable, 1);
}

void ccv_nnc_tensor_variable_detach(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t tensor_variable)
{
	// This cannot be an alias.
	assert(!tensor_variable->alias_index_ref);
	// If no computation done yet, mark this as constant.
	if (tensor_variable->symbol.d == CCV_NNC_NO_TENSOR_SYMBOL)
	{
		tensor_variable->type = CCV_NNC_TENSOR_CONSTANT;
		return;
	}
	// Otherwise, we need to do some book keeping updates to make sure it doesn't participate gradient computation any more.
	ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, tensor_variable->symbol.d);
	// Because tensor variable cannot be alias, its bind cannot have alias pointer.
	assert(!bind->alias_ref);
	// Go through to break ties between sources and destinations.
	int i, j;
	if (bind->sources && bind->destinations)
	{
		for (i = 0; i < bind->sources->rnum; i++)
		{
			const int s = *(int*)ccv_array_get(bind->sources, i);
			const int* outputs; int output_size;
			const ccv_nnc_graph_exec_symbol_t s_symbol = {
				.d = s,
				.graph = graph->tape
			};
			ccv_nnc_graph_exec_symbol_io(graph->tape, s_symbol, 0, 0, &outputs, &output_size);
			for (j = 0; j < bind->destinations->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(bind->destinations, j);
				const ccv_nnc_graph_exec_symbol_t d_symbol = {
					.d = d,
					.graph = graph->tape
				};
				const int* inputs; int input_size;
				ccv_nnc_graph_exec_symbol_io(graph->tape, d_symbol, &inputs, &input_size, 0, 0);
				int x, y;
				int flag = 0; // Whether we find a symbol that connects source and destination but not the current one we detach. If found, we cannot break the tie between s_symbol and d_symbol.
				for (x = 0; !flag && x < output_size; x++)
				{
					ccv_nnc_tensor_symbol_t x_symbol = ccv_nnc_tensor_symbol_alias_to(graph->tape, (ccv_nnc_tensor_symbol_t){
						.d = outputs[x],
						.graph = graph->tape
					});
					if (x_symbol.d == CCV_NNC_NO_TENSOR_SYMBOL)
					{
						x_symbol.d = outputs[x];
						x_symbol.graph = graph->tape;
					}
					if (x_symbol.d == tensor_variable->symbol.d || x_symbol.d == CCV_NNC_NO_TENSOR_SYMBOL)
						continue;
					for (y = 0; !flag && y < input_size; y++)
					{
						ccv_nnc_tensor_symbol_t y_symbol = ccv_nnc_tensor_symbol_alias_to(graph->tape, (ccv_nnc_tensor_symbol_t){
							.d = inputs[y],
							.graph = graph->tape
						});
						if (y_symbol.d == CCV_NNC_NO_TENSOR_SYMBOL)
						{
							y_symbol.d = inputs[y];
							y_symbol.graph = graph->tape;
						}
						if (y_symbol.d == tensor_variable->symbol.d || y_symbol.d == CCV_NNC_NO_TENSOR_SYMBOL)
							continue;
						flag = (x_symbol.d == y_symbol.d);
					}
				}
				if (!flag)
					ccv_nnc_graph_exec_symbol_disjoin(graph->tape, s_symbol, d_symbol);
			}
		}
	}
	const int sources_and_is_only_output = (bind->sources && bind->sources->rnum > 0) && _ccv_nnc_tensor_variable_is_only_output(graph, bind, tensor_variable->symbol.d);
	if (!bind->sources || bind->sources->rnum == 0 || sources_and_is_only_output)
	{
		int ws_start = -1;
		_ccv_nnc_tensor_bind_trace_forward_to_free(graph, tensor_variable, bind, bind, &ws_start, 1);
		// Because we are detaching from the graph, there is no need to forward trace to see if it is not used and
		// then to remove the source execs. We can remove them right now, breaking the graph in two. That is why
		// we called trace backward to free regardless the outcome of the forward to free.
		if (ws_start == -1)
		{
			if (!graph->ws)
				graph->ws = ccv_array_new(sizeof(int), bind->destinations ? bind->destinations->rnum : 0, 0);
			ccv_array_t* const ws = graph->ws;
			ccv_array_clear(ws);
			if (bind->sources)
				for (i = 0; i < bind->sources->rnum; i++)
					ccv_array_add_unique_int(ws, *(int*)ccv_array_get(bind->sources, i));
			ws_start = 0;
		}
		_ccv_nnc_tensor_bind_trace_backward_to_free(graph, graph->ws, ws_start);
	}
	// If now bind has no relevant sources or destinations, we can safely free the underlying tensor symbol.
	if ((!bind->sources || bind->sources->rnum == 0) && (!bind->destinations || bind->destinations->rnum == 0))
	{
		_ccv_nnc_tensor_variable_graph_bind_free(graph, bind, 1);
		ccv_nnc_tensor_symbol_free(graph->tape, tensor_variable->symbol);
		tensor_variable->type = CCV_NNC_TENSOR_CONSTANT;
		tensor_variable->symbol = NO_TENSOR_SYMBOL;
		return;
	}
	// Mark both as constant, such that even if it cannot be freed now, it can be freed as soon as possible later.
	bind->type = CCV_NNC_TENSOR_CONSTANT;
	tensor_variable->type = CCV_NNC_TENSOR_CONSTANT;
}

void ccv_nnc_dynamic_graph_has_effect_to_tensor_variables(const ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_variable_t* const source_variables, const int source_variable_size, const ccv_nnc_tensor_variable_t* const destination_variables, const int destination_variable_size, uint64_t* const bitmask)
{
	int i, j;
	ccv_array_t* const sources_destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), source_variable_size + destination_variable_size, 0);
	for (i = 0; i < source_variable_size; i++)
	{
		if (source_variables[i]->symbol.d < 0)
			continue;
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, source_variables[i]->symbol.d);
		if (bind->destinations && bind->destinations->rnum > 0)
			for (j = 0; j < bind->destinations->rnum; j++)
			{
				// It is ok to have duplicate symbols.
				const int d = *(int*)ccv_array_get(bind->destinations, j);
				ccv_nnc_graph_exec_symbol_t symbol = {
					.d = d,
					.graph = graph->tape
				};
				ccv_array_push(sources_destinations, &symbol);
			}
	}
	const int source_size = sources_destinations->rnum;
	for (i = 0; i < destination_variable_size; i++)
	{
		if (destination_variables[i]->symbol.d < 0)
			continue;
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, destination_variables[i]->symbol.d);
		if (bind->sources && bind->sources->rnum > 0)
			for (j = 0; j < bind->sources->rnum; j++)
			{
				// It is ok to have duplicate symbols.
				const int d = *(int*)ccv_array_get(bind->sources, j);
				ccv_nnc_graph_exec_symbol_t symbol = {
					.d = d,
					.graph = graph->tape
				};
				ccv_array_push(sources_destinations, &symbol);
			}
	}
	const int destination_size = sources_destinations->rnum - source_size;
	if (source_size == 0 || destination_size == 0)
	{
		ccv_array_free(sources_destinations);
		return;
	}
	const int bitmask_size = ((source_size + 63) >> 6);
	assert(bitmask_size < 256);
	uint64_t exec_bitmask[bitmask_size];
	ccv_nnc_symbolic_graph_sources_to_destinations(graph->tape, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources_destinations, 0), source_size, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources_destinations, source_size), destination_size, exec_bitmask);
	int k = 0;
	for (i = 0; i < source_variable_size; i++)
	{
		if (source_variables[i]->symbol.d < 0)
		{
			bitmask[i >> 6] &= ~((uint64_t)1 << (i & 63));
			continue;
		}
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(graph->binds, source_variables[i]->symbol.d);
		int flag = 0;
		if (bind->destinations && bind->destinations->rnum > 0)
		{
			assert(k <= source_size - bind->destinations->rnum);
			for (j = 0; !flag && j < bind->destinations->rnum; j++)
				flag = (((uint64_t)1 << ((k + j) & 63)) & exec_bitmask[(k + j) >> 6]);
			k += bind->destinations->rnum;
		}
		if (flag)
			bitmask[i >> 6] |= ((uint64_t)1 << (i & 63));
		else
			bitmask[i >> 6] &= ~((uint64_t)1 << (i & 63));
	}
	ccv_array_free(sources_destinations);
}

int ccv_nnc_dynamic_graph_bookkeeping_count(const ccv_nnc_dynamic_graph_t* const graph, const int type)
{
	return ccv_nnc_symbolic_graph_active_symbol_count(graph->tape, type);
}

void ccv_nnc_dynamic_graph_dot(const ccv_nnc_dynamic_graph_t* const graph, const int flags, FILE* out)
{
	ccv_nnc_symbolic_graph_dot(graph->tape, flags, out);
}

void ccv_nnc_dynamic_graph_format(const ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_symbolic_graph_format_f format_fn, void* const context)
{
	ccv_nnc_symbolic_graph_format(graph->tape, 0, 0, 0, 0, format_fn, context);
}
