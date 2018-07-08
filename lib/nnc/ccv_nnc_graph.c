#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"

ccv_nnc_graph_t* ccv_nnc_graph_new(void)
{
	ccv_nnc_graph_t* graph = (ccv_nnc_graph_t*)cccalloc(1, sizeof(ccv_nnc_graph_t));
	graph->exec_info = ccv_array_new(sizeof(ccv_nnc_graph_exec_info_t), 5, 0);
	return graph;
}

void ccv_nnc_graph_set_sources(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t* const sources, const int source_size)
{
	if (!graph->sources)
		graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), source_size, 0);
	else
		ccv_array_clear(graph->sources);
	int i;
	for (i = 0; i < source_size; i++)
		ccv_array_push(graph->sources, sources + i);
	graph->sequential = 0;
}

ccv_nnc_graph_exec_t* ccv_nnc_graph_sources(const ccv_nnc_graph_t* const graph)
{
	return graph->sources ? (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, 0) : 0;
}

int ccv_nnc_graph_source_size(const ccv_nnc_graph_t* const graph)
{
	return graph->sources ? graph->sources->rnum : 0;
}

void ccv_nnc_graph_set_destinations(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	if (!graph->destinations)
		graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), destination_size, 0);
	else
		ccv_array_clear(graph->sources);
	int i;
	for (i = 0; i < destination_size; i++)
		ccv_array_push(graph->destinations, destinations + i);
	graph->sequential = 0;
}

ccv_nnc_graph_exec_t* ccv_nnc_graph_destinations(const ccv_nnc_graph_t* const graph)
{
	return graph->destinations ? (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, 0) : 0;
}

int ccv_nnc_graph_destination_size(const ccv_nnc_graph_t* const graph)
{
	return graph->destinations ? graph->destinations->rnum : 0;
}

void ccv_nnc_graph_exec_set(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, const ccv_nnc_cmd_t cmd)
{
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	exec_info->cmd = cmd;
}

void ccv_nnc_graph_exec_set_hint(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, const ccv_nnc_hint_t hint)
{
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	exec_info->hint = hint;
}

static int _ccv_nnc_tensor_multiview_level_count(const ccv_nnc_tensor_multiview_t* const mv)
{
	if (!CCV_IS_TENSOR_MULTIVIEW(mv))
		return 1;
	const int count = mv->kind + mv->repeat;
	int i, c = 0;
	for (i = 0; i < count; i++)
	{
		ccv_nnc_tensor_t* tv = CCV_NNC_MULTIVIEW_DATA(mv)[i];
		if (tv == CCV_NNC_TENSOR_PLACEHOLDER)
			c = ccv_max(c, 1);
		else
			c = ccv_max(c, _ccv_nnc_tensor_multiview_level_count((ccv_nnc_tensor_multiview_t*)tv));
	}
	return c + 1;
}

static ccv_nnc_graph_tensor_wrap_t* _ccv_nnc_graph_tensor_wrap_new(const ccv_nnc_tensor_multiview_t* const mv)
{
	const int level_count = _ccv_nnc_tensor_multiview_level_count(mv);
	ccv_nnc_graph_tensor_wrap_t* tensor_wrap = (ccv_nnc_graph_tensor_wrap_t*)ccmalloc(sizeof(ccv_nnc_graph_tensor_wrap_t) + sizeof(ccv_nnc_tensor_t*) * (level_count - 1));
	tensor_wrap->update_required = 0;
	tensor_wrap->count = level_count;
	tensor_wrap->index = 0;
	tensor_wrap->tensors[0] = (ccv_nnc_tensor_t*)mv;
	return tensor_wrap;
}

static void _ccv_nnc_graph_exec_rewind(ccv_nnc_graph_exec_info_t* const info)
{
	if (!info->tensor_wrap_size)
		return;
	int i;
	// Rewind from tensor wraps.
	for (i = 0; i < info->input_size; i++)
		if (info->tensor_wraps[i])
			info->inputs[i] = info->tensor_wraps[i]->tensors[0];
	const int d = info->input_size;
	for (i = 0; i < info->output_size; i++)
		if (info->tensor_wraps[d + i])
			info->outputs[i] = info->tensor_wraps[d + i]->tensors[0];
	const int dd = info->input_size + info->output_size;
	for (i = 0; i < info->update_size; i++)
		if (info->tensor_wraps[dd + i])
			info->updates[i] = info->tensor_wraps[dd + i]->tensors[0];
}

static void _ccv_nnc_graph_tensor_wrap_free(ccv_nnc_graph_tensor_wrap_t* const tensor_wrap)
{
	ccfree(tensor_wrap);
}

static void _ccv_nnc_graph_redo_tensor_wraps(ccv_nnc_graph_exec_info_t* const info)
{
	int i;
	int has_wrap = 0;
	for (i = 0; i < info->input_size && !has_wrap; i++)
		has_wrap = (info->inputs[i] &&
			CCV_IS_TENSOR_MULTIVIEW(info->inputs[i]) &&
			((ccv_nnc_tensor_multiview_t*)info->inputs[i])->anchor != CCV_NNC_MULTIVIEW_PHI);
	for (i = 0; i < info->output_size && !has_wrap; i++)
		has_wrap = (info->outputs[i] &&
			CCV_IS_TENSOR_MULTIVIEW(info->outputs[i]) &&
			((ccv_nnc_tensor_multiview_t*)info->outputs[i])->anchor != CCV_NNC_MULTIVIEW_PHI);
	for (i = 0; i < info->update_size && !has_wrap; i++)
		has_wrap = (info->updates[i] &&
			CCV_IS_TENSOR_MULTIVIEW(info->updates[i]) &&
			((ccv_nnc_tensor_multiview_t*)info->updates[i])->anchor != CCV_NNC_MULTIVIEW_PHI);
	if (has_wrap)
	{
		const int tensor_wrap_size = info->input_size + info->output_size + info->update_size;
		if (info->tensor_wraps)
		{
			info->tensor_wraps = (ccv_nnc_graph_tensor_wrap_t**)ccrealloc(info->tensor_wraps, sizeof(ccv_nnc_graph_tensor_wrap_t*) * tensor_wrap_size);
			for (i = info->tensor_wrap_size; i < tensor_wrap_size; i++)
				info->tensor_wraps[i] = 0;
		} else
			info->tensor_wraps = (ccv_nnc_graph_tensor_wrap_t**)cccalloc(tensor_wrap_size, sizeof(ccv_nnc_graph_tensor_wrap_t*));
		info->tensor_wrap_size = tensor_wrap_size;
		for (i = 0; i < info->input_size; i++)
			if (info->inputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(info->inputs[i]) &&
					((ccv_nnc_tensor_multiview_t*)info->inputs[i])->anchor != CCV_NNC_MULTIVIEW_PHI)
				{
					if (!info->tensor_wraps[i] || info->inputs[i] != info->tensor_wraps[i]->tensors[0])
					{
						if (info->tensor_wraps[i])
							_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[i]);
						info->tensor_wraps[i] = _ccv_nnc_graph_tensor_wrap_new((ccv_nnc_tensor_multiview_t*)info->inputs[i]);
					}
				} else {
					if (info->tensor_wraps[i])
						_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[i]);
					info->tensor_wraps[i] = 0;
				}
			}
		const int d = info->input_size;
		for (i = 0; i < info->output_size; i++)
			if (info->outputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(info->outputs[i]) &&
					((ccv_nnc_tensor_multiview_t*)info->outputs[i])->anchor != CCV_NNC_MULTIVIEW_PHI)
				{
					if (!info->tensor_wraps[d + i] || info->outputs[i] != info->tensor_wraps[d + i]->tensors[0])
					{
						if (info->tensor_wraps[d + i])
							_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[d + i]);
						info->tensor_wraps[d + i] = _ccv_nnc_graph_tensor_wrap_new((ccv_nnc_tensor_multiview_t*)info->outputs[i]);
					}
				} else {
					if (info->tensor_wraps[d + i])
						_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[d + i]);
					info->tensor_wraps[d + i] = 0;
				}
			}
		const int dd = info->input_size + info->output_size;
		for (i = 0; i < info->update_size; i++)
			if (CCV_IS_TENSOR_MULTIVIEW(info->updates[i]) &&
				((ccv_nnc_tensor_multiview_t*)info->updates[i])->anchor != CCV_NNC_MULTIVIEW_PHI)
			{
				if (!info->tensor_wraps[dd + i] || info->updates[i] != info->tensor_wraps[dd + i]->tensors[0])
				{
					if (info->tensor_wraps[dd + i])
						_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[dd + i]);
					info->tensor_wraps[dd + i] = _ccv_nnc_graph_tensor_wrap_new((ccv_nnc_tensor_multiview_t*)info->updates[i]);
				}
			} else {
				if (info->tensor_wraps[dd + i])
					_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[dd + i]);
				info->tensor_wraps[dd + i] = 0;
			}
	} else {
		for (i = 0; i < info->tensor_wrap_size; i++)
			if (info->tensor_wraps[i])
				_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[i]);
		ccfree(info->tensor_wraps);
		info->tensor_wrap_size = 0;
		info->tensor_wraps = 0;
	}
}

static void _ccv_nnc_graph_deregister_tensor_wraps(ccv_nnc_graph_t* graph, const ccv_nnc_graph_exec_t exec)
{
	ccv_nnc_graph_t* p = graph;
	do {
		int i;
		// Remove from the array.
		if (p->exec_wraps)
			for (i = 0; i < p->exec_wraps->rnum; i++)
			{
				ccv_nnc_graph_exec_t* const exec_wrap = (ccv_nnc_graph_exec_t*)ccv_array_get(p->exec_wraps, i);
				if (exec_wrap->d == exec.d && exec_wrap->graph == graph)
				{
					--p->exec_wraps->rnum;
					if (i < p->exec_wraps->rnum)
						memcpy(exec_wrap, exec_wrap + 1, sizeof(ccv_nnc_graph_exec_t) * (p->exec_wraps->rnum - i));
					break;
				}
			}
		p = p->p;
	} while (p);
}

static void _ccv_nnc_graph_register_tensor_wraps(ccv_nnc_graph_t* graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_graph_exec_info_t* const info)
{
	assert(info->tensor_wrap_size > 0);
	ccv_nnc_graph_t* p = graph;
	do {
		if (!p->exec_wraps)
		{
			p->exec_wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), 0, 0);
			ccv_array_push(p->exec_wraps, &exec);
		} else {
			int i;
			int has_exec_wrap = 0;
			for (i = 0; !has_exec_wrap && i < p->exec_wraps->rnum; i++)
			{
				ccv_nnc_graph_exec_t* exec_wrap = (ccv_nnc_graph_exec_t*)ccv_array_get(p->exec_wraps, i);
				has_exec_wrap = (exec_wrap->d == exec.d && exec_wrap->graph == exec.graph);
			}
			if (!has_exec_wrap)
				ccv_array_push(p->exec_wraps, &exec);
		}
		p = p->p;
	} while (p);
}

void ccv_nnc_graph_exec_set_io_flags(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, const int* const input_flags, const int input_flag_size, const int* const output_flags, const int output_flag_size)
{
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* const info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	assert(input_flag_size <= info->input_size);
	assert(output_flag_size <= info->output_size);
	if (info->input_size + info->output_size == 0)
		return;
	if (!info->input_flags)
	{
		info->input_flags = (int*)cccalloc(info->input_size + info->output_size, sizeof(int));
		info->output_flags = info->input_flags + info->input_size;
	}
	if (input_flag_size > 0)
		memcpy(info->input_flags, input_flags, sizeof(int) * input_flag_size);
	if (output_flag_size > 0)
		memcpy(info->output_flags, output_flags, sizeof(int) * output_flag_size);
}

void ccv_nnc_graph_exec_set_peer(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, const ccv_nnc_graph_exec_t peer_exec)
{
	assert(exec.graph == graph);
	assert(exec.d >= 0);
	assert(exec.d < graph->exec_info->rnum);
	assert(peer_exec.graph == graph || peer_exec.graph == graph->peer);
	assert(peer_exec.d >= 0);
	if (peer_exec.graph == graph)
		{ assert(peer_exec.d < graph->exec_info->rnum); }
	else
		{ assert(peer_exec.d < graph->peer->exec_info->rnum); }
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	exec_info->peer_ref = peer_exec.d + 1;
}

void ccv_nnc_graph_exec_set_io(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* const info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	// De-register from the graph if it contains multiview tensors.
	if (info->tensor_wrap_size)
		_ccv_nnc_graph_deregister_tensor_wraps(graph, exec);
	// In case it is already executed, rewind.
	_ccv_nnc_graph_exec_rewind(info);
	if (input_size == 0 && output_size == 0)
	{
		if (info->input_size > 0 || info->output_size > 0)
			ccfree(info->inputs);
		info->inputs = 0;
		info->outputs = 0;
		info->input_size = 0;
		info->output_size = 0;
		_ccv_nnc_graph_redo_tensor_wraps(info);
		if (info->tensor_wrap_size)
			_ccv_nnc_graph_register_tensor_wraps(graph, exec, info);
		return;
	}
	if (info->inputs)
		info->inputs = (ccv_nnc_tensor_t**)ccrealloc(info->inputs, sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
	else
		info->inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
	info->outputs = info->inputs + input_size;
	if (inputs)
		memcpy(info->inputs, inputs, sizeof(ccv_nnc_tensor_t*) * input_size);
	if (outputs)
		memcpy(info->outputs, outputs, sizeof(ccv_nnc_tensor_t*) * output_size);
	int i;
	int tensor_memory = 0, tensor_formats = 0, tensor_datatypes = 0;
	for (i = 0; i < input_size + output_size; i++)
		if (info->inputs[i])
			tensor_memory |= CCV_TENSOR_GET_MEMORY(info->inputs[i]->info.type), tensor_formats |= info->inputs[i]->info.format, tensor_datatypes |= info->inputs[i]->info.datatype;
	info->cmd.backend = ccv_nnc_cmd_find_backend(info->cmd, tensor_memory, tensor_formats, tensor_datatypes);
	info->input_size = input_size;
	info->output_size = output_size;
	_ccv_nnc_graph_redo_tensor_wraps(info);
	// Register again if the tensor wraps exist.
	if (info->tensor_wrap_size)
		_ccv_nnc_graph_register_tensor_wraps(graph, exec, info);
	// Free flags.
	if (info->input_flags)
	{
		ccfree(info->input_flags);
		info->input_flags = info->output_flags = 0;
	}
}

void ccv_nnc_graph_exec_add_update(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_tensor_t* const update)
{
	assert(CCV_IS_TENSOR_MULTIVIEW(update));
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* const info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	const int register_tensor_wraps = !info->tensor_wrap_size;
	const int update_index = info->update_size;
	++info->update_size;
	if (info->updates)
		info->updates = (ccv_nnc_tensor_t**)ccrealloc(info->updates, sizeof(ccv_nnc_tensor_t*) * info->update_size);
	else
		info->updates = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * info->update_size);
	info->updates[update_index] = update;
	_ccv_nnc_graph_redo_tensor_wraps(info);
	if (register_tensor_wraps)
		_ccv_nnc_graph_register_tensor_wraps(graph, exec, info);
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_new(ccv_nnc_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int d = graph->exec_info->rnum;
	ccv_nnc_graph_exec_info_t info = {
		.cmd = cmd,
		.hint = hint,
		.input_size = input_size,
		.output_size = output_size,
	};
	assert(inputs || input_size == 0);
	assert(outputs || output_size == 0);
	if (input_size > 0 || output_size > 0)
	{
		info.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
		info.outputs = info.inputs + input_size;
		if (inputs)
			memcpy(info.inputs, inputs, sizeof(ccv_nnc_tensor_t*) * input_size);
		if (outputs)
			memcpy(info.outputs, outputs, sizeof(ccv_nnc_tensor_t*) * output_size);
		info.input_size = input_size;
		info.output_size = output_size;
		int i;
		int tensor_memory = 0, tensor_formats = 0, tensor_datatypes = 0;
		for (i = 0; i < input_size + output_size; i++)
			if (info.inputs[i])
				tensor_memory |= CCV_TENSOR_GET_MEMORY(info.inputs[i]->info.type), tensor_formats |= info.inputs[i]->info.format, tensor_datatypes |= info.inputs[i]->info.datatype;
		info.cmd.backend = ccv_nnc_cmd_find_backend(info.cmd, tensor_memory, tensor_formats, tensor_datatypes);
	}
	ccv_nnc_graph_exec_t exec = {
		.d = d,
		.graph = graph,
	};
	_ccv_nnc_graph_redo_tensor_wraps(&info);
	// Add itself to the graph's wraps array, this will help the run time when we run the graph and do unwrapping.
	if (info.tensor_wrap_size)
		_ccv_nnc_graph_register_tensor_wraps(graph, exec, &info);
	ccv_array_push(graph->exec_info, &info);
	return exec;
}

void ccv_nnc_graph_add_carry_over(ccv_nnc_graph_t* const graph, const ccv_nnc_tensor_t* const from, const ccv_nnc_tensor_t* const to)
{
	ccv_nnc_graph_tensor_carry_over_t carry_over = {
		.from = _ccv_nnc_graph_tensor_wrap_new((ccv_nnc_tensor_multiview_t*)from),
		.to = _ccv_nnc_graph_tensor_wrap_new((ccv_nnc_tensor_multiview_t*)to)
	};
	if (!graph->carry_overs)
		graph->carry_overs = ccv_array_new(sizeof(ccv_nnc_graph_tensor_carry_over_t), 0, 0);
	ccv_array_push(graph->carry_overs, &carry_over);
}

int ccv_nnc_graph_exec_concat(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t source, const ccv_nnc_graph_exec_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_info->rnum);
	assert(destination.d < graph->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* src_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, source.d);
	if (src_info->outgoings == 0)
		src_info->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	else {
		int i;
		// Check if this is already connected, if so, skip.
		for (i = 0; i < src_info->outgoings->rnum; i++)
			if (*(int*)ccv_array_get(src_info->outgoings, i) == destination.d)
				return -1;
	}
	ccv_array_push(src_info->outgoings, &destination.d);
	graph->sequential = 0;
	return 0;
}

int ccv_nnc_graph_exec_disjoin(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t source, const ccv_nnc_graph_exec_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_info->rnum);
	assert(destination.d < graph->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* src_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, source.d);
	if (!src_info->outgoings)
		return -1;
	int i, j = -1;
	// Check if this is already connected, if so, skip.
	for (i = 0; i < src_info->outgoings->rnum; i++)
		if (*(int*)ccv_array_get(src_info->outgoings, i) == destination.d)
		{
			j = i;
			break;
		}
	if (j < 0)
		return -1;
	if (j < src_info->outgoings->rnum - 1)
		*(int*)ccv_array_get(src_info->outgoings, j) = *(int*)ccv_array_get(src_info->outgoings, src_info->outgoings->rnum - 1);
	--src_info->outgoings->rnum;
	ccv_nnc_graph_exec_info_t* dest_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, destination.d);
	if (dest_info->outgoings)
		for (i = 0; i < dest_info->outgoings->rnum; i++)
			ccv_array_add_unique_int(src_info->outgoings, *(int*)ccv_array_get(dest_info->outgoings, i));
	graph->sequential = 0;
	return 0;
}

int ccv_nnc_graph_exec_size(const ccv_nnc_graph_t* const graph)
{
	return graph->exec_info ? graph->exec_info->rnum : 0;
}

void ccv_nnc_graph_sequential(ccv_nnc_graph_t* const graph, int* const exec_cvt, const int exec_cvt_size)
{
	assert(exec_cvt_size == graph->exec_info->rnum);
	assert(graph->sources && graph->sources->rnum);
	assert(graph->destinations && graph->destinations->rnum);
	int i, j;
	for (i = 0; i < exec_cvt_size; i++)
		exec_cvt[i] = -1;
	ccv_array_t* exec_info = ccv_array_new(sizeof(ccv_nnc_graph_exec_info_t), graph->exec_info->rnum, 0);
	// If there are breakpoints, it is more complicated, we first start to the breakpoints, and then continue from the breakpoints to the destinations.
	if (graph->breakpoint_size)
	{
		ccv_nnc_graph_visit_t* visit = ccv_nnc_graph_visit_new(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, 0), graph->sources->rnum, graph->breakpoints, graph->breakpoint_size, 0);
		for (i = 0; i < graph->breakpoint_size; i++)
			exec_cvt[graph->breakpoints[i].d] = -2; // Mark this as breakpoints, so we will skip the first round.
		ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), node, idx) {
			assert(!node->peer_ref); // If node has a peer ref, we cannot fix it up.
			if (exec_cvt[idx] == -2) // Skip breakpoint.
				continue;
			// Loop over node and push to the array.
			ccv_array_push(exec_info, node);
			// Go to its sub-graph to fix exec_idx
			for (i = 0; i < node->graph_ref_size; i++)
			{
				const int graph_ref = CCV_NNC_GRAPH_REF(node)[i] - 1;
				if (graph_ref >= 0)
				{
					ccv_nnc_graph_t* const sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, graph_ref);
					sub_graph->exec_idx = exec_info->rnum;
				}
			}
			exec_cvt[idx] = exec_info->rnum - 1;
		} ccv_nnc_graph_visit_endfor
		ccv_nnc_graph_visit_free(visit);
		graph->breakpoint_offset = exec_info->rnum;
		visit = ccv_nnc_graph_visit_new(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph->breakpoints, graph->breakpoint_size, (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, 0), graph->destinations->rnum, 0);
		ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), node, idx) {
			assert(!node->peer_ref); // If node has a peer ref, we cannot fix it up.
			// Loop over node and push to the array.
			ccv_array_push(exec_info, node);
			// Go to its sub-graph to fix exec_idx
			for (i = 0; i < node->graph_ref_size; i++)
			{
				const int graph_ref = CCV_NNC_GRAPH_REF(node)[i] - 1;
				if (graph_ref >= 0)
				{
					ccv_nnc_graph_t* const sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, graph_ref);
					sub_graph->exec_idx = exec_info->rnum;
				}
			}
			exec_cvt[idx] = exec_info->rnum - 1;
		} ccv_nnc_graph_visit_endfor
		ccv_nnc_graph_visit_free(visit);
		for (i = 0; i < graph->breakpoint_size; i++)
			{ assert(exec_cvt[graph->breakpoints[i].d] >= 0); } // All breakpoints should be assigned.
	} else {
		ccv_nnc_graph_visit_t* visit = ccv_nnc_graph_visit_new(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, 0), graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, 0), graph->destinations->rnum, 0);
		ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), node, idx) {
			assert(!node->peer_ref); // If node has a peer ref, we cannot fix it up.
			// Loop over node and push to the array.
			ccv_array_push(exec_info, node);
			// Go to its sub-graph to fix exec_idx
			for (i = 0; i < node->graph_ref_size; i++)
			{
				const int graph_ref = CCV_NNC_GRAPH_REF(node)[i] - 1;
				if (graph_ref >= 0)
				{
					ccv_nnc_graph_t* const sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, graph_ref);
					sub_graph->exec_idx = exec_info->rnum;
				}
			}
			exec_cvt[idx] = exec_info->rnum - 1;
		} ccv_nnc_graph_visit_endfor
		ccv_nnc_graph_visit_free(visit);
	}
	assert(graph->exec_info->rnum == exec_info->rnum);
	ccv_array_free(graph->exec_info);
	graph->exec_info = exec_info;
	for (i = 0; i < graph->sources->rnum; i++)
	{
		ccv_nnc_graph_exec_t* const source = (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, i);
		source->d = exec_cvt[source->d];
	}
	for (i = 0; i < graph->destinations->rnum; i++)
	{
		ccv_nnc_graph_exec_t* const destination = (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, i);
		destination->d = exec_cvt[destination->d];
	}
	// Update all outgoings to reflect the latest.
	for (i = 0; i < exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* const info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(exec_info, i);
		if (info->outgoings)
			for (j = 0; j < info->outgoings->rnum; j++)
				*(int*)ccv_array_get(info->outgoings, j) = exec_cvt[*(int*)ccv_array_get(info->outgoings, j)];
	}
	graph->sequential = 1;
}

static void _ccv_nnc_graph_dot_exec(const int index, const ccv_nnc_graph_exec_info_t* const exec_info, const int flags, FILE* out)
{
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	fprintf(out, "node%d", index);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		fputs("|Command: ", out);
		fputs(ccv_nnc_cmd_name(exec_info->cmd.cmd), out);
		fputc('}', out);
	}
}

static void _ccv_nnc_graph_dot_tensor(const int index, const ccv_nnc_tensor_t* const tensor, const int zone, const int flags, const int depth, FILE* out)
{
	// if it has an alias pointer, or, it is a long form.
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	int is_tensor_view = CCV_IS_TENSOR_VIEW(tensor);
	if (is_tensor_view)
		fprintf(out, "tensorview%d", index);
	else
		fprintf(out, "tensor%d", index);
	int i;
	for (i = 0; i < depth; i++) // Print subscription to denote depth.
		fputc('\'', out);
	if (CCV_GET_TAPE_ALLOC(tensor->type))
		fputs(" (t)", out);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		fprintf(out, "|zone%d", zone);
		for (i = 0; i < depth; i++) // Print subscription to denote depth.
			fputc('\'', out);
		uintptr_t aptr = (uintptr_t)tensor->data.u8;
		const int* ainc = is_tensor_view ? ((ccv_nnc_tensor_view_t*)(tensor))->inc : tensor->info.dim;
		// For the last one, we don't extend to full ainc.
		size_t ainc_size = (ccv_nnc_dimension_count(ainc) - ainc[0] + tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(tensor->type);
		// Print out the range as well.
		fprintf(out, "|{%#010x|%#010x}|%d", (uint32_t)aptr, (uint32_t)(aptr + ainc_size - 1), tensor->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && tensor->info.dim[i]; i++)
			fprintf(out, "x%d", tensor->info.dim[i]);
		fputc('}', out);
	}
}

typedef struct {
	int index;
	int name;
	int zone;
	uintptr_t tensor_ref;
	uintptr_t start_ptr;
	uintptr_t end_ptr;
} ccv_nnc_tensor_dot_t;

typedef struct {
	ccv_nnc_tensor_dot_t* dots;
	int* remap;
	int* rename_zone;
	int* rename_index;
} ccv_nnc_tensor_dot_recovery_t;

// First sort by start_ptr, then sort by tensor ptr (so that we will have the same tensor sorted to one cluster).
#define less_than(i1, i2, aux) ((i1).start_ptr < (i2).start_ptr || ((i1).start_ptr == (i2).start_ptr && (i1).tensor_ref < (i2).tensor_ref))
static CCV_IMPLEMENT_QSORT(_ccv_nnc_tensor_dot_sort_by_ptr, ccv_nnc_tensor_dot_t, less_than)
#undef less_than

static int _ccv_nnc_graph_dot_tensor_multiview_count(const ccv_nnc_tensor_multiview_t* const mv)
{
	if (!CCV_IS_TENSOR_MULTIVIEW(mv))
		return 1;
	const int count = mv->kind + mv->repeat;
	int i, c = 0;
	for (i = 0; i < count; i++)
		c += _ccv_nnc_graph_dot_tensor_multiview_count((ccv_nnc_tensor_multiview_t*)CCV_NNC_MULTIVIEW_DATA(mv)[i]);
	return c;
}

static void _ccv_nnc_graph_dot_tensor_multiview_tensor_dots(const ccv_nnc_tensor_multiview_t* const mv, ccv_nnc_tensor_dot_t* const tensor_dots, int* tensor_index)
{
	const int count = mv->kind + mv->repeat;
	int i;
	for (i = 0; i < count; i++)
		if (CCV_IS_TENSOR_MULTIVIEW(CCV_NNC_MULTIVIEW_DATA(mv)[i]))
			_ccv_nnc_graph_dot_tensor_multiview_tensor_dots((ccv_nnc_tensor_multiview_t*)CCV_NNC_MULTIVIEW_DATA(mv)[i], tensor_dots, tensor_index);
		else {
			tensor_dots[*tensor_index].name = *tensor_index;
			tensor_dots[*tensor_index].start_ptr =  (uintptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i]->data.u8;
			// Because tv's pointer will get updated, it is not correct in this case to have one tensor_ref.
			tensor_dots[*tensor_index].tensor_ref = tensor_dots[*tensor_index].start_ptr;
			const size_t dim_size = ccv_nnc_dimension_count(CCV_NNC_MULTIVIEW_DATA(mv)[i]->info.dim) * CCV_GET_DATA_TYPE_SIZE(CCV_NNC_MULTIVIEW_DATA(mv)[i]->type);
			tensor_dots[*tensor_index].end_ptr = tensor_dots[*tensor_index].start_ptr + dim_size - 1;
			++(*tensor_index);
		}
}

static ccv_nnc_tensor_dot_recovery_t _ccv_nnc_graph_tensor_dot_recovery(const ccv_nnc_graph_t* const graph)
{
	int i, j;
	// Recover tensor relationships for all tensors referenced in the graph.
	// Most notably, we have to give these indexes, and find if they point to
	// the same memory region, and whether they overlap. These information
	// are lost since we converted from symbolic form to the execution form.
	// and here we do our best to recover because that is easier to understand
	// if we want to present the graph visually (also, we don't want to put this
	// information into the tensor or execution graph to avoid overhead, thus,
	// recovering is the best we can do).
	int tensor_count = 0;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		for (j = 0; j < exec_info->input_size; j++)
			if (exec_info->inputs[j])
				tensor_count += CCV_IS_TENSOR_MULTIVIEW(exec_info->inputs[j]) ? _ccv_nnc_graph_dot_tensor_multiview_count((ccv_nnc_tensor_multiview_t*)exec_info->inputs[j]) : 1;
		for (j = 0; j < exec_info->output_size; j++)
			if (exec_info->outputs[j])
				tensor_count += CCV_IS_TENSOR_MULTIVIEW(exec_info->outputs[j]) ? _ccv_nnc_graph_dot_tensor_multiview_count((ccv_nnc_tensor_multiview_t*)exec_info->outputs[j]) : 1;
	}
	ccv_nnc_tensor_dot_t* tensor_dots = tensor_count > 0 ? (ccv_nnc_tensor_dot_t*)ccmalloc(sizeof(ccv_nnc_tensor_dot_t) * tensor_count) : 0;
	int k = 0;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		for (j = 0; j < exec_info->input_size; j++)
		{
			ccv_nnc_tensor_t* tensor = exec_info->inputs[j];
			if (!tensor)
				continue;
			if (CCV_IS_TENSOR_MULTIVIEW(tensor))
				_ccv_nnc_graph_dot_tensor_multiview_tensor_dots((ccv_nnc_tensor_multiview_t*)tensor, tensor_dots, &k);
			else {
				tensor_dots[k].name = k;
				tensor_dots[k].tensor_ref = (uintptr_t)tensor;
				tensor_dots[k].start_ptr = (uintptr_t)tensor->data.u8;
				const int* inc = CCV_IS_TENSOR_VIEW(tensor) ? ((ccv_nnc_tensor_view_t*)tensor)->inc : tensor->info.dim;
				const size_t inc_size = (ccv_nnc_dimension_count(inc) - inc[0] + tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(tensor->type);
				tensor_dots[k].end_ptr = tensor_dots[k].start_ptr + inc_size - 1;
				++k;
			}
		}
		for (j = 0; j < exec_info->output_size; j++)
		{
			ccv_nnc_tensor_t* tensor = exec_info->outputs[j];
			if (!tensor)
				continue;
			if (CCV_IS_TENSOR_MULTIVIEW(tensor))
				_ccv_nnc_graph_dot_tensor_multiview_tensor_dots((ccv_nnc_tensor_multiview_t*)tensor, tensor_dots, &k);
			else {
				tensor_dots[k].name = k;
				tensor_dots[k].tensor_ref = (uintptr_t)tensor;
				tensor_dots[k].start_ptr = (uintptr_t)tensor->data.u8;
				const int* inc = CCV_IS_TENSOR_VIEW(tensor) ? ((ccv_nnc_tensor_view_t*)tensor)->inc : tensor->info.dim;
				const size_t inc_size = (ccv_nnc_dimension_count(inc) - inc[0] + tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(tensor->type);
				tensor_dots[k].end_ptr = tensor_dots[k].start_ptr + inc_size - 1;
				++k;
			}
		}
	}
	tensor_count = k; // We may over count, now shrink.
	// To group overlap memory into one zone, we sort it by start ptr first (secondary by the tensor pointer).
	_ccv_nnc_tensor_dot_sort_by_ptr(tensor_dots, tensor_count, 0);
	int index = 0, zone = 0;
	uintptr_t tensor_ref = tensor_count > 0 ? tensor_dots[0].tensor_ref : 0;
	uintptr_t end_ptr = tensor_count > 0 ? tensor_dots[0].end_ptr : 0;
	// Then, it is trivial, we go by end ptr. If the next start ptr is still within the end ptr (start ptr <= end ptr),
	// they are the same zone.
	for (i = 0; i < tensor_count; i++)
	{
		if (tensor_dots[i].tensor_ref != tensor_ref)
		{
			tensor_ref = tensor_dots[i].tensor_ref;
			++index;
		}
		if (tensor_dots[i].start_ptr > end_ptr)
		{
			end_ptr = ccv_max(end_ptr, tensor_dots[i].end_ptr);
			++zone;
		}
		tensor_dots[i].index = index;
		tensor_dots[i].zone = zone;
	}
	// We already have index and zone assigned, but the problem is that these are not very human interpretable (because
	// it follows the pointer from low to high, not the tensor creation order). The following code renamed both the index
	// and the zone so that it is much more understandable.
	const int index_count = index + 1;
	const int zone_count = zone + 1;
	int* remap = (int*)ccmalloc(sizeof(int) * (tensor_count + index_count + zone_count));
	int* rename_index = remap + tensor_count;
	int* rename_zone = rename_index + index_count;
	for (i = 0; i < tensor_count; i++)
		remap[tensor_dots[i].name] = i;
	for (i = 0; i < index_count; i++)
		rename_index[i] = -1;
	for (i = 0; i < zone_count; i++)
		rename_zone[i] = -1;
	index = 0;
	zone = 0;
	for (i = 0; i < tensor_count; i++)
	{
		ccv_nnc_tensor_dot_t* tensor_dot = tensor_dots + remap[i];
		if (rename_index[tensor_dot->index] == -1)
			rename_index[tensor_dot->index] = index++;
		if (rename_zone[tensor_dot->zone] == -1)
			rename_zone[tensor_dot->zone] = zone++;
	}
	ccv_nnc_tensor_dot_recovery_t recovery = {
		.dots = tensor_dots,
		.remap = remap,
		.rename_index = rename_index,
		.rename_zone = rename_zone,
	};
	return recovery;
}

static void _ccv_nnc_graph_tensor_dot_recovery_free(const ccv_nnc_tensor_dot_recovery_t recovery)
{
	ccfree(recovery.dots);
	ccfree(recovery.remap);
}

static void _ccv_nnc_graph_dot_tensor_multiview_one(const ccv_nnc_tensor_multiview_t* const mv, const ccv_nnc_tensor_dot_recovery_t recovery, const int depth, int* tensor_index, FILE* out)
{
	const int count = mv->kind + mv->repeat;
	int i, j;
	fputs("|{", out);
	for (i = 0; i < count; i++)
		if (CCV_IS_TENSOR_MULTIVIEW(CCV_NNC_MULTIVIEW_DATA(mv)[i]))
		{
			fprintf(out, "{%d", i);
			if (mv->kind == CCV_NNC_MULTIVIEW_K0N || (mv->kind == CCV_NNC_MULTIVIEW_K1N && i > 0))
				fputc('*', out); // Denotes that we loop on this.
			_ccv_nnc_graph_dot_tensor_multiview_one((ccv_nnc_tensor_multiview_t*)CCV_NNC_MULTIVIEW_DATA(mv)[i], recovery, depth, tensor_index, out);
			if (i == count - 1)
				fputc('}', out);
			else
				fputs("}|", out);
		} else {
			fprintf(out, "{%d", i);
			if (mv->kind == CCV_NNC_MULTIVIEW_K0N || (mv->kind == CCV_NNC_MULTIVIEW_K1N && i > 0))
				fputc('*', out); // Denotes that we loop on this.
			const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[*tensor_index];
			fprintf(out, "|zone%d", recovery.rename_zone[tensor_dot->zone]);
			for (j = 0; j < depth; j++)
				fputc('\'', out);
			uintptr_t aptr = (uintptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i]->data.u8;
			// For the last one, we don't extend to full ainc.
			size_t dim_size = ccv_nnc_dimension_count(CCV_NNC_MULTIVIEW_DATA(mv)[i]->info.dim) * CCV_GET_DATA_TYPE_SIZE(CCV_NNC_MULTIVIEW_DATA(mv)[i]->type);
			// Print out the range as well.
			fprintf(out, "|{%#010x|%#010x}", (uint32_t)aptr, (uint32_t)(aptr + dim_size - 1));
			++(*tensor_index);
			if (i == count - 1)
				fputc('}', out);
			else
				fputs("}|", out);
		}
	fputc('}', out);
}

static void _ccv_nnc_graph_dot_tensor_multiview(const ccv_nnc_tensor_multiview_t* const mv, const ccv_nnc_tensor_dot_recovery_t recovery, const int flags, const int depth, int* tensor_index, FILE* out)
{
	// if it has an alias pointer, or, it is a long form.
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[*tensor_index];
	fprintf(out, "multiview%d", recovery.rename_index[tensor_dot->index]);
	int i;
	for (i = 0; i < depth; i++) // Print subscription to denote depth.
		fputc('\'', out);
	if (CCV_GET_TAPE_ALLOC(mv->type))
		fputs(" (t)", out);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		_ccv_nnc_graph_dot_tensor_multiview_one(mv, recovery, depth, tensor_index, out);
		const ccv_nnc_tensor_t* root = (ccv_nnc_tensor_t*)mv;
		while (CCV_IS_TENSOR_MULTIVIEW(root))
			root = CCV_NNC_MULTIVIEW_DATA((ccv_nnc_tensor_multiview_t*)root)[0];
		fprintf(out, "|%d", root->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && root->info.dim[i]; i++)
			fprintf(out, "x%d", root->info.dim[i]);
		fputc('}', out);
	} else
		*tensor_index += _ccv_nnc_graph_dot_tensor_multiview_count(mv);
}

static void _ccv_nnc_graph_dot_node(const ccv_nnc_graph_exec_info_t* const exec_info, const int exec_index, const ccv_nnc_tensor_dot_recovery_t recovery, const int flags, const int depth, FILE* out, int* const tensor_index)
{
	fprintf(out, "node%d [shape=record,label=\"", exec_index);
	_ccv_nnc_graph_dot_exec(exec_index, exec_info, flags, out);
	int i;
	int k = *tensor_index;
	if (exec_info->input_size > 0)
	{
		fputs("|{Input", out);
		for (i = 0; i < exec_info->input_size; i++)
			if (exec_info->inputs[i])
			{
				fputc('|', out);
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->inputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->inputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->inputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputs("|-", out);
		fputc('}', out);
	}
	if (exec_info->output_size > 0)
	{
		fputs("|{Output", out);
		for (i = 0; i < exec_info->output_size; i++)
			if (exec_info->outputs[i])
			{
				fputc('|', out);
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->outputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->outputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->outputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputs("|-", out);
		fputc('}', out);
	}
	fputs("\"];\n", out);
	*tensor_index = k;
}

static void _ccv_nnc_graph_dot_while_label(const ccv_nnc_graph_exec_info_t* const exec_info, const int exec_index, const ccv_nnc_tensor_dot_recovery_t recovery, const ccv_nnc_graph_t* const while_graph, const int flags, const int depth, FILE* out, int* tensor_index)
{
	int i;
	fprintf(out, "label=<<b>while%d </b>Command: ", exec_index);
	fputs(ccv_nnc_cmd_name(exec_info->cmd.cmd), out);
	fputs(">;\n", out);
	fprintf(out, "label%d [shape=record,label=\"{", exec_index);
	int k = *tensor_index;
	if (exec_info->input_size > 0)
	{
		fputs("{Input|{", out);
		for (i = 0; i < exec_info->input_size; i++)
		{
			if (i > 0)
				fputc('|', out);
			if (exec_info->inputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->inputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->inputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->inputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputc('-', out);
		}
		fputs("}}", out);
	}
	if (exec_info->output_size > 0)
	{
		if (exec_info->input_size > 0)
			fputs("|", out);
		fputs("{Output|{", out);
		for (i = 0; i < exec_info->output_size; i++)
		{
			if (i > 0)
				fputc('|', out);
			if (exec_info->outputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->outputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->outputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->outputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputc('-', out);
		}
		fputs("}}", out);
	}
	fputs("}\"];\n", out);
	*tensor_index = k;
}

static void _ccv_nnc_graph_dot_case_of_label(const ccv_nnc_graph_exec_info_t* const exec_info, const int exec_index, const ccv_nnc_tensor_dot_recovery_t recovery, const int flags, const int depth, FILE* out, int* tensor_index)
{
	int i;
	fprintf(out, "label=<<b>caseof%d </b>Command: ", exec_index);
	fputs(ccv_nnc_cmd_name(exec_info->cmd.cmd), out);
	fputs(">;\n", out);
	fprintf(out, "label%d [shape=record,label=\"{", exec_index);
	int k = *tensor_index;
	if (exec_info->input_size > 0)
	{
		fputs("{Input|{", out);
		for (i = 0; i < exec_info->input_size; i++)
		{
			if (i > 0)
				fputc('|', out);
			if (exec_info->inputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->inputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->inputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->inputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputc('-', out);
		}
		fputs("}}", out);
	}
	if (exec_info->output_size > 0)
	{
		if (exec_info->input_size > 0)
			fputs("|", out);
		fputs("{Output|{", out);
		for (i = 0; i < exec_info->output_size; i++)
		{
			if (i > 0)
				fputc('|', out);
			if (exec_info->outputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->outputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->outputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->outputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputc('-', out);
		}
		fputs("}}", out);
	}
	fputs("}\"];\n", out);
	*tensor_index = k;
}

static void _ccv_nnc_graph_dot_sub_graphs(const ccv_nnc_graph_exec_info_t* const exec_info, const ccv_nnc_tensor_dot_recovery_t p_recovery, const ccv_array_t* const sub_graphs, const int flags, const int depth, FILE* out, int* tensor_index, int* exec_index)
{
	if (exec_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
	{
		fprintf(out, "subgraph cluster%d {\nstyle=\"rounded\";\nnode%d [style=invisible];\n", *exec_index, *exec_index);
		const ccv_nnc_graph_t* const while_graph = *(ccv_nnc_graph_t**)ccv_array_get(sub_graphs, CCV_NNC_GRAPH_REF(exec_info)[0] - 1);
		// Output this node info within this subgraph.
		_ccv_nnc_graph_dot_while_label(exec_info, *exec_index, p_recovery, while_graph, flags, depth - 1 /* Label all references to its level above. */, out, tensor_index);
	} else if (exec_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF) {
		fprintf(out, "subgraph cluster%d {\nstyle=\"rounded\";\nnode%d [style=invisible];\n", *exec_index, *exec_index);
		_ccv_nnc_graph_dot_case_of_label(exec_info, *exec_index, p_recovery, flags, depth - 1 /* Label all references to its level above. */, out, tensor_index);
	}
	++(*exec_index);
	int p;
	for (p = 0; p < exec_info->graph_ref_size; p++)
	{
		if (exec_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			fprintf(out, "subgraph cluster%d {\nstyle=\"rounded\";\nnode%d [style=invisible];\nlabel=\"\"\n", *exec_index, *exec_index);
			++(*exec_index);
		}
		const ccv_nnc_graph_t* const graph = *(ccv_nnc_graph_t**)ccv_array_get(sub_graphs, CCV_NNC_GRAPH_REF(exec_info)[p] - 1);
		ccv_nnc_tensor_dot_recovery_t recovery = _ccv_nnc_graph_tensor_dot_recovery(graph);
		int i, j;
		int k = 0;
		int* node_id = (int*)ccmalloc(sizeof(int) * graph->exec_info->rnum);
		// Output styles.
		for (i = 0; i < graph->exec_info->rnum; i++)
		{
			node_id[i] = *exec_index;
			ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
			if (CCV_NNC_GRAPH_REF(exec_info)[0])
				_ccv_nnc_graph_dot_sub_graphs(exec_info, recovery, graph->sub_graphs, flags, depth + 1, out, &k, exec_index);
			else {
				_ccv_nnc_graph_dot_node(exec_info, *exec_index, recovery, flags, depth, out, &k);
				++(*exec_index);
			}
		}
		// Output connections.
		for (i = 0; i < graph->exec_info->rnum; i++)
		{
			ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
			if (exec_info->outgoings)
				for (j = 0; j < exec_info->outgoings->rnum; j++)
				{
					const int outgoing_idx = *(int*)ccv_array_get(exec_info->outgoings, j);
					const ccv_nnc_graph_exec_info_t* const outgoing_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, outgoing_idx);
					// If both are sub-graphs, have both tail and head specified.
					if (CCV_NNC_GRAPH_REF(exec_info)[0] && CCV_NNC_GRAPH_REF(outgoing_info)[0])
						fprintf(out, "node%d -> node%d [ltail=cluster%d,lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i], node_id[outgoing_idx]);
					else if (CCV_NNC_GRAPH_REF(exec_info)[0] && !CCV_NNC_GRAPH_REF(outgoing_info)[0])
						fprintf(out, "node%d -> node%d [ltail=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i]);
					else if (!CCV_NNC_GRAPH_REF(exec_info)[0] && CCV_NNC_GRAPH_REF(outgoing_info)[0])
						fprintf(out, "node%d -> node%d [lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[outgoing_idx]);
					else
						fprintf(out, "node%d -> node%d;\n", node_id[i], node_id[outgoing_idx]);
				}
		}
		fputs("}\n", out);
		_ccv_nnc_graph_tensor_dot_recovery_free(recovery);
		ccfree(node_id);
	}
	// Extra subgraph cluster.
	if (exec_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		fputs("}\n", out);
}

void ccv_nnc_graph_dot(const ccv_nnc_graph_t* const graph, const int flags, FILE* out)
{
	fputs("digraph G {\ncompound=true;\n", out);
	ccv_nnc_tensor_dot_recovery_t recovery = _ccv_nnc_graph_tensor_dot_recovery(graph);
	int i, j;
	int k = 0, c = 0;
	int* node_id = (int*)ccmalloc(sizeof(int) * graph->exec_info->rnum);
	// Output styles.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		node_id[i] = c;
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (CCV_NNC_GRAPH_REF(exec_info)[0])
			_ccv_nnc_graph_dot_sub_graphs(exec_info, recovery, graph->sub_graphs, flags, 1, out, &k, &c);
		else {
			_ccv_nnc_graph_dot_node(exec_info, c, recovery, flags, 0, out, &k);
			++c;
		}
	}
	// Output connections.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (exec_info->outgoings)
			for (j = 0; j < exec_info->outgoings->rnum; j++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(exec_info->outgoings, j);
				const ccv_nnc_graph_exec_info_t* const outgoing_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, outgoing_idx);
				// If both are sub-graphs, have both tail and head specified.
				if (CCV_NNC_GRAPH_REF(exec_info)[0] && CCV_NNC_GRAPH_REF(outgoing_info)[0])
					fprintf(out, "node%d -> node%d [ltail=cluster%d,lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i], node_id[outgoing_idx]);
				else if (CCV_NNC_GRAPH_REF(exec_info)[0] && !CCV_NNC_GRAPH_REF(outgoing_info)[0])
					fprintf(out, "node%d -> node%d [ltail=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i]);
				else if (!CCV_NNC_GRAPH_REF(exec_info)[0] && CCV_NNC_GRAPH_REF(outgoing_info)[0])
					fprintf(out, "node%d -> node%d [lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[outgoing_idx]);
				else
					fprintf(out, "node%d -> node%d;\n", node_id[i], node_id[outgoing_idx]);
			}
	}
	fputs("}\n", out);
	_ccv_nnc_graph_tensor_dot_recovery_free(recovery);
	ccfree(node_id);
}

void ccv_nnc_graph_autotune(ccv_nnc_graph_t* const graph, const size_t max_workspace_size, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	// exec current node, for synchronous CPU execution, no stream unit.
#define visitor(node, idx, ...) \
	do { \
		node->cmd = ccv_nnc_cmd_autotune(node->cmd, max_workspace_size, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0); \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, 0, visitor);
#undef visitor
}

void ccv_nnc_graph_free(ccv_nnc_graph_t* const graph)
{
	int i, j;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t *info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (info->_heap_graph_ref)
			ccfree(info->_heap_graph_ref);
		ccv_array_t* outgoings = info->outgoings;
		if (outgoings)
			ccv_array_free(outgoings);
		// We allocate inputs & outputs in continuous fashion, therefore, only need to free the input array.
		if (info->inputs)
			ccfree(info->inputs);
		if (info->input_flags)
			ccfree(info->input_flags);
		if (info->updates)
			ccfree(info->updates);
		if (info->tensor_wrap_size)
		{
			for (j = 0; j < info->tensor_wrap_size; j++)
				if (info->tensor_wraps[j])
					_ccv_nnc_graph_tensor_wrap_free(info->tensor_wraps[j]);
			ccfree(info->tensor_wraps);
		}
		if ((info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) && info->p_while.inputs)
			ccfree(info->p_while.inputs);
	}
	if (graph->breakpoints)
		ccfree(graph->breakpoints);
	if (graph->sources)
		ccv_array_free(graph->sources);
	if (graph->destinations)
		ccv_array_free(graph->destinations);
	if (graph->exec_wraps)
		ccv_array_free(graph->exec_wraps);
	if (graph->carry_overs)
	{
		for (i = 0; i < graph->carry_overs->rnum; i++)
		{
			ccv_nnc_graph_tensor_carry_over_t* const carry_over = (ccv_nnc_graph_tensor_carry_over_t*)ccv_array_get(graph->carry_overs, i);
			_ccv_nnc_graph_tensor_wrap_free(carry_over->from);
			_ccv_nnc_graph_tensor_wrap_free(carry_over->to);
		}
		ccv_array_free(graph->carry_overs);
	}
	if (graph->sub_graphs)
	{
		for (i = 0; i < graph->sub_graphs->rnum; i++)
			ccv_nnc_graph_free(*(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, i));
		ccv_array_free(graph->sub_graphs);
	}
	ccv_array_free(graph->exec_info);
	ccfree(graph);
}
