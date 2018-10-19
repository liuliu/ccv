#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"
#include "_ccv_nnc_stream.h"

#pragma mark - Level-2 API

static void _ccv_nnc_unwrap_tensor_wrap(const ccv_nnc_graph_t* const graph, const int64_t count, const int64_t reverse_count, ccv_nnc_graph_tensor_wrap_t* const tensor_wrap)
{
	ccv_nnc_tensor_t* tensor = tensor_wrap->tensors[tensor_wrap->index];
	while (CCV_IS_TENSOR_MULTIVIEW(tensor) &&
		   (((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph ||
			((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph->peer))
	{
		// If the anchor is from the peer, we use the reverse_count instead (we are looking it up).
		const int i = (int)((((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph) ? count : reverse_count);
		ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
		const int off = mv->kind;
		const int mod = mv->repeat;
		tensor = CCV_NNC_MULTIVIEW_DATA(mv)[i >= off ? ((i - off) % mod) + off : i]; // Unwrap.
		// If reached the root.
		if (!CCV_IS_TENSOR_MULTIVIEW(tensor))
			tensor_wrap->update_required = 1; // Need to update tensor updates.
		++tensor_wrap->index;
		tensor_wrap->tensors[tensor_wrap->index] = tensor;
		assert(tensor_wrap->index < tensor_wrap->count);
	}
}

static void _ccv_nnc_graph_unwrap_sub_graph(const ccv_nnc_graph_t* const graph, const int64_t count, const int64_t reverse_count, const ccv_nnc_graph_t* const sub_graph)
{
	int i;
	if (sub_graph->carry_overs)
		for (i = 0; i < sub_graph->carry_overs->rnum; i++)
		{
			ccv_nnc_graph_tensor_carry_over_t* const carry_over = (ccv_nnc_graph_tensor_carry_over_t*)ccv_array_get(sub_graph->carry_overs, i);
			_ccv_nnc_unwrap_tensor_wrap(graph, count, reverse_count, carry_over->from);
			_ccv_nnc_unwrap_tensor_wrap(graph, count, reverse_count, carry_over->to);
		}
	if (sub_graph->sub_graphs)
		for (i = 0; i < sub_graph->sub_graphs->rnum; i++)
			_ccv_nnc_graph_unwrap_sub_graph(graph, count, reverse_count, *(ccv_nnc_graph_t**)ccv_array_get(sub_graph->sub_graphs, i));
}

static void _ccv_nnc_graph_unwrap(const ccv_nnc_graph_t* const graph, const int64_t count, const int64_t reverse_count)
{
	if (!graph->tensor_wraps_refs)
		return;
	int i, j;
	for (i = 0; i < graph->tensor_wraps_refs->rnum; i++)
	{
		const ccv_nnc_graph_tensor_wraps_ref_t* const tensor_wraps_ref = (const ccv_nnc_graph_tensor_wraps_ref_t*)ccv_array_get(graph->tensor_wraps_refs, i);
		const ccv_nnc_graph_t* const sub_graph = tensor_wraps_ref->graph;
		ccv_nnc_graph_tensor_wrap_array_t* const tensor_wrap_array = *(ccv_nnc_graph_tensor_wrap_array_t**)ccv_array_get(sub_graph->tensor_wraps, tensor_wraps_ref->d);
		if (tensor_wrap_array)
			for (j = 0; j < tensor_wrap_array->size; j++)
			{
				ccv_nnc_graph_tensor_wrap_t* const tensor_wrap = tensor_wrap_array->tensor_wraps[j];
				if (!tensor_wrap)
					continue;
				_ccv_nnc_unwrap_tensor_wrap(graph, count, reverse_count, tensor_wrap);
			}
	}
	_ccv_nnc_graph_unwrap_sub_graph(graph, count, reverse_count, graph);
}

static void _ccv_nnc_graph_transit_move_to(const ccv_nnc_graph_t* const graph)
{
	int i;
	if (graph->carry_overs)
		for (i = 0; i < graph->carry_overs->rnum; i++)
		{
			ccv_nnc_graph_tensor_carry_over_t* const carry_over = (ccv_nnc_graph_tensor_carry_over_t*)ccv_array_get(graph->carry_overs, i);
			ccv_nnc_tensor_t* it = (ccv_nnc_tensor_t*)(carry_over->to->tensors[carry_over->to->index]);
			assert(!CCV_IS_TENSOR_MULTIVIEW(it));
			it->data = carry_over->transit;
		}
}

static void _ccv_nnc_graph_from_move_transit(const ccv_nnc_graph_t* const graph)
{
	int i;
	if (graph->carry_overs)
		for (i = 0; i < graph->carry_overs->rnum; i++)
		{
			ccv_nnc_graph_tensor_carry_over_t* const carry_over = (ccv_nnc_graph_tensor_carry_over_t*)ccv_array_get(graph->carry_overs, i);
			ccv_nnc_tensor_t* it = (ccv_nnc_tensor_t*)(carry_over->from->tensors[carry_over->from->index]);
			assert(!CCV_IS_TENSOR_MULTIVIEW(it));
			carry_over->transit = it->data;
		}
}

static void _ccv_nnc_rewrap_tensor_wrap(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_tensor_wrap_t* const tensor_wrap)
{
	while (tensor_wrap->index > 0 && CCV_IS_TENSOR_MULTIVIEW(tensor_wrap->tensors[tensor_wrap->index - 1]) &&
			(((ccv_nnc_tensor_multiview_t*)tensor_wrap->tensors[tensor_wrap->index - 1])->anchor == (intptr_t)graph ||
			 ((ccv_nnc_tensor_multiview_t*)tensor_wrap->tensors[tensor_wrap->index - 1])->anchor == (intptr_t)graph->peer))
		--tensor_wrap->index;
}

static void _ccv_nnc_graph_rewrap_sub_graph(const ccv_nnc_graph_t* const graph, const ccv_nnc_graph_t* const sub_graph)
{
	int i;
	if (sub_graph->carry_overs)
		for (i = 0; i < sub_graph->carry_overs->rnum; i++)
		{
			ccv_nnc_graph_tensor_carry_over_t* const carry_over = (ccv_nnc_graph_tensor_carry_over_t*)ccv_array_get(sub_graph->carry_overs, i);
			_ccv_nnc_rewrap_tensor_wrap(graph, carry_over->from);
			_ccv_nnc_rewrap_tensor_wrap(graph, carry_over->to);
		}
	if (sub_graph->sub_graphs)
		for (i = 0; i < sub_graph->sub_graphs->rnum; i++)
			_ccv_nnc_graph_rewrap_sub_graph(graph, *(ccv_nnc_graph_t**)ccv_array_get(sub_graph->sub_graphs, i));
}

static void _ccv_nnc_graph_rewrap(const ccv_nnc_graph_t* const graph) // Call this method at the end to roll the wrap_ptr back
{
	if (!graph->tensor_wraps_refs)
		return;
	int i, j;
	for (i = 0; i < graph->tensor_wraps_refs->rnum; i++)
	{
		const ccv_nnc_graph_tensor_wraps_ref_t* const tensor_wraps_ref = (const ccv_nnc_graph_tensor_wraps_ref_t*)ccv_array_get(graph->tensor_wraps_refs, i);
		const ccv_nnc_graph_t* const sub_graph = tensor_wraps_ref->graph;
		ccv_nnc_graph_tensor_wrap_array_t* const tensor_wrap_array = *(ccv_nnc_graph_tensor_wrap_array_t**)ccv_array_get(sub_graph->tensor_wraps, tensor_wraps_ref->d);
		if (tensor_wrap_array)
			for (j = 0; j < tensor_wrap_array->size; j++)
			{
				ccv_nnc_graph_tensor_wrap_t* const tensor_wrap = tensor_wrap_array->tensor_wraps[j];
				if (!tensor_wrap)
					continue;
				_ccv_nnc_rewrap_tensor_wrap(graph, tensor_wrap);
			}
	}
	_ccv_nnc_graph_rewrap_sub_graph(graph, graph);
}

static void _ccv_nnc_graph_exec_unwrap_io(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node)
{
	if (!node->tensor_wraps_ref)
		return;
	int i;
	ccv_nnc_graph_tensor_wrap_array_t* const tensor_wrap_array = *(ccv_nnc_graph_tensor_wrap_array_t**)ccv_array_get(graph->tensor_wraps, node->tensor_wraps_ref - 1);
	ccv_nnc_graph_tensor_wrap_t** const tensor_wraps = tensor_wrap_array->tensor_wraps;
	for (i = 0; i < tensor_wrap_array->size; i++)
		if (tensor_wraps[i])
		{
			assert(tensor_wraps[i]->index > 0);
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)(tensor_wraps[i]->tensors[tensor_wraps[i]->index - 1]);
			assert(CCV_IS_TENSOR_MULTIVIEW(mv));
			// Only now set the mv->it, because now this node is about to get executed.
			mv->it = tensor_wraps[i]->tensors[tensor_wraps[i]->index];
			assert(!CCV_IS_TENSOR_MULTIVIEW(mv->it));
		}
	for (i = 0; i < node->input_size; i++)
		if (tensor_wraps[i])
			node->inputs[i] = tensor_wraps[i]->tensors[tensor_wraps[i]->index];
	const int d = node->input_size;
	for (i = 0; i < node->output_size; i++)
		if (tensor_wraps[d + i])
			node->outputs[i] = tensor_wraps[d + i]->tensors[tensor_wraps[d + i]->index];
}

static void _ccv_nnc_graph_exec_unwrap_while_expr(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node)
{
	assert(node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE);
	if (!node->p_while.tensor_wraps_ref)
		return;
	int i;
	ccv_nnc_graph_tensor_wrap_array_t* const tensor_wrap_array = *(ccv_nnc_graph_tensor_wrap_array_t**)ccv_array_get(graph->tensor_wraps, node->p_while.tensor_wraps_ref - 1);
	ccv_nnc_graph_tensor_wrap_t** const tensor_wraps = tensor_wrap_array->tensor_wraps;
	for (i = 0; i < tensor_wrap_array->size; i++)
		if (tensor_wraps[i])
		{
			assert(tensor_wraps[i]->index > 0);
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)(tensor_wraps[i]->tensors[tensor_wraps[i]->index - 1]);
			assert(CCV_IS_TENSOR_MULTIVIEW(mv));
			// Only now set the mv->it, because now this node is about to get executed.
			mv->it = tensor_wraps[i]->tensors[tensor_wraps[i]->index];
			assert(!CCV_IS_TENSOR_MULTIVIEW(mv->it));
		}
	for (i = 0; i < node->p_while.input_size; i++)
		if (tensor_wraps[i])
			node->p_while.inputs[i] = tensor_wraps[i]->tensors[tensor_wraps[i]->index];
}

static void _ccv_nnc_graph_exec_unwrap_phi(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_info_t* const node, const int ref)
{
	int i;
	// If the output tensor is a phi multi-view tensor, we update our selection to all the subscribers.
	for (i = 0; i < node->output_size; i++)
		if (CCV_IS_TENSOR_MULTIVIEW(node->outputs[i]) &&
			((ccv_nnc_tensor_multiview_t*)node->outputs[i])->anchor == CCV_NNC_MULTIVIEW_PHI)
		{
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)node->outputs[i];
			mv->it = CCV_NNC_MULTIVIEW_DATA(mv)[ref >= 0];
			ccv_nnc_tensor_multiview_synchronize(mv);
		}
}

static void _ccv_nnc_graph_exec_begin_synchronize_multiviews(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node)
{
	if (!node->tensor_wraps_ref)
		return;
	int i;
	ccv_nnc_graph_tensor_wrap_array_t* const tensor_wrap_array = *(ccv_nnc_graph_tensor_wrap_array_t**)ccv_array_get(graph->tensor_wraps, node->tensor_wraps_ref - 1);
	ccv_nnc_graph_tensor_wrap_t** const tensor_wraps = tensor_wrap_array->tensor_wraps;
	for (i = 0; i < tensor_wrap_array->size; i++)
		if (tensor_wraps[i] && tensor_wraps[i]->update_required)
		{
			assert(tensor_wraps[i]->index > 0);
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)(tensor_wraps[i]->tensors[tensor_wraps[i]->index - 1]);
			// Now update the final pointer.
			ccv_nnc_tensor_multiview_synchronize(mv);
			tensor_wraps[i]->update_required = 0; // Reset, no need to update.
		}
}

static void _ccv_nnc_print_tensor_verbose(const ccv_nnc_tensor_t* const tensor)
{
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) != CCV_TENSOR_CPU_MEMORY)
		return;
	int i;
	switch (tensor->info.datatype)
	{
		case CCV_32F:
			for (i = 0; i < ccv_min(tensor->info.dim[0], 3); i++)
				PRINT(CCV_CLI_VERBOSE, " %f", tensor->data.f32[i]);
			break;
		case CCV_64F:
			for (i = 0; i < ccv_min(tensor->info.dim[0], 3); i++)
				PRINT(CCV_CLI_VERBOSE, " %f", tensor->data.f64[i]);
			break;
		case CCV_32S:
			for (i = 0; i < ccv_min(tensor->info.dim[0], 3); i++)
				PRINT(CCV_CLI_VERBOSE, " %d", tensor->data.i32[i]);
			break;
		case CCV_64S:
			for (i = 0; i < ccv_min(tensor->info.dim[0], 3); i++)
				PRINT(CCV_CLI_VERBOSE, " %lld", (long long)tensor->data.i64[i]);
			break;
		case CCV_8U:
			for (i = 0; i < ccv_min(tensor->info.dim[0], 3); i++)
				PRINT(CCV_CLI_VERBOSE, " %d", (int)tensor->data.u8[i]);
			break;
	}
	if (ccv_nnc_tensor_count(tensor->info) > 3)
		PRINT(CCV_CLI_VERBOSE, " ..");
}

typedef struct {
	ccv_nnc_graph_t* graph;
	int exec_idx;
	ccv_nnc_graph_exec_info_t* exec;
	ccv_nnc_tensor_tape_t* tensor_tape;
	ccv_nnc_stream_context_t* stream_context;
	int flags;
} ccv_nnc_graph_topsorted_run_coro_t;

static void _ccv_nnc_graph_topsorted_run_coro(ccv_nnc_stream_task_t* const self, void* const userdata);

typedef struct {
	ccv_nnc_graph_t* graph;
	int exec_idx;
	ccv_nnc_graph_exec_info_t* exec;
	ccv_nnc_tensor_t* const* inputs;
	ccv_nnc_tensor_tape_t* tensor_tape;
	ccv_nnc_stream_context_t* stream_context;
	int flags;
} ccv_nnc_graph_exec_cases_of_coro_t;

static void _ccv_nnc_graph_exec_cases_of_coro(ccv_nnc_stream_task_t* const self, void* const userdata)
{
	const ccv_nnc_graph_exec_cases_of_coro_t* const params = (ccv_nnc_graph_exec_cases_of_coro_t*)userdata;
	ccv_nnc_graph_t* const graph = params->graph;
	const int exec_idx = params->exec_idx;
	ccv_nnc_graph_exec_info_t* const exec = params->exec;
	ccv_nnc_tensor_t* const* const inputs = params->inputs;
	ccv_nnc_tensor_tape_t* const tensor_tape = params->tensor_tape;
	ccv_nnc_stream_context_t* const stream_context = params->stream_context;
	const int flags = params->flags;
	// Wait until this stream context is done.
	ccv_nnc_stream_task_synchronize(self, stream_context);
	int ref;
	if (exec->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
	{
		ref = exec->case_of.offset + exec->case_of.expr(inputs, exec->input_size, exec->case_of.data);
		if (tensor_tape)
			ccv_nnc_tensor_tape_set_numbering(tensor_tape, graph, (ccv_nnc_graph_exec_t){
				.d = exec_idx,
				.graph = graph,
			}, ref);
	} else {
		assert(exec->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
		assert(tensor_tape);
		ref = ccv_nnc_tensor_tape_numbering(tensor_tape, graph, (ccv_nnc_graph_exec_t){
				.d = exec_idx,
				.graph = graph,
			});
	}
	if (ref >= 0)
	{
		assert(ref < exec->graph_ref_size);
		ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(exec)[ref] - 1);
		assert(graph->streams[exec->schedule.stream] == sub_graph->streams[0]);
		ccv_nnc_graph_topsorted_run_coro_t params = {
			.graph = sub_graph,
			.exec_idx = exec_idx,
			.exec = exec,
			.tensor_tape = tensor_tape,
			.stream_context = graph->streams[exec->schedule.stream],
			.flags = flags
		};
		// Directly call it.
		_ccv_nnc_graph_topsorted_run_coro(self, &params);
	}
	_ccv_nnc_graph_exec_unwrap_phi(graph, exec, ref);
}

static inline ccv_nnc_stream_task_t* _ccv_nnc_graph_exec_run_task(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node, const int idx, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_scheduler_t* const scheduler, const int flags)
{
	_ccv_nnc_graph_exec_unwrap_io(graph, node);
	ccv_nnc_tensor_t** inputs = node->inputs;
	ccv_nnc_tensor_t** outputs = inputs + node->input_size;
	if (tensor_tape)
		ccv_nnc_tensor_tape_io(tensor_tape, graph, node->input_flags, inputs, node->input_size, node->output_flags, outputs, node->output_size);
	/* Broadcast the updates to all subscribed references for input / output, even though at th
	 * time output is not written yet, propagate pointer change is still valid. */
	_ccv_nnc_graph_exec_begin_synchronize_multiviews(graph, node);
	if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
	{
		if (node->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			ccv_nnc_stream_context_t* const node_stream = graph->streams[node->schedule.stream];
			ccv_nnc_graph_exec_cases_of_coro_t params = {
				.graph = graph,
				.exec_idx = idx,
				.exec = node,
				.inputs = inputs,
				.tensor_tape = tensor_tape,
				.stream_context = node_stream,
				.flags = flags,
			};
			ccv_nnc_stream_task_t* const task = ccv_nnc_stream_task_new(scheduler, _ccv_nnc_graph_exec_cases_of_coro, &params, 0);
			ccv_nnc_stream_task_resume(task);
			return task;
		} else if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) {
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[0] - 1);
			assert(graph->streams[node->schedule.stream] == sub_graph->streams[0]);
			ccv_nnc_graph_topsorted_run_coro_t params = {
				.graph = sub_graph,
				.exec_idx = idx,
				.exec = node,
				.tensor_tape = tensor_tape,
				.stream_context = graph->streams[node->schedule.stream],
				.flags = flags
			};
			ccv_nnc_stream_task_t* const task = ccv_nnc_stream_task_new(scheduler, _ccv_nnc_graph_topsorted_run_coro, &params, 0);
			ccv_nnc_stream_task_resume(task);
			return task;
		}
	} else {
		ccv_nnc_stream_context_t* const node_stream = graph->streams[node->schedule.stream];
		int i;
		for (i = 0; i < node->schedule.wait_size; i++)
			ccv_nnc_stream_context_wait_signal(node_stream, graph->signals[node->schedule.waits[i]]);
		ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, node_stream);
		if (node->schedule.sign >= 0)
			ccv_nnc_stream_context_emit_signal(node_stream, graph->signals[node->schedule.sign]);
	}
	return 0;
}

static void _ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const exec_info, ccv_nnc_graph_exec_info_t* const node, ccv_nnc_stream_task_t* const task)
{
	int i;
	if (node->outgoings)
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, i);
			ccv_nnc_graph_exec_info_t* const outgoing_node = exec_info + outgoing_idx;
			// An outgoing stream can be blocked by multiple other tasks from other streams. But it is OK,
			// because on next round of execution, that one will be marked as blocked again.
			graph->block_stream_tasks[outgoing_node->schedule.stream] = task;
		}
}

static void _ccv_nnc_graph_wait_any_sub_tasks(ccv_nnc_stream_task_t* const self, ccv_nnc_graph_t* const graph, ccv_nnc_stream_task_t* const* const sub_tasks, const int sub_task_size, ccv_nnc_graph_exec_info_t* const exec_info, const int* const pending_nodes, const int pending_node_size)
{
	int i, j;
	if (sub_task_size)
		ccv_nnc_stream_task_wait_any(self, sub_tasks, sub_task_size);
	for (i = 0; i < sub_task_size; i++)
		if (sub_tasks[i]->done)
			for (j = 0; j < pending_node_size; j++)
			{
				ccv_nnc_graph_exec_info_t* const node = exec_info + pending_nodes[j];
				if (graph->block_stream_tasks[node->schedule.stream] == sub_tasks[i])
					graph->block_stream_tasks[node->schedule.stream] = 0;
			}
}

static void _ccv_nnc_graph_exec_run_loop(ccv_nnc_stream_task_t* const self, ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const exec_info, const int start_index, const int exec_info_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags)
{
	int i;
	int sub_task_size = 0;
	ccv_nnc_stream_task_t** const sub_tasks = (ccv_nnc_stream_task_t**)ccv_nnc_graph_buffer(graph, sizeof(ccv_nnc_stream_task_t*) * (graph->sub_graphs ? graph->sub_graphs->rnum : 0) + sizeof(int) * exec_info_size * 2);
	int* pending_nodes[2];
	pending_nodes[0] = (int*)(sub_tasks + (graph->sub_graphs ? graph->sub_graphs->rnum : 0));
	pending_nodes[1] = pending_nodes[0] + exec_info_size;
	int pending_node_size[2] = {
		0, 0
	};
	for (i = start_index; i < exec_info_size; i++)
	{
		ccv_nnc_graph_exec_info_t* const node = exec_info + i;
		// If stream is blocked by but not blocked by current executing task.
		if (graph->block_stream_tasks[node->schedule.stream])
		{
			pending_nodes[0][pending_node_size[0]++] = i;
			_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(graph, exec_info, node, graph->block_stream_tasks[node->schedule.stream]);
			continue;
		}
		ccv_nnc_stream_task_t* const task = _ccv_nnc_graph_exec_run_task(graph, node, i, tensor_tape, self->super, flags);
		if (task && !task->done)
		{
			sub_tasks[sub_task_size++] = task;
			graph->block_stream_tasks[node->schedule.stream] = task;
			_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(graph, exec_info, node, task);
		}
	}
	_ccv_nnc_graph_wait_any_sub_tasks(self, graph, sub_tasks, sub_task_size, exec_info, pending_nodes[0], pending_node_size[0]);
	int p = 0, q = 1;
	while (pending_node_size[p] > 0)
	{
		pending_node_size[q] = 0;
		sub_task_size = 0;
		for (i = 0; i < pending_node_size[p]; i++)
		{
			const int idx = pending_nodes[p][i];
			ccv_nnc_graph_exec_info_t* const node = exec_info + idx;
			if (graph->block_stream_tasks[node->schedule.stream])
			{
				_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(graph, exec_info, node, graph->block_stream_tasks[node->schedule.stream]);
				pending_nodes[q][pending_node_size[q]++] = idx;
				continue;
			}
			ccv_nnc_stream_task_t* const task = _ccv_nnc_graph_exec_run_task(graph, node, idx, tensor_tape, self->super, flags);
			if (task && !task->done)
			{
				sub_tasks[sub_task_size++] = task;
				graph->block_stream_tasks[node->schedule.stream] = task;
				_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(graph, exec_info, node, task);
			}
		}
		int t;
		CCV_SWAP(p, q, t);
		_ccv_nnc_graph_wait_any_sub_tasks(self, graph, sub_tasks, sub_task_size, exec_info, pending_nodes[p], pending_node_size[p]);
	}
}

static void _ccv_nnc_graph_topsorted_run_coro(ccv_nnc_stream_task_t* const self, void* const userdata)
{
	const ccv_nnc_graph_topsorted_run_coro_t* const params = (ccv_nnc_graph_topsorted_run_coro_t*)userdata;
	ccv_nnc_graph_t* const graph = params->graph;
	const int exec_idx = params->exec_idx;
	ccv_nnc_graph_exec_info_t* const exec = params->exec;
	ccv_nnc_tensor_tape_t* const tensor_tape = params->tensor_tape;
	ccv_nnc_stream_context_t* const stream_context = params->stream_context;
	const int flags = params->flags;
	int i;
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0);
	if (exec_idx == -1)
	{
		if (stream_context->main)
		{
			ccv_nnc_stream_task_t* const previous_main = stream_context->main;
			stream_context->main = self;
			// Wait the previous task to be done. This makes sure that our graph run is serial on the same stream.
			assert(!previous_main->done);
			ccv_nnc_stream_task_wait_any(self, &previous_main, 1);
		} else
			stream_context->main = self;
	}
	if (exec && (exec->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
	{
		assert(exec->p_while.expr);
		int64_t count = 0;
		// This is a forward while loop. Backward while loop will just consult its peering part.
		if (exec->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
		{
			const int graph_breakpoint_size = graph->breakpoint_offset + graph->breakpoint_size;
			for (;; ++count)
			{
				graph->while_count = count;
				if (tensor_tape)
					ccv_nnc_tensor_tape_set_numbering(tensor_tape, graph->p, (ccv_nnc_graph_exec_t){
						.d = exec_idx,
						.graph = graph->p,
					}, count);
				_ccv_nnc_graph_unwrap(graph, count, 0);
				if (count > 0)
					_ccv_nnc_graph_transit_move_to(graph);
				_ccv_nnc_graph_exec_run_loop(self, graph, exec_info, 0, graph_breakpoint_size, tensor_tape, flags);
				// Reached breakpoints, now check the breakpoint, if not met, break out.
				// Wait until everything on the stream is executed.
				for (i = graph->breakpoint_offset; i < graph_breakpoint_size; i++)
					ccv_nnc_stream_task_synchronize(self, graph->streams[exec_info[i].schedule.stream]);
				_ccv_nnc_graph_exec_unwrap_while_expr(graph, exec);
				if (!exec->p_while.expr(exec->p_while.inputs, exec->p_while.input_size, exec->p_while.data))
				{
					_ccv_nnc_graph_rewrap(graph);
					// If we break from here, it is ok because all the streams are waited.
					break;
				}
				_ccv_nnc_graph_exec_run_loop(self, graph, exec_info, graph_breakpoint_size, graph->exec_info->rnum, tensor_tape, flags);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
			}
		} else {
			// For backward graph, no need to evaluate the while expr.
			assert(exec->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
			assert(graph->peer);
			assert(tensor_tape);
			count = 0;
			int64_t reverse_count = graph->while_count = ccv_nnc_tensor_tape_numbering(tensor_tape, graph->p, (ccv_nnc_graph_exec_t){
					.d = exec_idx,
					.graph = graph->p,
				});
			_ccv_nnc_graph_unwrap(graph, count, reverse_count);
			_ccv_nnc_graph_exec_run_loop(self, graph, exec_info, graph->breakpoint_offset, graph->exec_info->rnum, tensor_tape, flags);
			_ccv_nnc_graph_from_move_transit(graph);
			_ccv_nnc_graph_rewrap(graph);
			for (count = 1; reverse_count > 0; ++count)
			{
				graph->while_count = --reverse_count;
				_ccv_nnc_graph_unwrap(graph, count, reverse_count);
				_ccv_nnc_graph_transit_move_to(graph);
				_ccv_nnc_graph_exec_run_loop(self, graph, exec_info, 0, graph->exec_info->rnum, tensor_tape, flags);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
			}
		}
		for (i = 0; i < graph->wait_size; i++)
			ccv_nnc_stream_context_wait_signal(graph->streams[0], graph->signals[graph->waits[i]]);
	} else {
		graph->while_count = 0;
		_ccv_nnc_graph_exec_run_loop(self, graph, exec_info, 0, graph->exec_info->rnum, tensor_tape, flags);
		for (i = 0; i < graph->wait_size; i++)
			ccv_nnc_stream_context_wait_signal(graph->streams[0], graph->signals[graph->waits[i]]);
	}
	if (stream_context != graph->streams[0])
	{
		ccv_nnc_stream_context_emit_signal(graph->streams[0], graph->extern_signal);
		ccv_nnc_stream_context_wait_signal(stream_context, graph->extern_signal);
	}
	// Reset main to 0 if it is current me.
	if (exec_idx == -1 && stream_context->main == self)
		stream_context->main = 0;
}

static int _ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size);

static inline void _ccv_nnc_graph_exec_run(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node, const int idx, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags)
{
	int i;
	_ccv_nnc_graph_exec_unwrap_io(graph, node);
	ccv_nnc_tensor_t** inputs = node->inputs;
	ccv_nnc_tensor_t** outputs = inputs + node->input_size;
	if (tensor_tape)
		ccv_nnc_tensor_tape_io(tensor_tape, graph, node->input_flags, inputs, node->input_size, node->output_flags, outputs, node->output_size);
	/* Broadcast the updates to all subscribed references for input / output, even though at th
	 * time output is not written yet, propagate pointer change is still valid. */
	_ccv_nnc_graph_exec_begin_synchronize_multiviews(graph, node);
	if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
	{
		assert(!stream_context); // This doesn't work properly with stream context.
		if (node->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			int ref;
			if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
			{
				ref = node->case_of.offset + node->case_of.expr(inputs, node->input_size, node->case_of.data);
				if (tensor_tape)
					ccv_nnc_tensor_tape_set_numbering(tensor_tape, graph, (ccv_nnc_graph_exec_t){
						.d = idx,
						.graph = graph,
					}, ref);
			} else {
				assert(node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
				assert(tensor_tape);
				ref = ccv_nnc_tensor_tape_numbering(tensor_tape, graph, (ccv_nnc_graph_exec_t){
						.d = idx,
						.graph = graph,
					});
			}
			if (ref >= 0)
			{
				assert(ref < node->graph_ref_size);
				ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[ref] - 1);
				_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, tensor_tape, stream_context, flags, 0, 0, 0, 0);
			}
			_ccv_nnc_graph_exec_unwrap_phi(graph, node, ref);
		} else if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) {
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[0] - 1);
			_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, tensor_tape, stream_context, flags, 0, 0, 0, 0);
		}
	} else {
		PRINT(CCV_CLI_VERBOSE, "%s [%d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, node->input_size, node->output_size);
		for (i = 0; i < node->input_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)", i + 1, inputs[i], (inputs[i] ? inputs[i]->data.u8 : 0));
			if (inputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(inputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
		ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, stream_context);
		for (i = 0; i < node->output_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)", i + 1, outputs[i], (outputs[i] ? outputs[i]->data.u8 : 0));
			if (outputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(outputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
	}
}

static inline void _ccv_nnc_graph_topsorted_run(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags)
{
	int i;
	if (exec && (exec->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
	{
		assert(!stream_context); // This doesn't work properly with stream context.
		assert(exec->p_while.expr);
		int64_t count = 0;
		// This is a forward while loop. Backward while loop will just consult its peering part.
		if (exec->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
		{
			const int graph_breakpoint_size = graph->breakpoint_offset + graph->breakpoint_size;
			for (;; ++count)
			{
				graph->while_count = count;
				if (tensor_tape)
					ccv_nnc_tensor_tape_set_numbering(tensor_tape, graph->p, (ccv_nnc_graph_exec_t){
						.d = exec_idx,
						.graph = graph->p,
					}, count);
				_ccv_nnc_graph_unwrap(graph, count, 0);
				if (count > 0)
					_ccv_nnc_graph_transit_move_to(graph);
				for (i = 0; i < graph_breakpoint_size; i++)
					_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, tensor_tape, stream_context, flags);
				_ccv_nnc_graph_exec_unwrap_while_expr(graph, exec);
				// Reached breakpoints, now check the breakpoint, if not met, break out.
				if (!exec->p_while.expr(exec->p_while.inputs, exec->p_while.input_size, exec->p_while.data))
				{
					_ccv_nnc_graph_rewrap(graph);
					break;
				}
				for (i = graph_breakpoint_size; i < graph->exec_info->rnum; i++)
					_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, tensor_tape, stream_context, flags);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
			}
		} else {
			// For backward graph, no need to evaluate the while expr.
			assert(exec->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
			assert(graph->peer);
			assert(tensor_tape);
			count = 0;
			int64_t reverse_count = graph->while_count = ccv_nnc_tensor_tape_numbering(tensor_tape, graph->p, (ccv_nnc_graph_exec_t){
					.d = exec_idx,
					.graph = graph->p,
				});
			_ccv_nnc_graph_unwrap(graph, count, reverse_count);
			for (i = graph->breakpoint_offset; i < graph->exec_info->rnum; i++)
				_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, tensor_tape, stream_context, flags);
			_ccv_nnc_graph_from_move_transit(graph);
			_ccv_nnc_graph_rewrap(graph);
			for (count = 1; reverse_count > 0; ++count)
			{
				graph->while_count = --reverse_count;
				_ccv_nnc_graph_unwrap(graph, count, reverse_count);
				_ccv_nnc_graph_transit_move_to(graph);
				for (i = 0; i < graph->exec_info->rnum; i++)
					_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, tensor_tape, stream_context, flags);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
			}
		}
	} else {
		graph->while_count = 0;
		for (i = 0; i < graph->exec_info->rnum; i++)
			_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, tensor_tape, stream_context, flags);
	}
}

static inline void _ccv_nnc_graph_run_slow_path(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	int i, j;
	const ccv_nnc_graph_exec_t* const graph_sources = sources ? sources : (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, 0);
	const int graph_source_size = source_size ? source_size : graph->sources->rnum;
	const ccv_nnc_graph_exec_t* const graph_destinations = destinations ? destinations : (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, 0);
	const int graph_destination_size = destination_size ? destination_size : graph->destinations->rnum;
#define visitor(node, idx, ...) \
	_ccv_nnc_graph_exec_run(graph, node, idx, tensor_tape, stream_context, flags)
	if (exec && (exec->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
	{
		assert(!stream_context); // This doesn't work properly with stream context.
		assert(exec->p_while.expr);
		int64_t count = 0;
		// This is a forward while loop. Backward while loop will just consult its peering part.
		if (exec->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
		{
			ccv_array_t* follows = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), graph->breakpoint_size, 0);
			for (i = 0; i < graph->breakpoint_size; i++)
			{
				const ccv_nnc_graph_exec_info_t* const exec_info = (const ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, graph->breakpoints->d);
				if (exec_info->outgoings)
					for (j = 0; j < exec_info->outgoings->rnum; j++)
					{
						const ccv_nnc_graph_exec_t exec = {
							.d = *(int*)ccv_array_get(exec_info->outgoings, j),
							.graph = graph,
						};
						ccv_array_push(follows, &exec);
					}
			}
			for (;; ++count)
			{
				graph->while_count = count;
				if (tensor_tape)
					ccv_nnc_tensor_tape_set_numbering(tensor_tape, graph->p, (ccv_nnc_graph_exec_t){
						.d = exec_idx,
						.graph = graph->p,
					}, count);
				_ccv_nnc_graph_unwrap(graph, count, 0);
				if (count > 0)
					_ccv_nnc_graph_transit_move_to(graph);
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph_sources, graph_source_size, graph->breakpoints, graph->breakpoint_size, 0, visitor);
				_ccv_nnc_graph_exec_unwrap_while_expr(graph, exec);
				// Reached breakpoints, now check the breakpoint, if not met, break out.
				if (!exec->p_while.expr(exec->p_while.inputs, exec->p_while.input_size, exec->p_while.data))
				{
					_ccv_nnc_graph_rewrap(graph);
					break;
				}
				if (follows->rnum > 0)
					CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(follows, 0), follows->rnum, graph_destinations, graph_destination_size, 0, visitor);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
			}
			ccv_array_free(follows);
		} else {
			// For backward graph, no need to evaluate the while expr.
			assert(exec->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
			assert(graph->peer);
			assert(tensor_tape);
			count = 0;
			int64_t reverse_count = graph->while_count = ccv_nnc_tensor_tape_numbering(tensor_tape, graph->p, (ccv_nnc_graph_exec_t){
					.d = exec_idx,
					.graph = graph->p,
				});
			_ccv_nnc_graph_unwrap(graph, count, reverse_count);
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph->breakpoints, graph->breakpoint_size, graph_destinations, graph_destination_size, 1, visitor);
			_ccv_nnc_graph_from_move_transit(graph);
			_ccv_nnc_graph_rewrap(graph);
			for (count = 1; reverse_count > 0; ++count)
			{
				graph->while_count = --reverse_count;
				_ccv_nnc_graph_unwrap(graph, count, reverse_count);
				_ccv_nnc_graph_transit_move_to(graph);
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph_sources, graph_source_size, graph_destinations, graph_destination_size, 0, visitor);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
			}
		}
	} else {
		graph->while_count = 0;
		CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph_sources, graph_source_size, graph_destinations, graph_destination_size, 0, visitor);
	}
#undef visitor
}

static int _ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	assert((sources == 0 && source_size == 0) || (sources && source_size));
	assert((destinations == 0 && destination_size == 0) || (destinations && destination_size));
	const ccv_nnc_graph_exec_t* const graph_sources = sources ? sources : (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, 0);
	const int graph_source_size = source_size ? source_size : graph->sources->rnum;
	const ccv_nnc_graph_exec_t* const graph_destinations = destinations ? destinations : (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, 0);
	const int graph_destination_size = destination_size ? destination_size : graph->destinations->rnum;
	int i;
	for (i = 0; i < graph_source_size; i++)
		if (graph_sources[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < graph_destination_size; i++)
		if (graph_destinations[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	// When topsorted is true, there is no memory allocation when run the graph.
	const int topsorted = (!sources && !destinations && graph->topsorted);
	if (topsorted)
		_ccv_nnc_graph_topsorted_run(graph, exec_idx, exec, tensor_tape, stream_context, flags);
	else
		_ccv_nnc_graph_run_slow_path(graph, exec_idx, exec, inputs, input_size, outputs, output_size, tensor_tape, stream_context, flags, sources, source_size, destinations, destination_size);
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	if (stream_context && graph->topsorted && source_size == 0 && destination_size == 0)
	{
		ccv_nnc_stream_scheduler_t* const scheduler = ccv_nnc_stream_context_get_scheduler(stream_context);
		ccv_nnc_graph_topsorted_run_coro_t params = {
			.graph = graph,
			.exec_idx = -1,
			.exec = 0,
			.tensor_tape = tensor_tape,
			.stream_context = stream_context,
			.flags = flags
		};
		ccv_nnc_stream_task_t* const task = ccv_nnc_stream_task_new(scheduler, _ccv_nnc_graph_topsorted_run_coro, &params, sizeof(params));
		ccv_nnc_stream_schedule_task(scheduler, task);
		return CCV_NNC_EXEC_SUCCESS;
	} else
		return _ccv_nnc_graph_run(graph, -1, 0, 0, 0, 0, 0, tensor_tape, stream_context, flags, sources, source_size, destinations, destination_size);
}
