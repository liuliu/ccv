#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"

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
	if (!graph->exec_wraps)
		return;
	int i, j;
	for (i = 0; i < graph->exec_wraps->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->exec_wraps, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		for (j = 0; j < exec_info->tensor_wrap_size; j++)
		{
			ccv_nnc_graph_tensor_wrap_t* const tensor_wrap = exec_info->tensor_wraps[j];
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
	if (!graph->exec_wraps)
		return;
	int i, j;
	for (i = 0; i < graph->exec_wraps->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->exec_wraps, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		for (j = 0; j < exec_info->tensor_wrap_size; j++)
		{
			ccv_nnc_graph_tensor_wrap_t* const tensor_wrap = exec_info->tensor_wraps[j];
			if (!tensor_wrap)
				continue;
			_ccv_nnc_rewrap_tensor_wrap(graph, tensor_wrap);
		}
	}
	_ccv_nnc_graph_rewrap_sub_graph(graph, graph);
}

static void _ccv_nnc_graph_exec_unwrap_io(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node)
{
	if (!node->tensor_wrap_size)
		return;
	int i;
	ccv_nnc_graph_tensor_wrap_t** const tensor_wraps = node->tensor_wraps;
	for (i = 0; i < node->tensor_wrap_size; i++)
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
	if (!node->tensor_wrap_size)
		return;
	int i;
	ccv_nnc_graph_tensor_wrap_t** const tensor_wraps = node->tensor_wraps;
	for (i = 0; i < node->tensor_wrap_size; i++)
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

static int _ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int exec_idx, const ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size);

static inline void _ccv_nnc_graph_exec_run(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node, const int idx, const int depth, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags)
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
		if (node->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			int ref;
			if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
			{
				ref = node->case_of.offset + node->case_of.expr(inputs + node->case_of.argument.offset, node->case_of.argument.size, node->case_of.data);
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
				_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, tensor_tape, flags, 0, 0, 0, 0);
			}
			_ccv_nnc_graph_exec_unwrap_phi(graph, node, ref);
		} else if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) {
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[0] - 1);
			_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, tensor_tape, flags, 0, 0, 0, 0);
		}
	} else {
		PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, depth, node->input_size, node->output_size);
		for (i = 0; i < node->input_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)", i + 1, inputs[i], (inputs[i] ? inputs[i]->data.u8 : 0));
			if (inputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(inputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
		ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, 0);
		for (i = 0; i < node->output_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)", i + 1, outputs[i], (outputs[i] ? outputs[i]->data.u8 : 0));
			if (outputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(outputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
	}
}

static int _ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int exec_idx, const ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	assert((sources == 0 && source_size == 0) || (sources && source_size));
	assert((destinations == 0 && destination_size == 0) || (destinations && destination_size));
	const ccv_nnc_graph_exec_t* const graph_sources = sources ? sources : (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, 0);
	const int graph_source_size = source_size ? source_size : graph->sources->rnum;
	const ccv_nnc_graph_exec_t* const graph_destinations = destinations ? destinations : (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, 0);
	const int graph_destination_size = destination_size ? destination_size : graph->destinations->rnum;
	int i, j;
	for (i = 0; i < graph_source_size; i++)
		if (graph_sources[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < graph_destination_size; i++)
		if (graph_destinations[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	// When sequential is true, there is no memory allocation when run the graph.
	const int sequential = (!sources && !destinations && graph->sequential);
#define visitor(node, idx, depth, ...) \
	_ccv_nnc_graph_exec_run(graph, node, idx, depth, tensor_tape, flags)
	if (exec && (exec->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
	{
		assert(exec->p_while.expr);
		int64_t count = 0;
		// This is a forward while loop. Backward while loop will just consult its peering part.
		if (exec->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
		{
			if (sequential)
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
						_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, 0, tensor_tape, flags);
					// Reached breakpoints, now check the breakpoint, if not met, break out.
					if (!exec->p_while.expr(exec->p_while.inputs, exec->p_while.input_size, exec->p_while.data))
					{
						_ccv_nnc_graph_rewrap(graph);
						break;
					}
					for (i = graph_breakpoint_size; i < graph->exec_info->rnum; i++)
						_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, 0, tensor_tape, flags);
					_ccv_nnc_graph_from_move_transit(graph);
					_ccv_nnc_graph_rewrap(graph);
				}
			} else {
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
			if (sequential)
			{
				_ccv_nnc_graph_unwrap(graph, count, reverse_count);
				for (i = graph->breakpoint_offset; i < graph->exec_info->rnum; i++)
					_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, 0, tensor_tape, flags);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
				for (count = 1; reverse_count > 0; ++count)
				{
					graph->while_count = --reverse_count;
					_ccv_nnc_graph_unwrap(graph, count, reverse_count);
					_ccv_nnc_graph_transit_move_to(graph);
					for (i = 0; i < graph->exec_info->rnum; i++)
						_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, 0, tensor_tape, flags);
					_ccv_nnc_graph_from_move_transit(graph);
					_ccv_nnc_graph_rewrap(graph);
				}
			} else {
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
		}
	} else {
		graph->while_count = 0;
		if (sequential)
			for (i = 0; i < graph->exec_info->rnum; i++)
				_ccv_nnc_graph_exec_run(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i), i, 0, tensor_tape, flags);
		else
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph_sources, graph_source_size, graph_destinations, graph_destination_size, 0, visitor);
	}
#undef visitor
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	return _ccv_nnc_graph_run(graph, -1, 0, 0, 0, 0, 0, tensor_tape, flags, sources, source_size, destinations, destination_size);
}
