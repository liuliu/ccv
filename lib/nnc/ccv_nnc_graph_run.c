#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"

static void _ccv_nnc_unwrap_tensor_tree(const ccv_nnc_graph_t* const graph, const int64_t count, const int64_t reverse_count, ccv_nnc_graph_tensor_tree_t* const tensor_tree)
{
	ccv_nnc_tensor_t* tensor = tensor_tree->tensors[tensor_tree->index];
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
			tensor_tree->broadcast_required = 1; // Need to broadcast tensor updates.
		++tensor_tree->index;
		tensor_tree->tensors[tensor_tree->index] = tensor;
		assert(tensor_tree->index < tensor_tree->count);
	}
}

static void _ccv_nnc_graph_unwrap_sub_graph(const ccv_nnc_graph_t* const graph, const int64_t count, const int64_t reverse_count, const ccv_nnc_graph_t* const sub_graph)
{
	int i;
	if (sub_graph->moves)
		for (i = 0; i < sub_graph->moves->rnum; i++)
		{
			ccv_nnc_graph_tensor_move_t* const move = (ccv_nnc_graph_tensor_move_t*)ccv_array_get(sub_graph->moves, i);
			_ccv_nnc_unwrap_tensor_tree(graph, count, reverse_count, move->from);
			_ccv_nnc_unwrap_tensor_tree(graph, count, reverse_count, move->to);
		}
	if (sub_graph->sub_graphs)
		for (i = 0; i < sub_graph->sub_graphs->rnum; i++)
			_ccv_nnc_graph_unwrap_sub_graph(graph, count, reverse_count, *(ccv_nnc_graph_t**)ccv_array_get(sub_graph->sub_graphs, i));
}

static void _ccv_nnc_graph_unwrap(const ccv_nnc_graph_t* const graph, const int64_t count, const int64_t reverse_count)
{
	if (!graph->tree_execs)
		return;
	int i, j;
	for (i = 0; i < graph->tree_execs->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->tree_execs, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		for (j = 0; j < exec_info->tensor_tree_size; j++)
		{
			ccv_nnc_graph_tensor_tree_t* const tensor_tree = exec_info->tensor_trees[j];
			if (!tensor_tree)
				continue;
			_ccv_nnc_unwrap_tensor_tree(graph, count, reverse_count, tensor_tree);
		}
	}
	_ccv_nnc_graph_unwrap_sub_graph(graph, count, reverse_count, graph);
}

static void _ccv_nnc_graph_transit_move_to(const ccv_nnc_graph_t* const graph)
{
	int i;
	if (graph->moves)
		for (i = 0; i < graph->moves->rnum; i++)
		{
			ccv_nnc_graph_tensor_move_t* const move = (ccv_nnc_graph_tensor_move_t*)ccv_array_get(graph->moves, i);
			ccv_nnc_tensor_t* it = (ccv_nnc_tensor_t*)(move->to->tensors[move->to->index]);
			assert(!CCV_IS_TENSOR_MULTIVIEW(it));
			it->data = move->transit;
		}
}

static void _ccv_nnc_graph_from_move_transit(const ccv_nnc_graph_t* const graph)
{
	int i;
	if (graph->moves)
		for (i = 0; i < graph->moves->rnum; i++)
		{
			ccv_nnc_graph_tensor_move_t* const move = (ccv_nnc_graph_tensor_move_t*)ccv_array_get(graph->moves, i);
			ccv_nnc_tensor_t* it = (ccv_nnc_tensor_t*)(move->from->tensors[move->from->index]);
			assert(!CCV_IS_TENSOR_MULTIVIEW(it));
			move->transit = it->data;
		}
}

static void _ccv_nnc_rewrap_tensor_tree(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_tensor_tree_t* const tensor_tree)
{
	while (tensor_tree->index > 0 && CCV_IS_TENSOR_MULTIVIEW(tensor_tree->tensors[tensor_tree->index - 1]) &&
			(((ccv_nnc_tensor_multiview_t*)tensor_tree->tensors[tensor_tree->index - 1])->anchor == (intptr_t)graph ||
			 ((ccv_nnc_tensor_multiview_t*)tensor_tree->tensors[tensor_tree->index - 1])->anchor == (intptr_t)graph->peer))
		--tensor_tree->index;
}

static void _ccv_nnc_graph_rewrap_sub_graph(const ccv_nnc_graph_t* const graph, const ccv_nnc_graph_t* const sub_graph)
{
	int i;
	if (sub_graph->moves)
		for (i = 0; i < sub_graph->moves->rnum; i++)
		{
			ccv_nnc_graph_tensor_move_t* const move = (ccv_nnc_graph_tensor_move_t*)ccv_array_get(sub_graph->moves, i);
			_ccv_nnc_rewrap_tensor_tree(graph, move->from);
			_ccv_nnc_rewrap_tensor_tree(graph, move->to);
		}
	if (sub_graph->sub_graphs)
		for (i = 0; i < sub_graph->sub_graphs->rnum; i++)
			_ccv_nnc_graph_rewrap_sub_graph(graph, *(ccv_nnc_graph_t**)ccv_array_get(sub_graph->sub_graphs, i));
}

static void _ccv_nnc_graph_rewrap(const ccv_nnc_graph_t* const graph) // Call this method at the end to roll the wrap_ptr back
{
	if (!graph->tree_execs)
		return;
	int i, j;
	for (i = 0; i < graph->tree_execs->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->tree_execs, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		for (j = 0; j < exec_info->tensor_tree_size; j++)
		{
			ccv_nnc_graph_tensor_tree_t* const tensor_tree = exec_info->tensor_trees[j];
			if (!tensor_tree)
				continue;
			_ccv_nnc_rewrap_tensor_tree(graph, tensor_tree);
		}
	}
	_ccv_nnc_graph_rewrap_sub_graph(graph, graph);
}

static void _ccv_nnc_graph_exec_unwrap_io(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node)
{
	if (!node->tensor_tree_size)
		return;
	int i;
	ccv_nnc_graph_tensor_tree_t** const tensor_trees = node->tensor_trees;
	for (i = 0; i < node->tensor_tree_size; i++)
		if (tensor_trees[i])
		{
			assert(tensor_trees[i]->index > 0);
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)(tensor_trees[i]->tensors[tensor_trees[i]->index - 1]);
			assert(CCV_IS_TENSOR_MULTIVIEW(mv));
			// Only now set the mv->it, because now this node is about to get executed.
			mv->it = tensor_trees[i]->tensors[tensor_trees[i]->index];
			assert(!CCV_IS_TENSOR_MULTIVIEW(mv->it));
		}
	for (i = 0; i < node->input_size; i++)
		if (tensor_trees[i])
			node->inputs[i] = tensor_trees[i]->tensors[tensor_trees[i]->index];
	const int d = node->input_size;
	for (i = 0; i < node->output_size; i++)
		if (tensor_trees[d + i])
			node->outputs[i] = tensor_trees[d + i]->tensors[tensor_trees[d + i]->index];
}

static void _ccv_nnc_graph_exec_unwrap_phi(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_info_t* const node, const int ref)
{
	int i;
	// If the output tensor is a phi multi-view tensor, we broadcast our selection to all the subscribers.
	for (i = 0; i < node->output_size; i++)
		if (CCV_IS_TENSOR_MULTIVIEW(node->outputs[i]) &&
			((ccv_nnc_tensor_multiview_t*)node->outputs[i])->anchor == CCV_NNC_MULTIVIEW_PHI)
		{
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)node->outputs[i];
			mv->it = CCV_NNC_MULTIVIEW_DATA(mv)[ref >= 0];
			ccv_nnc_tensor_multiview_synchronize(mv, CCV_NNC_MULTIVIEW_EXEC_BEGIN_SYNC);
		}
}

static void _ccv_nnc_graph_exec_begin_synchronize_multiviews(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node)
{
	if (!node->tensor_tree_size)
		return;
	int i;
	ccv_nnc_graph_tensor_tree_t** const tensor_trees = node->tensor_trees;
	for (i = 0; i < node->tensor_tree_size; i++)
		if (tensor_trees[i] && tensor_trees[i]->broadcast_required)
		{
			assert(tensor_trees[i]->index > 0);
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)(tensor_trees[i]->tensors[tensor_trees[i]->index - 1]);
			// Now broadcast the final pointer.
			ccv_nnc_tensor_multiview_synchronize(mv, CCV_NNC_MULTIVIEW_EXEC_BEGIN_SYNC);
			tensor_trees[i]->broadcast_required = 0; // Reset, no need to broadcast.
		}
}

static void _ccv_nnc_print_tensor_verbose(const ccv_nnc_tensor_t* const tensor)
{
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

static int _ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int exec_idx, const ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	int i, j;
	for (i = 0; i < source_size; i++)
		if (sources[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < destination_size; i++)
		if (destinations[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
#define visitor(node, idx, depth, ...) \
	do { \
		_ccv_nnc_graph_exec_unwrap_io(graph, node); \
		ccv_nnc_tensor_t** inputs = node->inputs; \
		ccv_nnc_tensor_t** outputs = inputs + node->input_size; \
		if (tensor_tape) \
			ccv_nnc_tensor_tape_io(tensor_tape, graph, node->input_flags, inputs, node->input_size, node->output_flags, outputs, node->output_size); \
		/* Broadcast the updates to all subscribed references for input / output, even though at this
		 * time output is not written yet, propagate pointer change is still valid. */ \
		_ccv_nnc_graph_exec_begin_synchronize_multiviews(graph, node); \
		if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD) \
		{ \
			if (node->flags & CCV_NNC_GRAPH_EXEC_CASE_OF) \
			{ \
				int ref; \
				if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD) \
				{ \
					ref = node->case_of.offset + node->case_of.expr(inputs + node->case_of.argument.offset, node->case_of.argument.size, node->case_of.data); \
					if (tensor_tape) \
						ccv_nnc_tensor_tape_set_numbering(tensor_tape, graph, (ccv_nnc_graph_exec_t){ \
							.d = idx, \
							.graph = graph, \
						}, ref); \
				} else { \
					assert(node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD); \
					assert(tensor_tape); \
					ref = ccv_nnc_tensor_tape_numbering(tensor_tape, graph, (ccv_nnc_graph_exec_t){ \
							.d = idx, \
							.graph = graph, \
						}); \
				} \
				if (ref >= 0) \
				{ \
					assert(ref < node->graph_ref_size); \
					ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[ref] - 1); \
					_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
				} \
				_ccv_nnc_graph_exec_unwrap_phi(graph, node, ref); \
			} else if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) { \
				ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[0] - 1); \
				_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
			} \
		} else { \
			PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, depth, node->input_size, node->output_size); \
			for (i = 0; i < node->input_size; i++) \
			{ \
				PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)", i + 1, inputs[i], (inputs[i] ? inputs[i]->data.u8 : 0)); \
				if (inputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE)) \
					_ccv_nnc_print_tensor_verbose(inputs[i]); \
				PRINT(CCV_CLI_VERBOSE, "\n"); \
			} \
			ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, 0); \
			for (i = 0; i < node->output_size; i++) \
			{ \
				PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)", i + 1, outputs[i], (outputs[i] ? outputs[i]->data.u8 : 0)); \
				if (outputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE)) \
					_ccv_nnc_print_tensor_verbose(outputs[i]); \
				PRINT(CCV_CLI_VERBOSE, "\n"); \
			} \
		} \
	} while (0)
	if (exec && (exec->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
	{
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
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, graph->breakpoints, graph->breakpoint_size, 0, visitor);
				// Reached breakpoints, now check the breakpoint, if not met, break out.
				if (!exec->p_while.expr(exec->p_while.inputs, exec->p_while.input_size, exec->p_while.data))
				{
					_ccv_nnc_graph_rewrap(graph);
					break;
				}
				if (follows->rnum > 0)
					CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(follows, 0), follows->rnum, destinations, destination_size, 0, visitor);
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
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph->breakpoints, graph->breakpoint_size, destinations, destination_size, 1, visitor);
			_ccv_nnc_graph_from_move_transit(graph);
			_ccv_nnc_graph_rewrap(graph);
			for (count = 1; reverse_count > 0; ++count)
			{
				graph->while_count = --reverse_count;
				_ccv_nnc_graph_unwrap(graph, count, reverse_count);
				_ccv_nnc_graph_transit_move_to(graph);
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, 0, visitor);
				_ccv_nnc_graph_from_move_transit(graph);
				_ccv_nnc_graph_rewrap(graph);
			}
		}
	} else {
		graph->while_count = 0;
		CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, 0, visitor);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	return _ccv_nnc_graph_run(graph, -1, 0, 0, 0, 0, 0, tensor_tape, flags, sources, source_size, destinations, destination_size);
}
