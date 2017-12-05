#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_graph.h"

void ccv_nnc_tensor_multiview(ccv_nnc_tensor_t* data[], const uint8_t kind, const uint16_t repeat, const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	assert(kind == CCV_NNC_MULTIVIEW_K0N || kind == CCV_NNC_MULTIVIEW_K1N);
	assert(repeat > 0);
	tensor_multiview->type = CCV_TENSOR_MULTIVIEW;
	tensor_multiview->kind = kind;
	tensor_multiview->repeat = repeat;
	tensor_multiview->anchor = (intptr_t)graph;
	tensor_multiview->it = 0;
	tensor_multiview->p = 0;
	tensor_multiview->offset = 0;
	tensor_multiview->rtvs = 0;
	tensor_multiview->_heap_data = (repeat + kind <= sizeof(tensor_multiview->_inline_data) / sizeof(tensor_multiview->_inline_data[0])) ? 0 : ccmalloc(sizeof(ccv_nnc_tensor_t*) * (repeat + kind));
	int i;
	// Currently, only CCV_NNC_MULTIVIEW_K12 uses 3 tensors.
	for (i = 0; i < repeat + kind; i++)
	{
		CCV_NNC_MULTIVIEW_DATA(tensor_multiview)[i] = data[i];
		ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)data[i];
		if (data[i] != CCV_NNC_TENSOR_PLACEHOLDER && CCV_IS_TENSOR_MULTIVIEW(mv))
			mv->p = tensor_multiview;
	}
}

void ccv_nnc_tensor_multiview_free(const ccv_nnc_tensor_multiview_t tensor_multiview)
{
	if (tensor_multiview.rtvs)
		ccv_array_free(tensor_multiview.rtvs);
	if (tensor_multiview._heap_data)
		ccfree(tensor_multiview._heap_data);
}

void ccv_nnc_tensor_reference_to_multiview(ccv_nnc_tensor_multiview_t* const tensor_multiview, ccv_nnc_tensor_t* const tensor)
{
	if (!tensor_multiview->rtvs)
		tensor_multiview->rtvs = ccv_array_new(sizeof(ccv_nnc_tensor_t*), 0, 0);
	ccv_array_push(tensor_multiview->rtvs, &tensor);
}

void ccv_nnc_tensor_multiview_broadcast(ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	assert(tensor_multiview->it && !CCV_IS_TENSOR_MULTIVIEW(tensor_multiview->it));
	// Update the pointer on tv only if it is not a single tensor pointer.
	unsigned char* const data = tensor_multiview->it->data.u8 - tensor_multiview->offset;
	const ccv_nnc_tensor_multiview_t* c = tensor_multiview;
	int i;
	do {
		if (c->rtvs)
			for (i = 0; i < c->rtvs->rnum; i++)
			{
				ccv_nnc_tensor_t* tensor = *(ccv_nnc_tensor_t**)ccv_array_get(c->rtvs, i);
				if (CCV_IS_TENSOR_VIEW(tensor))
				{
					ccv_nnc_tensor_view_t* tensor_view = (ccv_nnc_tensor_view_t*)tensor;
					tensor_view->data.u8 = data + tensor_view->off;
				} else
					tensor->data.u8 = data;
			}
		c = c->p;
	} while (c);
}

ccv_nnc_graph_exec_t ccv_nnc_graph_while(ccv_nnc_graph_t* const graph, const uint32_t cmd, ccv_nnc_graph_t* const while_graph)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	ccv_nnc_graph_exec_t while_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_info_t* while_exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, while_exec.d);
	while_exec_info->flags |= CCV_NNC_GRAPH_EXEC_P_WHILE;
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_graph_t*), 1, 0);
	int i;
	if (while_graph->nest_execs)
	{
		// Copy wraps from sub graph to parent graph.
		if (!graph->nest_execs)
			graph->nest_execs = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), while_graph->nest_execs->rnum, 0);
		for (i = 0; i < while_graph->nest_execs->rnum; i++)
			ccv_array_push(graph->nest_execs, ccv_array_get(while_graph->nest_execs, i));
	}
	ccv_array_push(graph->sub_graphs, &while_graph);
	while_graph->p = graph;
	while_graph->p_idx = graph->sub_graphs->rnum;
	while_graph->exec_idx = while_exec.d + 1;
	while_exec_info->graph_ref_size = 1;
	CCV_NNC_GRAPH_REF(while_exec_info)[0] = graph->sub_graphs->rnum;
	return while_exec;
}

ccv_nnc_graph_t* ccv_nnc_graph_from_graph_exec(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_t exec)
{
	assert(exec.graph == graph);
	assert(exec.d < graph->exec_info->rnum);
	assert(graph->sub_graphs);
	ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	assert(CCV_NNC_GRAPH_REF(exec_info)[0]);
	const int graph_ref = CCV_NNC_GRAPH_REF(exec_info)[0] - 1;
	assert(graph_ref < graph->sub_graphs->rnum);
	ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, graph_ref);
	return sub_graph;
}

void ccv_nnc_graph_set_while_expr(ccv_nnc_graph_t* const while_graph, const ccv_nnc_graph_while_f while_expr, const void* const while_data, const ccv_nnc_graph_exec_t* const breakpoints, const int breakpoint_size)
{
	assert(while_graph->p);
	const int exec_idx = while_graph->exec_idx - 1;
	assert(exec_idx >= 0 && exec_idx < while_graph->p->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(while_graph->p->exec_info, exec_idx);
	exec_info->p_while.expr = while_expr;
	exec_info->p_while.data = while_data;
	assert(breakpoint_size > 0);
	while_graph->breakpoint_size = breakpoint_size;
	while_graph->breakpoints = (ccv_nnc_graph_exec_t*)((while_graph->breakpoints) ? ccrealloc(while_graph->breakpoints, sizeof(ccv_nnc_graph_exec_t) * breakpoint_size) : ccmalloc(sizeof(ccv_nnc_graph_exec_t) * breakpoint_size));
	memcpy(while_graph->breakpoints, breakpoints, sizeof(ccv_nnc_graph_exec_t) * breakpoint_size);
}

static void _ccv_nnc_graph_unwrap(const ccv_nnc_graph_t* const graph, const int count)
{
	if (!graph->nest_execs)
		return;
	int i, j;
	for (i = 0; i < graph->nest_execs->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->nest_execs, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		for (j = 0; j < exec_info->tensor_nest_size; j++)
		{
			ccv_nnc_graph_tensor_nest_t* const tensor_nest = exec_info->tensor_nests[j];
			if (!tensor_nest)
				continue;
			ccv_nnc_tensor_t* tensor = tensor_nest->tensors[tensor_nest->index];
			while (CCV_IS_TENSOR_MULTIVIEW(tensor) &&
				   (((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph ||
					((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph->peer))
			{
				ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
				const int off = mv->kind;
				const int mod = mv->repeat;
				tensor = CCV_NNC_MULTIVIEW_DATA(mv)[count >= off ? ((count - off) % mod) + off : count]; // Unwrap.
				// If reached the root.
				if (!CCV_IS_TENSOR_MULTIVIEW(tensor))
					tensor_nest->broadcast_required = 1; // Need to broadcast tensor updates.
				++tensor_nest->index;
				tensor_nest->tensors[tensor_nest->index] = tensor;
				assert(tensor_nest->index < tensor_nest->count);
			}
		}
	}
}

static void _ccv_nnc_graph_rewrap(const ccv_nnc_graph_t* const graph) // Call this method at the end to roll the wrap_ptr back
{
	if (!graph->nest_execs)
		return;
	int i, j;
	for (i = 0; i < graph->nest_execs->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->nest_execs, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		for (j = 0; j < exec_info->tensor_nest_size; j++)
		{
			ccv_nnc_graph_tensor_nest_t* const tensor_nest = exec_info->tensor_nests[j];
			if (!tensor_nest)
				continue;
			while (tensor_nest->index > 0 && CCV_IS_TENSOR_MULTIVIEW(tensor_nest->tensors[tensor_nest->index - 1]) &&
					(((ccv_nnc_tensor_multiview_t*)tensor_nest->tensors[tensor_nest->index - 1])->anchor == (intptr_t)graph ||
					 ((ccv_nnc_tensor_multiview_t*)tensor_nest->tensors[tensor_nest->index - 1])->anchor == (intptr_t)graph->peer))
				--tensor_nest->index;
		}
	}
}

static void _ccv_nnc_graph_exec_unwrap_io(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node)
{
	if (!node->tensor_nest_size)
		return;
	int i;
	ccv_nnc_graph_tensor_nest_t** const tensor_nests = node->tensor_nests;
	for (i = 0; i < node->tensor_nest_size; i++)
		if (tensor_nests[i])
		{
			assert(tensor_nests[i]->index > 0);
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)(tensor_nests[i]->tensors[tensor_nests[i]->index - 1]);
			assert(CCV_IS_TENSOR_MULTIVIEW(mv));
			assert(mv->anchor == (intptr_t)graph || mv->anchor == (intptr_t)graph->peer);
			// Only now set the mv->it, because now this node is about to get executed.
			mv->it = tensor_nests[i]->tensors[tensor_nests[i]->index];
		}
	for (i = 0; i < node->input_size; i++)
		if (tensor_nests[i])
			node->inputs[i] = tensor_nests[i]->tensors[tensor_nests[i]->index];
	const int d = node->input_size;
	for (i = 0; i < node->output_size; i++)
		if (tensor_nests[d + i])
			node->outputs[i] = tensor_nests[d + i]->tensors[tensor_nests[d + i]->index];
}

static void _ccv_nnc_graph_exec_broadcast(ccv_nnc_graph_exec_info_t* const node)
{
	if (!node->tensor_nest_size)
		return;
	int i;
	ccv_nnc_graph_tensor_nest_t** const tensor_nests = node->tensor_nests;
	for (i = 0; i < node->tensor_nest_size; i++)
		if (tensor_nests[i] && tensor_nests[i]->broadcast_required)
		{
			assert(tensor_nests[i]->index > 0);
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)(tensor_nests[i]->tensors[tensor_nests[i]->index - 1]);
			// Now broadcast the final pointer.
			ccv_nnc_tensor_multiview_broadcast(mv);
			tensor_nests[i]->broadcast_required = 0; // Reset, no need to broadcast.
		}
}

static int _ccv_nnc_graph_while_run(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	int i, j;
	for (i = 0; i < source_size; i++)
		if (sources[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < destination_size; i++)
		if (destinations[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
#define visitor(node, idx, d, ...) \
	do { \
		_ccv_nnc_graph_exec_unwrap_io(graph, node); \
		ccv_nnc_tensor_t** inputs = node->inputs; \
		ccv_nnc_tensor_t** outputs = inputs + node->input_size; \
		if (tensor_tape) \
			ccv_nnc_tensor_tape_io(tensor_tape, graph, node->input_flags, inputs, node->input_size, node->output_flags, outputs, node->output_size); \
		/* Broadcast the updates to all subscribed references for input / output, even though at this
		 * time output is not written yet, propagate pointer change is still valid. */ \
		_ccv_nnc_graph_exec_broadcast(node); \
		if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD) \
		{ \
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[0] - 1); \
			_ccv_nnc_graph_while_run(sub_graph, node, inputs, node->input_size, outputs, node->output_size, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
		} else { \
			PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, d, node->input_size, node->output_size); \
			for (i = 0; i < node->input_size; i++) \
				PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)\n", i + 1, inputs[i], (inputs[i] ? inputs[i]->data.u8 : 0)); \
			for (i = 0; i < node->output_size; i++) \
				PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)\n", i + 1, outputs[i], (outputs[i] ? outputs[i]->data.u8 : 0)); \
			ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, 0); \
		} \
	} while (0)
	if (exec && exec->p_while.expr)
	{
		uint64_t count = 0;
		ccv_nnc_tensor_t count_tensor = ccv_nnc_tensor(&count, ONE_CPU_TENSOR(1, 1, 1), 0);
		ccv_nnc_tensor_t* special_tensors[] = { &count_tensor };
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
					ccv_nnc_tensor_tape_set_while_count(tensor_tape, graph, count);
				_ccv_nnc_graph_unwrap(graph, count);
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, graph->breakpoints, graph->breakpoint_size, 0, visitor);
				// Reached breakpoints, now check the breakpoint, if not met, break out.
				if (!exec->p_while.expr(special_tensors, 1, inputs, input_size, outputs, output_size, exec->p_while.data))
				{
					_ccv_nnc_graph_rewrap(graph);
					break;
				}
				if (follows->rnum > 0)
					CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(follows, 0), follows->rnum, destinations, destination_size, 0, visitor);
				_ccv_nnc_graph_rewrap(graph);
			}
			ccv_array_free(follows);
		} else {
			// For backward graph, no need to evaluate the while expr.
			assert(exec->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
			assert(graph->peer);
			assert(tensor_tape);
			graph->while_count = count = ccv_nnc_tensor_tape_while_count(tensor_tape, graph);
			_ccv_nnc_graph_unwrap(graph, count);
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, graph->breakpoints, graph->breakpoint_size, destinations, destination_size, 1, visitor);
			_ccv_nnc_graph_rewrap(graph);
			if (count > 0)
				do {
					graph->while_count = --count;
					_ccv_nnc_graph_unwrap(graph, count);
					CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, 0, visitor);
					_ccv_nnc_graph_rewrap(graph);
				} while (count > 0);
		}
	} else {
		graph->while_count = 0;
		CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, 0, visitor);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	return _ccv_nnc_graph_while_run(graph, 0, 0, 0, 0, 0, tensor_tape, flags, sources, source_size, destinations, destination_size);
}
