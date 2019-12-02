#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"
#include "_ccv_nnc_stream.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif

#pragma mark - Level-2 API

static void _ccv_nnc_unwrap_tensor_wrap(const ccv_nnc_graph_t* const graph, const int64_t count, const int64_t reverse_count, ccv_nnc_graph_tensor_wrap_t* const tensor_wrap)
{
	ccv_nnc_tensor_t* tensor = tensor_wrap->tensors[tensor_wrap->index];
	while (CCV_IS_TENSOR_MULTIVIEW(tensor) &&
		   (((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph ||
			((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph->pair))
	{
		// If the anchor is from the pair, we use the reverse_count instead (we are looking it up).
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
			 ((ccv_nnc_tensor_multiview_t*)tensor_wrap->tensors[tensor_wrap->index - 1])->anchor == (intptr_t)graph->pair))
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
	if (tensor->info.dim[0] <= 0)
		return;
	int i;
	const int len = ccv_min(tensor->info.dim[0], 3);
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
	{
#ifdef HAVE_CUDA
		switch (tensor->info.datatype)
		{
			case CCV_16F: {
				uint16_t data[len];
				cumemcpy(data, CCV_TENSOR_CPU_MEMORY, tensor->data.f16, tensor->info.type, len * sizeof(uint16_t));
				float fp32[len];
				ccv_half_precision_to_float(data, fp32, len);
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %f", fp32[i]);
				break;
			}
			case CCV_32F: {
				float data[len];
				cumemcpy(data, CCV_TENSOR_CPU_MEMORY, tensor->data.f32, tensor->info.type, len * sizeof(float));
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %f", data[i]);
				break;
			}
			case CCV_64F: {
				double data[len];
				cumemcpy(data, CCV_TENSOR_CPU_MEMORY, tensor->data.f64, tensor->info.type, len * sizeof(double));
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %f", data[i]);
				break;
			}
			case CCV_32S: {
				int data[len];
				cumemcpy(data, CCV_TENSOR_CPU_MEMORY, tensor->data.i32, tensor->info.type, len * sizeof(int));
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %d", data[i]);
				break;
			}
			case CCV_64S: {
				int64_t data[len];
				cumemcpy(data, CCV_TENSOR_CPU_MEMORY, tensor->data.i64, tensor->info.type, len * sizeof(int64_t));
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %lld", (long long)data[i]);
				break;
			}
			case CCV_8U: {
				uint8_t data[len];
				cumemcpy(data, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, len * sizeof(uint8_t));
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %d", (int)data[i]);
				break;
			}
		}
		if (ccv_nnc_tensor_count(tensor->info) > 3)
			PRINT(CCV_CLI_VERBOSE, " ..");
#endif
	} else if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_CPU_MEMORY) {
		switch (tensor->info.datatype)
		{
			case CCV_16F: {
				float fp32[len];
				ccv_half_precision_to_float((uint16_t*)tensor->data.f16, fp32, len);
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %f", fp32[i]);
				break;
			}
			case CCV_32F:
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %f", tensor->data.f32[i]);
				break;
			case CCV_64F:
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %f", tensor->data.f64[i]);
				break;
			case CCV_32S:
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %d", tensor->data.i32[i]);
				break;
			case CCV_64S:
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %lld", (long long)tensor->data.i64[i]);
				break;
			case CCV_8U:
				for (i = 0; i < len; i++)
					PRINT(CCV_CLI_VERBOSE, " %d", (int)tensor->data.u8[i]);
				break;
		}
		if (ccv_nnc_tensor_count(tensor->info) > 3)
			PRINT(CCV_CLI_VERBOSE, " ..");
	}
}

static co_decl(_ccv_nnc_graph_topsorted_run_coro, (ccv_nnc_graph_t* const graph, const int exec_idx, const ccv_nnc_graph_static_schedule_t* const schedule, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags));

static co_decl_task(_ccv_nnc_graph_exec_cases_of_coro, (ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, const ccv_nnc_graph_exec_schedule_t* const schd, ccv_nnc_tensor_t* const* const inputs, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, int flags), private(
	int ref;
	ccv_nnc_graph_t* sub_graph;
)) {
	// Wait until this stream context is done.
	co_stream_await(CO_P(stream_context));
	if (CO_P(exec)->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
	{
		CO_V(ref) = CO_P(exec)->case_of.offset + CO_P(exec)->case_of.expr(CO_P(inputs), CO_P(exec)->input_size, CO_P(exec)->case_of.data);
		if (CO_P(tensor_tape))
			ccv_nnc_tensor_tape_set_numbering(CO_P(tensor_tape), CO_P(graph), (ccv_nnc_graph_exec_t){
				.d = CO_P(exec_idx),
				.graph = CO_P(graph),
			}, CO_V(ref));
	} else {
		assert(CO_P(exec)->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
		assert(CO_P(tensor_tape));
		CO_V(ref) = ccv_nnc_tensor_tape_numbering(CO_P(tensor_tape), CO_P(graph), (ccv_nnc_graph_exec_t){
				.d = CO_P(exec_idx),
				.graph = CO_P(graph),
			});
	}
	if (CO_V(ref) >= 0)
	{
		assert(CO_V(ref) < CO_P(exec)->graph_ref_size);
		CO_V(sub_graph) = *(ccv_nnc_graph_t**)ccv_array_get(CO_P(graph)->sub_graphs, CCV_NNC_GRAPH_REF(CO_P(exec))[CO_V(ref)] - 1);
		assert(CO_P(schd)->stream_size == 1);
		assert(CO_P(graph)->streams[SCHEDULE_STREAMS(*CO_P(schd))[0]] == CO_V(sub_graph)->streams[0]);
		co_apply(_ccv_nnc_graph_topsorted_run_coro, (CO_V(sub_graph), CO_P(exec_idx), CO_V(sub_graph)->default_schedule, CO_P(exec), CO_P(tensor_tape), CO_P(graph)->streams[SCHEDULE_STREAMS(*CO_P(schd))[0]], CO_P(flags)));
	}
	_ccv_nnc_graph_exec_unwrap_phi(CO_P(graph), CO_P(exec), CO_V(ref));
} co_end()

typedef struct {
	ccv_nnc_graph_t* graph;
	const ccv_nnc_graph_exec_schedule_t* node;
	ccv_nnc_stream_context_t* stream;
} ccv_nnc_graph_neighbor_context_discovery_t;

static ccv_nnc_stream_context_t* _ccv_nnc_graph_neighbor_context_discovery(const int device_id, void* const context)
{
	const ccv_nnc_graph_neighbor_context_discovery_t* const discovery = (ccv_nnc_graph_neighbor_context_discovery_t*)context;
	if (CCV_STREAM_GET_DEVICE_ID(ccv_nnc_stream_context_type(discovery->stream)) == device_id)
		return discovery->stream;
	ccv_nnc_graph_t* const graph = discovery->graph;
	const ccv_nnc_graph_exec_schedule_t* const node = discovery->node;
	int i;
	// First try to find in other streams of the same node.
	for (i = 0; i < node->stream_size; i++)
	{
		ccv_nnc_stream_context_t* const stream = graph->streams[SCHEDULE_STREAMS(*node)[i]];
		if (CCV_STREAM_GET_DEVICE_ID(ccv_nnc_stream_context_type(stream)) == device_id)
			return stream;
	}
	// If cannot find, try to find in all the wait streams.
	for (i = 0; i < node->wait_size; i++)
	{
		ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_signal_get_emitter(graph->signals[node->waits[i]]);
		if (stream_context && CCV_STREAM_GET_DEVICE_ID(ccv_nnc_stream_context_type(stream_context)) == device_id)
			return stream_context;
	}
	return 0;
}

static co_routine_t* _ccv_nnc_graph_exec_run_task(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const node, const ccv_nnc_graph_exec_schedule_t* const schd, const int idx, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags)
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
			ccv_nnc_stream_context_t* const node_stream = graph->streams[SCHEDULE_STREAMS(*schd)[0]];
			return co_new(_ccv_nnc_graph_exec_cases_of_coro, (graph, idx, node, schd, inputs, tensor_tape, node_stream, flags));
		} else if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) {
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[0] - 1);
			assert(graph->streams[SCHEDULE_STREAMS(*schd)[0]] == sub_graph->streams[0]);
			return co_new(_ccv_nnc_graph_topsorted_run_coro, (sub_graph, idx, sub_graph->default_schedule, node, tensor_tape, graph->streams[SCHEDULE_STREAMS(*schd)[0]], flags));
		}
	} else {
		PRINT(CCV_CLI_VERBOSE, "%s [%d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, node->input_size, node->output_size);
		int i, j;
		for (i = 0; i < schd->stream_size; i++)
		{
			ccv_nnc_stream_context_t* const stream = graph->streams[SCHEDULE_STREAMS(*schd)[i]];
			for (j = 0; j < schd->wait_size; j++)
				ccv_nnc_stream_context_wait_signal(stream, graph->signals[schd->waits[j]]);
		}
		for (i = 0; i < node->input_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p:%d)", i + 1, inputs[i], (inputs[i] ? inputs[i]->data.u8 : 0), (inputs[i] ? CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type) : -1));
			if (inputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(inputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
		ccv_nnc_stream_context_t* const node_stream = graph->streams[SCHEDULE_STREAMS(*schd)[0]];
		ccv_nnc_graph_neighbor_context_discovery_t discovery_context = {
			.graph = graph,
			.node = schd,
			.stream = node_stream
		};
		ccv_nnc_stream_context_set_neighbor_discovery(node_stream, _ccv_nnc_graph_neighbor_context_discovery, &discovery_context);
		ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, node_stream);
		for (i = 0; i < node->output_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p:%d)", i + 1, outputs[i], (outputs[i] ? outputs[i]->data.u8 : 0), (outputs[i] ? CCV_TENSOR_GET_DEVICE_ID(outputs[i]->info.type) : -1));
			if (outputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(outputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
		for (i = 0; i < schd->stream_size; i++)
			if (SCHEDULE_SIGNALS(*schd)[i] >= 0)
			{
				ccv_nnc_stream_context_t* const stream = graph->streams[SCHEDULE_STREAMS(*schd)[i]];
				ccv_nnc_stream_context_emit_signal(stream, graph->signals[SCHEDULE_SIGNALS(*schd)[i]]);
			}
	}
	return 0;
}

static void _ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_schedule_t* const schd_info, ccv_nnc_graph_exec_info_t* const node, co_routine_t* const task)
{
	int i, j;
	if (node->outgoings)
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, i);
			const ccv_nnc_graph_exec_schedule_t* const outgoing_schd = schd_info + outgoing_idx;
			// An outgoing stream can be blocked by multiple other tasks from other streams. But it is OK,
			// because on next round of execution, that one will be marked as blocked again.
			for (j = 0; j < outgoing_schd->stream_size; j++)
				graph->block_stream_tasks[SCHEDULE_STREAMS(*outgoing_schd)[j]] = task;
		}
}

static co_decl_task(_ccv_nnc_graph_wait_any_sub_tasks, (ccv_nnc_graph_t* const graph, co_routine_t* const* const sub_tasks, const int sub_task_size, const ccv_nnc_graph_exec_schedule_t* const schd_info, const int* const pending_nodes, const int pending_node_size), private(
)) {
	if (CO_P(sub_task_size))
		co_await_any(CO_P(sub_tasks), CO_P(sub_task_size));
	// This is not good, these local variables need to be in the private section.
	// I got away with it because there is no yield or resume or apply or any after await above.
	int i, j, k;
	for (i = 0; i < CO_P(sub_task_size); i++)
		if (co_is_done(CO_P(sub_tasks)[i]))
		{
			for (j = 0; j < CO_P(pending_node_size); j++)
			{
				const ccv_nnc_graph_exec_schedule_t* const node = CO_P(schd_info) + CO_P(pending_nodes)[j];
				for (k = 0; k < node->stream_size; k++)
					if (CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*node)[k]] == CO_P(sub_tasks)[i])
						CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*node)[k]] = 0;
			}
			co_free(CO_P(sub_tasks)[i]);
		}
} co_end()

static co_decl_task(_ccv_nnc_graph_exec_run_loop, (ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const exec_info, const ccv_nnc_graph_exec_schedule_t* const schd_info, const int* const psort, const int start_index, const int exec_info_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags), private(
	int i, p, q;
	int sub_task_size;
	co_routine_t** sub_tasks;
	int* pending_nodes[2];
	int pending_node_size[2];
	int idx;
	ccv_nnc_graph_exec_info_t* node;
	const ccv_nnc_graph_exec_schedule_t* schd;
	co_routine_t* task;
)) {
	CO_V(sub_task_size) = 0;
	CO_V(sub_tasks) = (co_routine_t**)ccv_nnc_graph_buffer(CO_P(graph), sizeof(co_routine_t*) * (CO_P(graph)->sub_graphs ? CO_P(graph)->sub_graphs->rnum : 0) + sizeof(int) * CO_P(exec_info_size) * 2);
	CO_V(pending_nodes)[0] = (int*)(CO_V(sub_tasks) + (CO_P(graph)->sub_graphs ? CO_P(graph)->sub_graphs->rnum : 0));
	CO_V(pending_nodes)[1] = CO_V(pending_nodes)[0] + CO_P(exec_info_size);
	CO_V(pending_node_size)[0] = 0;
	CO_V(pending_node_size)[1] = 0;
	for (CO_V(i) = CO_P(start_index); CO_V(i) < CO_P(exec_info_size); CO_V(i)++)
	{
		CO_V(idx) = CO_P(psort) ? CO_P(psort)[CO_V(i)] : CO_V(i);
		CO_V(node) = CO_P(exec_info) + CO_V(idx);
		CO_V(schd) = CO_P(schd_info) + CO_V(idx);
		// If stream is blocked by but not blocked by current executing task.
		int blocked = 0, j;
		for (j = 0; j < CO_V(schd)->stream_size; j++)
			if (CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*CO_V(schd))[j]])
			{
				CO_V(pending_nodes)[0][CO_V(pending_node_size)[0]++] = CO_V(idx);
				_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(CO_P(graph), CO_P(schd_info), CO_V(node), CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*CO_V(schd))[j]]);
				blocked = 1;
			}
		if (blocked)
			continue;
		CO_V(task) = _ccv_nnc_graph_exec_run_task(CO_P(graph), CO_V(node), CO_V(schd), CO_V(idx), CO_P(tensor_tape), CO_P(flags));
		if (CO_V(task))
		{
			co_resume(CO_V(task));
			if (!co_is_done(CO_V(task)))
			{
				CO_V(sub_tasks)[CO_V(sub_task_size)++] = CO_V(task);
				int j;
				for (j = 0; j < CO_V(schd)->stream_size; j++)
					CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*CO_V(schd))[j]] = CO_V(task);
				_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(CO_P(graph), CO_P(schd_info), CO_V(node), CO_V(task));
			} else
				co_free(CO_V(task));
		}
	}
	co_apply(_ccv_nnc_graph_wait_any_sub_tasks, (CO_P(graph), CO_V(sub_tasks), CO_V(sub_task_size), CO_P(schd_info), CO_V(pending_nodes)[0], CO_V(pending_node_size)[0]));
	CO_V(p) = 0;
	CO_V(q) = 1;
	while (CO_V(pending_node_size)[CO_V(p)] > 0)
	{
		CO_V(pending_node_size)[CO_V(q)] = 0;
		CO_V(sub_task_size) = 0;
		for (CO_V(i) = 0; CO_V(i) < CO_V(pending_node_size)[CO_V(p)]; CO_V(i)++)
		{
			CO_V(idx) = CO_V(pending_nodes)[CO_V(p)][CO_V(i)];
			CO_V(node) = CO_P(exec_info) + CO_V(idx);
			CO_V(schd) = CO_P(schd_info) + CO_V(idx);
			int blocked = 0, j;
			for (j = 0; j < CO_V(schd)->stream_size; j++)
				if (CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*CO_V(schd))[j]])
				{
					_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(CO_P(graph), CO_P(schd_info), CO_V(node), CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*CO_V(schd))[j]]);
					CO_V(pending_nodes)[CO_V(q)][CO_V(pending_node_size)[CO_V(q)]++] = CO_V(idx);
					blocked = 1;
				}
			if (blocked)
				continue;
			CO_V(task) = _ccv_nnc_graph_exec_run_task(CO_P(graph), CO_V(node), CO_V(schd), CO_V(idx), CO_P(tensor_tape), CO_P(flags));
			if (CO_V(task))
			{
				co_resume(CO_V(task));
				if (!co_is_done(CO_V(task)))
				{
					CO_V(sub_tasks)[CO_V(sub_task_size)++] = CO_V(task);
					for (j = 0; j < CO_V(schd)->stream_size; j++)
						CO_P(graph)->block_stream_tasks[SCHEDULE_STREAMS(*CO_V(schd))[j]] = CO_V(task);
					_ccv_nnc_graph_mark_outgoing_streams_blocked_by_task(CO_P(graph), CO_P(schd_info), CO_V(node), CO_V(task));
				} else
					co_free(CO_V(task));
			}
		}
		int t;
		CCV_SWAP(CO_V(p), CO_V(q), t);
		co_apply(_ccv_nnc_graph_wait_any_sub_tasks, (CO_P(graph), CO_V(sub_tasks), CO_V(sub_task_size), CO_P(schd_info), CO_V(pending_nodes)[CO_V(p)], CO_V(pending_node_size)[CO_V(p)]));
	}
} co_end()

co_task(_ccv_nnc_graph_topsorted_run_coro, (ccv_nnc_graph_t* const graph, const int exec_idx, const ccv_nnc_graph_static_schedule_t* const schedule, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context, const int flags), private(
	ccv_nnc_graph_exec_info_t* exec_info;
	const ccv_nnc_graph_exec_schedule_t* schd_info;
	co_routine_t* previous_main;
	int stream_0;
	// while loop
	int64_t count, reverse_count;
	int graph_breakpoint_size;
	int i, j;
)) {
	assert(CO_P(graph)->stream_size > 0);
	int i;
	// Assign the resource container pointer.
	for (i = 0; i < CO_P(graph)->stream_size; i++)
		CO_P(graph)->streams[i]->resource_container = CO_P(stream_context)->_inline_container;
	CO_V(exec_info) = (ccv_nnc_graph_exec_info_t*)ccv_array_get(CO_P(graph)->exec_info, 0);
	CO_V(schd_info) = CO_P(schedule)->exec_info;
	CO_V(stream_0) = CO_P(schedule)->stream_0;
	if (CO_P(exec_idx) == -1)
	{
		if (CO_P(stream_context)->main)
		{
			CO_V(previous_main) = CO_P(stream_context)->main;
			CO_P(stream_context)->main = co_self();
			// Wait the previous task to be done. This makes sure that our graph run is serial on the same stream.
			assert(!co_is_done(CO_V(previous_main)));
			co_await(CO_V(previous_main));
			co_free(CO_V(previous_main));
		} else
			CO_P(stream_context)->main = co_self();
		if (CO_P(stream_context) != CO_P(graph)->streams[CO_V(stream_0)])
		{
			// Make sure when we start work on streams[0], the current stream context is done.
			ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_get_signal(CO_P(stream_context), (int64_t)(intptr_t)CO_P(graph));
			ccv_nnc_stream_context_emit_signal(CO_P(stream_context), signal);
			ccv_nnc_stream_context_wait_signal(CO_P(graph)->streams[CO_V(stream_0)], signal);
		}
	} else {
		assert(CO_P(stream_context) == CO_P(graph)->streams[0]);
	}
	if (CO_P(exec) && (CO_P(exec)->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
	{
		assert(CO_P(schedule) == CO_P(graph)->default_schedule);
		assert(CO_P(exec)->p_while.expr);
		CO_V(count) = 0;
		// This is a forward while loop. Backward while loop will just consult its pairing part.
		if (CO_P(exec)->cmd.cmd == CCV_NNC_GRAPH_FORWARD)
		{
			CO_V(graph_breakpoint_size) = CO_P(graph)->breakpoint_offset + CO_P(graph)->breakpoint_size;
			for (;; ++CO_V(count))
			{
				CO_P(graph)->while_count = CO_V(count);
				if (CO_P(tensor_tape))
					ccv_nnc_tensor_tape_set_numbering(CO_P(tensor_tape), CO_P(graph)->p, (ccv_nnc_graph_exec_t){
						.d = CO_P(exec_idx),
						.graph = CO_P(graph)->p,
					}, CO_V(count));
				_ccv_nnc_graph_unwrap(CO_P(graph), CO_V(count), 0);
				if (CO_V(count) > 0)
					_ccv_nnc_graph_transit_move_to(CO_P(graph));
				co_apply(_ccv_nnc_graph_exec_run_loop, (CO_P(graph), CO_V(exec_info), CO_V(schd_info), 0, 0, CO_V(graph_breakpoint_size), CO_P(tensor_tape), CO_P(flags)));
				// Reached breakpoints, now check the breakpoint, if not met, break out.
				// Wait until everything on the stream is executed.
				for (CO_V(i) = CO_P(graph)->breakpoint_offset; CO_V(i) < CO_V(graph_breakpoint_size); CO_V(i)++)
					for (CO_V(j) = 0; CO_V(j) < CO_V(schd_info)[CO_V(i)].stream_size; CO_V(j)++)
						co_stream_await(CO_P(graph)->streams[SCHEDULE_STREAMS(CO_V(schd_info)[CO_V(i)])[CO_V(j)]]);
				_ccv_nnc_graph_exec_unwrap_while_expr(CO_P(graph), CO_P(exec));
				if (!CO_P(exec)->p_while.expr(CO_P(exec)->p_while.inputs, CO_P(exec)->p_while.input_size, CO_P(exec)->p_while.data))
				{
					_ccv_nnc_graph_rewrap(CO_P(graph));
					// If we break from here, it is ok because all the streams are waited.
					break;
				}
				co_apply(_ccv_nnc_graph_exec_run_loop, (CO_P(graph), CO_V(exec_info), CO_V(schd_info), 0, CO_V(graph_breakpoint_size), CO_P(graph)->exec_info->rnum, CO_P(tensor_tape), CO_P(flags)));
				_ccv_nnc_graph_from_move_transit(CO_P(graph));
				_ccv_nnc_graph_rewrap(CO_P(graph));
			}
		} else {
			// For backward graph, no need to evaluate the while expr.
			assert(CO_P(exec)->cmd.cmd == CCV_NNC_GRAPH_BACKWARD);
			assert(CO_P(graph)->pair);
			assert(CO_P(tensor_tape));
			CO_V(count) = 0;
			CO_V(reverse_count) = CO_P(graph)->while_count = ccv_nnc_tensor_tape_numbering(CO_P(tensor_tape), CO_P(graph)->p, (ccv_nnc_graph_exec_t){
					.d = CO_P(exec_idx),
					.graph = CO_P(graph)->p,
				});
			_ccv_nnc_graph_unwrap(CO_P(graph), CO_V(count), CO_V(reverse_count));
			co_apply(_ccv_nnc_graph_exec_run_loop, (CO_P(graph), CO_V(exec_info), CO_V(schd_info), 0, CO_P(graph)->breakpoint_offset, CO_P(graph)->exec_info->rnum, CO_P(tensor_tape), CO_P(flags)));
			_ccv_nnc_graph_from_move_transit(CO_P(graph));
			_ccv_nnc_graph_rewrap(CO_P(graph));
			for (CO_V(count) = 1; CO_V(reverse_count) > 0; ++CO_V(count))
			{
				CO_P(graph)->while_count = --CO_V(reverse_count);
				_ccv_nnc_graph_unwrap(CO_P(graph), CO_V(count), CO_V(reverse_count));
				_ccv_nnc_graph_transit_move_to(CO_P(graph));
				co_apply(_ccv_nnc_graph_exec_run_loop, (CO_P(graph), CO_V(exec_info), CO_V(schd_info), 0, 0, CO_P(graph)->exec_info->rnum, CO_P(tensor_tape), CO_P(flags)));
				_ccv_nnc_graph_from_move_transit(CO_P(graph));
				_ccv_nnc_graph_rewrap(CO_P(graph));
			}
		}
		int i;
		for (i = 0; i < CO_P(schedule)->wait_size; i++)
			ccv_nnc_stream_context_wait_signal(CO_P(graph)->streams[0], CO_P(graph)->signals[CO_P(schedule)->waits[i]]);
	} else {
		CO_P(graph)->while_count = 0;
		co_apply(_ccv_nnc_graph_exec_run_loop, (CO_P(graph), CO_V(exec_info), CO_V(schd_info), CO_P(schedule)->psort, 0, CO_P(schedule)->psort ? CO_P(schedule)->psort_size : CO_P(schedule)->exec_info_size, CO_P(tensor_tape), CO_P(flags)));
		int i;
		for (i = 0; i < CO_P(schedule)->wait_size; i++)
			ccv_nnc_stream_context_wait_signal(CO_P(graph)->streams[CO_V(stream_0)], CO_P(graph)->signals[CO_P(schedule)->waits[i]]);
	}
	if (CO_P(stream_context) != CO_P(graph)->streams[CO_V(stream_0)])
	{
		assert(CO_P(exec_idx) == -1);
		ccv_nnc_stream_context_emit_signal(CO_P(graph)->streams[CO_V(stream_0)], CO_P(schedule)->signal_synced);
		ccv_nnc_stream_context_wait_signal(CO_P(stream_context), CO_P(schedule)->signal_synced);
	}
	// Reset main to 0 if it is current me.
	if (CO_P(exec_idx) == -1 && CO_P(stream_context)->main == co_self())
		CO_P(stream_context)->main = 0;
} co_end()

static int _ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context);

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
				_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, flags, 0, 0, 0, 0, tensor_tape, stream_context);
			}
			_ccv_nnc_graph_exec_unwrap_phi(graph, node, ref);
		} else if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) {
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[0] - 1);
			_ccv_nnc_graph_run(sub_graph, idx, node, inputs, node->input_size, outputs, node->output_size, flags, 0, 0, 0, 0, tensor_tape, stream_context);
		}
	} else {
		PRINT(CCV_CLI_VERBOSE, "%s [%d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, node->input_size, node->output_size);
		for (i = 0; i < node->input_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p:%d)", i + 1, inputs[i], (inputs[i] ? inputs[i]->data.u8 : 0), (inputs[i] ? CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type) : -1));
			if (inputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(inputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
		ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, stream_context);
		for (i = 0; i < node->output_size; i++)
		{
			PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p:%d)", i + 1, outputs[i], (outputs[i] ? outputs[i]->data.u8 : 0), (outputs[i] ? CCV_TENSOR_GET_DEVICE_ID(outputs[i]->info.type) : -1));
			if (outputs[i] && CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE))
				_ccv_nnc_print_tensor_verbose(outputs[i]);
			PRINT(CCV_CLI_VERBOSE, "\n");
		}
	}
}

static inline void _ccv_nnc_graph_topsorted_run(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, const int flags, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	if (exec && (exec->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
	{
		assert(!stream_context); // This doesn't work properly with stream context.
		assert(exec->p_while.expr);
		int64_t count = 0;
		// This is a forward while loop. Backward while loop will just consult its pairing part.
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
			assert(graph->pair);
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

static inline void _ccv_nnc_graph_run_slow_path(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
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
		// This is a forward while loop. Backward while loop will just consult its pairing part.
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
			assert(graph->pair);
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

static int _ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int exec_idx, ccv_nnc_graph_exec_info_t* const exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
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
		_ccv_nnc_graph_topsorted_run(graph, exec_idx, exec, flags, tensor_tape, stream_context);
	else
		_ccv_nnc_graph_run_slow_path(graph, exec_idx, exec, inputs, input_size, outputs, output_size, flags, sources, source_size, destinations, destination_size, tensor_tape, stream_context);
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_run(ccv_nnc_graph_t* const graph, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
{
	if (stream_context && graph->topsorted && graph->stream_size > 0 && graph->default_schedule && source_size == 0 && destination_size == 0)
	{
		co_scheduler_t* const scheduler = ccv_nnc_stream_context_get_scheduler(stream_context);
		co_routine_t* const task = co_new(_ccv_nnc_graph_topsorted_run_coro, (graph, -1, graph->default_schedule, 0, tensor_tape, stream_context, flags));
		co_schedule(scheduler, task);
		// I don't need to worry about freeing this task, it will free itself at the end.
		return CCV_NNC_EXEC_SUCCESS;
	} else
		return _ccv_nnc_graph_run(graph, -1, 0, 0, 0, 0, 0, flags, sources, source_size, destinations, destination_size, tensor_tape, 0 /* In this case, we don't support stream context yet. */);
}

int ccv_nnc_graph_run_with_schedule(ccv_nnc_graph_t* const graph, const int flags, const ccv_nnc_graph_static_schedule_t* const _schedule, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const _stream_context)
{
	assert(graph->topsorted);
	assert(graph->stream_size > 0);
	const ccv_nnc_graph_static_schedule_t* const schedule = _schedule ? _schedule : graph->default_schedule;
	assert(schedule);
	ccv_nnc_stream_context_t* const stream_context = _stream_context ? _stream_context : graph->streams[schedule->stream_0];
	co_scheduler_t* const scheduler = ccv_nnc_stream_context_get_scheduler(stream_context);
	co_routine_t* const task = co_new(_ccv_nnc_graph_topsorted_run_coro, (graph, -1, schedule, 0, tensor_tape, stream_context, flags));
	co_schedule(scheduler, task);
		// I don't need to worry about freeing this task, it will free itself at the end.
	if (!_stream_context) // If no stream context provided, this is a sync operation.
		ccv_nnc_stream_context_wait(stream_context);
	return CCV_NNC_EXEC_SUCCESS;
}
