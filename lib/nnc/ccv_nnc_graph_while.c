#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_graph.h"

void ccv_nnc_tensor_multiview(ccv_nnc_tensor_t* const tv, ccv_numeric_data_t data[], const int kind, const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	assert(kind == CCV_NNC_MULTIVIEW_K01 || kind == CCV_NNC_MULTIVIEW_K11 || kind == CCV_NNC_MULTIVIEW_K02 || kind == CCV_NNC_MULTIVIEW_K12);
	tensor_multiview->type = CCV_TENSOR_MULTIVIEW;
	tensor_multiview->kind = kind;
	tensor_multiview->anchor = (intptr_t)graph;
	tensor_multiview->tv = tv;
	tensor_multiview->p = 0;
	tensor_multiview->offset = 0;
	tensor_multiview->rtvs = 0;
	int i;
	// Currently, only CCV_NNC_MULTIVIEW_K12 uses 3 tensors.
	const int kindz[] = {
		1, 2, 2, 3
	};
	for (i = 0; i < kindz[kind]; i++)
	{
		tensor_multiview->data[i] = data[i];
		if (!tv)
		{
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)data[i].ptr;
			assert(CCV_IS_TENSOR_MULTIVIEW(mv));
			mv->p = tensor_multiview;
		}
	}
}

void ccv_nnc_tensor_multiview_free(const ccv_nnc_tensor_multiview_t tensor_multiview)
{
	if (tensor_multiview.rtvs)
		ccv_array_free(tensor_multiview.rtvs);
}

void ccv_nnc_tensor_reference_to_multiview(ccv_nnc_tensor_multiview_t* const tensor_multiview, const off_t offset, ccv_nnc_tensor_t* const tensor)
{
	assert(!CCV_IS_TENSOR_MULTIVIEW(tensor));
	ccv_nnc_tensor_reference_t tensor_reference = {
		.offset = offset,
		.tensor = tensor,
	};
	if (!tensor_multiview->rtvs)
		tensor_multiview->rtvs = ccv_array_new(sizeof(ccv_nnc_tensor_reference_t), 0, 0);
	ccv_array_push(tensor_multiview->rtvs, &tensor_reference);
}

void ccv_nnc_tensor_multiview_broadcast(const ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	// Update the pointer on tv.
	tensor_multiview->tv->data = tensor_multiview->it;
	unsigned char* const data = tensor_multiview->it.u8 - tensor_multiview->offset;
	const ccv_nnc_tensor_multiview_t* c = tensor_multiview;
	int i;
	do {
		if (c->rtvs)
			for (i = 0; i < c->rtvs->rnum; i++)
			{
				ccv_nnc_tensor_reference_t* reference = (ccv_nnc_tensor_reference_t*)ccv_array_get(c->rtvs, i);
				reference->tensor->data.u8 = data + reference->offset;
			}
		c = c->p;
	} while (c);
}

ccv_nnc_graph_exec_t ccv_nnc_graph_while(ccv_nnc_graph_t* const graph, uint32_t cmd, ccv_nnc_graph_t* const while_graph)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	ccv_nnc_graph_exec_t while_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_info_t* while_exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, while_exec.d);
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_graph_t*), 1, 0);
	int i;
	if (while_graph->wraps)
	{
		// Copy wraps from sub graph to parent graph.
		if (!graph->wraps)
			graph->wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), while_graph->wraps->rnum, 0);
		for (i = 0; i < while_graph->wraps->rnum; i++)
			ccv_array_push(graph->wraps, ccv_array_get(while_graph->wraps, i));
	}
	ccv_array_push(graph->sub_graphs, &while_graph);
	while_graph->p = graph;
	while_graph->exec_idx = while_exec.d + 1;
	while_exec_info->graph_ref = graph->sub_graphs->rnum;
	return while_exec;
}

void ccv_nnc_graph_set_while_expr(ccv_nnc_graph_t* const while_graph, const ccv_nnc_graph_while_f while_expr, const void* const while_data, const ccv_nnc_graph_exec_t* const cond_evals, const int cond_eval_size)
{
	while_graph->while_data = while_data;
	while_graph->while_expr = while_expr;
	assert(cond_eval_size > 0);
	while_graph->cond_eval_size = cond_eval_size;
	while_graph->cond_evals = (ccv_nnc_graph_exec_t*)((while_graph->cond_evals) ? ccrealloc(while_graph->cond_evals, sizeof(ccv_nnc_graph_exec_t) * cond_eval_size) : ccmalloc(sizeof(ccv_nnc_graph_exec_t) * cond_eval_size));
	memcpy(while_graph->cond_evals, cond_evals, sizeof(ccv_nnc_graph_exec_t) * cond_eval_size);
}

#define TAG_TENSOR_REQUIRE_BROADCAST(x) (ccv_nnc_tensor_t*)((intptr_t)(x) | 1)
#define UNTAG_TENSOR_REQUIRE_BROADCAST(x) (ccv_nnc_tensor_t*)((intptr_t)(x) & ~(intptr_t)1)
#define IS_TAGGED_TENSOR_REQUIRE_BROADCAST(x) ((intptr_t)(x) & 1)

static void _ccv_nnc_graph_unwrap(const ccv_nnc_graph_t* const graph, const int count)
{
	if (!graph->wraps)
		return;
	int i, j;
	for (i = 0; i < graph->wraps->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->wraps, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		ccv_nnc_tensor_t** const tensors = exec_info->inputs + (exec_info->input_size + exec_info->output_size) * exec_info->wrap_ptr;
		const int tensor_size = exec_info->input_size + exec_info->output_size;
		int rewrap = 0;
		for (j = 0; j < tensor_size && !rewrap; j++)
			// If I have a multi-view tensor and this multi-view tensor need to be unwrapped at this level (wrap_anchor)
			if (CCV_IS_TENSOR_MULTIVIEW(tensors[j]) && ((ccv_nnc_tensor_multiview_t*)tensors[j])->anchor == (intptr_t)graph)
				rewrap = 1;
		if (rewrap)
		{
			// Unwrap tensors at this level.
			++exec_info->wrap_ptr;
			ccv_nnc_tensor_t** const unwrap_tensors = exec_info->inputs + (exec_info->input_size + exec_info->output_size) * exec_info->wrap_ptr;
			for (j = 0; j < tensor_size; j++)
			{
				assert(!IS_TAGGED_TENSOR_REQUIRE_BROADCAST(tensors[j])); // I cannot encounter a tagged pointer.
				if (CCV_IS_TENSOR_MULTIVIEW(tensors[j]) && ((ccv_nnc_tensor_multiview_t*)tensors[j])->anchor == (intptr_t)graph)
				{
					// This can be unwrapped, do that.
					ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensors[j];
					const int off = (mv->kind >> 1) & 1;
					const int mask = mv->kind & 1;
					// If reached the root.
					if (mv->tv)
					{
						// Update the pointer
						mv->it = mv->data[count >= off ? ((count - off) & mask) + off : count]; // See the comment of the CCV_NNC_MULTIVIEW_KXX enum for why the computation carried out this way.
						unwrap_tensors[j] = TAG_TENSOR_REQUIRE_BROADCAST(tensors[j]); // Keep it dirty yet, will unwrap the first time encountered it in actual execution, using tagged pointer to keep track.
						// In this way, I can broadcast the pointer change only when executing it, to avoid early abortion causing no pointer
						// update is needed.
					} else
						unwrap_tensors[j] = (ccv_nnc_tensor_t*)mv->data[count >= off ? ((count - off) & mask) + off : count].ptr; // Unwrap.
				} else
					unwrap_tensors[j] = tensors[j]; // Just copy it over
			}
		}
	}
}

static void _ccv_nnc_graph_rewrap(const ccv_nnc_graph_t* const graph) // Call this method at the end to roll the wrap_ptr back
{
	if (!graph->wraps)
		return;
	int i, j;
	for (i = 0; i < graph->wraps->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->wraps, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		if (exec_info->wrap_ptr > 0)
		{
			ccv_nnc_tensor_t** const tensors = exec_info->inputs + (exec_info->input_size + exec_info->output_size) * (exec_info->wrap_ptr - 1);
			const int tensor_size = exec_info->input_size + exec_info->output_size;
			int rewrap = 0;
			for (j = 0; j < tensor_size && !rewrap; j++)
				// If I have a multi-view tensor and this multi-view tensor need to be unwrapped at this level (wrap_anchor)
				if (CCV_IS_TENSOR_MULTIVIEW(tensors[j]) && ((ccv_nnc_tensor_multiview_t*)tensors[j])->anchor == (intptr_t)graph)
					rewrap = 1;
			// If I did rewrap before, pop the pointer.
			if (rewrap)
				--exec_info->wrap_ptr;
		}
		assert(exec_info->wrap_ptr >= 0);
	}
}

static int _ccv_nnc_graph_while_run(const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	assert(tensor_tape == 0); // Cannot handle tensor tape yet.
	int i, j;
	for (i = 0; i < source_size; i++)
		if (sources[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < destination_size; i++)
		if (destinations[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
#define visitor(node, idx, d, ...) \
	do { \
		ccv_nnc_tensor_t** inputs = node->inputs + (node->input_size + node->output_size) * node->wrap_ptr; \
		ccv_nnc_tensor_t** outputs = inputs + node->input_size; \
 		/* Broadcast the updates to all subscribed references for input / output, even though at this
		 * time output is not written yet, propagate pointer change is still valid. */ \
		for (i = 0; i < node->input_size; i++) \
			if (IS_TAGGED_TENSOR_REQUIRE_BROADCAST(inputs[i])) \
			{ \
				ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)UNTAG_TENSOR_REQUIRE_BROADCAST(inputs[i]); \
				assert(CCV_IS_TENSOR_MULTIVIEW(mv)); \
				if (mv->tv) /* This is marked dirty. Unwrap it and broadcast.*/ \
					ccv_nnc_tensor_multiview_broadcast(mv), inputs[i] = mv->tv; \
			} \
		for (i = 0; i < node->output_size; i++) \
			if (IS_TAGGED_TENSOR_REQUIRE_BROADCAST(outputs[i])) \
			{ \
				ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)UNTAG_TENSOR_REQUIRE_BROADCAST(outputs[i]); \
				assert(CCV_IS_TENSOR_MULTIVIEW(mv)); \
				if (mv->tv) /* This is marked dirty. Unwrap it and broadcast.*/ \
					ccv_nnc_tensor_multiview_broadcast(mv), outputs[i] = mv->tv; \
			} \
		if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD) \
		{ \
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, node->graph_ref - 1); \
			_ccv_nnc_graph_while_run(sub_graph, inputs, node->input_size, outputs, node->output_size, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
		} else { \
			PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, d, node->input_size, node->output_size); \
			for (i = 0; i < node->input_size; i++) \
				PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)\n", i + 1, inputs[i], (inputs[i] ? node->inputs[i]->data.u8 : 0)); \
			for (i = 0; i < node->output_size; i++) \
				PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)\n", i + 1, outputs[i], (outputs[i] ? node->outputs[i]->data.u8 : 0)); \
			ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, 0); \
		} \
	} while (0)
	if (graph->while_expr)
	{
		// This is a while loop.
		ccv_array_t* follows = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), graph->cond_eval_size, 0);
		for (i = 0; i < graph->cond_eval_size; i++)
		{
			const ccv_nnc_graph_exec_info_t* const exec_info = (const ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, graph->cond_evals->d);
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
		uint64_t count = 0;
		ccv_nnc_tensor_t count_tensor = ccv_nnc_tensor(&count, ONE_CPU_TENSOR(1, 1, 1), 0);
		ccv_nnc_tensor_t* special_tensors[] = { &count_tensor };
		for (;; ++count)
		{
			_ccv_nnc_graph_unwrap(graph, count);
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, graph->cond_evals, graph->cond_eval_size, visitor);
			// Reached cond_evals, now check the cond_eval, if not met, break out.
			if (!graph->while_expr(special_tensors, 1, inputs, input_size, outputs, output_size, graph->while_data))
			{
				_ccv_nnc_graph_rewrap(graph);
				break;
			}
			if (follows->rnum > 0)
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(follows, 0), follows->rnum, destinations, destination_size, visitor);
			_ccv_nnc_graph_rewrap(graph);
		}
		ccv_array_free(follows);
	} else {
		CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_while_run(const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	return _ccv_nnc_graph_while_run(graph, 0, 0, 0, 0, tensor_tape, flags, sources, source_size, destinations, destination_size);
}
