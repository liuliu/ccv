#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_graph.h"

ccv_nnc_tensor_multiview_t ccv_nnc_tensor_multiview(ccv_nnc_tensor_t* const tv, const int versions, const int repeats, ccv_numeric_data_t* const data)
{
	ccv_nnc_tensor_multiview_t tensor_multiview;
	tensor_multiview.type = CCV_TENSOR_MULTIVIEW;
	tensor_multiview.data = data;
	tensor_multiview.versions = versions;
	tensor_multiview.repeats = repeats;
	tensor_multiview.tv = tv;
	return tensor_multiview;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_while(ccv_nnc_graph_t* const graph, uint32_t cmd, ccv_nnc_graph_t* const while_graph, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size, const ccv_nnc_graph_exec_t* const cond_evals, const int cond_eval_size, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_graph_while_f while_expr, const void* const while_data)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	ccv_nnc_graph_exec_t while_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, inputs, input_size, outputs, output_size);
	ccv_nnc_graph_exec_info_t* while_exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, while_exec.d);
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_graph_t*), 1, 0);
	while_graph->while_data = while_data;
	while_graph->while_expr = while_expr;
	assert(cond_eval_size > 0);
	while_graph->cond_eval_size = cond_eval_size;
	while_graph->cond_evals = (ccv_nnc_graph_exec_t*)((while_graph->cond_evals) ? ccrealloc(while_graph->cond_evals, sizeof(ccv_nnc_graph_exec_t) * cond_eval_size) : ccmalloc(sizeof(ccv_nnc_graph_exec_t) * cond_eval_size));
	if (!while_graph->sources)
		while_graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), source_size, 0);
	else
		ccv_array_clear(while_graph->sources);
	int i;
	for (i = 0; i < source_size; i++)
		ccv_array_push(while_graph->sources, sources + i);
	if (!while_graph->destinations)
		while_graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), destination_size, 0);
	else
		ccv_array_clear(while_graph->destinations);
	for (i = 0; i < destination_size; i++)
		ccv_array_push(while_graph->destinations, destinations + i);
	memcpy(while_graph->cond_evals, cond_evals, sizeof(ccv_nnc_graph_exec_t) * cond_eval_size);
	ccv_array_push(graph->sub_graphs, &while_graph);
	while_exec_info->graph_ref = graph->sub_graphs->rnum;
	return while_exec;
}

static void _ccv_nnc_graph_exec_unwrap_inputs_outputs(const ccv_nnc_graph_exec_info_t* const graph_exec, const uint64_t count, ccv_nnc_tensor_t*** inputs, ccv_nnc_tensor_t*** outputs)
{
	assert(inputs);
	assert(outputs);
	if (graph_exec->wrapped)
	{
		*inputs = graph_exec->unwrapped_inputs;
		*outputs = graph_exec->unwrapped_outputs;
		int i;
		for (i = 0; i < graph_exec->input_size; i++)
			if (CCV_IS_TENSOR_MULTIVIEW(graph_exec->inputs[i]))
			{
				ccv_nnc_tensor_multiview_t* tmv = (ccv_nnc_tensor_multiview_t*)graph_exec->inputs[i];
				const int ver = count >= tmv->versions ? (count - (tmv->versions - tmv->repeats)) % tmv->repeats + (tmv->versions - tmv->repeats) : count;
				// Update the data pointer to the latest version.
				if (CCV_IS_TENSOR_MULTIVIEW(tmv->tv))
					// If it points to a multiview again, change that multiview's data ptr to this newer version.
					((ccv_nnc_tensor_multiview_t*)(tmv->tv))->data = tmv->data[ver].ptr;
				else
					tmv->tv->data = tmv->data[ver];
				graph_exec->unwrapped_inputs[i] = tmv->tv;
			} else
				graph_exec->unwrapped_inputs[i] = graph_exec->inputs[i];
		for (i = 0; i < graph_exec->output_size; i++)
			if (CCV_IS_TENSOR_MULTIVIEW(graph_exec->outputs[i]))
			{
				ccv_nnc_tensor_multiview_t* tmv = (ccv_nnc_tensor_multiview_t*)graph_exec->outputs[i];
				const int ver = count >= tmv->versions ? (count - (tmv->versions - tmv->repeats)) % tmv->repeats + (tmv->versions - tmv->repeats) : count;
				// Update the data pointer to the latest version.
				if (CCV_IS_TENSOR_MULTIVIEW(tmv->tv))
					// If it points to a multiview again, change that multiview's data ptr to this newer version.
					((ccv_nnc_tensor_multiview_t*)(tmv->tv))->data = tmv->data[ver].ptr;
				else
					tmv->tv->data = tmv->data[ver];
				graph_exec->unwrapped_outputs[i] = tmv->tv;
			} else
				graph_exec->unwrapped_outputs[i] = graph_exec->outputs[i];
		return;
	}
	*inputs = graph_exec->inputs;
	*outputs = graph_exec->outputs;
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
	if (graph->while_expr)
	{
#define visitor(node, idx, d, ...) \
		do { \
			ccv_nnc_tensor_t** unwrapped_inputs; \
			ccv_nnc_tensor_t** unwrapped_outputs; \
			/* Do the unwrap at this point. Note that unwrapping will use the current count to
			 * compute a data pointer, and will change the underlying tensor's data pointer, that
			 * means you cannot pass that tensor along while updating the count in a while loop.
			 * This is OK because every time we update the count, we made sure that we reached the
			 * end of the destinations, therefore, no ops are currently in progress to use these
			 * tensors. */ \
			_ccv_nnc_graph_exec_unwrap_inputs_outputs(node, count, &unwrapped_inputs, &unwrapped_outputs); \
			if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD) \
			{ \
				ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, node->graph_ref - 1); \
				_ccv_nnc_graph_while_run(sub_graph, unwrapped_inputs, node->input_size, unwrapped_outputs, node->output_size, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
			} else { \
				PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, d, node->input_size, node->output_size); \
				for (i = 0; i < node->input_size; i++) \
					PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)\n", i + 1, node->inputs[i], (node->inputs[i] ? node->inputs[i]->data.u8 : 0)); \
				for (i = 0; i < node->output_size; i++) \
					PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)\n", i + 1, node->outputs[i], (node->outputs[i] ? node->outputs[i]->data.u8 : 0)); \
				ccv_nnc_cmd_exec(node->cmd, node->hint, flags, unwrapped_inputs, node->input_size, unwrapped_outputs, node->output_size, 0); \
			} \
		} while (0)
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
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, graph->cond_evals, graph->cond_eval_size, visitor);
			// Reached cond_evals, now check the cond_eval, if not met, break out.
			if (!graph->while_expr(special_tensors, 1, inputs, input_size, outputs, output_size, graph->while_data))
				break;
			if (follows->rnum > 0)
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(follows, 0), follows->rnum, destinations, destination_size, visitor);
		}
		ccv_array_free(follows);
#undef visitor
	} else {
#define visitor(node, idx, d, ...) \
		do { \
			if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD) \
			{ \
				ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, node->graph_ref - 1); \
				_ccv_nnc_graph_while_run(sub_graph, node->inputs, node->input_size, node->outputs, node->output_size, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
			} else { \
				PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, d, node->input_size, node->output_size); \
				for (i = 0; i < node->input_size; i++) \
					PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)\n", i + 1, node->inputs[i], (node->inputs[i] ? node->inputs[i]->data.u8 : 0)); \
				for (i = 0; i < node->output_size; i++) \
					PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)\n", i + 1, node->outputs[i], (node->outputs[i] ? node->outputs[i]->data.u8 : 0)); \
				ccv_nnc_cmd_exec(node->cmd, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0); \
			} \
		} while (0)
		CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_while_run(const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	return _ccv_nnc_graph_while_run(graph, 0, 0, 0, 0, tensor_tape, flags, sources, source_size, destinations, destination_size);
}
