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

ccv_nnc_graph_exec_t ccv_nnc_graph_while(ccv_nnc_graph_t* const graph, uint32_t cmd, ccv_nnc_graph_t* const while_graph, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size, const ccv_nnc_graph_exec_t* const conditionals, const int conditional_size, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_graph_while_f while_func, const void* const while_data)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	ccv_nnc_graph_exec_t while_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, inputs, input_size, outputs, output_size);
	ccv_nnc_graph_exec_info_t* while_exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, while_exec.d);
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_graph_t*), 1, 0);
	while_graph->while_data = while_data;
	while_graph->while_func = while_func;
	assert(conditional_size > 0);
	while_graph->conditional_size = conditional_size;
	while_graph->conditionals = (ccv_nnc_graph_exec_t*)((while_graph->conditionals) ? ccrealloc(while_graph->conditionals, sizeof(ccv_nnc_graph_exec_t) * conditional_size) : ccmalloc(sizeof(ccv_nnc_graph_exec_t) * conditional_size));
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
	memcpy(while_graph->conditionals, conditionals, sizeof(ccv_nnc_graph_exec_t) * conditional_size);
	ccv_array_push(graph->sub_graphs, &while_graph);
	while_exec_info->graph_ref = graph->sub_graphs->rnum;
	return while_exec;
}

int ccv_nnc_graph_while_run(const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	// TODO: some error checking.
	assert(tensor_tape == 0); // Cannot handle tensor tape yet.
	// This is a while loop.
#define visitor(node, ...) \
	do { \
		if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD) \
		{ \
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, node->graph_ref - 1); \
			ccv_nnc_graph_while_run(sub_graph, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
		} else \
			ccv_nnc_cmd_exec(node->cmd, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0); \
	} while (0)
	if (graph->while_func)
	{
		int i, j;
		ccv_array_t* follows = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), graph->conditional_size, 0);
		for (i = 0; i < graph->conditional_size; i++)
		{
			const ccv_nnc_graph_exec_info_t* const exec_info = (const ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, graph->conditionals->d);
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
		for (;;)
		{
			++count;
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, graph->conditionals, graph->conditional_size, visitor);
			// Reached conditionals, now check the conditional, if not met, break out.
			if (!graph->while_func(special_tensors, 1, 0, 0, 0, 0, graph->while_data))
				break;
			if (follows->rnum > 0)
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(follows, 0), follows->rnum, destinations, destination_size, visitor);
			// TODO: Change versions on the tensor, so it point to a different part of the data.
		}
		ccv_array_free(follows);
	} else {
		CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
	}
#undef visitor
	return CCV_NNC_EXEC_SUCCESS;
}
