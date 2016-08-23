#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

typedef struct {
	int input_size;
	int output_size;
	ccv_nnc_tensor_t** inputs;
	ccv_nnc_tensor_t** outputs;
	ccv_array_t* outgoings; // outgoing nodes
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
} ccv_nnc_graph_exec_info_t;

struct ccv_nnc_graph_s {
	ccv_array_t* exec_info; // deferred exec info
};

ccv_nnc_graph_t* ccv_nnc_graph_new(void)
{
	ccv_nnc_graph_t* graph = (ccv_nnc_graph_t*)ccmalloc(sizeof(ccv_nnc_graph_t));
	graph->exec_info = ccv_array_new(sizeof(ccv_nnc_graph_exec_info_t), 5, 0);
	return graph;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec(const ccv_nnc_graph_t* graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size > 0 || output_size > 0);
	int d = graph->exec_info->rnum;
	ccv_nnc_graph_exec_info_t info = {
		.cmd = cmd,
		.hint = hint,
		.input_size = input_size,
		.output_size = output_size,
		.outgoings = 0,
	};
	info.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
	memcpy(info.inputs, inputs, sizeof(ccv_nnc_tensor_t*) * input_size);
	info.outputs = info.inputs + input_size;
	memcpy(info.outputs, outputs, sizeof(ccv_nnc_tensor_t*) * output_size);
	ccv_array_push(graph->exec_info, &info);
	ccv_nnc_graph_exec_t exec = {
		.d = d,
		.graph = graph,
	};
	return exec;
}

int ccv_nnc_graph_exec_concat(const ccv_nnc_graph_t* graph, const ccv_nnc_graph_exec_t source, const ccv_nnc_graph_exec_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_info->rnum);
	assert(destination.d < graph->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* src_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, source.d);
	if (src_info->outgoings == 0)
		src_info->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	ccv_array_push(src_info->outgoings, &destination.d);
	return 0;
}

void ccv_nnc_graph_run(const ccv_nnc_graph_t* graph, const int flags, const ccv_nnc_graph_exec_t* sources, const int source_size, const ccv_nnc_graph_exec_t* destinations, const int destination_size)
{
	// exec current node, for synchronous CPU execution, no stream unit.
#define visitor(node, ...) \
	ccv_nnc_cmd_exec(node->cmd, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
}

void ccv_nnc_graph_free(ccv_nnc_graph_t* graph)
{
	int i;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t *info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		ccv_array_t* outgoings = info->outgoings;
		if (outgoings)
			ccv_array_free(outgoings);
		// We allocate inputs & outputs in continuous fashion, therefore, only need to free the input array.
		ccfree(info->inputs);
	}
	ccv_array_free(graph->exec_info);
	ccfree(graph);
}
