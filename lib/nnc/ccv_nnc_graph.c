#include "ccv_nnc.h"

typedef struct {
	int input_size;
	int output_size;
	ccv_nnc_tensor_t** inputs;
	ccv_nnc_tensor_t** outputs;
	ccv_array_t* outgoings; // outgoing nodes
	ccv_nnc_hint_t hint;
	ccv_nnc_cmd_t cmd;
} ccv_nnc_graph_back_t;

struct ccv_nnc_graph_s {
	ccv_array_t* bn;
};

ccv_nnc_graph_t* ccv_nnc_graph_new(void)
{
	ccv_nnc_graph_t* graph = (ccv_nnc_graph_t*)ccmalloc(sizeof(ccv_nnc_graph_t));
	graph->bn = ccv_array_new(sizeof(ccv_nnc_graph_back_t), 5, 0);
	return graph;
}

ccv_nnc_graph_node_t ccv_nnc_graph_node(const ccv_nnc_graph_t* graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size > 0 || output_size > 0);
	int d = graph->bn->rnum;
	ccv_nnc_graph_back_t back = {
		.cmd = cmd,
		.hint = hint,
		.input_size = input_size,
		.output_size = output_size,
		.outgoings = 0,
	};
	back.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
	back.outputs = back.inputs + input_size;
	ccv_array_push(graph->bn, &back);
	ccv_nnc_graph_node_t node = {
		.d = d,
		.graph = graph,
	};
	return node;
}

int ccv_nnc_graph_node_concat(const ccv_nnc_graph_node_t source, const ccv_nnc_graph_node_t destination)
{
	assert(source.graph == destination.graph);
	const ccv_nnc_graph_t* graph = source.graph;
	assert(source.d < graph->bn->rnum);
	ccv_nnc_graph_back_t* srcb = (ccv_nnc_graph_back_t*)ccv_array_get(graph->bn, source.d);
	if (srcb->outgoings == 0)
		srcb->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	assert(destination.d < graph->bn->rnum);
	ccv_array_push(srcb->outgoings, &destination.d);
	return 0;
}

void ccv_nnc_graph_run(const ccv_nnc_graph_t* graph, const ccv_nnc_graph_node_t* sources, const int source_size, const ccv_nnc_graph_node_t* destinations, const int destination_size, int flags)
{
	// Statistics of how many incoming edges for all nodes of a graph.
	int32_t* incomings = (int32_t*)alloca(sizeof(int32_t) * graph->bn->rnum);
	memset(incomings, 0, sizeof(int32_t) * graph->bn->rnum);
	int i, j;
	for (i = 0; i < graph->bn->rnum; i++)
	{
		ccv_nnc_graph_back_t* node = (ccv_nnc_graph_back_t*)ccv_array_get(graph->bn, i);
		if (node->outgoings)
			for (j = 0; j < node->outgoings->rnum; j++)
				++incomings[*(int*)ccv_array_get(node->outgoings, j)];
	}
	// After we have that statistics, we can do topsort and run the command.
}

void ccv_nnc_graph_free(ccv_nnc_graph_t* graph)
{
	int i;
	for (i = 0; i < graph->bn->rnum; i++)
	{
		ccv_nnc_graph_back_t *back = (ccv_nnc_graph_back_t*)ccv_array_get(graph->bn, i);
		ccv_array_t* outgoings = back->outgoings;
		if (outgoings)
			ccv_array_free(outgoings);
		// We allocate inputs & outputs in continuous fashion.
		ccfree(back->inputs);
	}
	ccv_array_free(graph->bn);
	ccfree(graph);
}
