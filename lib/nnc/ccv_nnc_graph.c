#include "ccv_nnc.h"
#include "ccv_internal.h"

typedef struct {
	int input_size;
	int output_size;
	int flags;
	ccv_nnc_tensor_t** inputs;
	ccv_nnc_tensor_t** outputs;
	ccv_array_t* outgoings; // outgoing nodes
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
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

ccv_nnc_graph_node_t ccv_nnc_graph_node(const ccv_nnc_graph_t* graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size > 0 || output_size > 0);
	int d = graph->bn->rnum;
	ccv_nnc_graph_back_t back = {
		.cmd = cmd,
		.hint = hint,
		.flags = flags,
		.input_size = input_size,
		.output_size = output_size,
		.outgoings = 0,
	};
	back.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
	memcpy(back.inputs, inputs, sizeof(ccv_nnc_tensor_t*) * input_size);
	back.outputs = back.inputs + input_size;
	memcpy(back.outputs, outputs, sizeof(ccv_nnc_tensor_t*) * output_size);
	ccv_array_push(graph->bn, &back);
	ccv_nnc_graph_node_t node = {
		.d = d,
		.graph = graph,
	};
	return node;
}

int ccv_nnc_graph_node_concat(const ccv_nnc_graph_t* graph, const ccv_nnc_graph_node_t source, const ccv_nnc_graph_node_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->bn->rnum);
	ccv_nnc_graph_back_t* srcb = (ccv_nnc_graph_back_t*)ccv_array_get(graph->bn, source.d);
	if (srcb->outgoings == 0)
		srcb->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	assert(destination.d < graph->bn->rnum);
	ccv_array_push(srcb->outgoings, &destination.d);
	return 0;
}

typedef struct {
	int8_t d; // tag if this is the destination node.
	int32_t c; // number of incoming edges.
} ccv_nnc_incoming_t;

void ccv_nnc_graph_run(const ccv_nnc_graph_t* graph, const int flags, const ccv_nnc_graph_node_t* sources, const int source_size, const ccv_nnc_graph_node_t* destinations, const int destination_size)
{
	// Statistics of how many incoming edges for all nodes of a graph.
	ccv_nnc_incoming_t* incomings = (ccv_nnc_incoming_t*)alloca(sizeof(ccv_nnc_incoming_t) * graph->bn->rnum);
	memset(incomings, 0, sizeof(ccv_nnc_incoming_t) * graph->bn->rnum);
	int i, j;
	for (i = 0; i < graph->bn->rnum; i++)
	{
		ccv_nnc_graph_back_t* node = (ccv_nnc_graph_back_t*)ccv_array_get(graph->bn, i);
		if (node->outgoings)
			for (j = 0; j < node->outgoings->rnum; j++)
				++incomings[*(int*)ccv_array_get(node->outgoings, j)].c;
	}
	for (i = 0; i < destination_size; i++)
	{
		assert(destinations[i].graph == graph);
		// tagging destination nodes.
		incomings[destinations[i].d].d = 1;
	}
	// After we have that statistics, we can do topsort and run the command.
	int32_t* exists[2];
	exists[0] = (int32_t*)alloca(sizeof(int32_t) * graph->bn->rnum * 2);
	exists[1] = exists[0] + graph->bn->rnum;
	for (i = 0; i < source_size; i++)
	{
		assert(sources[i].graph == graph);
		exists[0][i] = sources[i].d;
	}
	int exist_size[2] = {
		source_size,
		0,
	};
	int p = 0, q = 1; // ping, pong swap.
	while (exist_size[p] > 0)
	{
		exist_size[q] = 0;
		for (i = 0; i < exist_size[p]; i++)
		{
			ccv_nnc_graph_back_t* node = (ccv_nnc_graph_back_t*)ccv_array_get(graph->bn, exists[p][i]);
			// exec current node.
			ccv_nnc_cmd_exec(node->cmd, node->hint, node->flags, node->inputs, node->input_size, node->outputs, node->output_size);
			if (node->outgoings)
				for (j = 0; j < node->outgoings->rnum; j++)
				{
					int d = *(int*)ccv_array_get(node->outgoings, j);
					--incomings[d].c;
					// If all incoming edges are consumed, and this is not the destination node, push it into next round
					if (incomings[d].c == 0 && !incomings[d].d)
					{
						exists[q][exist_size[q]] = d;
						++exist_size[q];
					}
				}
		}
		// swap p and q.
		CCV_SWAP(p, q, i /* using i as temp holder */);
	}
	for (i = 0; i < destination_size; i++)
	{
		assert(destinations[i].graph == graph);
		// tagging destination nodes.
		assert(incomings[destinations[i].d].c == 0);
		// fetch the back node for destination nodes
		ccv_nnc_graph_back_t* node = (ccv_nnc_graph_back_t*)ccv_array_get(graph->bn, destinations[i].d);
		// exec current node.
		ccv_nnc_cmd_exec(node->cmd, node->hint, node->flags, node->inputs, node->input_size, node->outputs, node->output_size);
	}
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
		// We allocate inputs & outputs in continuous fashion, therefore, only need to free the input array.
		ccfree(back->inputs);
	}
	ccv_array_free(graph->bn);
	ccfree(graph);
}
