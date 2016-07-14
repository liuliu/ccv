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

ccv_nnc_graph_exec_t ccv_nnc_graph_exec(const ccv_nnc_graph_t* graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size > 0 || output_size > 0);
	int d = graph->exec_info->rnum;
	ccv_nnc_graph_exec_info_t info = {
		.cmd = cmd,
		.hint = hint,
		.flags = flags,
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

typedef struct {
	int8_t d; // tag if this is the destination node.
	int32_t c; // number of incoming edges.
} ccv_nnc_incoming_t;

void ccv_nnc_graph_run(const ccv_nnc_graph_t* graph, const int flags, const ccv_nnc_graph_exec_t* sources, const int source_size, const ccv_nnc_graph_exec_t* destinations, const int destination_size)
{
	// Statistics of how many incoming edges for all nodes of a graph.
	ccv_nnc_incoming_t* incomings = (ccv_nnc_incoming_t*)alloca(sizeof(ccv_nnc_incoming_t) * graph->exec_info->rnum);
	memset(incomings, 0, sizeof(ccv_nnc_incoming_t) * graph->exec_info->rnum);
	int i, j;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (info->outgoings)
			for (j = 0; j < info->outgoings->rnum; j++)
				++incomings[*(int*)ccv_array_get(info->outgoings, j)].c;
	}
	for (i = 0; i < destination_size; i++)
	{
		assert(destinations[i].graph == graph);
		// tagging destination nodes.
		incomings[destinations[i].d].d = 1;
	}
	// After we have that statistics, we can do topsort and run the command.
	int32_t* exists[2];
	exists[0] = (int32_t*)alloca(sizeof(int32_t) * graph->exec_info->rnum * 2);
	exists[1] = exists[0] + graph->exec_info->rnum;
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
			ccv_nnc_graph_exec_info_t* info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exists[p][i]);
			// exec current node, for synchronous CPU execution, no stream unit.
			ccv_nnc_cmd_exec(info->cmd, info->hint, info->flags, info->inputs, info->input_size, info->outputs, info->output_size, 0);
			if (info->outgoings)
				for (j = 0; j < info->outgoings->rnum; j++)
				{
					int d = *(int*)ccv_array_get(info->outgoings, j);
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
		// fetch the info for destination nodes
		ccv_nnc_graph_exec_info_t* info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, destinations[i].d);
		// exec current node, for synchronous CPU execution, no stream unit.
		ccv_nnc_cmd_exec(info->cmd, info->hint, info->flags, info->inputs, info->input_size, info->outputs, info->output_size, 0);
	}
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
