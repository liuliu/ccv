#include "ccv_nnc.h"

typedef struct ccv_nnc_net_graph_node_backing_s {
	int input_size;
	int output_size;
	ccv_nnc_tensor_t** inputs;
	ccv_nnc_tensor_t** outputs;
	ccv_array_t* incomings; // incoming nodes
	ccv_array_t* outgoings; // outgoing nodes
	ccv_nnc_net_command_t command;
} ccv_nnc_net_graph_node_backing_t;

struct ccv_nnc_net_graph_s {
	ccv_array_t* nodes;
};

ccv_nnc_net_graph_t* ccv_nnc_net_graph_new(void)
{
	ccv_nnc_net_graph_t* graph = (ccv_nnc_net_graph_t*)ccmalloc(sizeof(ccv_nnc_net_graph_t));
	graph->nodes = ccv_array_new(sizeof(ccv_nnc_net_graph_node_backing_t), 5, 0);
	return graph;
}

ccv_nnc_net_graph_node_t ccv_nnc_net_graph_node(const ccv_nnc_net_graph_t* graph, const ccv_nnc_net_command_t command, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	int i = graph->nodes->rnum;
	ccv_nnc_net_graph_node_backing_t node_backing = {
		.command = command,
		.input_size = input_size,
		.output_size = output_size,
		.incomings = 0,
		.outgoings = 0,
	};
	node_backing.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
	node_backing.outputs = node_backing.inputs + input_size;
	ccv_array_push(graph->nodes, &node_backing);
	ccv_nnc_net_graph_node_t node = {
		.i = i,
		.graph = graph,
	};
	return node;
}

int ccv_nnc_net_node_concat(const ccv_nnc_net_graph_node_t source, const ccv_nnc_net_graph_node_t destination)
{
	assert(source.graph == destination.graph);
	return 0;
}

void ccv_nnc_net_graph_run(const ccv_nnc_net_graph_t* graph, const ccv_nnc_net_graph_node_t* sources, const int source_size, const ccv_nnc_net_graph_t destinations, const int destination_size)
{
}

void ccv_nnc_net_graph_free(ccv_nnc_net_graph_t* graph)
{
	int i;
	for (i = 0; i < graph->nodes->rnum; i++)
	{
		ccv_nnc_net_graph_node_backing_t *node_backing = (ccv_nnc_net_graph_node_backing_t*)ccv_array_get(graph->nodes, i);
		// We allocate inputs & outputs in continuous fashion.
		ccfree(node_backing->inputs);
	}
	ccv_array_free(graph->nodes);
	ccfree(graph);
}
