#include "ccv_nnc.h"

typedef struct ccv_nnc_net_switch_s {
	struct ccv_nnc_net_switch_s** incomings;
	int incoming_size;
	struct ccv_nnc_net_switch_s** outgoings;
	int outgoing_size;
	ccv_nnc_tensor_t** inputs;
	int input_size;
	ccv_nnc_net_command_t command;
	ccv_nnc_tensor_t** outputs;
	int output_size;
} ccv_nnc_net_switch_t;

struct ccv_nnc_net_graph_s {
	ccv_nnc_net_switch_t** incomings;
	int incoming_size;
	ccv_nnc_net_switch_t** outgoings;
	int outgoing_size;
};

static ccv_nnc_net_switch_t* _ccv_nnc_net_switch_new(int incoming_size, int outgoing_size, const ccv_nnc_net_command_t command, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	ccv_nnc_net_switch_t* net_switch = (ccv_nnc_net_switch_t*)ccmalloc(sizeof(ccv_nnc_net_switch_t) + sizeof(ccv_nnc_net_switch_t*) * (incoming_size + outgoing_size) + sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
	net_switch->incomings = (ccv_nnc_net_switch_t**)(net_switch + 1);
	net_switch->outgoings = net_switch->incomings + incoming_size;
	net_switch->inputs = (ccv_nnc_tensor_t**)(net_switch->outgoings + outgoing_size);
	net_switch->outputs = net_switch->inputs + input_size;
	net_switch->command = command;
	net_switch->incoming_size = incoming_size;
	net_switch->outgoing_size = outgoing_size;
	net_switch->input_size = input_size;
	net_switch->output_size = output_size;
	return net_switch;
}

static ccv_nnc_net_switch_t* _ccv_nnc_net_switch_dup(ccv_nnc_net_switch_t* a, ccv_nnc_net_switch_t** outgoings, int outgoing_size)
{
	ccv_nnc_net_switch_t* b;
	if (a->output_size > 0)
	{
		// Simply dup the output_size, this case also need to dup the output
		b = _ccv_nnc_net_switch_new(a->incoming_size, a->outgoing_size, a->command, a->inputs, a->input_size, a->outputs, a->output_size);
	} else {
		b = _ccv_nnc_net_switch_new(a->incoming_size, outgoing_size, a->command, a->inputs, a->input_size, a->outputs, a->output_size);
		// Copy over the outgoings.
		if (outgoing_size > 0)
			memcpy(b->outgoings, outgoings, sizeof(ccv_nnc_net_switch_t*) * outgoing_size);
	}
	return b;
}

ccv_nnc_net_graph_t* ccv_nnc_net_graph_new(const ccv_nnc_net_command_t command, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	ccv_nnc_net_switch_t* net_switch = _ccv_nnc_net_switch_new(0, 0, command, inputs, input_size, outputs, output_size);
	ccv_nnc_net_graph_t* net_graph = (ccv_nnc_net_graph_t*)ccmalloc(sizeof(ccv_nnc_net_graph_t) + sizeof(ccv_nnc_net_switch_t*) * 2);
	net_graph->incomings = (ccv_nnc_net_switch_t**)(net_graph + 1);
	net_graph->incomings[0] = net_switch;
	net_graph->incoming_size = 1;
	net_graph->outgoings = net_graph->incomings + net_graph->incoming_size;
	net_graph->outgoings[0] = net_switch;
	net_graph->outgoing_size = 1;
	return net_graph;
}

ccv_nnc_net_graph_t* ccv_nnc_net_graph_concat(ccv_nnc_net_graph_t* const* inputs, const int input_size, const ccv_nnc_net_graph_t* output)
{
	int i;
	int outgoing_size = 0;
	for (i = 0; i < input_size; i++)
		outgoing_size += inputs[i]->outgoing_size;
	ccv_nnc_net_graph_t* net_graph = (ccv_nnc_net_graph_t*)ccmalloc(sizeof(ccv_nnc_net_graph_t) + outgoing_size * sizeof(ccv_nnc_net_switch_t*));
	net_graph->outgoings = (ccv_nnc_net_switch_t**)(net_graph + 1);
	net_graph->outgoing_size = outgoing_size;
	int c = 0;
	for (i = 0; i < input_size; i++)
	{
		assert(inputs[i]->outgoing_size > 0);
		c += inputs[i]->outgoing_size;
	}
	// c is the total outgoing nodes from all inputs.
	for (i = 0; i < output->outgoing_size; i++)
	{
	}
	return net_graph;
}

void ccv_nnc_net_graph_run(const ccv_nnc_net_graph_t* graph)
{
}

void ccv_nnc_net_graph_free(ccv_nnc_net_graph_t* graph)
{
}
