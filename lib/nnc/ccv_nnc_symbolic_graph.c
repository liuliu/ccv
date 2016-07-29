#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

typedef struct {
	ccv_nnc_tensor_param_t info;
} ccv_nnc_tensor_symbol_info_t;

typedef struct {
	int input_size;
	int output_size;
	int* inputs;
	int* outputs;
	ccv_array_t* outgoings; // outgoing nodes
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
} ccv_nnc_graph_exec_symbol_info_t;

struct ccv_nnc_symbolic_graph_s {
	ccv_array_t* tensor_symbol_info;
	ccv_array_t* exec_symbol_info;
};

struct ccv_nnc_tensor_arena_s {
	ccv_array_t* tensor_info;
};

const ccv_nnc_tensor_param_t ccv_nnc_tensor_auto = {0};

int ccv_nnc_is_tensor_auto(const ccv_nnc_tensor_param_t params)
{
	return (memcmp(&params, &ccv_nnc_tensor_auto, sizeof(ccv_nnc_tensor_param_t)) == 0);
}

ccv_nnc_symbolic_graph_t* ccv_nnc_symbolic_graph_new(void)
{
	ccv_nnc_symbolic_graph_t* graph = ccmalloc(sizeof(ccv_nnc_symbolic_graph_t));
	graph->tensor_symbol_info = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_info_t), 5, 0);
	graph->exec_symbol_info = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_info_t), 5, 0);
	return graph;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_param_t info)
{
	ccv_nnc_tensor_symbol_t symbol = {
		.info = info,
		.d = graph->tensor_symbol_info->rnum,
		.graph = graph
	};
	ccv_nnc_tensor_symbol_info_t symbol_info = {
		.info = info
	};
	ccv_array_push(graph->tensor_symbol_info, &symbol_info);
	return symbol;
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_cmd_t cmd, ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* outputs, const int output_size)
{
	ccv_nnc_graph_exec_symbol_t symbol = {
		.d = graph->exec_symbol_info->rnum,
		.graph = graph
	};
	ccv_nnc_graph_exec_symbol_info_t symbol_info = {
		.input_size = input_size,
		.output_size = output_size,
		.outgoings = 0,
		.cmd = cmd,
		.hint = ccv_nnc_no_hint
	};
	symbol_info.inputs = ccmalloc(sizeof(int) * (input_size + output_size));
	int i;
	for (i = 0; i < input_size; i++)
		symbol_info.inputs[i] = inputs[i].d;
	symbol_info.outputs = symbol_info.inputs + input_size;
	for (i = 0; i < output_size; i++)
		symbol_info.outputs[i] = outputs[i].d;
	ccv_array_push(graph->exec_symbol_info, &symbol_info);
	return symbol;
}

int ccv_nnc_graph_exec_symbol_set_hint(const ccv_nnc_symbolic_graph_t* graph, ccv_nnc_graph_exec_symbol_t exec, ccv_nnc_hint_t hint)
{
	assert(graph == exec.graph);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	symbol_info->hint = hint;
	return 0;
}

int ccv_nnc_tensor_symbol_set(const ccv_nnc_symbolic_graph_t* graph, ccv_nnc_tensor_symbol_t tensor, const ccv_nnc_tensor_param_t info)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	symbol_info->info = info;
	return 0;
}

int ccv_nnc_graph_exec_symbol_concat(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t source, const ccv_nnc_graph_exec_symbol_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_symbol_info->rnum);
	assert(destination.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* src_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, source.d);
	if (src_symbol_info->outgoings == 0)
		src_symbol_info->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	ccv_array_push(src_symbol_info->outgoings, &destination.d);
	return 0;
}

void ccv_nnc_symbolic_graph_compile(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, ccv_nnc_graph_t** graph_ref, ccv_nnc_tensor_arena_t** tensor_arena_ref)
{
	// First, fill all the "auto" holes.
	// This is the symbol table that with "auto" info filled up.
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * graph->tensor_symbol_info->rnum);
	memcpy(tensor_symbol_info, graph->tensor_symbol_info->data, sizeof(ccv_nnc_tensor_symbol_info_t) * graph->tensor_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * graph->exec_symbol_info->rnum);
	memcpy(exec_symbol_info, graph->exec_symbol_info->data, sizeof(ccv_nnc_graph_exec_symbol_info_t) * graph->exec_symbol_info->rnum);

	int i;
	// Materialize auto hints.
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		// If there is no hint and we have input and output tensor specified.
		if (ccv_nnc_is_no_hint(exec_symbol_info[i].hint) &&
				exec_symbol_info[i].input_size > 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].inputs[0]].info) &&
				exec_symbol_info[i].output_size > 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].outputs[0]].info))
			exec_symbol_info[i].hint = ccv_nnc_hint_auto(exec_symbol_info[i].cmd.info, tensor_symbol_info[exec_symbol_info[i].inputs[0]].info, tensor_symbol_info[exec_symbol_info[i].outputs[0]].info);

	// Materialize auto tensors. This need to go with the topological order.
#define visitor(node, ...) \
	do { \
		if (node->input_size > 0 && node->output_size > 0) \
			tensor_symbol_info[node->outputs[0]].info = ccv_nnc_hint_tensor_auto(node->cmd, tensor_symbol_info[node->inputs[0]].info, node->hint); \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor

	// Now, collect information about the tensor liveness.
	typedef struct {
		int s;
		int t;
	} ccv_tensor_liveness_t;
	ccv_tensor_liveness_t* tensor_liveness = (ccv_tensor_liveness_t*)ccmalloc(sizeof(ccv_tensor_liveness_t) * graph->tensor_symbol_info->rnum);
	for (i = 0; i < graph->tensor_symbol_info->rnum; i++)
		tensor_liveness[i].s = -1;
#define visitor(node, _, level) \
	do { \
		for (i = 0; i < node->input_size; i++) \
		{ \
			if (tensor_liveness[node->inputs[i]].s < 0) \
				tensor_liveness[node->inputs[i]].s = level; \
			tensor_liveness[node->inputs[i]].t = level; \
		} \
		for (i = 0; i < node->output_size; i++) \
		{ \
			if (tensor_liveness[node->outputs[i]].s < 0) \
				tensor_liveness[node->outputs[i]].s = level; \
			tensor_liveness[node->outputs[i]].t = level; \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	ccfree(tensor_liveness);
	ccfree(tensor_symbol_info);
	ccfree(exec_symbol_info);
}

void ccv_nnc_symbolic_graph_free(ccv_nnc_symbolic_graph_t* graph)
{
	int i;
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
	{
		ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		ccv_array_t* outgoings = symbol_info->outgoings;
		if (outgoings)
			ccv_array_free(outgoings);
		// We allocate inputs & outputs in continuous fashion, therefore, only need to free the input array.
		ccfree(symbol_info->inputs);
	}
	ccv_array_free(graph->tensor_symbol_info);
	ccv_array_free(graph->exec_symbol_info);
	ccfree(graph);
}

ccv_nnc_tensor_t* ccv_nnc_tensor_from_symbol(const ccv_nnc_tensor_arena_t* tensor_arena, const ccv_nnc_tensor_symbol_t symbol)
{
	return 0;
}

void ccv_nnc_tensor_arena_free(ccv_nnc_tensor_arena_t* tensor_arena)
{
	ccfree(tensor_arena);
}
