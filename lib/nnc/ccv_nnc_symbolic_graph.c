#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
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
};

const ccv_nnc_tensor_param_t ccv_nnc_tensor_auto = {0};

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

void ccv_nnc_tensor_arena_free(ccv_nnc_tensor_arena_t* tensor_arena)
{
}
