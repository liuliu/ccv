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

static void _ccv_nnc_graph_dot_exec(const int index, const ccv_nnc_graph_exec_info_t* exec_info, const int flags, FILE* out)
{
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	fprintf(out, "node%d", index);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		fputs("|Command: ", out);
		fputs(ccv_nnc_cmd_name(exec_info->cmd.cmd), out);
		fputc('}', out);
	}
}

static void _ccv_nnc_graph_dot_tensor(const int index, const ccv_nnc_tensor_t* tensor, const int zone, const int flags, FILE* out)
{
	// if it has an alias pointer, or, it is a long form.
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	int is_tensor_view = CCV_IS_TENSOR_VIEW(tensor);
	if (is_tensor_view)
		fprintf(out, "tensorview%d", index);
	else
		fprintf(out, "tensor%d", index);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		fprintf(out, "|zone%d", zone);
		uintptr_t aptr = (uintptr_t)tensor->data.u8;
		const int* ainc = is_tensor_view ? ((ccv_nnc_tensor_view_t*)(tensor))->inc : tensor->info.dim;
		// For the last one, we don't extend to full ainc.
		size_t ainc_size = (ccv_nnc_dimension_count(ainc) - ainc[0] + tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(tensor->type);
		// Print out the range as well.
		fprintf(out, "|{%#010x|%#010x}", (uint32_t)aptr, (uint32_t)(aptr + ainc_size));
		int i;
		fprintf(out, "|%d", tensor->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && tensor->info.dim[i]; i++)
			fprintf(out, "x%d", tensor->info.dim[i]);
		fputc('}', out);
	}
}

typedef struct {
	int index;
	ccv_nnc_tensor_t* tensor;
} ccv_nnc_tensor_dot_t;

static int _ccv_nnc_tensor_zoning(const void* a, const void* b, void* data)
{
	ccv_nnc_tensor_dot_t* dot_a = (ccv_nnc_tensor_dot_t*)a;
	ccv_nnc_tensor_dot_t* dot_b = (ccv_nnc_tensor_dot_t*)b;
	if (dot_a->tensor == dot_b->tensor)
		return 1;
	uintptr_t aptr = (uintptr_t)dot_a->tensor->data.u8;
	const int* ainc = CCV_IS_TENSOR_VIEW(dot_a->tensor) ? ((ccv_nnc_tensor_view_t*)(dot_a->tensor))->inc : dot_a->tensor->info.dim;
	size_t ainc_size = (ccv_nnc_dimension_count(ainc) - ainc[0] + dot_a->tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(dot_a->tensor->type);
	uintptr_t bptr = (uintptr_t)dot_b->tensor->data.u8;
	const int* binc = CCV_IS_TENSOR_VIEW(dot_b->tensor) ? ((ccv_nnc_tensor_view_t*)(dot_b->tensor))->inc : dot_b->tensor->info.dim;
	size_t binc_size = (ccv_nnc_dimension_count(binc) - binc[0] + dot_b->tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(dot_a->tensor->type);
	return ccv_max(aptr, bptr) < ccv_min(aptr + ainc_size, bptr + binc_size);
}

void ccv_nnc_graph_dot(const ccv_nnc_graph_t* graph, const int flags, FILE* out)
{
	fputs("digraph G {\n", out);
	int i, j;
	// Recover tensor relationships for all tensors referenced in the graph.
	// Most notably, we have to give these indexes, and find if they point to
	// the same memory region, and whether they overlap. These information
	// are lost since we converted from symbolic form to the execution form.
	// and here we do our best to recover because that is easier to understand
	// if we want to present the graph visually (also, we don't want to put this
	// information into the tensor or execution graph to avoid overhead, thus,
	// recovering is the best we can do).
	ccv_array_t* tensor_dots = ccv_array_new(sizeof(ccv_nnc_tensor_dot_t), graph->exec_info->rnum * 2, 0);
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		for (j = 0; j < exec_info->input_size; j++)
		{
			ccv_nnc_tensor_dot_t tensor_dot = {
				.index = -1,
				.tensor = exec_info->inputs[j]
			};
			ccv_array_push(tensor_dots, &tensor_dot);
		}
		for (j = 0; j < exec_info->output_size; j++)
		{
			ccv_nnc_tensor_dot_t tensor_dot = {
				.index = -1,
				.tensor = exec_info->outputs[j]
			};
			ccv_array_push(tensor_dots, &tensor_dot);
		}
	}
	int k = 0;
	// Using a simple double for loop to find duplicate tensors, and assign index to these.
	for (i = 0; i < tensor_dots->rnum; i++)
	{
		ccv_nnc_tensor_dot_t* tensor_dot_i = (ccv_nnc_tensor_dot_t*)ccv_array_get(tensor_dots, i);
		if (tensor_dot_i->index == -1)
			tensor_dot_i->index = k++; // Assign out the new index.
		for (j = i + 1; j < tensor_dots->rnum; j++)
		{
			ccv_nnc_tensor_dot_t* tensor_dot_j = (ccv_nnc_tensor_dot_t*)ccv_array_get(tensor_dots, j);
			// They are the same, because tensor_dot_j is later than tensor_dot_i, it will take tensor_dot_i's index.
			if (tensor_dot_j->tensor == tensor_dot_i->tensor)
				tensor_dot_j->index = tensor_dot_i->index;
		}
	}
	ccv_array_t* tensor_zones = 0;
	// We may not need this if we don't need LONG_GRAPH
	if (flags & CCV_NNC_LONG_DOT_GRAPH)
		ccv_array_group(tensor_dots, &tensor_zones, _ccv_nnc_tensor_zoning, 0);
	k = 0;
	// Output styles.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		fprintf(out, "node%d [shape=Mrecord,label=\"", i);
		_ccv_nnc_graph_dot_exec(i, exec_info, flags, out);
		if (exec_info->input_size > 0)
		{
			fputs("|{Input", out);
			for (j = 0; j < exec_info->input_size; j++)
			{
				fputc('|', out);
				ccv_nnc_tensor_dot_t* tensor_dot = (ccv_nnc_tensor_dot_t*)ccv_array_get(tensor_dots, k);
				int zone = tensor_zones ? *(int*)ccv_array_get(tensor_zones, k) : 0;
				_ccv_nnc_graph_dot_tensor(tensor_dot->index, exec_info->inputs[j], zone, flags, out);
				++k;
			}
			fputc('}', out);
		}
		if (exec_info->output_size > 0)
		{
			fputs("|{Output", out);
			for (j = 0; j < exec_info->output_size; j++)
			{
				fputc('|', out);
				ccv_nnc_tensor_dot_t* tensor_dot = (ccv_nnc_tensor_dot_t*)ccv_array_get(tensor_dots, k);
				int zone = tensor_zones ? *(int*)ccv_array_get(tensor_zones, k) : 0;
				_ccv_nnc_graph_dot_tensor(tensor_dot->index, exec_info->outputs[j], zone, flags, out);
				++k;
			}
			fputc('}', out);
		}
		fputs("\"];\n", out);
	}
	// Output connections.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (exec_info->outgoings)
			for (j = 0; j < exec_info->outgoings->rnum; j++)
				fprintf(out, "node%d -> node%d;\n", i, *(int*)ccv_array_get(exec_info->outgoings, j));
	}
	fputs("}\n", out);
	ccv_array_free(tensor_dots);
	if (tensor_zones)
		ccv_array_free(tensor_zones);
}

void ccv_nnc_graph_autotune(const ccv_nnc_graph_t* graph, const size_t max_workspace_size, const int flags, const ccv_nnc_graph_exec_t* sources, const int source_size, const ccv_nnc_graph_exec_t* destinations, const int destination_size)
{
	// exec current node, for synchronous CPU execution, no stream unit.
#define visitor(node, idx, ...) \
	do { \
		node->cmd = ccv_nnc_cmd_autotune(node->cmd, max_workspace_size, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0); \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
}

void ccv_nnc_graph_run(const ccv_nnc_graph_t* graph, const int flags, const ccv_nnc_graph_exec_t* sources, const int source_size, const ccv_nnc_graph_exec_t* destinations, const int destination_size)
{
	// exec current node, for synchronous CPU execution, no stream unit.
#define visitor(node, ...) \
	do { \
		ccv_nnc_cmd_exec(node->cmd, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0); \
	} while (0)
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
