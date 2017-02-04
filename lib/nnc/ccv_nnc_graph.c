#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"

ccv_nnc_graph_t* ccv_nnc_graph_new(void)
{
	ccv_nnc_graph_t* graph = (ccv_nnc_graph_t*)cccalloc(1, sizeof(ccv_nnc_graph_t));
	graph->exec_info = ccv_array_new(sizeof(ccv_nnc_graph_exec_info_t), 5, 0);
	return graph;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_new(ccv_nnc_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int d = graph->exec_info->rnum;
	ccv_nnc_graph_exec_info_t info = {
		.cmd = cmd,
		.hint = hint,
		.input_size = input_size,
		.output_size = output_size,
	};
	assert(inputs || input_size == 0);
	assert(outputs || output_size == 0);
	if (input_size > 0 || output_size > 0)
	{
		int i;
		for (i = 0; i < input_size && !info.backed; i++)
			if (CCV_IS_TENSOR_MULTIVIEW(inputs[i]))
				info.backed = 1;
		for (i = 0; i < output_size && !info.backed; i++)
			if (CCV_IS_TENSOR_MULTIVIEW(outputs[i]))
				info.backed = 1;
		if (info.backed)
		{
			info.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size) * 2);
			info.outputs = info.inputs + input_size;
			info.backed_inputs = info.outputs + output_size;
			info.backed_outputs = info.backed_inputs + input_size;
		} else {
			info.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size));
			info.outputs = info.inputs + input_size;
		}
		if (inputs)
			memcpy(info.inputs, inputs, sizeof(ccv_nnc_tensor_t*) * input_size);
		if (outputs)
			memcpy(info.outputs, outputs, sizeof(ccv_nnc_tensor_t*) * output_size);
	}
	ccv_array_push(graph->exec_info, &info);
	ccv_nnc_graph_exec_t exec = {
		.d = d,
		.graph = graph,
	};
	return exec;
}

int ccv_nnc_graph_exec_concat(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t source, const ccv_nnc_graph_exec_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_info->rnum);
	assert(destination.d < graph->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* src_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, source.d);
	if (src_info->outgoings == 0)
		src_info->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	else {
		int i;
		// Check if this is already connected, if so, skip.
		for (i = 0; i < src_info->outgoings->rnum; i++)
			if (*(int*)ccv_array_get(src_info->outgoings, i) == destination.d)
				return -1;
	}
	ccv_array_push(src_info->outgoings, &destination.d);
	return 0;
}

int ccv_nnc_graph_exec_disjoin(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t source, const ccv_nnc_graph_exec_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_info->rnum);
	assert(destination.d < graph->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* src_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, source.d);
	if (!src_info->outgoings)
		return -1;
	int i, j = -1;
	// Check if this is already connected, if so, skip.
	for (i = 0; i < src_info->outgoings->rnum; i++)
		if (*(int*)ccv_array_get(src_info->outgoings, i) == destination.d)
		{
			j = i;
			break;
		}
	if (j < 0)
		return -1;
	if (j < src_info->outgoings->rnum - 1)
		*(int*)ccv_array_get(src_info->outgoings, j) = *(int*)ccv_array_get(src_info->outgoings, src_info->outgoings->rnum - 1);
	--src_info->outgoings->rnum;
	return 0;
}

static void _ccv_nnc_graph_dot_exec(const int index, const ccv_nnc_graph_exec_info_t* const exec_info, const int flags, FILE* out)
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

static void _ccv_nnc_graph_dot_tensor(const int index, const ccv_nnc_tensor_t* const tensor, const int zone, const int flags, FILE* out)
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
		fprintf(out, "|{%#010x|%#010x}", (uint32_t)aptr, (uint32_t)(aptr + ainc_size - 1));
		int i;
		fprintf(out, "|%d", tensor->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && tensor->info.dim[i]; i++)
			fprintf(out, "x%d", tensor->info.dim[i]);
		fputc('}', out);
	}
}

typedef struct {
	int index;
	int name;
	int zone;
	ccv_nnc_tensor_t* tensor;
	uintptr_t start_ptr;
	uintptr_t end_ptr;
} ccv_nnc_tensor_dot_t;

// First sort by start_ptr, then sort by tensor ptr (so that we will have the same tensor sorted to one cluster).
#define less_than(i1, i2, aux) ((i1).start_ptr < (i2).start_ptr || ((i1).start_ptr == (i2).start_ptr && (i1).tensor < (i2).tensor))
static CCV_IMPLEMENT_QSORT(_ccv_nnc_tensor_dot_sort_by_ptr, ccv_nnc_tensor_dot_t, less_than)
#undef less_than

void ccv_nnc_graph_dot(const ccv_nnc_graph_t* const graph, const int flags, FILE* out)
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
	int tensor_count = 0;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		tensor_count += exec_info->input_size + exec_info->output_size;
	}
	ccv_nnc_tensor_dot_t* tensor_dots = tensor_count > 0 ? (ccv_nnc_tensor_dot_t*)ccmalloc(sizeof(ccv_nnc_tensor_dot_t) * tensor_count) : 0;
	int k = 0;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		for (j = 0; j < exec_info->input_size; j++)
		{
			ccv_nnc_tensor_t* tensor = exec_info->inputs[j];
			if (tensor)
			{
				tensor_dots[k].name = k;
				tensor_dots[k].tensor = tensor;
				tensor_dots[k].start_ptr = (uintptr_t)tensor->data.u8;
				const int* inc = CCV_IS_TENSOR_VIEW(tensor) ? ((ccv_nnc_tensor_view_t*)tensor)->inc : tensor->info.dim;
				const size_t inc_size = (ccv_nnc_dimension_count(inc) - inc[0] + tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(tensor->type);
				tensor_dots[k].end_ptr = tensor_dots[k].start_ptr + inc_size - 1;
				++k;
			}
		}
		for (j = 0; j < exec_info->output_size; j++)
		{
			ccv_nnc_tensor_t* tensor = exec_info->outputs[j];
			if (tensor)
			{
				tensor_dots[k].name = k;
				tensor_dots[k].tensor = tensor;
				tensor_dots[k].start_ptr = (uintptr_t)tensor->data.u8;
				const int* inc = CCV_IS_TENSOR_VIEW(tensor) ? ((ccv_nnc_tensor_view_t*)tensor)->inc : tensor->info.dim;
				const size_t inc_size = (ccv_nnc_dimension_count(inc) - inc[0] + tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(tensor->type);
				tensor_dots[k].end_ptr = tensor_dots[k].start_ptr + inc_size - 1;
				++k;
			}
		}
	}
	tensor_count = k; // We may over count, now shrink.
	// To group overlap memory into one zone, we sort it by start ptr first (secondary by the tensor pointer).
	_ccv_nnc_tensor_dot_sort_by_ptr(tensor_dots, tensor_count, 0);
	int index = 0, zone = 0;
	ccv_nnc_tensor_t* tensor = tensor_dots[0].tensor;
	uintptr_t end_ptr = tensor_dots[0].end_ptr;
	// Then, it is trivial, we go by end ptr. If the next start ptr is still within the end ptr (start ptr <= end ptr),
	// they are the same zone.
	for (i = 0; i < tensor_count; i++)
	{
		if (tensor_dots[i].tensor != tensor)
		{
			tensor = tensor_dots[i].tensor;
			++index;
		}
		if (tensor_dots[i].start_ptr > end_ptr)
		{
			end_ptr = ccv_max(end_ptr, tensor_dots[i].end_ptr);
			++zone;
		}
		tensor_dots[i].index = index;
		tensor_dots[i].zone = zone;
	}
	// We already have index and zone assigned, but the problem is that these are not very human interpretable (because
	// it follows the pointer from low to high, not the tensor creation order). The following code renamed both the index
	// and the zone so that it is much more understandable.
	const int index_count = index + 1;
	const int zone_count = zone + 1;
	int* remap = (int*)ccmalloc(sizeof(int) * (tensor_count + index_count + zone_count));
	int* rename_index = remap + tensor_count;
	int* rename_zone = rename_index + index_count;
	for (i = 0; i < tensor_count; i++)
		remap[tensor_dots[i].name] = i;
	for (i = 0; i < index_count; i++)
		rename_index[i] = -1;
	for (i = 0; i < zone_count; i++)
		rename_zone[i] = -1;
	index = 0;
	zone = 0;
	for (i = 0; i < tensor_count; i++)
	{
		ccv_nnc_tensor_dot_t* tensor_dot = tensor_dots + remap[i];
		if (rename_index[tensor_dot->index] == -1)
			rename_index[tensor_dot->index] = index++;
		if (rename_zone[tensor_dot->zone] == -1)
			rename_zone[tensor_dot->zone] = zone++;
	}
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
				if (exec_info->inputs[j])
				{
					fputc('|', out);
					ccv_nnc_tensor_dot_t* tensor_dot = tensor_dots + remap[k];
					_ccv_nnc_graph_dot_tensor(rename_index[tensor_dot->index], exec_info->inputs[j], rename_zone[tensor_dot->zone], flags, out);
					++k;
				} else
					fputs("|-", out);
			fputc('}', out);
		}
		if (exec_info->output_size > 0)
		{
			fputs("|{Output", out);
			for (j = 0; j < exec_info->output_size; j++)
				if (exec_info->inputs[j])
				{
					fputc('|', out);
					ccv_nnc_tensor_dot_t* tensor_dot = tensor_dots + remap[k];
					_ccv_nnc_graph_dot_tensor(rename_index[tensor_dot->index], exec_info->outputs[j], rename_zone[tensor_dot->zone], flags, out);
					++k;
				} else
					fputs("|-", out);
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
	ccfree(tensor_dots);
	ccfree(remap);
}

void ccv_nnc_graph_autotune(ccv_nnc_graph_t* const graph, const size_t max_workspace_size, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	// exec current node, for synchronous CPU execution, no stream unit.
#define visitor(node, idx, ...) \
	do { \
		node->cmd = ccv_nnc_cmd_autotune(node->cmd, max_workspace_size, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0); \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
}

void ccv_nnc_graph_run(const ccv_nnc_graph_t* const graph, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	// exec current node, for synchronous CPU execution, no stream unit.
#define visitor(node, ...) \
	do { \
		ccv_nnc_cmd_exec(node->cmd, node->hint, flags, node->inputs, node->input_size, node->outputs, node->output_size, 0); \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
}

void ccv_nnc_graph_free(ccv_nnc_graph_t* const graph)
{
	int i;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t *info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		ccv_array_t* outgoings = info->outgoings;
		if (outgoings)
			ccv_array_free(outgoings);
		// We allocate inputs & outputs in continuous fashion, therefore, only need to free the input array.
		if (info->inputs)
			ccfree(info->inputs);
	}
	if (graph->conditionals)
		ccfree(graph->conditionals);
	if (graph->sources)
		ccv_array_free(graph->sources);
	if (graph->destinations)
		ccv_array_free(graph->destinations);
	if (graph->sub_graphs)
	{
		for (i = 0; i < graph->sub_graphs->rnum; i++)
			ccv_nnc_graph_free(*(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, i));
		ccv_array_free(graph->sub_graphs);
	}
	ccv_array_free(graph->exec_info);
	ccfree(graph);
}
