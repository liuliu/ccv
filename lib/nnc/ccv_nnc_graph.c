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

void ccv_nnc_graph_set_sources(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t* const sources, const int source_size)
{
	if (!graph->sources)
		graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), source_size, 0);
	else
		ccv_array_clear(graph->sources);
	int i;
	for (i = 0; i < source_size; i++)
		ccv_array_push(graph->sources, sources + i);
}

ccv_nnc_graph_exec_t* ccv_nnc_graph_sources(const ccv_nnc_graph_t* const graph)
{
	return graph->sources ? (ccv_nnc_graph_exec_t*)ccv_array_get(graph->sources, 0) : 0;
}

int ccv_nnc_graph_source_size(const ccv_nnc_graph_t* const graph)
{
	return graph->sources ? graph->sources->rnum : 0;
}

void ccv_nnc_graph_set_destinations(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	if (!graph->destinations)
		graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), destination_size, 0);
	else
		ccv_array_clear(graph->sources);
	int i;
	for (i = 0; i < destination_size; i++)
		ccv_array_push(graph->destinations, destinations + i);
}

ccv_nnc_graph_exec_t* ccv_nnc_graph_destinations(const ccv_nnc_graph_t* const graph)
{
	return graph->destinations ? (ccv_nnc_graph_exec_t*)ccv_array_get(graph->destinations, 0) : 0;
}

int ccv_nnc_graph_destination_size(const ccv_nnc_graph_t* const graph)
{
	return graph->destinations ? graph->destinations->rnum : 0;
}

static void _ccv_recursively_mark_as_anchored_for_multiview_wrap(ccv_nnc_tensor_multiview_t* const mv, intptr_t* const wrap_anchors, const int wrap_size)
{
	int i;
	for (i = 0; i < wrap_size; i++)
		if (mv->anchor == (wrap_anchors[i] & ~(intptr_t)1))
		{
			wrap_anchors[i] |= 1;
			break;
		}
	const int count = mv->kind + mv->repeat;
	for (i = 0; i < count; i++)
		if (!mv->tv && CCV_IS_TENSOR_MULTIVIEW(mv->data[i].ptr))
			_ccv_recursively_mark_as_anchored_for_multiview_wrap((ccv_nnc_tensor_multiview_t*)(mv->data[i].ptr), wrap_anchors, wrap_size);
}

void ccv_nnc_graph_exec_set_hint(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, const ccv_nnc_hint_t hint)
{
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	exec_info->hint = hint;
}

static void _ccv_nnc_graph_exec_info_set_io(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const info, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	assert(input_size > 0 || output_size > 0);
	int i;
	int has_wraps = 0;
	for (i = 0; i < input_size && !has_wraps; i++)
		if (inputs[i] && CCV_IS_TENSOR_MULTIVIEW(inputs[i]))
			has_wraps = 1;
	for (i = 0; i < output_size && !has_wraps; i++)
		if (outputs[i] && CCV_IS_TENSOR_MULTIVIEW(outputs[i]))
			has_wraps = 1;
	// It need to be handled specifically if it contains a wrap.
	info->io_wraps = 0;
	if (has_wraps)
	{
		// The logic as following, I assume I need to unwrap at each graph level (go through graph until reaches p = 0).
		// and then go over each input / output to see at which level we actually do the unwrapping.
		assert(graph->p);
		// Graph has to have attached to it parent, otherwise we cannot figure out the anchor point the wrap is pointing to.
		int wrap_size = 1;
		ccv_nnc_graph_t* p = graph->p;
		for (; p; p = p->p)
			++wrap_size;
		// I will take advantage of the fact that intptr are 4-bytes aligned.
		assert(wrap_size < 512); // We can only go 512 layer deep.
		intptr_t* wrap_anchors = (intptr_t*)alloca(sizeof(intptr_t) * wrap_size);
		for (p = graph, i = 0; p; p = p->p, i++)
			wrap_anchors[i] = (intptr_t)p;
		for (i = 0; i < input_size; i++)
			if (inputs[i] && CCV_IS_TENSOR_MULTIVIEW(inputs[i]))
				_ccv_recursively_mark_as_anchored_for_multiview_wrap((ccv_nnc_tensor_multiview_t*)inputs[i], wrap_anchors, wrap_size);
		for (i = 0; i < output_size; i++)
			if (outputs[i] && CCV_IS_TENSOR_MULTIVIEW(outputs[i]))
				_ccv_recursively_mark_as_anchored_for_multiview_wrap((ccv_nnc_tensor_multiview_t*)outputs[i], wrap_anchors, wrap_size);
		// Now all wrap_anchors are marked (with the least significant bit to be 1), compute the depth we required.
		for (i = 0; i < wrap_size; i++)
			if (wrap_anchors[i] & 1)
				++info->io_wraps;
	}
	// info.io_wraps gives out how deep the "stack" need to be. We "unwrap" tensors when we go deeper into each sub-graph (if it is wrapped into a multiview tensor structure), thus, a "stack" is a perfect analogy to express this kind of memory structure we use.
	if (info->inputs)
		info->inputs = (ccv_nnc_tensor_t**)ccrealloc(info->inputs, sizeof(ccv_nnc_tensor_t*) * (input_size + output_size) * (info->io_wraps + 1));
	else
		info->inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size) * (info->io_wraps + 1));
	info->outputs = info->inputs + input_size;
	if (inputs)
		memcpy(info->inputs, inputs, sizeof(ccv_nnc_tensor_t*) * input_size);
	if (outputs)
		memcpy(info->outputs, outputs, sizeof(ccv_nnc_tensor_t*) * output_size);
	info->input_size = input_size;
	info->output_size = output_size;
}

void ccv_nnc_graph_exec_set_io(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	int i;
	ccv_nnc_graph_exec_info_t* const info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	// Remove only it doesn't have both io_wraps and cast_wraps.
	if (info->io_wraps && !info->cast_wraps)
	{
		ccv_nnc_graph_t* p = graph;
		do {
			// Remove from the array.
			if (p->wraps)
				for (i = 0; i < p->wraps->rnum; i++)
				{
					ccv_nnc_graph_exec_t* const wrap_exec = (ccv_nnc_graph_exec_t*)ccv_array_get(p->wraps, i);
					if (wrap_exec->d == exec.d && wrap_exec->graph == graph)
					{
						--p->wraps->rnum;
						if (i < p->wraps->rnum)
							memcpy(wrap_exec, wrap_exec + 1, sizeof(ccv_nnc_graph_exec_t) * (p->wraps->rnum - i));
						break;
					}
				}
			p = p->p;
		} while (p);
	}
	if (input_size == 0 && output_size == 0)
	{
		if (info->input_size > 0 || info->output_size > 0)
			ccfree(info->inputs);
		info->inputs = 0;
		info->outputs = 0;
		info->input_size = 0;
		info->output_size = 0;
		info->io_wraps = 0;
		info->io_wrap_ptr = 0;
		return;
	}
	_ccv_nnc_graph_exec_info_set_io(graph, info, inputs, input_size, outputs, output_size);
	// If we don't remove it because cast_wraps is nil, then no need to add it back.
	if (info->io_wraps && !info->cast_wraps)
	{
		ccv_nnc_graph_t* p = graph;
		do {
			if (!p->wraps)
				p->wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), 0, 0);
			ccv_array_push(p->wraps, &exec);
			p = p->p;
		} while (p);
	}
}
static void _ccv_nnc_graph_exec_info_set_cast(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_info_t* const info, ccv_nnc_tensor_t* const* const casts, const int cast_size)
{
	assert(cast_size > 0);
	int i;
	int has_wraps = 0;
	for (i = 0; i < cast_size && !has_wraps; i++)
		if (casts[i] && CCV_IS_TENSOR_MULTIVIEW(casts[i]))
			has_wraps = 1;
	// It need to be handled specifically if it contains a wrap.
	info->cast_wraps = 0;
	if (has_wraps)
	{
		// The logic as following, I assume I need to unwrap at each graph level (go through graph until reaches p = 0).
		// and then go over each input / output to see at which level we actually do the unwrapping.
		assert(graph->p);
		// Graph has to have attached to it parent, otherwise we cannot figure out the anchor point the wrap is pointing to.
		int wrap_size = 1;
		ccv_nnc_graph_t* p = graph->p;
		for (; p; p = p->p)
			++wrap_size;
		// I will take advantage of the fact that intptr are 4-bytes aligned.
		assert(wrap_size < 512); // We can only go 512 layer deep.
		intptr_t* wrap_anchors = (intptr_t*)alloca(sizeof(intptr_t) * wrap_size);
		for (p = graph, i = 0; p; p = p->p, i++)
			wrap_anchors[i] = (intptr_t)p;
		for (i = 0; i < cast_size; i++)
			if (casts[i] && CCV_IS_TENSOR_MULTIVIEW(casts[i]))
				_ccv_recursively_mark_as_anchored_for_multiview_wrap((ccv_nnc_tensor_multiview_t*)casts[i], wrap_anchors, wrap_size);
		// Now all wrap_anchors are marked (with the least significant bit to be 1), compute the depth we required.
		for (i = 0; i < wrap_size; i++)
			if (wrap_anchors[i] & 1)
				++info->cast_wraps;
	}
	// info.cast_wraps gives out how deep the "stack" need to be. We "unwrap" tensors when we go deeper into each sub-graph (if it is wrapped into a multiview tensor structure), thus, a "stack" is a perfect analogy to express this kind of memory structure we use.
	if (info->casts)
		info->casts = (ccv_nnc_tensor_t**)ccrealloc(info->casts, sizeof(ccv_nnc_tensor_t*) * cast_size * (info->cast_wraps + 1));
	else
		info->casts = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * cast_size * (info->cast_wraps + 1));
	if (casts)
		memcpy(info->casts, casts, sizeof(ccv_nnc_tensor_t*) * cast_size);
	info->cast_size = cast_size;
}

void ccv_nnc_graph_exec_set_cast(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_tensor_t* const* const casts, const int cast_size)
{
	assert(exec.d < graph->exec_info->rnum);
	assert(exec.graph == graph);
	int i;
	ccv_nnc_graph_exec_info_t* const info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	// Remove only it doesn't have both io_wraps and cast_wraps.
	if (info->cast_wraps && !info->io_wraps)
	{
		ccv_nnc_graph_t* p = graph;
		do {
			// Remove from the array.
			if (p->wraps)
				for (i = 0; i < p->wraps->rnum; i++)
				{
					ccv_nnc_graph_exec_t* const wrap_exec = (ccv_nnc_graph_exec_t*)ccv_array_get(p->wraps, i);
					if (wrap_exec->d == exec.d && wrap_exec->graph == graph)
					{
						--p->wraps->rnum;
						if (i < p->wraps->rnum)
							memcpy(wrap_exec, wrap_exec + 1, sizeof(ccv_nnc_graph_exec_t) * (p->wraps->rnum - i));
						break;
					}
				}
			p = p->p;
		} while (p);
	}
	if (cast_size == 0)
	{
		if (info->cast_size > 0)
			ccfree(info->casts);
		info->casts = 0;
		info->cast_size = 0;
		info->cast_wraps = 0;
		info->cast_wrap_ptr = 0;
		return;
	}
	_ccv_nnc_graph_exec_info_set_cast(graph, info, casts, cast_size);
	// If we don't remove it because io_wraps is nil, then no need to add it back.
	if (info->cast_wraps && !info->io_wraps)
	{
		ccv_nnc_graph_t* p = graph;
		do {
			if (!p->wraps)
				p->wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), 0, 0);
			ccv_array_push(p->wraps, &exec);
			p = p->p;
		} while (p);
	}
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
		_ccv_nnc_graph_exec_info_set_io(graph, &info, inputs, input_size, outputs, output_size);
	ccv_array_push(graph->exec_info, &info);
	ccv_nnc_graph_exec_t exec = {
		.d = d,
		.graph = graph,
	};
	// Add itself to the graph's wraps array, this will help the run time when we run the graph and do unwrapping.
	if (info.io_wraps)
	{
		ccv_nnc_graph_t* p = graph;
		do {
			if (!p->wraps)
				p->wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), 0, 0);
			ccv_array_push(p->wraps, &exec);
			p = p->p;
		} while (p);
	}
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

static void _ccv_nnc_graph_dot_tensor(const int index, const ccv_nnc_tensor_t* const tensor, const int zone, const int flags, const int depth, FILE* out)
{
	// if it has an alias pointer, or, it is a long form.
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	int is_tensor_view = CCV_IS_TENSOR_VIEW(tensor);
	if (is_tensor_view)
		fprintf(out, "tensorview%d", index);
	else
		fprintf(out, "tensor%d", index);
	int i;
	for (i = 0; i < depth; i++) // Print subscription to denote depth.
		fputc('\'', out);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		fprintf(out, "|zone%d", zone);
		for (i = 0; i < depth; i++) // Print subscription to denote depth.
			fputc('\'', out);
		uintptr_t aptr = (uintptr_t)tensor->data.u8;
		const int* ainc = is_tensor_view ? ((ccv_nnc_tensor_view_t*)(tensor))->inc : tensor->info.dim;
		// For the last one, we don't extend to full ainc.
		size_t ainc_size = (ccv_nnc_dimension_count(ainc) - ainc[0] + tensor->info.dim[0]) * CCV_GET_DATA_TYPE_SIZE(tensor->type);
		// Print out the range as well.
		fprintf(out, "|{%#010x|%#010x}|%d", (uint32_t)aptr, (uint32_t)(aptr + ainc_size - 1), tensor->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && tensor->info.dim[i]; i++)
			fprintf(out, "x%d", tensor->info.dim[i]);
		fputc('}', out);
	}
}

typedef struct {
	int index;
	int name;
	int zone;
	uintptr_t tensor_ref;
	uintptr_t start_ptr;
	uintptr_t end_ptr;
} ccv_nnc_tensor_dot_t;

typedef struct {
	ccv_nnc_tensor_dot_t* dots;
	int* remap;
	int* rename_zone;
	int* rename_index;
} ccv_nnc_tensor_dot_recovery_t;

// First sort by start_ptr, then sort by tensor ptr (so that we will have the same tensor sorted to one cluster).
#define less_than(i1, i2, aux) ((i1).start_ptr < (i2).start_ptr || ((i1).start_ptr == (i2).start_ptr && (i1).tensor_ref < (i2).tensor_ref))
static CCV_IMPLEMENT_QSORT(_ccv_nnc_tensor_dot_sort_by_ptr, ccv_nnc_tensor_dot_t, less_than)
#undef less_than

static int _ccv_nnc_graph_dot_tensor_multiview_count(const ccv_nnc_tensor_multiview_t* const mv)
{
	assert(CCV_IS_TENSOR_MULTIVIEW(mv));
	const int count = mv->kind + mv->repeat;
	if (mv->tv)
		return count;
	int i, c = 0;
	for (i = 0; i < count; i++)
		c += _ccv_nnc_graph_dot_tensor_multiview_count(mv->data[i].ptr);
	return c;
}

static void _ccv_nnc_graph_dot_tensor_multiview_tensor_dots(const ccv_nnc_tensor_multiview_t* const mv, ccv_nnc_tensor_dot_t* const tensor_dots, int* tensor_index)
{
	const int count = mv->kind + mv->repeat;
	int i;
	if (mv->tv)
		for (i = 0; i < count; i++)
		{
			tensor_dots[*tensor_index].name = *tensor_index;
			if (CCV_NNC_MULTIVIEW_K01(mv))
			{
				tensor_dots[*tensor_index].tensor_ref = (uintptr_t)mv->tv;
				tensor_dots[*tensor_index].start_ptr =  (uintptr_t)mv->tv->data.u8;
			} else {
				tensor_dots[*tensor_index].start_ptr =  (uintptr_t)mv->data[i].ptr;
				// Because tv's pointer will get updated, it is not correct in this case to have one tensor_ref.
				tensor_dots[*tensor_index].tensor_ref = tensor_dots[*tensor_index].start_ptr;
			}
			const size_t dim_size = ccv_nnc_dimension_count(mv->tv->info.dim) * CCV_GET_DATA_TYPE_SIZE(mv->tv->type);
			tensor_dots[*tensor_index].end_ptr = tensor_dots[*tensor_index].start_ptr + dim_size - 1;
			++(*tensor_index);
		}
	else
		for (i = 0; i < count; i++)
			_ccv_nnc_graph_dot_tensor_multiview_tensor_dots((ccv_nnc_tensor_multiview_t*)mv->data[i].ptr, tensor_dots, tensor_index);
}

static ccv_nnc_tensor_dot_recovery_t _ccv_nnc_graph_tensor_dot_recovery(const ccv_nnc_graph_t* const graph)
{
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
		for (j = 0; j < exec_info->input_size; j++)
			if (exec_info->inputs[j])
				tensor_count += CCV_IS_TENSOR_MULTIVIEW(exec_info->inputs[j]) ? _ccv_nnc_graph_dot_tensor_multiview_count((ccv_nnc_tensor_multiview_t*)exec_info->inputs[j]) : 1;
		for (j = 0; j < exec_info->output_size; j++)
			if (exec_info->outputs[j])
				tensor_count += CCV_IS_TENSOR_MULTIVIEW(exec_info->outputs[j]) ? _ccv_nnc_graph_dot_tensor_multiview_count((ccv_nnc_tensor_multiview_t*)exec_info->outputs[j]) : 1;
	}
	ccv_nnc_tensor_dot_t* tensor_dots = tensor_count > 0 ? (ccv_nnc_tensor_dot_t*)ccmalloc(sizeof(ccv_nnc_tensor_dot_t) * tensor_count) : 0;
	int k = 0;
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		for (j = 0; j < exec_info->input_size; j++)
		{
			ccv_nnc_tensor_t* tensor = exec_info->inputs[j];
			if (!tensor)
				continue;
			if (CCV_IS_TENSOR_MULTIVIEW(tensor))
				_ccv_nnc_graph_dot_tensor_multiview_tensor_dots((ccv_nnc_tensor_multiview_t*)tensor, tensor_dots, &k);
			else {
				tensor_dots[k].name = k;
				tensor_dots[k].tensor_ref = (uintptr_t)tensor;
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
			if (!tensor)
				continue;
			if (CCV_IS_TENSOR_MULTIVIEW(tensor))
				_ccv_nnc_graph_dot_tensor_multiview_tensor_dots((ccv_nnc_tensor_multiview_t*)tensor, tensor_dots, &k);
			else {
				tensor_dots[k].name = k;
				tensor_dots[k].tensor_ref = (uintptr_t)tensor;
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
	uintptr_t tensor_ref = tensor_count > 0 ? tensor_dots[0].tensor_ref : 0;
	uintptr_t end_ptr = tensor_count > 0 ? tensor_dots[0].end_ptr : 0;
	// Then, it is trivial, we go by end ptr. If the next start ptr is still within the end ptr (start ptr <= end ptr),
	// they are the same zone.
	for (i = 0; i < tensor_count; i++)
	{
		if (tensor_dots[i].tensor_ref != tensor_ref)
		{
			tensor_ref = tensor_dots[i].tensor_ref;
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
	ccv_nnc_tensor_dot_recovery_t recovery = {
		.dots = tensor_dots,
		.remap = remap,
		.rename_index = rename_index,
		.rename_zone = rename_zone,
	};
	return recovery;
}

static void _ccv_nnc_graph_tensor_dot_recovery_free(const ccv_nnc_tensor_dot_recovery_t recovery)
{
	ccfree(recovery.dots);
	ccfree(recovery.remap);
}

static void _ccv_nnc_graph_dot_tensor_multiview_one(const ccv_nnc_tensor_multiview_t* const mv, const ccv_nnc_tensor_dot_recovery_t recovery, const int depth, int* tensor_index, FILE* out)
{
	const int count = mv->kind + mv->repeat;
	int i, j;
	fputs("|{", out);
	if (mv->tv)
	{
		for (i = 0; i < count; i++)
		{
			fprintf(out, "{%d", i);
			if (mv->kind == CCV_NNC_MULTIVIEW_K0N || (mv->kind == CCV_NNC_MULTIVIEW_K1N && i > 0))
				fputc('*', out); // Denotes that we loop on this.
			const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[*tensor_index];
			fprintf(out, "|zone%d", recovery.rename_zone[tensor_dot->zone]);
			for (j = 0; j < depth; j++)
				fputc('\'', out);
			uintptr_t aptr = (uintptr_t)(CCV_NNC_MULTIVIEW_K01(mv) ? mv->tv->data.u8 : mv->data[i].ptr);
			// For the last one, we don't extend to full ainc.
			size_t dim_size = ccv_nnc_dimension_count(mv->tv->info.dim) * CCV_GET_DATA_TYPE_SIZE(mv->tv->type);
			// Print out the range as well.
			fprintf(out, "|{%#010x|%#010x}", (uint32_t)aptr, (uint32_t)(aptr + dim_size - 1));
			++(*tensor_index);
			if (i == count - 1)
				fputc('}', out);
			else
				fputs("}|", out);
		}
	} else {
		for (i = 0; i < count; i++)
		{
			fprintf(out, "{%d", i);
			if (mv->kind == CCV_NNC_MULTIVIEW_K0N || (mv->kind == CCV_NNC_MULTIVIEW_K1N && i > 0))
				fputc('*', out); // Denotes that we loop on this.
			_ccv_nnc_graph_dot_tensor_multiview_one((ccv_nnc_tensor_multiview_t*)mv->data[i].ptr, recovery, depth, tensor_index, out);
			if (i == count - 1)
				fputc('}', out);
			else
				fputs("}|", out);
		}
	}
	fputc('}', out);
}

static void _ccv_nnc_graph_dot_tensor_multiview(const ccv_nnc_tensor_multiview_t* const mv, const ccv_nnc_tensor_dot_recovery_t recovery, const int flags, const int depth, int* tensor_index, FILE* out)
{
	// if it has an alias pointer, or, it is a long form.
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[*tensor_index];
	fprintf(out, "multiview%d", recovery.rename_index[tensor_dot->index]);
	int i;
	for (i = 0; i < depth; i++) // Print subscription to denote depth.
		fputc('\'', out);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		_ccv_nnc_graph_dot_tensor_multiview_one(mv, recovery, depth, tensor_index, out);
		const ccv_nnc_tensor_multiview_t* root = mv;
		while (!root->tv)
			root = (ccv_nnc_tensor_multiview_t*)(root->data[0].ptr);
		fprintf(out, "|%d", root->tv->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && root->tv->info.dim[i]; i++)
			fprintf(out, "x%d", root->tv->info.dim[i]);
		fputc('}', out);
	} else
		*tensor_index += _ccv_nnc_graph_dot_tensor_multiview_count(mv);
}

static void _ccv_nnc_graph_dot_node(const ccv_nnc_graph_exec_info_t* const exec_info, const int exec_index, const ccv_nnc_tensor_dot_recovery_t recovery, const int flags, const int depth, FILE* out, int* const tensor_index)
{
	fprintf(out, "node%d [shape=record,label=\"", exec_index);
	_ccv_nnc_graph_dot_exec(exec_index, exec_info, flags, out);
	int i;
	int k = *tensor_index;
	if (exec_info->input_size > 0)
	{
		fputs("|{Input", out);
		for (i = 0; i < exec_info->input_size; i++)
			if (exec_info->inputs[i])
			{
				fputc('|', out);
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->inputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->inputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->inputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputs("|-", out);
		fputc('}', out);
	}
	if (exec_info->output_size > 0)
	{
		fputs("|{Output", out);
		for (i = 0; i < exec_info->output_size; i++)
			if (exec_info->inputs[i])
			{
				fputc('|', out);
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->outputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->outputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->outputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputs("|-", out);
		fputc('}', out);
	}
	fputs("\"];\n", out);
	*tensor_index = k;
}

static void _ccv_nnc_graph_dot_while_label(const ccv_nnc_graph_exec_info_t* const exec_info, const int exec_index, const ccv_nnc_tensor_dot_recovery_t recovery, const ccv_nnc_graph_t* const while_graph, const int flags, const int depth, FILE* out, int* tensor_index)
{
	int i;
	fprintf(out, "label=<<b>while%d</b>>;\n", exec_index);
	fputs("label [shape=record,label=\"{", out);
	int k = *tensor_index;
	if (exec_info->input_size > 0)
	{
		fputs("{Input|{", out);
		for (i = 0; i < exec_info->input_size; i++)
		{
			if (i > 0)
				fputc('|', out);
			if (exec_info->inputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->inputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->inputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->inputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputc('-', out);
		}
		fputs("}}", out);
	}
	if (exec_info->output_size > 0)
	{
		if (exec_info->input_size > 0)
			fputs("|", out);
		fputs("{Output|{", out);
		for (i = 0; i < exec_info->output_size; i++)
		{
			if (i > 0)
				fputc('|', out);
			if (exec_info->outputs[i])
			{
				if (CCV_IS_TENSOR_MULTIVIEW(exec_info->outputs[i]))
					_ccv_nnc_graph_dot_tensor_multiview((ccv_nnc_tensor_multiview_t*)exec_info->outputs[i], recovery, flags, depth, &k, out);
				else {
					const ccv_nnc_tensor_dot_t* const tensor_dot = recovery.dots + recovery.remap[k];
					_ccv_nnc_graph_dot_tensor(recovery.rename_index[tensor_dot->index], exec_info->outputs[i], recovery.rename_zone[tensor_dot->zone], flags, depth, out);
					++k;
				}
			} else
				fputc('-', out);
		}
		fputs("}}", out);
	}
	fputs("}\"];\n", out);
	*tensor_index = k;
}

static void _ccv_nnc_graph_dot_sub_graph(const ccv_nnc_graph_exec_info_t* const exec_info, const ccv_nnc_tensor_dot_recovery_t p_recovery, const ccv_nnc_graph_t* const graph, const int flags, const int depth, FILE* out, int* tensor_index, int* exec_index)
{
	fprintf(out, "subgraph cluster%d {\nstyle=\"rounded\";\nnode%d [style=invisible];\n", *exec_index, *exec_index);
	// Output this node info within this subgraph.
	_ccv_nnc_graph_dot_while_label(exec_info, *exec_index, p_recovery, graph, flags, depth - 1 /* Label all references to its level above. */, out, tensor_index);
	++(*exec_index);
	ccv_nnc_tensor_dot_recovery_t recovery = _ccv_nnc_graph_tensor_dot_recovery(graph);
	int i, j;
	int k = 0;
	int* node_id = (int*)ccmalloc(sizeof(int) * graph->exec_info->rnum);
	// Output styles.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		node_id[i] = *exec_index;
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (exec_info->graph_ref)
		{
			const ccv_nnc_graph_t* const while_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, exec_info->graph_ref - 1);
			_ccv_nnc_graph_dot_sub_graph(exec_info, recovery, while_graph, flags, depth + 1, out, &k, exec_index);
		} else {
			_ccv_nnc_graph_dot_node(exec_info, *exec_index, recovery, flags, depth, out, &k);
			++(*exec_index);
		}
	}
	// Output connections.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (exec_info->outgoings)
			for (j = 0; j < exec_info->outgoings->rnum; j++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(exec_info->outgoings, j);
				const ccv_nnc_graph_exec_info_t* const outgoing_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, outgoing_idx);
				// If both are sub-graphs, have both tail and head specified.
				if (exec_info->graph_ref && outgoing_info->graph_ref)
					fprintf(out, "node%d -> node%d [ltail=cluster%d,lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i], node_id[outgoing_idx]);
				else if (exec_info->graph_ref && !outgoing_info->graph_ref)
					fprintf(out, "node%d -> node%d [ltail=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i]);
				else if (!exec_info->graph_ref && outgoing_info->graph_ref)
					fprintf(out, "node%d -> node%d [lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[outgoing_idx]);
				else
					fprintf(out, "node%d -> node%d;\n", node_id[i], node_id[outgoing_idx]);
			}
	}
	fputs("}\n", out);
	_ccv_nnc_graph_tensor_dot_recovery_free(recovery);
	ccfree(node_id);
}

void ccv_nnc_graph_dot(const ccv_nnc_graph_t* const graph, const int flags, FILE* out)
{
	fputs("digraph G {\ncompound=true;\n", out);
	ccv_nnc_tensor_dot_recovery_t recovery = _ccv_nnc_graph_tensor_dot_recovery(graph);
	int i, j;
	int k = 0, c = 0;
	int* node_id = (int*)ccmalloc(sizeof(int) * graph->exec_info->rnum);
	// Output styles.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		node_id[i] = c;
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (exec_info->graph_ref)
		{
			const ccv_nnc_graph_t* const while_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, exec_info->graph_ref - 1);
			_ccv_nnc_graph_dot_sub_graph(exec_info, recovery, while_graph, flags, 1, out, &k, &c);
		} else {
			_ccv_nnc_graph_dot_node(exec_info, c, recovery, flags, 0, out, &k);
			++c;
		}
	}
	// Output connections.
	for (i = 0; i < graph->exec_info->rnum; i++)
	{
		ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, i);
		if (exec_info->outgoings)
			for (j = 0; j < exec_info->outgoings->rnum; j++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(exec_info->outgoings, j);
				const ccv_nnc_graph_exec_info_t* const outgoing_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, outgoing_idx);
				// If both are sub-graphs, have both tail and head specified.
				if (exec_info->graph_ref && outgoing_info->graph_ref)
					fprintf(out, "node%d -> node%d [ltail=cluster%d,lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i], node_id[outgoing_idx]);
				else if (exec_info->graph_ref && !outgoing_info->graph_ref)
					fprintf(out, "node%d -> node%d [ltail=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[i]);
				else if (!exec_info->graph_ref && outgoing_info->graph_ref)
					fprintf(out, "node%d -> node%d [lhead=cluster%d];\n", node_id[i], node_id[outgoing_idx], node_id[outgoing_idx]);
				else
					fprintf(out, "node%d -> node%d;\n", node_id[i], node_id[outgoing_idx]);
			}
	}
	fputs("}\n", out);
	_ccv_nnc_graph_tensor_dot_recovery_free(recovery);
	ccfree(node_id);
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
	int i;
#define visitor(node, idx, d, ...) \
	do { \
		PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, d, node->input_size, node->output_size); \
		for (i = 0; i < node->input_size; i++) \
			PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)\n", i + 1, node->inputs[i], (node->inputs[i] ? node->inputs[i]->data.u8 : 0)); \
		for (i = 0; i < node->output_size; i++) \
			PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)\n", i + 1, node->outputs[i], (node->outputs[i] ? node->outputs[i]->data.u8 : 0)); \
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
		if (info->casts)
			ccfree(info->casts);
	}
	if (graph->cond_evals)
		ccfree(graph->cond_evals);
	if (graph->sources)
		ccv_array_free(graph->sources);
	if (graph->destinations)
		ccv_array_free(graph->destinations);
	if (graph->wraps)
		ccv_array_free(graph->wraps);
	if (graph->sub_graphs)
	{
		for (i = 0; i < graph->sub_graphs->rnum; i++)
			ccv_nnc_graph_free(*(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, i));
		ccv_array_free(graph->sub_graphs);
	}
	ccv_array_free(graph->exec_info);
	ccfree(graph);
}
