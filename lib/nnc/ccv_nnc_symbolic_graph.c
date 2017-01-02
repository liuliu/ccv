#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "ccv_nnc_symbolic_graph_internal.h"

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
	graph->forward_symbol_size = 0;
	graph->backward_tensor_symbols = 0;
	graph->backward_symbol_size = 0;
	graph->backward_exec_symbols = 0;
	return graph;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_nnc_tensor_symbol_t symbol = {
		.info = info,
		.d = graph->tensor_symbol_info->rnum,
		.graph = graph
	};
	ccv_nnc_tensor_symbol_info_t symbol_info = {
		.alias_ref = 0,
		.info = info,
		.name = 0
	};
	if (name)
	{
		size_t n = strnlen(name, 63) + 1;
		symbol_info.name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		strncpy(symbol_info.name, name, n);
	}
	ccv_array_push(graph->tensor_symbol_info, &symbol_info);
	return symbol;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_alias_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* const name)
{
	assert(tensor_symbol.graph == graph);
	int d = tensor_symbol.d;
	assert(d >= 0 && d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* info_d = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
	// Find the root tensor that is not an alias.
	while (info_d->alias_ref)
	{
		d = info_d->alias_ref - 1;
		assert(d >= 0 && d < graph->tensor_symbol_info->rnum);
		info_d = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
	}
	ccv_nnc_tensor_symbol_t alias = {
		.info = info,
		.d = graph->tensor_symbol_info->rnum,
		.graph = graph
	};
	// Alias comes in two shapes: 1). the total tensor count is strictly smaller or equal to, and without ofs; 2). with ofs, and each dimension is strictly smaller or equal to.
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		assert(info.dim[i] + ofs[i] <= inc[i]);
	}
	assert(ccv_nnc_dimension_count(inc) <= ccv_nnc_tensor_count(((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d))->info));
	ccv_nnc_tensor_symbol_info_t alias_info = {
		.alias_ref = d + 1,
		.info = info,
		.name = 0
	};
	if (name)
	{
		size_t n = strnlen(name, 63) + 1;
		alias_info.name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		strncpy(alias_info.name, name, n);
	}
	memcpy(alias_info.ofs, ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(alias_info.inc, inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	ccv_array_push(graph->tensor_symbol_info, &alias_info);
	return alias;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_resolve_alias(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor_alias)
{
	assert(graph == tensor_alias.graph);
	assert(tensor_alias.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* alias_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_alias.d);
	assert(alias_info->alias_ref);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, alias_info->alias_ref - 1);
	ccv_nnc_tensor_symbol_t symbol = {
		.info = symbol_info->info,
		.d = alias_info->alias_ref - 1,
		.graph = graph
	};
	return symbol;
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
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
		.hint = ccv_nnc_no_hint,
		.name = 0
	};
	if (name)
	{
		size_t n = strnlen(name, 63) + 1;
		symbol_info.name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		strncpy(symbol_info.name, name, n);
	}
	if (input_size > 0 || output_size > 0)
	{
		symbol_info.inputs = ccmalloc(sizeof(int) * (input_size + output_size));
		symbol_info.outputs = symbol_info.inputs + input_size;
	}
	int i;
	for (i = 0; i < input_size; i++)
		symbol_info.inputs[i] = inputs[i].d;
	for (i = 0; i < output_size; i++)
		symbol_info.outputs[i] = outputs[i].d;
	ccv_array_push(graph->exec_symbol_info, &symbol_info);
	return symbol;
}

ccv_nnc_cmd_t ccv_nnc_graph_exec_symbol_cmd(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec)
{
	assert(graph == exec.graph);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	return symbol_info->cmd;
}

int ccv_nnc_graph_exec_symbol_set_hint(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t exec, const ccv_nnc_hint_t hint)
{
	assert(graph == exec.graph);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	symbol_info->hint = hint;
	return 0;
}

int ccv_nnc_tensor_symbol_set(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor, const ccv_nnc_tensor_param_t info)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	symbol_info->info = info;
	return 0;
}

int ccv_nnc_tensor_symbol_set_flags(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor, const int flags)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	symbol_info->flags = flags;
	return 0;
}

int ccv_nnc_tensor_symbol_flag(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t tensor, const int flags)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	return !!(symbol_info->flags & flags);
}

int ccv_nnc_graph_exec_symbol_concat(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t source, const ccv_nnc_graph_exec_symbol_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_symbol_info->rnum);
	assert(destination.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* src_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, source.d);
	if (!src_symbol_info->outgoings)
		src_symbol_info->outgoings = ccv_array_new(sizeof(int32_t), 1, 0);
	else {
		int i;
		// Check if this is already connected, if so, skip.
		for (i = 0; i < src_symbol_info->outgoings->rnum; i++)
			if (*(int*)ccv_array_get(src_symbol_info->outgoings, i) == destination.d)
				return -1;
	}
	ccv_array_push(src_symbol_info->outgoings, &destination.d);
	return 0;
}

int ccv_nnc_graph_exec_symbol_disjoin(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t source, const ccv_nnc_graph_exec_symbol_t destination)
{
	assert(graph == source.graph);
	assert(graph == destination.graph);
	assert(source.d < graph->exec_symbol_info->rnum);
	assert(destination.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* src_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, source.d);
	if (!src_symbol_info->outgoings)
		return -1;
	int i, j = -1;
	// Check if this is already connected, if so, skip.
	for (i = 0; i < src_symbol_info->outgoings->rnum; i++)
		if (*(int*)ccv_array_get(src_symbol_info->outgoings, i) == destination.d)
		{
			j = i;
			break;
		}
	if (j < 0)
		return -1;
	if (j < src_symbol_info->outgoings->rnum - 1)
		*(int*)ccv_array_get(src_symbol_info->outgoings, j) = *(int*)ccv_array_get(src_symbol_info->outgoings, src_symbol_info->outgoings->rnum - 1);
	--src_symbol_info->outgoings->rnum;
	return 0;
}

int ccv_nnc_graph_exec_symbol_autogen(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* execs, const int exec_size)
{
	int i, j, x, y;
	for (i = 0; i < exec_size; i++)
	{
		assert(execs[i].graph == graph);
		assert(execs[i].d >= 0);
		assert(execs[i].d < graph->exec_symbol_info->rnum);
	}
	for (i = 0; i < exec_size; i++)
	{
		int a_idx = execs[i].d;
		ccv_nnc_graph_exec_symbol_info_t* a_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, a_idx);
		for (j = i + 1; j < exec_size;j++)
		{
			int b_idx = execs[j].d;
			// Skip if they are the same.
			if (a_idx == b_idx)
				continue;
			ccv_nnc_graph_exec_symbol_info_t* b_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, b_idx);
			int b_to_a = 0;
			for (x = 0; x < a_symbol_info->input_size && !b_to_a; x++)
			{
				int a = a_symbol_info->inputs[x];
				// Handle alias as well.
				ccv_nnc_tensor_symbol_info_t* a_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, a);
				if (a_tensor_info->alias_ref)
					a = a_tensor_info->alias_ref - 1;
				for (y = 0; y < b_symbol_info->output_size && !b_to_a; y++)
				{
					int b = b_symbol_info->outputs[y];
					ccv_nnc_tensor_symbol_info_t* b_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, b);
					if (b_tensor_info->alias_ref)
						b = b_tensor_info->alias_ref - 1;
					if (a == b)
						// This two have matching inputs and outputs, thus, you can concat b to a.
						b_to_a = 1;
				}
			}
			if (b_to_a)
				ccv_nnc_graph_exec_symbol_concat(graph, execs[j], execs[i]);
			int a_to_b = 0;
			for (x = 0; x < a_symbol_info->output_size && !a_to_b; x++)
			{
				int a = a_symbol_info->outputs[x];
				// Handle alias as well.
				ccv_nnc_tensor_symbol_info_t* a_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, a);
				if (a_tensor_info->alias_ref)
					a = a_tensor_info->alias_ref - 1;
				for (y = 0; y < b_symbol_info->input_size && !a_to_b; y++)
				{
					int b = b_symbol_info->inputs[y];
					ccv_nnc_tensor_symbol_info_t* b_tensor_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, b);
					if (b_tensor_info->alias_ref)
						b = b_tensor_info->alias_ref - 1;
					if (a == b)
						// This two have matching inputs and outputs, thus, you can concat b to a.
						a_to_b = 1;
				}
			}
			if (a_to_b)
				ccv_nnc_graph_exec_symbol_concat(graph, execs[i], execs[j]);
		}
	}
	return 0;
}

static void _ccv_nnc_symbolic_graph_dot_exec_symbol(const int index, const ccv_nnc_graph_exec_symbol_info_t* const symbol_info, const int flags, FILE* out)
{
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
		fputc('{', out);
	if (symbol_info->name)
		fputs(symbol_info->name, out);
	else
		fprintf(out, "node%d", index);
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		fputs("|Command: ", out);
		fputs(ccv_nnc_cmd_name(symbol_info->cmd.cmd), out);
		fputc('}', out);
	}
}

static void _ccv_nnc_symbolic_graph_dot_tensor_symbol(const int index, const ccv_nnc_tensor_symbol_info_t* const symbol_info, const ccv_nnc_tensor_symbol_info_t* const alias_info, const int flags, FILE* out)
{
	// if it has an alias pointer, or, it is a long form.
	if (flags == CCV_NNC_LONG_DOT_GRAPH || alias_info)
		fputc('{', out);
	if (symbol_info->name)
		fputs(symbol_info->name, out);
	else
		fprintf(out, "tensor%d", index);
	if (flags == CCV_NNC_LONG_DOT_GRAPH && (symbol_info->flags & CCV_NNC_SYM_TENSOR_INIT_ZEROS))
		fputs(" (0)", out); // Output if it is zero init'ed.
	if (alias_info)
	{
		fputs("|as. ", out);
		if (alias_info->name)
			fputs(alias_info->name, out);
		else
			fprintf(out, "tensor%d", symbol_info->alias_ref - 1);
		if (flags == CCV_NNC_LONG_DOT_GRAPH && (alias_info->flags & CCV_NNC_SYM_TENSOR_INIT_ZEROS))
			fputs(" (0)", out); // Output if it is zero init'ed.
	}
	if (flags == CCV_NNC_LONG_DOT_GRAPH)
	{
		int i;
		fprintf(out, "|%d", symbol_info->info.dim[0]);
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && symbol_info->info.dim[i]; i++)
			fprintf(out, "x%d", symbol_info->info.dim[i]);
	}
	if (flags == CCV_NNC_LONG_DOT_GRAPH || alias_info)
		fputc('}', out);
}

void ccv_nnc_symbolic_graph_dot(const ccv_nnc_symbolic_graph_t* const graph, const int flags, FILE* out)
{
	fputs("digraph G {\n", out);
	int i, j;
	// Output styles.
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
	{
		ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		fprintf(out, "node%d [shape=Mrecord,label=\"", i);
		_ccv_nnc_symbolic_graph_dot_exec_symbol(i, exec_symbol_info, flags, out);
		if (exec_symbol_info->input_size > 0)
		{
			fputs("|{Input", out);
			for (j = 0; j < exec_symbol_info->input_size; j++)
			{
				if (exec_symbol_info->inputs[j] >= 0)
				{
					fputc('|', out);
					ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, exec_symbol_info->inputs[j]);
					ccv_nnc_tensor_symbol_info_t* alias_symbol_info = tensor_symbol_info->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_symbol_info->alias_ref - 1) : 0;
					_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->inputs[j], tensor_symbol_info, alias_symbol_info, flags, out);
				} else
					fputs("|-", out);
			}
			fputc('}', out);
		}
		if (exec_symbol_info->output_size > 0)
		{
			fputs("|{Output", out);
			for (j = 0; j < exec_symbol_info->output_size; j++)
			{
				if (exec_symbol_info->outputs[j] >= 0)
				{
					fputc('|', out);
					ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, exec_symbol_info->outputs[j]);
					ccv_nnc_tensor_symbol_info_t* alias_symbol_info = tensor_symbol_info->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_symbol_info->alias_ref - 1) : 0;
					_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->outputs[j], tensor_symbol_info, alias_symbol_info, flags, out);
				} else
					fputs("|-", out);
			}
			fputc('}', out);
		}
		fputs("\"];\n", out);
	}
	// Output connections.
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
	{
		ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		if (exec_symbol_info->outgoings)
			for (j = 0; j < exec_symbol_info->outgoings->rnum; j++)
				fprintf(out, "node%d -> node%d;\n", i, *(int*)ccv_array_get(exec_symbol_info->outgoings, j));
	}
	fputs("}\n", out);
}

void ccv_nnc_symbolic_graph_free(ccv_nnc_symbolic_graph_t* const graph)
{
	int i;
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
	{
		ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		if (symbol_info->name)
			ccfree(symbol_info->name);
		ccv_array_t* outgoings = symbol_info->outgoings;
		if (outgoings)
			ccv_array_free(outgoings);
		// We allocate inputs & outputs in continuous fashion, therefore, only need to free the input array.
		ccfree(symbol_info->inputs);
	}
	for (i = 0; i < graph->tensor_symbol_info->rnum; i++)
	{
		ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, i);
		if (symbol_info->name)
			ccfree(symbol_info->name);
	}
	ccv_array_free(graph->tensor_symbol_info);
	ccv_array_free(graph->exec_symbol_info);
	if (graph->backward_tensor_symbols)
		ccfree(graph->backward_tensor_symbols);
	ccfree(graph);
}

void ccv_nnc_symbolic_graph_symbol_organize(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info)
{
	memcpy(tensor_symbol_info, symbolic_graph->tensor_symbol_info->data, sizeof(ccv_nnc_tensor_symbol_info_t) * symbolic_graph->tensor_symbol_info->rnum);
	memcpy(exec_symbol_info, symbolic_graph->exec_symbol_info->data, sizeof(ccv_nnc_graph_exec_symbol_info_t) * symbolic_graph->exec_symbol_info->rnum);
	int i, max_input_size = 0, max_output_size = 0;
	// Materialize auto hints.
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
	{
		max_input_size = ccv_max(max_input_size, exec_symbol_info[i].input_size);
		max_output_size = ccv_max(max_output_size, exec_symbol_info[i].output_size);
		// If there is no hint and we have input and output tensor specified.
		if (ccv_nnc_is_no_hint(exec_symbol_info[i].hint) &&
			exec_symbol_info[i].input_size > 0 && exec_symbol_info[i].inputs[0] >= 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].inputs[0]].info) &&
			exec_symbol_info[i].output_size > 0 && exec_symbol_info[i].outputs[0] >= 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].outputs[0]].info))
			exec_symbol_info[i].hint = ccv_nnc_hint_auto(exec_symbol_info[i].cmd.info, tensor_symbol_info[exec_symbol_info[i].inputs[0]].info, tensor_symbol_info[exec_symbol_info[i].outputs[0]].info);
	}

	ccv_nnc_tensor_param_t* input_params = max_input_size > 0 ? (ccv_nnc_tensor_param_t*)ccmalloc(sizeof(ccv_nnc_tensor_param_t) * max_input_size) : 0;
	ccv_nnc_tensor_param_t* output_params = max_output_size > 0 ? (ccv_nnc_tensor_param_t*)ccmalloc(sizeof(ccv_nnc_tensor_param_t) * max_output_size) : 0;

	// Materialize auto tensors. This need to go with the topological order.
#define visitor(node, ...) \
	do { \
		if (node->input_size > 0 && node->output_size > 0) \
		{ \
			for (i = 0; i < node->input_size; i++) \
				input_params[i] = node->inputs[i] >= 0 ? tensor_symbol_info[node->inputs[i]].info : ccv_nnc_tensor_auto; \
			ccv_nnc_hint_tensor_auto(node->cmd, input_params, node->input_size, node->hint, output_params, node->output_size); \
			for (i = 0; i < node->output_size; i++) \
				/* Only assign the output parameters if the symbol itself is auto. */ \
				if (node->outputs[i] >= 0 && ccv_nnc_is_tensor_auto(tensor_symbol_info[node->outputs[i]].info)) \
					tensor_symbol_info[node->outputs[i]].info = output_params[i]; \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	if (input_params)
		ccfree(input_params);
	if (output_params)
		ccfree(output_params);
}
