#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif

typedef struct {
	int alias_ref;
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_param_t info;
	int flags;
	char* name;
} ccv_nnc_tensor_symbol_info_t;

typedef struct {
	int input_size;
	int output_size;
	int* inputs;
	int* outputs;
	ccv_array_t* outgoings; // outgoing nodes
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
	char* name;
} ccv_nnc_graph_exec_symbol_info_t;

struct ccv_nnc_symbolic_graph_s {
	ccv_array_t* tensor_symbol_info;
	ccv_array_t* exec_symbol_info;
	int forward_symbol_size;
	int* backward_tensor_symbols;
	int backward_symbol_size;
	int* backward_exec_symbols;
};

struct ccv_nnc_tensor_arena_s {
	int memory_type;
	int device_id;
	// This is a table of tensor references to real allocated tensors.
	int vt_tensor_rnum;
	ccv_nnc_tensor_t** vt_tensor;
	// This is the allocated non-continuous buffers.
	int buffer_rnum;
	uint8_t** buffer;
	uint64_t* buffer_size;
	// Real allocated tensor headers.
	ccv_nnc_tensor_view_t tensor[1];
};

struct ccv_nnc_graph_exec_arena_s {
	int graph_exec_rnum;
	ccv_nnc_graph_exec_t source;
	ccv_nnc_graph_exec_t destination;
	ccv_nnc_graph_exec_t graph_exec[1];
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
	graph->forward_symbol_size = 0;
	graph->backward_tensor_symbols = 0;
	graph->backward_symbol_size = 0;
	graph->backward_exec_symbols = 0;
	return graph;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_new(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_param_t info, const char* name)
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

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_alias_new(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_symbol_t tensor_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* name)
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

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_resolve_alias(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_symbol_t tensor_alias)
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

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol_new(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_cmd_t cmd, ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* outputs, const int output_size, const char* name)
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

ccv_nnc_cmd_t ccv_nnc_graph_exec_symbol_cmd(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t exec)
{
	assert(graph == exec.graph);
	assert(exec.d < graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec.d);
	return symbol_info->cmd;
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

int ccv_nnc_tensor_symbol_set_flags(const ccv_nnc_symbolic_graph_t* graph, ccv_nnc_tensor_symbol_t tensor, const int flags)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	symbol_info->flags = flags;
	return 0;
}

int ccv_nnc_tensor_symbol_flag(const ccv_nnc_symbolic_graph_t* graph, ccv_nnc_tensor_symbol_t tensor, const int flags)
{
	assert(graph == tensor.graph);
	assert(tensor.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_info_t* symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor.d);
	return !!(symbol_info->flags & flags);
}

int ccv_nnc_graph_exec_symbol_concat(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t source, const ccv_nnc_graph_exec_symbol_t destination)
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

int ccv_nnc_graph_exec_symbol_disjoin(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t source, const ccv_nnc_graph_exec_symbol_t destination)
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

int ccv_nnc_graph_exec_symbol_autogen(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t* execs, const int exec_size)
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

static void _ccv_nnc_symbolic_graph_dot_exec_symbol(const int index, const ccv_nnc_graph_exec_symbol_info_t* symbol_info, const int flags, FILE* out)
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

static void _ccv_nnc_symbolic_graph_dot_tensor_symbol(const int index, const ccv_nnc_tensor_symbol_info_t* symbol_info, const ccv_nnc_tensor_symbol_info_t* alias_info, const int flags, FILE* out)
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

void ccv_nnc_symbolic_graph_dot(const ccv_nnc_symbolic_graph_t* graph, const int flags, FILE* out)
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
				ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, exec_symbol_info->inputs[j]);
				fputc('|', out);
				ccv_nnc_tensor_symbol_info_t* alias_symbol_info = tensor_symbol_info->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_symbol_info->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->inputs[j], tensor_symbol_info, alias_symbol_info, flags, out);
			}
			fputc('}', out);
		}
		if (exec_symbol_info->output_size > 0)
		{
			fputs("|{Output", out);
			for (j = 0; j < exec_symbol_info->output_size; j++)
			{
				ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, exec_symbol_info->outputs[j]);
				fputc('|', out);
				ccv_nnc_tensor_symbol_info_t* alias_symbol_info = tensor_symbol_info->alias_ref ? (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_symbol_info->alias_ref - 1) : 0;
				_ccv_nnc_symbolic_graph_dot_tensor_symbol(exec_symbol_info->outputs[j], tensor_symbol_info, alias_symbol_info, flags, out);
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

typedef struct {
	int flag;
	int ref;
	ccv_array_t* head; // The head nodes (it could be multiple if from the graph, one cannot determine which is the first).
	ccv_array_t* tail; // The tail nodes (it could be multiple if from the graph, one cannot determine which is the last).
} ccv_nnc_tensor_expect_t;

enum {
	UNASSIGNED = 0x1,
	ALIAS = 0x2,
	CONST_TENSOR = 0x3,
};

#define TENSOR_EXPECT_UNASSIGNED(t) (t.flag == UNASSIGNED)
#define TENSOR_EXPECT_ALIAS(t) (t.flag == ALIAS)
#define TENSOR_EXPECT_CONST(t) (t.flag == CONST_TENSOR)
#define TENSOR_EXPECT_UNUSED(t) (t.flag == UNUSED)
#define TENSOR_EXPECT_COMPUTABLE(t) (!TENSOR_EXPECT_ALIAS(t) && !TENSOR_EXPECT_UNASSIGNED(t))

static void _ccv_array_replace_or_insert_int(ccv_array_t* ints, const int idx, const int outgoing)
{
	int i;
	for (i = 0; i < ints->rnum; i++)
		if (*(int*)ccv_array_get(ints, i) == idx)
		{
			*(int*)ccv_array_get(ints, i) = outgoing;
			return;
		}
	ccv_array_push(ints, &outgoing);
}

typedef struct {
	int index;
	int companion; // The companion node index (the node that doesn't interfere with current one).
	uint64_t size;
} ccv_nnc_tensor_opt_t;

#define more_than(i1, i2, aux) ((i1).size >= (i2).size)
static CCV_IMPLEMENT_QSORT(_ccv_nnc_tensor_opt_sort_by_size, ccv_nnc_tensor_opt_t, more_than)
#undef more_than

// If every a's head is deterministically after b's tail
static int _ccv_nnc_tensor_expect_head_after_tail(const ccv_sparse_matrix_t* exec_dep, const ccv_nnc_tensor_expect_t a, const ccv_nnc_tensor_expect_t b)
{
	assert(a.head);
	assert(b.tail);
	int x, y;
	for (x = 0; x < a.head->rnum; x++)
		for (y = 0; y < b.tail->rnum; y++)
		{
			ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, *(int*)ccv_array_get(a.head, x), *(int*)ccv_array_get(b.tail, y));
			if (!cell.i32 || cell.i32[0] == 0)
				return 0;
		}
	// We've entered this nested-for loop, therefore, it must be verifiably, deterministically after b's tail now.
	return (a.head->rnum > 0 && b.tail->rnum > 0);
}

static ccv_nnc_tensor_arena_t* _ccv_nnc_tensor_arena_new(const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const int tensor_symbol_info_size, const ccv_sparse_matrix_t* exec_dep, const ccv_nnc_tensor_expect_t* tensor_expect, ccv_array_t** alloc_dep)
{
	// Compute how many dis-continuous buffers are needed.
	// We prefer to have several dis-continuous buffers instead of one big buffer because
	// in this way, we can rely on system memory allocators (jemalloc, tcmalloc, or CUDA's allocator)
	// to fully utilize memory.
	int i, j, k;
	uint64_t* tensor_size = (uint64_t*)ccmalloc(sizeof(uint64_t) * tensor_symbol_info_size);
	int computable_tensor_size = 0, available_tensor_size = 0;
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (!TENSOR_EXPECT_UNASSIGNED(tensor_expect[i]))
		{
			// Tensors that we need the header info.
			++available_tensor_size;
			if (!TENSOR_EXPECT_ALIAS(tensor_expect[i]))
			{
				// Tensors that we actually need to compute (exclude the alias).
				++computable_tensor_size;
				// Cache tensor size (assuming it is 32F, and align to 16 bytes).
				tensor_size[i] = ((uint64_t)CCV_GET_DATA_TYPE_SIZE(CCV_32F) * ccv_nnc_tensor_count(tensor_symbol_info[i].info) + 15) / 16 * 16;
			}
		}
	ccv_sparse_matrix_t* tensor_itf = ccv_sparse_matrix_new(tensor_symbol_info_size, tensor_symbol_info_size, CCV_8U | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	// Overlap count.
	for (i = 0; i < tensor_symbol_info_size; i++)
		for (j = i + 1; j < tensor_symbol_info_size; j++)
			if (TENSOR_EXPECT_COMPUTABLE(tensor_expect[i]) && TENSOR_EXPECT_COMPUTABLE(tensor_expect[j]))
			{
				// If either of the tensor is const, it must interfere with each other.
				const uint8_t one = 1;
				if (TENSOR_EXPECT_CONST(tensor_expect[i]) || TENSOR_EXPECT_CONST(tensor_expect[j]))
					ccv_set_sparse_matrix_cell(tensor_itf, i, j, &one);
				else {
					// Otherwise, check to see if they interfere (default to yes).
					// If any of the i's head is deterministically later than j's tail
					// or any of the i's tail is deterministically earlier than j's head, they don't interfere.
					int i_hop_j = _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[i], tensor_expect[j]);
					int j_hop_i = _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[j], tensor_expect[i]);
					// It cannot be that both i can hop to j can j can hop to i.
					assert(!(i_hop_j > 0 && j_hop_i > 0));
					if (!i_hop_j && !j_hop_i)
						ccv_set_sparse_matrix_cell(tensor_itf, i, j, &one);
				}
			}
	int* oc = (int*)cccalloc(tensor_symbol_info_size, sizeof(int));
	for (i = 0; i < tensor_symbol_info_size; i++)
		for (j = 0; j < tensor_symbol_info_size; j++)
			// If these two tensors are still alive, analyze them.
			if (i != j && TENSOR_EXPECT_COMPUTABLE(tensor_expect[i]) && TENSOR_EXPECT_COMPUTABLE(tensor_expect[j]))
			{
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(i, j), ccv_max(i, j));
				// If their life time overlaps, compute how many tensors it overlap.
				if (cell.u8 && cell.u8[0] == 1)
					++oc[i];
			}
	int* assigned = (int*)cccalloc(tensor_symbol_info_size, sizeof(int));
	uint64_t* allocated_offset = (uint64_t*)cccalloc(tensor_symbol_info_size, sizeof(uint64_t));
	uint64_t* allocated_size = (uint64_t*)cccalloc(tensor_symbol_info_size, sizeof(uint64_t));
	int num_assigned = 0; 
	// I can do a bit optimization here to assign out const tensor first, but heck, this just works for now.
	// Allocation graph (assuming there is a source node, and a destination node, which is 0, and (tensor_symbol_info_size + 1)
	// The first channel denotes the bytes available for allocation,
	// the second channel denotes the offset available for the allocation,
	ccv_sparse_matrix_t* alloc = ccv_sparse_matrix_new(tensor_symbol_info_size + 2, tensor_symbol_info_size + 2, CCV_64S | CCV_C2, CCV_SPARSE_ROW_MAJOR, 0);
	ccv_array_t* opt = ccv_array_new(sizeof(ccv_nnc_tensor_opt_t), 1, 0);
	for (j = 0; j < computable_tensor_size;)
	{
		// Find the one with largest overlap, and it is not assigned.
		int max_oc = 0;
		ccv_array_clear(opt);
		for (i = 0; i < tensor_symbol_info_size; i++)
			if (oc[i] >= max_oc && TENSOR_EXPECT_COMPUTABLE(tensor_expect[i]) && !assigned[i])
			{
				ccv_nnc_tensor_opt_t a = {
					.size = tensor_size[i],
					.index = i,
					.companion = -1,
				};
				// In case we have a tie, take them all in the array.
				if (oc[i] > max_oc)
					ccv_array_clear(opt), max_oc = oc[i];
				ccv_array_push(opt, &a);
			}
		assert(opt->rnum > 0);
		// Go through opt array, find all tensors that doesn't interfere with it, and have tensor size larger than it.
		// Push them with the "companion" into the opt array as well.
		int rnum = opt->rnum;
		for (i = 0; i < rnum; i++)
		{
			// Copy it out, because after insertion, it may hold invalid pointer.
			ccv_nnc_tensor_opt_t a = *(ccv_nnc_tensor_opt_t*)ccv_array_get(opt, i);
			for (k = 0; k < tensor_symbol_info_size; k++)
				// Find non-overlapping tensor that has larger size (of course, is unassigned).
				if (TENSOR_EXPECT_COMPUTABLE(tensor_expect[k]) && !assigned[k] && tensor_size[k] > a.size)
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(a.index, k), ccv_max(a.index, k));
					// Good, push to opt array.
					if (!cell.u8 || cell.u8[0] == 0)
					{
						ccv_nnc_tensor_opt_t b = a;
						b.companion = k;
						b.size = tensor_size[k];
						ccv_array_push(opt, &b);
					}
				}
		}
		// Order opt array by the size.
		_ccv_nnc_tensor_opt_sort_by_size((ccv_nnc_tensor_opt_t*)opt->data, opt->rnum, 0);
		// Assuming all tensors has the same data format (32F), therefore, we only need to consider the dimensional size.
		// Go through opt array again, this time, it is ordered by size, therefore, if we found a place to insert, we are good.
		int min_y = 0, min_x = tensor_symbol_info_size + 1, min_i = -1, min_hop = exec_dep->rows * 3;
		uint64_t min_val[2] = {
			0, 0
		};
		for (i = 0; i < opt->rnum; i++)
		{
			ccv_nnc_tensor_opt_t a = *(ccv_nnc_tensor_opt_t*)ccv_array_get(opt, i);
			// Determine the order between the two.
			int a_hop_c = 0;
			int c_hop_a = 0;
			if (a.companion >= 0)
			{
				a_hop_c = _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[a.companion], tensor_expect[a.index]);
				c_hop_a = _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[a.index], tensor_expect[a.companion]);
				// You can only hop from one direction, otherwise we have a loop.
				assert((a_hop_c > 0 && c_hop_a == 0) || (a_hop_c == 0 && c_hop_a > 0));
			}
#define for_block(y, x, val) do { \
				/* y is always earlier than x, but this is hard to assert now. */ \
				/* If this edge satisfy the requirement, now we need to find the ones with tightest possible bounds. */ \
				/* Thus, the hop between y and x (through a) should be smallest ones. */ \
				if (((uint64_t*)val)[0] >= a.size) \
				{ \
					if (a.companion < 0) \
					{ \
						int y_hop_a = (y == 0) ? exec_dep->rows : _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[a.index], tensor_expect[y - 1]); \
						int a_hop_x = (x == tensor_symbol_info_size + 1) ? exec_dep->rows : _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[x - 1], tensor_expect[a.index]); \
						int hop = y_hop_a + a_hop_x; \
						/* a.index doesn't overlap with y and x (in between) */ \
						if ((y == 0 || y_hop_a) && (x == tensor_symbol_info_size + 1 || a_hop_x) && hop < min_hop) \
							min_y = y, min_x = x, min_hop = hop, \
							min_val[0] = ((uint64_t*)val)[0], min_val[1] = ((uint64_t*)val)[1]; \
					} else { \
						/* a.index doesn't overlap with y and x (in between) */ \
						/* a.companion doesn't overlap with y and x (in between) as well */ \
						/* because we know a.index is before a.companion (a can hop to c), */ \
						/* we can check if y can hop to a and then c can hop to x to determine. */ \
						if (a_hop_c > 0) \
						{ \
							int y_hop_a = (y == 0) ? exec_dep->rows : _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[a.index], tensor_expect[y - 1]); \
							int c_hop_x = (x == tensor_symbol_info_size + 1) ? exec_dep->rows : _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[x - 1], tensor_expect[a.companion]); \
							int hop = y_hop_a + c_hop_x; \
							if ((y == 0 || y_hop_a) && (x == tensor_symbol_info_size + 1 || c_hop_x) && hop < min_hop) \
								min_y = y, min_x = x, min_hop = hop, \
								min_val[0] = ((uint64_t*)val)[0], min_val[1] = ((uint64_t*)val)[1]; \
						} else { \
							int y_hop_c = (y == 0) ? exec_dep->rows : _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[a.companion], tensor_expect[y - 1]); \
							int a_hop_x = (x == tensor_symbol_info_size + 1) ? exec_dep->rows : _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[x - 1], tensor_expect[a.index]); \
							int hop = y_hop_c + a_hop_x; \
							if ((y == 0 || y_hop_c) && (x == tensor_symbol_info_size + 1 || a_hop_x) && hop < min_hop) \
								min_y = y, min_x = x, min_hop = hop, \
								min_val[0] = ((uint64_t*)val)[0], min_val[1] = ((uint64_t*)val)[1]; \
						} \
					} \
				} \
			} while (0)
			CCV_SPARSE_FOREACH(alloc, for_block);
#undef for_block
			// If I found a place, stop, and exit.
			if (min_y > 0 || min_x < tensor_symbol_info_size + 1)
			{
				min_i = i;
				break;
			}
		}
		// If I cannot find a place, then start a new connection between min_y and min_x (a new assignment group).
		// and default to largest size available.
		ccv_nnc_tensor_opt_t a = *(ccv_nnc_tensor_opt_t*)ccv_array_get(opt, ccv_max(0, min_i));
		if (min_i == -1)
		{
			allocated_size[num_assigned] = a.size;
			++num_assigned;
		}
		int assign_group = num_assigned;
		if (min_y > 0)
		{
			assign_group = assigned[min_y - 1];
			// The y and x should belong to the same assigned group.
			assert(min_x == tensor_symbol_info_size + 1 || assigned[min_x - 1] == assign_group);
		} else if (min_x < tensor_symbol_info_size + 1)
			assign_group = assigned[min_x - 1];
		// Assign out the selected one.
		assigned[a.index] = assign_group;
		// The offset for this one, should be either 0 (started a new group, when min_i == -1), or the offset on this edge.
		allocated_offset[a.index] = min_val[1];
		for (i = 0; i < tensor_symbol_info_size; i++)
			if (!assigned[i] && TENSOR_EXPECT_COMPUTABLE(tensor_expect[i]))
			{
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(i, a.index), ccv_max(i, a.index));
				if (cell.u8 && cell.u8[0] == 1)
					--oc[i];
			}
		// Assign out companion as well.
		if (a.companion >= 0)
		{
			assigned[a.companion] = assign_group;
			// The offset for this one, should be either 0 (started a new group, when min_i == -1), or the offset on this edge.
			allocated_offset[a.companion] = min_val[1];
			for (i = 0; i < tensor_symbol_info_size; i++)
				if (!assigned[i] && TENSOR_EXPECT_COMPUTABLE(tensor_expect[i]))
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(i, a.companion), ccv_max(i, a.companion));
					if (cell.u8 && cell.u8[0] == 1)
						--oc[i];
				}
		}
		// If min_y is source and min_x is destination, we don't need to do anything, otherwise, decrease the weight on that edge.
		if (min_y != 0 || min_x != tensor_symbol_info_size + 1)
		{
			uint64_t val[2] = {
				min_val[0], min_val[1]
			};
			assert(val[0] >= a.size);
			val[0] -= a.size;
			val[1] = val[1] + a.size; // Move the offset to the next one.
			ccv_set_sparse_matrix_cell(alloc, min_y, min_x, val);
		}
		// If a doesn't have a companion, simple, set the edge between min_y and the current one, current one and min_x,
		// with proper offset and size deduction.
		if (a.companion < 0)
		{
			uint64_t val[2] = {
				a.size, min_val[1] // keep the offset
			};
			ccv_set_sparse_matrix_cell(alloc, min_y, a.index + 1, val);
			ccv_set_sparse_matrix_cell(alloc, a.index + 1, min_x, val);
			// Move to the next available tensor.
			j++;
		} else {
			int a_hop_c = _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[a.companion], tensor_expect[a.index]);
			int c_hop_a = _ccv_nnc_tensor_expect_head_after_tail(exec_dep, tensor_expect[a.index], tensor_expect[a.companion]);
			// You can only hop from one direction, otherwise we have a loop.
			assert((a_hop_c > 0 && c_hop_a == 0) || (a_hop_c == 0 && c_hop_a > 0));
			if (a_hop_c > 0)
			{
				uint64_t val[2] = {
					tensor_size[a.index], min_val[1] // keep the offset
				};
				ccv_set_sparse_matrix_cell(alloc, min_y, a.index + 1, val);
				val[0] = a.size;
				assert(a.size == tensor_size[a.companion]);
				ccv_set_sparse_matrix_cell(alloc, a.index + 1, a.companion + 1, val);
				ccv_set_sparse_matrix_cell(alloc, a.companion + 1, min_x, val);
				if (a.size > tensor_size[a.index])
				{
					// residual size connection between min_y and companion.
					val[0] = a.size - tensor_size[a.index];
					// offset need to be updated as well.
					val[1] = min_val[1] + tensor_size[a.index];
					ccv_set_sparse_matrix_cell(alloc, min_y, a.companion + 1, val);
				}
			} else {
				uint64_t val[2] = {
					a.size, min_val[1] // keep the offset
				};
				assert(a.size == tensor_size[a.companion]);
				ccv_set_sparse_matrix_cell(alloc, min_y, a.companion + 1, val);
				val[0] = tensor_size[a.index];
				ccv_set_sparse_matrix_cell(alloc, a.companion + 1, a.index + 1, val);
				ccv_set_sparse_matrix_cell(alloc, a.index + 1, min_x, val);
				if (a.size > tensor_size[a.index])
				{
					// residual size connection between min_y and companion.
					val[0] = a.size - tensor_size[a.index];
					// offset need to be updated as well.
					val[1] = min_val[1] + tensor_size[a.index];
					ccv_set_sparse_matrix_cell(alloc, a.companion + 1, min_x, val);
				}
			}
			// Assigned out two tensors.
			j += 2;
		}
	}
	ccv_array_free(opt);
	ccv_matrix_free(tensor_itf);
#define for_block(y, x, val) do { \
		if (((uint64_t*)val)[0] > 0 && y > 0 && x < tensor_symbol_info_size + 1) \
		{ \
			if (!alloc_dep[x - 1]) \
				alloc_dep[x - 1] = ccv_array_new(sizeof(int), 1, 0); \
			_ccv_array_replace_or_insert_int(alloc_dep[x - 1], y - 1, y - 1); \
		} \
	} while (0)
	CCV_SPARSE_FOREACH(alloc, for_block);
#undef for_block
	ccv_matrix_free(alloc);
	ccfree(tensor_size);
	ccfree(oc);
	// All tensors assigned out, now, the num_assigned is the number of dis-continuous buffers,
	// Each tensor have the designation in assigned array, and offset in allocated_offset.
	ccv_nnc_tensor_arena_t* tensor_arena = (ccv_nnc_tensor_arena_t*)ccmalloc(sizeof(ccv_nnc_tensor_arena_t) + sizeof(ccv_nnc_tensor_t*) * tensor_symbol_info_size + sizeof(uint8_t*) * num_assigned + sizeof(uint64_t) * num_assigned + sizeof(ccv_nnc_tensor_view_t) * (available_tensor_size - 1));
	tensor_arena->vt_tensor = (ccv_nnc_tensor_t**)(tensor_arena->tensor + available_tensor_size);
	tensor_arena->buffer = (uint8_t**)(tensor_arena->vt_tensor + tensor_symbol_info_size);
	tensor_arena->buffer_size = (uint64_t*)(tensor_arena->buffer + num_assigned);
	tensor_arena->buffer_rnum = num_assigned;
	tensor_arena->vt_tensor_rnum = tensor_symbol_info_size;
	memcpy(tensor_arena->buffer_size, allocated_size, sizeof(uint64_t) * num_assigned);
	ccfree(allocated_size);
	int memory_type = CCV_TENSOR_GET_MEMORY(tensor_symbol_info[0].info.type);
	int device_id = CCV_TENSOR_GET_DEVICE_ID(tensor_symbol_info[0].info.type);
	for (i = 1; i < tensor_symbol_info_size; i++)
	{
		assert(CCV_TENSOR_GET_MEMORY(tensor_symbol_info[i].info.type) == memory_type);
		assert(CCV_TENSOR_GET_DEVICE_ID(tensor_symbol_info[i].info.type) == device_id);
	}
	tensor_arena->memory_type = memory_type;
	tensor_arena->device_id = device_id;
	// Now, allocate actual buffers.
#ifdef HAVE_CUDA
	if (memory_type == CCV_TENSOR_GPU_MEMORY)
	{
		for (i = 0; i < tensor_arena->buffer_rnum; i++)
			tensor_arena->buffer[i] = (uint8_t*)cumalloc(device_id, tensor_arena->buffer_size[i]);
	} else {
		assert(memory_type == CCV_TENSOR_CPU_MEMORY);
		for (i = 0; i < tensor_arena->buffer_rnum; i++)
			ccmemalign((void **)&tensor_arena->buffer[i], 16, tensor_arena->buffer_size[i]);
	}
#else
	assert(memory_type == CCV_TENSOR_CPU_MEMORY);
	for (i = 0; i < tensor_arena->buffer_rnum; i++)
		ccmemalign((void **)&tensor_arena->buffer[i], 16, tensor_arena->buffer_size[i]);
#endif
	j = 0;
	// Assigning out the tensors (in case of sharing tensors / in-place ops).
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (TENSOR_EXPECT_COMPUTABLE(tensor_expect[i]))
		{
			tensor_arena->vt_tensor[i] = (ccv_nnc_tensor_t*)&tensor_arena->tensor[j];
			// Also, set its allocations.
			assert(assigned[i] > 0);
			// Since tensor view is bit compatible with tensor, we can just cast.
			ccv_nnc_tensor_t tensor = ccv_nnc_tensor(tensor_arena->buffer[assigned[i] - 1] + allocated_offset[i], tensor_symbol_info[i].info, 0);
			memset(tensor_arena->tensor + j, 0, sizeof(ccv_nnc_tensor_view_t));
			memcpy(tensor_arena->tensor + j, &tensor, sizeof(ccv_nnc_tensor_t));
			assert(allocated_offset[i] + (((uint64_t)CCV_GET_DATA_TYPE_SIZE(CCV_32F) * ccv_nnc_tensor_count(tensor_symbol_info[i].info) + 15) / 16 * 16) <= tensor_arena->buffer_size[assigned[i] - 1]);
			++j;
		} else // Clean it out.
			tensor_arena->vt_tensor[i] = 0;
	ccfree(allocated_offset);
	ccfree(assigned);
	for (i = 0; i < tensor_symbol_info_size; i++)
		// It could be binded tensor (or unused), in that case, it doesn't have a ref.
		if (TENSOR_EXPECT_UNASSIGNED(tensor_expect[i]) && tensor_expect[i].ref)
		{
			// It must be available.
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_expect[tensor_expect[i].ref - 1]));
			assert(tensor_arena->vt_tensor[tensor_expect[i].ref - 1]);
			tensor_arena->vt_tensor[i] = tensor_arena->vt_tensor[tensor_expect[i].ref - 1];
		}
	// Now assigning out the tensor aliases.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (TENSOR_EXPECT_ALIAS(tensor_expect[i]))
		{
			assert(tensor_symbol_info[i].alias_ref);
			int alias_ref = tensor_symbol_info[i].alias_ref - 1;
			// It referenced to is not an alias.
			assert(tensor_arena->vt_tensor[alias_ref]);
			assert(!CCV_IS_TENSOR_VIEW(tensor_arena->vt_tensor[alias_ref]));
			tensor_arena->vt_tensor[i] = (ccv_nnc_tensor_t*)&tensor_arena->tensor[j];
			// If there is no ofs, and inc is the same as dim, we take a shortcut and just init as normal tensor.
			if (memcmp(ccv_nnc_no_ofs, tensor_symbol_info[i].ofs, sizeof(ccv_nnc_no_ofs)) == 0 &&
				memcmp(tensor_symbol_info[i].inc, tensor_symbol_info[i].info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
			{
				ccv_nnc_tensor_t tensor = ccv_nnc_tensor(tensor_arena->vt_tensor[alias_ref]->data.u8, tensor_symbol_info[i].info, 0);
				memset(tensor_arena->tensor + j, 0, sizeof(ccv_nnc_tensor_view_t));
				memcpy(tensor_arena->tensor + j, &tensor, sizeof(ccv_nnc_tensor_t));
			} else {
				// Otherwise initialize a tensor view
				// 1). Simple case, if the inc is equal to original tensor, just init a tensor view.
				if (memcmp(tensor_arena->vt_tensor[alias_ref]->info.dim, tensor_symbol_info[i].inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
					tensor_arena->tensor[j] = ccv_nnc_tensor_view(tensor_arena->vt_tensor[alias_ref], tensor_symbol_info[i].ofs, tensor_symbol_info[i].info.dim);
				else {
					// Otherwise, create the tensor first, and then create the tensor view off the new tensor.
					ccv_nnc_tensor_param_t info = tensor_symbol_info[i].info;
					memcpy(info.dim, tensor_symbol_info[i].inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
					assert(ccv_nnc_tensor_count(info) <= ccv_nnc_tensor_count(tensor_arena->vt_tensor[alias_ref]->info));
					ccv_nnc_tensor_t tensor = ccv_nnc_tensor(tensor_arena->vt_tensor[alias_ref]->data.u8, info, 0);
					tensor_arena->tensor[j] = ccv_nnc_tensor_view(&tensor, tensor_symbol_info[i].ofs, tensor_symbol_info[i].info.dim);
				}
			}
			++j;
		}
	return tensor_arena;
}

static void _ccv_nnc_tensor_expect_add_exec(const ccv_sparse_matrix_t* exec_dep, const int idx, ccv_nnc_tensor_expect_t tensor_expect)
{
	int i, found = 0;
	// Try to insert head.
	ccv_array_t* head = tensor_expect.head;
	for (i = 0; i < head->rnum;)
	{
		ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, *(int*)ccv_array_get(head, i), idx);
		if (cell.i32 && cell.i32[0] > 0)
		{
			/* If the current node is the parent of the head node, check if we found it or not. */
			/* If not found, replace the current one. */
			if (!found)
			{
				found = 1;
				*(int*)ccv_array_get(head, i) = idx;
			} else {
				/* Remove the current one, change the rnum. */
				if (i < head->rnum - 1)
					*(int*)ccv_array_get(head, i) = *(int*)ccv_array_get(head, head->rnum - 1);
				--head->rnum;
				continue;
			}
		} else {
			// If the head is the parent of the idx, we cannot add it to the array (it is deterministically later than head).
			cell = ccv_get_sparse_matrix_cell(exec_dep, idx, *(int*)ccv_array_get(head, i));
			if (cell.i32 && cell.i32[0] > 0)
			{
				found = 1;
				break;
			}
		}
		/* Advancing i. */
		++i;
	}
	/* If not found, push this idx to the end of the array. */
	if (!found)
		ccv_array_push(head, &idx);
	// Try to insert tail.
	found = 0;
	ccv_array_t* tail = tensor_expect.tail;
	for (i = 0; i < tail->rnum;)
	{
		ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, idx, *(int*)ccv_array_get(tail, i));
		if (cell.i32 && cell.i32[0] > 0)
		{
			/* If the current node is the child of the tail node, check if we found it or not. */
			/* If not found, replace the current one. */
			if (!found)
			{
				found = 1;
				*(int*)ccv_array_get(tail, i) = idx;
			} else {
				/* Remove the current one, change the rnum. */
				*(int*)ccv_array_get(tail, i) = *(int*)ccv_array_get(tail, tail->rnum - 1);
				--tail->rnum;
				continue;
			}
		} else {
			// If the tail is the child of the idx, we cannot add it to the array (it is deterministically earlier than tail).
			cell = ccv_get_sparse_matrix_cell(exec_dep, *(int*)ccv_array_get(tail, i), idx);
			if (cell.i32 && cell.i32[0] > 0)
			{
				found = 1;
				break;
			}
		}
		/* Advancing i. */
		++i;
	}
	/* If not found, push this idx to the end of the array. */
	if (!found)
		ccv_array_push(tail, &idx);
}

static void _ccv_nnc_symbolic_graph_auto_symbols(const ccv_nnc_symbolic_graph_t* symbolic_graph, const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info)
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
			exec_symbol_info[i].input_size > 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].inputs[0]].info) &&
			exec_symbol_info[i].output_size > 0 && !ccv_nnc_is_tensor_auto(tensor_symbol_info[exec_symbol_info[i].outputs[0]].info))
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
				input_params[i] = tensor_symbol_info[node->inputs[i]].info; \
			ccv_nnc_hint_tensor_auto(node->cmd, input_params, node->input_size, node->hint, output_params, node->output_size); \
			for (i = 0; i < node->output_size; i++) \
				/* Only assign the output parameters if the symbol itself is auto. */ \
				if (ccv_nnc_is_tensor_auto(tensor_symbol_info[node->outputs[i]].info)) \
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

ccv_nnc_tensor_t* ccv_nnc_tensor_from_symbol(const ccv_nnc_tensor_arena_t* tensor_arena, const ccv_nnc_tensor_symbol_t symbol)
{
	assert(symbol.d >= 0 && symbol.d < tensor_arena->vt_tensor_rnum);
	return tensor_arena->vt_tensor[symbol.d];
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_from_symbol(const ccv_nnc_graph_exec_arena_t* graph_exec_arena, const ccv_nnc_graph_exec_symbol_t symbol)
{
	assert(symbol.d >= 0 && symbol.d < graph_exec_arena->graph_exec_rnum);
	return graph_exec_arena->graph_exec[symbol.d];
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_source(const ccv_nnc_graph_exec_arena_t* graph_exec_arena)
{
	return graph_exec_arena->source;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_destination(const ccv_nnc_graph_exec_arena_t* graph_exec_arena)
{
	return graph_exec_arena->destination;
}

void ccv_nnc_symbolic_graph_compile(const ccv_nnc_symbolic_graph_t* symbolic_graph, const ccv_nnc_tensor_bind_t* tensor_binds, const int tensor_bind_size, const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, ccv_nnc_graph_t** graph_ref, ccv_nnc_tensor_arena_t** tensor_arena_ref, ccv_nnc_graph_exec_arena_t** graph_exec_arena_ref)
{
	assert(graph_ref);
	assert(tensor_arena_ref);
	assert(graph_exec_arena_ref);
	assert(source_size > 0);
	assert(destination_size > 0);
	// First, fill all the "auto" holes.
	// This is the symbol table that with "auto" info filled up.
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * symbolic_graph->tensor_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * symbolic_graph->exec_symbol_info->rnum);
	_ccv_nnc_symbolic_graph_auto_symbols(symbolic_graph, sources, source_size, destinations, destination_size, tensor_symbol_info, exec_symbol_info);

	int i, j, k;
	// Generate exec dependencies (or, in other words, partial ordering of executions).
	ccv_sparse_matrix_t* exec_dep = ccv_sparse_matrix_new(symbolic_graph->exec_symbol_info->rnum, symbolic_graph->exec_symbol_info->rnum, CCV_32S | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	int* buf = (int*)ccmalloc(sizeof(int) * symbolic_graph->exec_symbol_info->rnum * 2);
	int buf_size;
#define for_block(x, val) \
	do { \
		if (((int32_t*)val)[0] > 0) \
		{ \
			buf[buf_size * 2] = x; \
			buf[buf_size * 2 + 1] = ((int32_t*)val)[0] + 1; \
			++buf_size; \
		} \
	} while (0)
#define visitor(node, idx, _, term) \
	do { \
		buf_size = 0; /* save all its parent deps to this buffer */ \
		ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx); \
		if (vector) \
			CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block); \
		if (!node->outgoings) \
			break; \
		for (i = 0; i < node->outgoings->rnum; i++) \
		{ \
			int outgoing = *(int*)ccv_array_get(node->outgoings, i); \
			const int32_t one = 1; \
			ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, outgoing, idx); \
			/* If not found, set, if the current node is the destination node, no need to
			 * set itself as parent of subsequent nodes because its terminal nature. */ \
			if (!term && (!cell.i32 || cell.i32[0] == 0)) \
				ccv_set_sparse_matrix_cell(exec_dep, outgoing, idx, &one); \
			for (j = 0; j < buf_size; j++) /* set with all idx's dependencies as well */ \
			{ \
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, outgoing, buf[j * 2]); \
 				/* If not found, set */ \
				if (!cell.i32 || cell.i32[0] == 0) \
					ccv_set_sparse_matrix_cell(exec_dep, outgoing, buf[j * 2], &buf[j * 2 + 1]); \
				else { \
					/* Otherwise, set to the longest one */ \
					int32_t dep = ccv_max(cell.i32[0], buf[j * 2 + 1]); \
					ccv_set_sparse_matrix_cell(exec_dep, outgoing, buf[j * 2], &dep); \
				} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef for_block
#undef visitor
	ccfree(buf);
	// This struct is allocated earlier to collect information about the tensor's expected start / end execs.
	ccv_nnc_tensor_expect_t* tensor_expect = (ccv_nnc_tensor_expect_t*)cccalloc(symbolic_graph->tensor_symbol_info->rnum, sizeof(ccv_nnc_tensor_expect_t));
	// The reason is that I need to make everyone of them to be unassigned unless it is used somewhere. It
	// happens that I have to loop through all relevant node to find out if one is used or not.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		tensor_expect[i].flag = UNASSIGNED;
#define visitor(node, idx, ...) \
	do { \
		for (i = 0; i < node->input_size; i++) \
			tensor_expect[node->inputs[i]].flag = 0; \
		for (i = 0; i < node->output_size; i++) \
			tensor_expect[node->outputs[i]].flag = 0; \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Ignore tensors that are already binded, no matter if it is used or not.
	for (i = 0; i < tensor_bind_size; i++)
		tensor_expect[tensor_binds[i].symbol.d].flag = UNASSIGNED;
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		// Check no tensor info is auto now.
		assert(!ccv_nnc_is_tensor_auto(tensor_symbol_info[i].info));
		if (tensor_symbol_info[i].alias_ref)
		{
			// An alias cannot ref to another alias.
			assert(!tensor_symbol_info[tensor_symbol_info[i].alias_ref - 1].alias_ref);
			tensor_expect[i].flag = ALIAS;
		}
		// If this tensor is not expected to be unassigned, allocate the arrays for s and t.
		if (TENSOR_EXPECT_COMPUTABLE(tensor_expect[i]))
		{
			tensor_expect[i].head = ccv_array_new(sizeof(int), 0, 0);
			tensor_expect[i].tail = ccv_array_new(sizeof(int), 0, 0);
		}
	}
	// Collect head nodes and tail nodes for each tensor.
#define visitor(node, idx, ...) \
	do { \
		for (i = 0; i < node->input_size; i++) \
		{ \
			int d = node->inputs[i]; \
			if (TENSOR_EXPECT_ALIAS(tensor_expect[d])) \
				d = tensor_symbol_info[d].alias_ref - 1; \
			if (TENSOR_EXPECT_UNASSIGNED(tensor_expect[d])) \
				continue; \
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_expect[d])); \
			if (tensor_expect[d].head->rnum == 0) \
				tensor_expect[d].flag = CONST_TENSOR; \
			else \
				_ccv_nnc_tensor_expect_add_exec(exec_dep, idx, tensor_expect[d]); \
		} \
		for (i = 0; i < node->output_size; i++) \
		{ \
			int d = node->outputs[i]; \
			if (TENSOR_EXPECT_ALIAS(tensor_expect[d])) \
				d = tensor_symbol_info[d].alias_ref - 1; \
			/* If it is recognized as a const tensor, we can find it in the output pool because it may be in a RNN. */ \
			if (TENSOR_EXPECT_CONST(tensor_expect[d]) || \
				TENSOR_EXPECT_UNASSIGNED(tensor_expect[d])) \
				continue; \
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_expect[d])); \
			_ccv_nnc_tensor_expect_add_exec(exec_dep, idx, tensor_expect[d]); \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
#define visitor(node, idx, ...) \
	do { \
		/* Remove tensor symbols that is for in-place operations (and it matches the start, end tensor). */ \
		if (ccv_nnc_cmd_attr(node->cmd, CCV_NNC_CMD_ATTR_INPLACE)) \
		{ \
			int x, y; \
			for (x = 0; x < node->input_size; x++) \
			{ \
				/* If the input is not assigned, it can be referenced, find the referenced one */ \
				int ref = node->inputs[x]; \
				while (!TENSOR_EXPECT_COMPUTABLE(tensor_expect[ref]) && tensor_expect[ref].ref) \
					ref = tensor_expect[ref].ref - 1; \
				const ccv_nnc_tensor_symbol_info_t x_symbol = tensor_symbol_info[ref]; \
				if (!TENSOR_EXPECT_CONST(tensor_expect[ref]) && \
					TENSOR_EXPECT_COMPUTABLE(tensor_expect[ref]) && \
					tensor_expect[ref].tail->rnum == 1) \
					for (y = 0; y < node->output_size; y++) \
						/* Only proceed if the input symbol is different from the output symbol, */ \
						/* and the input symbol meets the output symbol exactly at the same spot. */ \
						if (ref != node->outputs[y] && \
							!TENSOR_EXPECT_CONST(tensor_expect[node->outputs[y]]) && \
							TENSOR_EXPECT_COMPUTABLE(tensor_expect[node->outputs[y]]) && \
							tensor_expect[node->outputs[y]].head->rnum == 1 && \
							*(int*)ccv_array_get(tensor_expect[ref].tail, 0) == *(int*)ccv_array_get(tensor_expect[node->outputs[y]].head, 0)) \
						{ \
							const ccv_nnc_tensor_symbol_info_t y_symbol = tensor_symbol_info[node->outputs[y]]; \
							/* If dimension matches perfectly, then we can assign y_symbol to x. */ \
							if (memcmp(x_symbol.info.dim, y_symbol.info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0) \
							{ \
								ccv_array_free(tensor_expect[ref].tail); \
								tensor_expect[ref].tail = tensor_expect[node->outputs[y]].tail; \
								/* Mark the original as unassigned, set its reference to the head of the current node. */ \
								ccv_array_free(tensor_expect[node->outputs[y]].head); \
								tensor_expect[node->outputs[y]].flag = UNASSIGNED; \
								tensor_expect[node->outputs[y]].ref = ref + 1; \
								tensor_expect[node->outputs[y]].head = 0; \
								tensor_expect[node->outputs[y]].tail = 0; \
							} \
						} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor

	// Now, everything is prepared, tensor life is analyzed, inplace operations are collapsed, all tensor symbols and hints
	// are automatically filled in. It is time to guess what's the best tensor placement and create the opaque tensor arena.
	// The alloc_dep will return the allocation dependencies, thus, which tensor is reused to the existing tensor.
	ccv_array_t** alloc_dep = (ccv_array_t**)cccalloc(symbolic_graph->tensor_symbol_info->rnum, sizeof(ccv_array_t*));
	ccv_nnc_tensor_arena_t* tensor_arena = _ccv_nnc_tensor_arena_new(tensor_symbol_info, symbolic_graph->tensor_symbol_info->rnum, exec_dep, tensor_expect, alloc_dep);
	// Handle binded tensors.
	for (i = 0; i < tensor_bind_size; i++)
	{
		// For binded tensors, it shouldn't be assigned yet.
		assert(tensor_arena->vt_tensor[tensor_binds[i].symbol.d] == 0);
		// I have to cast this, unfortunately.
		tensor_arena->vt_tensor[tensor_binds[i].symbol.d] = (ccv_nnc_tensor_t*)tensor_binds[i].tensor;
	}
	*tensor_arena_ref = tensor_arena;

	ccv_matrix_free(exec_dep);

	// The above handled tensor allocation, now we need to materialize the graph from symbolic to real.
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	*graph_ref = graph;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = (ccv_nnc_graph_exec_arena_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_arena_t) + sizeof(ccv_nnc_graph_exec_t) * (symbolic_graph->exec_symbol_info->rnum - 1));
	graph_exec_arena->graph_exec_rnum = symbolic_graph->exec_symbol_info->rnum;
	*graph_exec_arena_ref = graph_exec_arena;
	ccv_nnc_graph_exec_t* graph_exec = graph_exec_arena->graph_exec;
	int max_input_size = 0, max_output_size = 0;
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
	{
		max_input_size = ccv_max(max_input_size, exec_symbol_info[i].input_size);
		max_output_size = ccv_max(max_input_size, exec_symbol_info[i].output_size);
		graph_exec[i].graph = 0;
	}
	ccv_nnc_tensor_t** max_inputs = max_input_size > 0 ? (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_input_size) : 0;
	ccv_nnc_tensor_t** max_outputs = max_output_size > 0 ? (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_output_size) : 0;
#define visitor(node, idx, ...) \
	do { \
		if (CCV_NO_GRAPH_EXEC(graph_exec[idx])) \
		{ \
			for (i = 0; i < node->input_size; i++) \
				max_inputs[i] = tensor_arena->vt_tensor[node->inputs[i]]; \
			for (i = 0; i < node->output_size; i++) \
				max_outputs[i] = tensor_arena->vt_tensor[node->outputs[i]]; \
			graph_exec[idx] = ccv_nnc_graph_exec_new(graph, node->cmd, node->hint, max_inputs, node->input_size, max_outputs, node->output_size); \
		} \
		if (!node->outgoings) \
			break; \
		for (i = 0; i < node->outgoings->rnum; i++) \
		{ \
			int outgoing = *(int*)ccv_array_get(node->outgoings, i); \
			if (CCV_NO_GRAPH_EXEC(graph_exec[outgoing])) \
			{ \
				ccv_nnc_graph_exec_symbol_info_t* outgoing_node = exec_symbol_info + outgoing; \
				for (j = 0; j < outgoing_node->input_size; j++) \
					max_inputs[j] = tensor_arena->vt_tensor[outgoing_node->inputs[j]]; \
				for (j = 0; j < outgoing_node->output_size; j++) \
					max_outputs[j] = tensor_arena->vt_tensor[outgoing_node->outputs[j]]; \
				graph_exec[outgoing] = ccv_nnc_graph_exec_new(graph, outgoing_node->cmd, outgoing_node->hint, max_inputs, outgoing_node->input_size, max_outputs, outgoing_node->output_size); \
			} \
			ccv_nnc_graph_exec_concat(graph, graph_exec[idx], graph_exec[outgoing]); \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	int source_exec_created = 0;
	// After the graph is materialized, we need to handle the case that some of these tensors require to be initialized to zero before use.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		if (tensor_symbol_info[i].flags & CCV_NNC_SYM_TENSOR_INIT_ZEROS)
		{
			int ref = i;
			while (tensor_symbol_info[ref].alias_ref)
				ref = tensor_symbol_info[ref].alias_ref - 1;
			while (!TENSOR_EXPECT_COMPUTABLE(tensor_expect[ref]) && tensor_expect[ref].ref)
				ref = tensor_expect[ref].ref - 1;
			// This is not computable. It could be that we marked a const tensor as init zero.
			if (!TENSOR_EXPECT_COMPUTABLE(tensor_expect[ref]))
				continue;
			// If this tensor is not used by any exec, we don't need to init at all. Skip.
			if (!tensor_expect[ref].head || tensor_expect[ref].head->rnum == 0)
				continue;
			ccv_nnc_tensor_t* tensor = tensor_arena->vt_tensor[ref];
			// Now, we have the original tensor, we can get the actual tensor, and construct the set command.
			ccv_nnc_graph_exec_t set_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_SET_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, &tensor, 1);
			for (j = 0; j < tensor_expect[ref].head->rnum; j++)
			{
				const int outgoing = *(int*)ccv_array_get(tensor_expect[ref].head, j);
				ccv_nnc_graph_exec_concat(graph, set_exec, graph_exec[outgoing]);
			}
			int flag = 0;
			if (alloc_dep[ref])
				for (j = 0; j < alloc_dep[ref]->rnum; j++)
				{
					const int d = *(int*)ccv_array_get(alloc_dep[ref], j);
					// This is from alloc_dep, it should be computable.
					assert(TENSOR_EXPECT_COMPUTABLE(tensor_expect[d]));
					if (tensor_expect[d].tail)
						for (k = 0; k < tensor_expect[d].tail->rnum; k++)
						{
							const int incoming = *(int*)ccv_array_get(tensor_expect[d].tail, j);
							ccv_nnc_graph_exec_concat(graph, graph_exec[incoming], set_exec);
							flag = 1;
						}
				}
			// If cannot find a start node for this exec, we need to append it to the no-op of the start.
			if (!flag)
			{
				if (!source_exec_created)
				{
					graph_exec_arena->source = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
					source_exec_created = 1;
				}
				ccv_nnc_graph_exec_concat(graph, graph_exec_arena->source, set_exec);
			}
		}
	}
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		if (tensor_expect[i].head)
			ccv_array_free(tensor_expect[i].head);
		if (tensor_expect[i].tail)
			ccv_array_free(tensor_expect[i].tail);
		if (alloc_dep[i])
			ccv_array_free(alloc_dep[i]);
	}
	ccfree(alloc_dep);
	ccfree(tensor_expect);
	ccfree(tensor_symbol_info);
	if (max_inputs)
		ccfree(max_inputs);
	if (max_outputs)
		ccfree(max_outputs);
	ccfree(exec_symbol_info);
	// Create source / destination phony node. This is to facilitate use of compiled graph.
	// Also, this is needed if you have init zero execs.
	if (source_exec_created || source_size > 1)
	{
		if (!source_exec_created)
			graph_exec_arena->source = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
		for (i = 0; i < source_size; i++)
			ccv_nnc_graph_exec_concat(graph, graph_exec_arena->source, graph_exec[sources[i].d]);
	} else {
		assert(!source_exec_created);
		assert(source_size == 1);
		graph_exec_arena->source = graph_exec[sources[0].d];
	}
	if (destination_size == 1)
		graph_exec_arena->destination = graph_exec[destinations[0].d];
	else {
		graph_exec_arena->destination = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
		for (i = 0; i < destination_size; i++)
			ccv_nnc_graph_exec_concat(graph, graph_exec[destinations[i].d], graph_exec_arena->destination);
	}
}

void ccv_nnc_symbolic_graph_free(ccv_nnc_symbolic_graph_t* graph)
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

void ccv_nnc_tensor_arena_free(ccv_nnc_tensor_arena_t* tensor_arena)
{
	int i;
#ifdef HAVE_CUDA
	if (tensor_arena->memory_type == CCV_TENSOR_GPU_MEMORY)
	{
		for (i = 0; i < tensor_arena->buffer_rnum; i++)
			cufree(tensor_arena->device_id, tensor_arena->buffer[i]);
	} else {
		assert(tensor_arena->memory_type == CCV_TENSOR_CPU_MEMORY);
		for (i = 0; i < tensor_arena->buffer_rnum; i++)
			ccfree(tensor_arena->buffer[i]);
	}
#else
	assert(tensor_arena->memory_type == CCV_TENSOR_CPU_MEMORY);
	for (i = 0; i < tensor_arena->buffer_rnum; i++)
		ccfree(tensor_arena->buffer[i]);
#endif
	ccfree(tensor_arena);
}

void ccv_nnc_graph_exec_arena_free(ccv_nnc_graph_exec_arena_t* graph_exec_arena)
{
	ccfree(graph_exec_arena);
}

/**
 * Level-4 API
 */

typedef struct {
	int f_wrt; // Check if both f_symbols and wrt_symbols flow through this node.
	ccv_array_t* outgoings; // backward traverse nodes.
} ccv_nnc_graph_backward_info_t;

typedef struct {
	int input_size;
	int* inputs;
	int output;
	ccv_array_t* outgoings;
	float value;
	ccv_nnc_graph_exec_symbol_t symbol;
} ccv_nnc_graph_sum_or_set_exec_t;

typedef struct {
	int input_size;
	int output_size;
	int* inputs;
	int* outputs;
	ccv_array_t* outgoings;
	ccv_nnc_cmd_t cmd;
	ccv_nnc_graph_exec_symbol_t symbol;
} ccv_nnc_graph_autograd_exec_t;

typedef struct {
	int d; // The pointer to the forward level object.
	int alias_ref; // The alias ref to itself (autograd_tensor_symbol array).
	int flags; // Flags for this symbol.
	ccv_nnc_tensor_symbol_t symbol;
} ccv_nnc_autograd_tensor_symbol_t;

typedef struct {
	int d; // The tensor symbol ref.
	int x; // The exec symbol ref.
	ccv_array_t* exec_registry; // Additional exec symbol refs, similar to x, only useful for aliasing.
	ccv_array_t* alias_registry; // int point to all the alias (if this is not an alias). The alias is the object in autograd_tensor_symbol, you need another level of indirection to get the actual forward level alias.
} ccv_nnc_tensor_ref_t;

typedef struct {
	int c; // The start non-accumulated version.
	ccv_array_t* ref_version; // tensor ref point to the reverse tensor symbol.
} ccv_nnc_autograd_tensor_version_t;

typedef struct {
	int d;
	int alias_ref;
} ccv_nnc_sum_variable_t;

// This method tries to figure out if a set of aliases can cover the whole tensor dim.
// This is not a precise implementation though. The requirement is to answer this question
// with a given memory constraint, therefore, only allow up to 65536 different tensor locations.
// If you have more than that, it will assume that it doesn't have fully assigned aliases,
// and will return 0.

// Return 1 if inserted successfully.
static inline int _ccv_nnc_try_mix(int* md, const int ins, const int c)
{
	if (!c)
	{
		md[0] = ins;
		return 1;
	}
	int ll = 0, uu = c - 1;
	int mm;
	do {
		mm = ll + ((uu - ll) >> 1);
		if (ins == md[mm])
			return 0;
		else if (ins < md[mm])
			uu = mm - 1;
		else if (ins > md[mm])
			ll = mm + 1;
	} while (ll <= uu);
	if (ll < c)
		memmove(md + ll + 1, md + ll, sizeof(int) * (c - ll));
	md[ll] = ins;
	return 1;
}

static inline int _ccv_nnc_mix_idx(const int* md, const int ins, const int c)
{
	if (c <= 1)
		return 0;
	int ll = 0, uu = c - 1;
	int mm;
	do {
		mm = ll + ((uu - ll) >> 1);
		if (ins == md[mm])
			return mm;
		else if (ins < md[mm])
			uu = mm - 1;
		else if (ins > md[mm])
			ll = mm + 1;
	} while (ll <= uu);
	assert(0 && "Shouldn't reach here");
	return -1;
}

static inline void _ccv_nnc_try_set_pix_0(const int* ofs, const int* dim, const int* tensor_dim, int* const* scmd, const int* cube_dim, const int* cube_step, uint8_t* cube, int offset)
{
	const int s = (ofs[0] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[0], ofs[0], cube_dim[0]) + 1;
	const int d = ((ofs[0] + dim[0] == tensor_dim[0]) ? cube_dim[0] : _ccv_nnc_mix_idx(scmd[0], ofs[0] + ccv_max(1, dim[0]), cube_dim[0])) + 1;
	assert(s >= 0 && d > s);
	int i;
	for (i = s; i < d; i++)
		// Fill this pix. I can make this faster by loop through full ones (divided by 8), but too lazy.
		cube[(offset + i) >> 3] |= (1 << ((offset + i) & 0x7));
}

static inline void _ccv_nnc_try_set_pix_1(const int* ofs, const int* dim, const int* tensor_dim, int* const* scmd, const int* cube_dim, const int* cube_step, uint8_t* cube, int offset)
{
	const int s0 = (ofs[0] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[0], ofs[0], cube_dim[0]) + 1;
	const int d0 = ((ofs[0] + dim[0] == tensor_dim[0]) ? cube_dim[0] : _ccv_nnc_mix_idx(scmd[0], ofs[0] + ccv_max(1, dim[0]), cube_dim[0])) + 1;
	assert(s0 >= 0 && d0 > s0);
	const int s1 = (ofs[1] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[1], ofs[1], cube_dim[1]) + 1;
	const int d1 = ((ofs[1] + dim[1] == tensor_dim[1]) ? cube_dim[1] : _ccv_nnc_mix_idx(scmd[1], ofs[1] + ccv_max(1, dim[1]), cube_dim[1])) + 1;
	assert(s1 >= 0 && d1 > s1);
	int i, j;
	const int step1 = cube_step[1];
	if (step1 == d0 - s0)
	{
		// Faster one, we can simply loop through.
		const int len = d1 + (d1 - s1) * step1;
		for (i = s1; i < len; i++)
			cube[(offset + i) >> 3] |= (1 << ((offset + i) & 0x7));
	} else {
		// There are gaps, slow one.
		for (i = s1; i < d1; i++, offset += step1)
			for (j = s0; j < d0; j++)
				cube[(offset + j) >> 3] |= (1 << ((offset + j) & 0x7));
	}
}

static inline void _ccv_nnc_try_set_pix(const int* ofs, const int* dim, const int* tensor_dim, int* const* scmd, const int* cube_dim, const int* cube_step, uint8_t* cube, int offset, const int dim_idx)
{
	switch (dim_idx)
	{
		case 1:
			_ccv_nnc_try_set_pix_1(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, offset);
			return;
		case 0:
			_ccv_nnc_try_set_pix_0(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, offset);
			return;
	}
	int i;
	const int s = (ofs[dim_idx] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[dim_idx], ofs[dim_idx], cube_dim[dim_idx]) + 1;
	const int d = ((ofs[dim_idx] + dim[dim_idx] == tensor_dim[dim_idx]) ? cube_dim[dim_idx] : _ccv_nnc_mix_idx(scmd[dim_idx], ofs[dim_idx] + ccv_max(1, dim[dim_idx]), cube_dim[dim_idx])) + 1;
	assert(s >= 0 && d > s);
	for (i = s; i < d; i++)
		_ccv_nnc_try_set_pix(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, offset + i * cube_step[dim_idx], dim_idx - 1);
}

static int _ccv_nnc_tensor_ref_fully_assigned_with_aliases(const ccv_nnc_tensor_ref_t* tensor_ref, const ccv_array_t* autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info)
{
	// Only work with tensor_ref of aliases.
	assert(tensor_ref->alias_registry);
	const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
	assert(tensor_symbol_info[autograd->d].alias_ref == 0);
	const int* tensor_dim = tensor_symbol_info[autograd->d].info.dim;
	int i, j;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		if (memcmp(inc, tensor_dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0)
			return 0;
	}
	/* We need a solid cube (potentially hyper dimensional) to compute if there are overlaps.
	 * To make this cube as small as possible, we need to map the actual tensor dimension
	 * (therefore, we don't actually allocate the whole tensor to compute overlaps) to a smaller
	 * cube given the ofs and dim size of its aliases.
	 *
	 * The following code generated the dimension mapping (using scratch space) with binary search + insertion
	 * and then we fill the cube with a given tensor alias's dimensional information (ofs, dim).
	 * Afterwards, we simply need to check if the cube is totally filled up to know if this tensor
	 * is fully assigned with its aliases (if that is the case, we can skip zeroing for this tensor).
	 *
	 * There are several restrictions though to make this faster: 1). I cannot handle any cube that all side
	 * lengths combined larger than 1023 (scm only have 1024 scratch space). 2). I cannot handle any cube
	 * that the total volume is larger than 2048 * 8 (I only allocate 2K on stack for this).
	 * */
	int scm[1024]; // Having 1024 int scratch space for mapping dimensions. (Or sparse coordinate mapping).
	int cube_dim[CCV_NNC_MAX_DIM_ALLOC] = {0}; // Mapping dimension size.
	int cube_size = 1;
	int* scmptr = scm;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && tensor_dim[i]; i++)
	{
		int head = 0, tail = 0; // Note that we touched both the head and tail (otherwise this dimension is not fully covered).
		int len = 0;
		for (j = 0; j < tensor_ref->alias_registry->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, j);
			assert(d < autograd_tensor_symbol->rnum);
			const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
			assert(tensor_symbol_info[autograd->d].alias_ref);
			const int* ofs = tensor_symbol_info[autograd->d].ofs;
			const int* dim = tensor_symbol_info[autograd->d].info.dim;
			head = head || (ofs[i] == 0);
			tail = tail || (ofs[i] + ccv_max(1, dim[i]) == tensor_dim[i]);
			if (ofs[i] != 0)
				len += _ccv_nnc_try_mix(scmptr, ofs[i], len);
			if (scmptr - scm + len >= 1024) // Cannot handle that much, abort.
				return 0;
			if (ofs[i] + ccv_max(1, dim[i]) < tensor_dim[i])
				len += _ccv_nnc_try_mix(scmptr, ofs[i] + ccv_max(1, dim[i]), len);
			if (scmptr - scm + len >= 1024) // Cannot handle that much, abort.
				return 0;
		}
		if (!head || !tail)
			return 0;
		cube_size *= (len + 1);
		cube_dim[i] = len;
		scmptr += len; // Moving to next level.
	}
	// The cube map is too large, cannot do the computation, assume it is not fully assigned.
	if (cube_size > 2048 * 8)
		return 0;
	// binary map to see if it fills up.
	uint8_t* cube = (uint8_t*)alloca(sizeof(uint8_t) * ((cube_size + 7) >> 3));
	memset(cube, 0, sizeof(uint8_t) * ((cube_size + 7) >> 3));
	int* scmd[CCV_NNC_MAX_DIM_ALLOC] = {0}; // Sparse coordinate map at dimension x.
	int cube_step[CCV_NNC_MAX_DIM_ALLOC] = {0};
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && tensor_dim[i]; i++)
	{
		cube_step[i] = (i > 0) ? cube_step[i - 1] * (cube_dim[i - 1] + 1) : 1;
		scmd[i] = (i > 0) ? scmd[i - 1] + cube_dim[i - 1] : scm;
	}
	const int max_dim = i;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		_ccv_nnc_try_set_pix(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, 0, max_dim - 1);
	}
	// Compare to see now if the binary map filled up. If it filled up, we know it is fully assigned.
	for (i = 0; i < (cube_size >> 3); i++)
		if (cube[i] < 0xff)
			return 0;
	if ((cube_size & 0x7) > 0)
	{
		// Fetch the rest.
		uint8_t r = 0;
		for (i = 0; i < (cube_size & 0x7); i++)
			r |= (1 << i);
		assert(cube[((cube_size + 7) >> 3) - 1] <= r);
		if (cube[((cube_size + 7) >> 3) - 1] < r)
			return 0;
	}
	return 1;
}

static void _ccv_nnc_graph_sum_autograd_tensor_versions(const int idx, const int d, const int exec_symbol_size, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, ccv_nnc_autograd_tensor_version_t* tensor_ver, ccv_nnc_graph_autograd_exec_t* autograd_exec, ccv_array_t* autograd_tensor_symbol, ccv_array_t* sum_or_set_exec)
{
	int i, j;
	assert(tensor_ver->c < tensor_ver->ref_version->rnum);
	const int input_size = tensor_ver->ref_version->rnum - tensor_ver->c;
	int* inputs = (int*)ccmalloc(sizeof(int) * input_size);
	for (i = tensor_ver->c; i < tensor_ver->ref_version->rnum; i++)
		inputs[i] = ((ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i))->d;
	ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0};
	tensor_sym.d = d;
	ccv_array_push(autograd_tensor_symbol, &tensor_sym);
	ccv_nnc_graph_sum_or_set_exec_t sum_exec = {
		.input_size = input_size,
		.inputs = inputs,
		.output = autograd_tensor_symbol->rnum - 1
	};
	if (idx >= 0)
	{
		sum_exec.outgoings = ccv_array_new(sizeof(int), 1, 0);
		ccv_array_push(sum_exec.outgoings, &idx);
	}
	ccv_array_push(sum_or_set_exec, &sum_exec);
	const int outgoing = exec_symbol_size + sum_or_set_exec->rnum - 1;
	for (i = tensor_ver->c; i < tensor_ver->ref_version->rnum; i++)
	{
		const ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i);
		const int x = tensor_ref->x;
		assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
		if (x < exec_symbol_size)
		{
			ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
			if (!back_exec->outgoings)
				back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
			_ccv_array_replace_or_insert_int(back_exec->outgoings, idx, outgoing);
			// If this tensor have associated alias, we need to init it to zeros when it is allocated (we only need to set a flag here)
			// it is handled at compilation phase.
			if (tensor_ref->alias_registry &&
				// Loop over to see if this tensor is fully occupied to avoid extra zero step.
				!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info))
			{
				ccv_nnc_autograd_tensor_symbol_t* tensor_sym = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
				// By having alias_registry, what this symbol represents must not by an alias.
				assert(tensor_sym->alias_ref == 0);
				tensor_sym->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
			}
		} else {
			// This tensor_ref cannot be referenced by aliases at all (because it is generated by the sum operation, therefore, it is presented as a full in the first place).
			assert(!tensor_ref->alias_registry);
			ccv_nnc_graph_sum_or_set_exec_t* sum_or_set = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size);
			_ccv_array_replace_or_insert_int(sum_or_set->outgoings, idx, outgoing);
		}
		if (tensor_ref->exec_registry)
			for (j = 0; j < tensor_ref->exec_registry->rnum; j++)
			{
				const int x = *(int*)ccv_array_get(tensor_ref->exec_registry, j);
				assert(x >= 0);
				// The exec_registry can only be generated by alias registry, therefore, it cannot reference to a sum operation.
				assert(x < exec_symbol_size);
				ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
				if (!back_exec->outgoings)
					back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
				_ccv_array_replace_or_insert_int(back_exec->outgoings, idx, outgoing);
			}
	}
	const ccv_nnc_tensor_ref_t tensor_ref = {
		.d = autograd_tensor_symbol->rnum - 1,
		.x = outgoing
	};
	ccv_array_push(tensor_ver->ref_version, &tensor_ref);
	/* Move the c pointer up to the latest summed result. */
	tensor_ver->c = tensor_ver->ref_version->rnum - 1;
}

static int _ccv_nnc_tensor_ref_version_involve_alias(const ccv_nnc_tensor_ref_t* tensor_ref, const ccv_array_t* autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const ccv_nnc_tensor_symbol_info_t* alias)
{
	assert(alias->alias_ref > 0);
	// No alias_registry, must conflict (owns the whole band).
	if (!tensor_ref->alias_registry)
		return 1;
	int i, j;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		// This must reference to an alias.
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		// Only can compare if the inc is the same, otherwise, we can only assume it overlaps.
		if (memcmp(inc, alias->inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0)
			return 1;
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		int none = 0;
		for (j = 0; j < CCV_NNC_MAX_DIM_ALLOC && dim[j] && alias->info.dim[j]; j++)
			if (ccv_min(ofs[j] + dim[j], alias->ofs[j] + alias->info.dim[j]) <= ccv_max(ofs[j], alias->ofs[j]))
			{
				none = 1;
				break;
			}
		// If it overlaps with this alias, nope.
		if (!none)
			return 1;
	}
	// All aliases referenced by this ref_version doesn't overlap with the provided one, thus, there is no conflict at all.
	return 0;
}

static int _ccv_nnc_tensor_ref_version_find_alias(const ccv_nnc_tensor_ref_t* tensor_ref, const ccv_array_t* autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const ccv_nnc_tensor_symbol_info_t* alias)
{
	assert(alias->alias_ref > 0);
	// No alias_registry, thus, cannot find the exact matched alias.
	if (!tensor_ref->alias_registry)
		return -1;
	int i;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		// This must reference to an alias.
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		// If everything matches, this is the required alias.
		if (memcmp(inc, alias->inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0 &&
			memcmp(ofs, alias->ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0 &&
			memcmp(dim, alias->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
			return d;
	}
	return -1;
}

static int _ccv_nnc_tensor_ref_version_has_this_alias_exclusively(const ccv_nnc_tensor_ref_t* tensor_ref, const ccv_array_t* autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const ccv_nnc_tensor_symbol_info_t* alias)
{
	assert(alias->alias_ref > 0);
	// No alias_registry, thus, cannot find the exact matched alias.
	if (!tensor_ref->alias_registry)
		return 0;
	int i;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		// This must reference to an alias.
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		if (memcmp(inc, alias->inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0 ||
			memcmp(ofs, alias->ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0 ||
			memcmp(dim, alias->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0)
			return 0;
	}
	// If everything matches for every alias in registry, we can use any of the alias directly.
	return 1;
}

static int _ccv_nnc_graph_sum_autograd_tensor_versions_alias(const int idx, const int d, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const int exec_symbol_size, const ccv_nnc_tensor_symbol_info_t* alias, ccv_nnc_autograd_tensor_version_t* tensor_ver, ccv_nnc_graph_autograd_exec_t* autograd_exec, ccv_array_t* autograd_tensor_symbol, ccv_array_t* sum_or_set_exec)
{
	assert(tensor_ver->c < tensor_ver->ref_version->rnum);
	int i, j = 0;
	struct {
		int k;
		int i;
	} kd[tensor_ver->ref_version->rnum - tensor_ver->c];
	for (i = tensor_ver->c; i < tensor_ver->ref_version->rnum; i++)
	{
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i);
		const int k = _ccv_nnc_tensor_ref_version_find_alias(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, alias);
		if (k >= 0)
			kd[j++] = (typeof(kd[0])){
				.k = k, .i = i
			};
		else if (_ccv_nnc_tensor_ref_version_involve_alias(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, alias))
			kd[j++] = (typeof(kd[0])) {
				.k = -1, .i = i // It has dependency to the original tensor (non-alias) now, label this with highest bit.
			};
	}
	// Can only find one. This is the easy case, we can simply return that symbol (or its alias).
	if (j == 1)
	{
		if (kd[0].k >= 0)
			return kd[0].k; // Only can find one alias, that is the one.
		// Otherwise, need to create a new alias.
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[0].i);
		ccv_nnc_autograd_tensor_symbol_t* ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
		// Since we create new alias, we need to set the referenced one to be allocated with 0s.
		if (ref->alias_ref) // If this is an alias, it has to be zero initialized.
		{
			ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, ref->alias_ref - 1);
			assert(ref->alias_ref == 0); // This is original.
			ref->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
		} else if (tensor_ref->alias_registry && // Otherwise, to see if this symbol is fully occupied.
				// Loop over to see if this tensor is fully occupied to avoid extra zero step.
				!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info)) {
			ref->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
		}
		ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0};
		tensor_sym.d = d;
		tensor_sym.alias_ref = tensor_ref->d + 1;
		ccv_array_push(autograd_tensor_symbol, &tensor_sym);
		const int ad = autograd_tensor_symbol->rnum - 1;
		if (tensor_ref->alias_registry) // Only push this when it has an alias registry (otherwise it already conflict with everyone).
			ccv_array_push(tensor_ref->alias_registry, &ad);
		// The newly inserted tensor symbol.
		return ad;
	}
	// Otherwise, we need to create the sum operation out of these.
	const int input_size = j;
	int has_this_alias_exclusively = 1;
	int* inputs = input_size > 0 ? (int*)ccmalloc(sizeof(int) * input_size) : 0;
	for (i = 0; i < input_size; i++)
	{
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i);
		// Can take a fast path if every ref involved has the same alias, our sum operation can be faster (using alias directly).
		if (has_this_alias_exclusively && kd[i].k >= 0 && _ccv_nnc_tensor_ref_version_has_this_alias_exclusively(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, alias))
			inputs[i] = *(int*)ccv_array_get(tensor_ref->alias_registry, 0); // Assigning the alias.
		else {
			if (has_this_alias_exclusively)
			{
				has_this_alias_exclusively = 0;
				for (j = 0; j < i; j++)
					inputs[j] = ((ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i))->d;
			}
			inputs[i] = tensor_ref->d;
		}
	}
	ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0};
	tensor_sym.d = alias->alias_ref - 1;
	const int tensor_ref_d = autograd_tensor_symbol->rnum - 1;
	ccv_array_push(autograd_tensor_symbol, &tensor_sym);
	tensor_sym.d = d;
	tensor_sym.alias_ref = tensor_ref_d + 1;
	ccv_array_push(autograd_tensor_symbol, &tensor_sym);
	const int ad = autograd_tensor_symbol->rnum - 1;
	ccv_nnc_graph_sum_or_set_exec_t sum_exec = {
		.input_size = input_size,
		.inputs = inputs,
		.output = ad
	};
	if (idx >= 0)
	{
		sum_exec.outgoings = ccv_array_new(sizeof(int), 1, 0);
		ccv_array_push(sum_exec.outgoings, &idx);
	}
	ccv_array_push(sum_or_set_exec, &sum_exec);
	const int outgoing = exec_symbol_size + sum_or_set_exec->rnum - 1;
	int no_alias_registry = 0;
	for (i = 0; i < input_size; i++)
	{
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i);
		if (!has_this_alias_exclusively)
		{
			// If the sum operation is not operating on one alias. I need to zero this tensor out when it is first
			// allocated (see discussions around the flags I use).
			ccv_nnc_autograd_tensor_symbol_t* tensor_sym = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
			if (tensor_sym->alias_ref)
			{
				// Find the original tensor_sym and set its flags (I prefer to set flags on its original).
				ccv_nnc_autograd_tensor_symbol_t* ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_sym->alias_ref - 1);
				assert(ref->alias_ref == 0); // This is original.
				ref->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
			} else if (tensor_ref->alias_registry && // Otherwise, to see if this symbol is fully occupied.
					// Loop over to see if this tensor is fully occupied to avoid extra zero step.
					!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info)) {
				tensor_sym->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
			}
		}
		// Check to see if any of these tensors doesn't have alias.
		no_alias_registry |= (!tensor_ref->alias_registry);
		const int x = tensor_ref->x;
		assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
		if (x < exec_symbol_size)
		{
			ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
			if (!back_exec->outgoings)
				back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
			ccv_array_push(back_exec->outgoings, &outgoing);
		} else {
			ccv_nnc_graph_sum_or_set_exec_t* sum_or_set = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size);
			ccv_array_push(sum_or_set->outgoings, &outgoing);
		}
		if (tensor_ref->exec_registry)
			for (j = 0; j < tensor_ref->exec_registry->rnum; j++)
			{
				const int x = *(int*)ccv_array_get(tensor_ref->exec_registry, j);
				assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
				assert(x < exec_symbol_size); // exec_registry is only used by alias_registry, it simply cannot reference to a sum operation.
				ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
				if (!back_exec->outgoings)
					back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_push(back_exec->outgoings, &outgoing);
			}
	}
	const ccv_nnc_tensor_ref_t tensor_ref = {
		.d = tensor_ref_d,
		.x = outgoing,
		.exec_registry = 0, // I don't need to take execution dependencies because this tensor is generated by sum, therefore, we already take that dependency.
		.alias_registry = !no_alias_registry || has_this_alias_exclusively ? ccv_array_new(sizeof(int), 1, 0) : 0
	};
	// If there is no alias registry, then we take the whole tensor ref as one.
	if (!no_alias_registry || has_this_alias_exclusively)
	{
		// If this tensor ref contains multiple different types of alias, have to add them together (otherwise
		// the computation for if there is an empty slot in this tensor ref is not correct without all the
		// occupancy availability information).
		if (!has_this_alias_exclusively)
			for (i = 0; i < input_size; i++)
			{
				ccv_nnc_tensor_ref_t* ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i);
				assert(ref->alias_registry);
				// It may get duplicates. But whatever, won't matter the computation.
				for (j = 0; j < ref->alias_registry->rnum; j++)
					ccv_array_push(tensor_ref.alias_registry, ccv_array_get(ref->alias_registry, j));
			}
		ccv_array_push(tensor_ref.alias_registry, &ad);
	}
	assert(input_size <= tensor_ver->ref_version->rnum - tensor_ver->c);
	ccv_nnc_tensor_ref_t x;
	for (i = 0; i < input_size; i++)
		// If the current one (i + tensor_ver->c) is smaller than the one referenced to, exchange.
		if (kd[i].i > i + tensor_ver->c)
			CCV_SWAP(*(ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i + tensor_ver->c), *(ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i), x);
	ccv_array_push(tensor_ver->ref_version, &tensor_ref);
	// We've consumed input_size tensor refs, now move c up to the pointer of non-consumed tensors.
	tensor_ver->c += input_size;
	return ad;
}

void ccv_nnc_symbolic_graph_backward(ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, const ccv_nnc_tensor_symbol_t* f_symbols, const int f_symbol_size, const ccv_nnc_tensor_symbol_t* wrt_symbols, const int wrt_symbol_size)
{
	// First, fill all the "auto" holes.
	// This is the symbol table that with "auto" info filled up.
	const int tensor_symbol_size = graph->tensor_symbol_info->rnum;
	const int exec_symbol_size = graph->exec_symbol_info->rnum;
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * tensor_symbol_size);
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * exec_symbol_size);
	_ccv_nnc_symbolic_graph_auto_symbols(graph, sources, source_size, destinations, destination_size, tensor_symbol_info, exec_symbol_info);
	// Now, for each one of these, find a reverse graph.
	ccv_nnc_graph_backward_info_t* backward_info = (ccv_nnc_graph_backward_info_t*)cccalloc(exec_symbol_size, sizeof(ccv_nnc_graph_backward_info_t));
	int i, j;
#define visitor(node, idx, ...) \
	do { \
		assert(ccv_nnc_cmd_is_forward(node->cmd)); \
		if (node->outgoings) \
			for (i = 0; i < node->outgoings->rnum; i++) \
			{ \
				int d = *(int*)ccv_array_get(node->outgoings, i); \
				if (backward_info[d].outgoings == 0) \
					backward_info[d].outgoings = ccv_array_new(sizeof(int32_t), 1, 0); \
				ccv_array_push(backward_info[d].outgoings, &idx); \
			} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, exec_symbol_size, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Find the f_symbols, and tag its flows.
#define visitor(node, idx, ...) \
	do { \
		int f = node->f_wrt & 0x1; \
		for (i = 0; i < exec_symbol_info[idx].output_size && !f; i++) \
			for (j = 0; j < f_symbol_size && !f; j++) \
				if (exec_symbol_info[idx].outputs[i] == f_symbols[j].d) \
					f = 1; \
		if (f) \
		{ \
			node->f_wrt |= f; \
			if (node->outgoings) \
				for (i = 0; i < node->outgoings->rnum; i++) \
				{ \
					int d = *(int*)ccv_array_get(node->outgoings, i); \
					backward_info[d].f_wrt |= f; \
				} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, backward_info, exec_symbol_size, destinations, destination_size, sources, source_size, visitor);
#undef visitor
	// Find the wrt_symbols, and tag its flows.
#define visitor(node, idx, ...) \
	do { \
		int wrt = backward_info[idx].f_wrt & 0x2; \
		for (i = 0; i < node->input_size && !wrt; i++) \
			for (j = 0; j < wrt_symbol_size && !wrt; j++) \
				if (node->inputs[i] == wrt_symbols[j].d) \
					wrt = 0x2; \
		if (wrt) \
		{ \
			backward_info[idx].f_wrt |= wrt; \
			if (node->outgoings) \
				for (i = 0; i < node->outgoings->rnum; i++) \
				{ \
					int d = *(int*)ccv_array_get(node->outgoings, i); \
					backward_info[d].f_wrt |= wrt; \
				} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, exec_symbol_size, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Now, only the flow from f_symbols back to wrt_symbols are interested to us.
	// Visit the graph in reverse order, build the AD nodes.
	ccv_nnc_graph_autograd_exec_t* autograd_exec = (ccv_nnc_graph_autograd_exec_t*)cccalloc(exec_symbol_size, sizeof(ccv_nnc_graph_autograd_exec_t));
	for (i = 0; i < exec_symbol_size; i++)
		if (backward_info[i].f_wrt == 0x3 && backward_info[i].outgoings)
		{
			// Copy over the outgoing bits.
			autograd_exec[i].outgoings = ccv_array_new(sizeof(int), backward_info[i].outgoings->rnum, 0);
			for (j = 0; j < backward_info[i].outgoings->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(backward_info[i].outgoings, j);
				// Only push the outgoing node if it is in the f_wrt path.
				if (backward_info[d].f_wrt == 0x3)
					ccv_array_push(autograd_exec[i].outgoings, &d);
			}
		}
	ccv_nnc_autograd_tensor_version_t* autograd_tensor_version = (ccv_nnc_autograd_tensor_version_t*)cccalloc(tensor_symbol_size, sizeof(ccv_nnc_autograd_tensor_version_t));
	ccv_array_t* autograd_tensor_symbol = ccv_array_new(sizeof(ccv_nnc_autograd_tensor_symbol_t), tensor_symbol_size, 0);
	ccv_array_t* sum_or_set_exec = ccv_array_new(sizeof(ccv_nnc_graph_sum_or_set_exec_t), 0, 0);
#define visitor(node, idx, ...) \
	do { \
		/* This is required by both f flow and wrt flow, therefore, an interest to us */ \
		if (node->f_wrt == 0x3) \
		{ \
			const ccv_nnc_graph_exec_symbol_info_t* forw_exec = exec_symbol_info + idx; \
			ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + idx; \
			back_exec->cmd = forw_exec->cmd; \
			back_exec->cmd.cmd += 1; /* Backward command is the one after forward command. */ \
			assert(ccv_nnc_cmd_is_backward(back_exec->cmd)); \
			back_exec->output_size = forw_exec->input_size; \
			back_exec->input_size = forw_exec->output_size; \
			back_exec->inputs = ccmalloc(sizeof(int) * (back_exec->input_size + back_exec->output_size)); \
			back_exec->outputs = back_exec->inputs + back_exec->input_size; \
			/* Need to compute input before we compute output */ \
			for (i = 0; i < forw_exec->output_size; i++) \
			{ \
				const int d = forw_exec->outputs[i]; \
				const int alias_ref = tensor_symbol_info[d].alias_ref; \
				ccv_nnc_autograd_tensor_version_t* tensor_ver = alias_ref ? autograd_tensor_version + (alias_ref - 1) : autograd_tensor_version + d; \
				/* Initialization tensor, should corresponding to f symbols */ \
				if (!tensor_ver->ref_version) \
				{ \
					ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0}; \
					if (!alias_ref) \
					{ \
						tensor_sym.d = d; \
						ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
						const ccv_nnc_tensor_ref_t tensor_ref = { \
							.d = autograd_tensor_symbol->rnum - 1, \
							.x = idx, \
							.alias_registry = 0 \
						}; \
						tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
						ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
					} else { \
						tensor_sym.d = alias_ref - 1; \
						ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
						const ccv_nnc_tensor_ref_t tensor_ref = { \
							.d = autograd_tensor_symbol->rnum - 1, \
							.x = idx, \
							.alias_registry = ccv_array_new(sizeof(int), 1, 0) \
						}; \
						/* This is f symbols, must be the same size as the original for this alias to be anywhere usable. */ \
						assert(ccv_nnc_tensor_count(tensor_symbol_info[alias_ref - 1].info) == ccv_nnc_tensor_count(tensor_symbol_info[d].info)); \
						tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
						ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
						tensor_sym.d = d; /* set back */ \
						tensor_sym.alias_ref = tensor_ref.d + 1; \
						ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
						const int ad = autograd_tensor_symbol->rnum - 1; \
						ccv_array_push(tensor_ref.alias_registry, &ad); \
					} \
				} \
				/* The simplest case (most common), it is not an alias. */ \
				if (!alias_ref) \
				{ \
					/* Even simpler, this only have one reference tensor, thus, pass this as input. */ \
					if (tensor_ver->c == tensor_ver->ref_version->rnum - 1) \
					{ \
						ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c); \
						/* There are alias associated with this tensor ref, zero it out when this tensor is allocated. */ \
						/* This is is required. Consider the case that we have an alias of this tensor used somehwere */ \
						/* on forward pass, when we compute backward, we have that alias computed first, however, its */ \
						/* underlying tensor is not zero initialized, and we will end up with garbage values here. */ \
						if (tensor_ref->alias_registry && \
							/* Loop over to see if this tensor is fully occupied to avoid extra zero step. */ \
							!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info)) \
						{ \
							ccv_nnc_autograd_tensor_symbol_t* tensor_sym = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d); \
							assert(tensor_sym->alias_ref == 0); \
							tensor_sym->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS; \
						} \
						back_exec->inputs[i] = tensor_ref->d; \
					} else { \
						/* Otherwise, we need to sum them up, and then pass the summed result to the computation. */ \
						_ccv_nnc_graph_sum_autograd_tensor_versions(idx, d, exec_symbol_size, tensor_symbol_info, tensor_ver, autograd_exec, autograd_tensor_symbol, sum_or_set_exec); \
						ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c); \
						back_exec->inputs[i] = tensor_ref->d; \
					} \
				} else \
					/* If this is an alias, go through all available tensor ref versions */ \
					back_exec->inputs[i] = _ccv_nnc_graph_sum_autograd_tensor_versions_alias(idx, d, tensor_symbol_info, exec_symbol_size, tensor_symbol_info + d, tensor_ver, autograd_exec, autograd_tensor_symbol, sum_or_set_exec); \
			} \
			for (i = 0; i < forw_exec->input_size; i++) \
			{ \
				const int d = forw_exec->inputs[i]; \
				const int alias_ref = tensor_symbol_info[d].alias_ref; \
				ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0}; \
				tensor_sym.d = d; \
				/* The simplest case (most common), it is not an alias. */ \
				if (!alias_ref) \
				{ \
					ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
					const ccv_nnc_tensor_ref_t tensor_ref = { \
						.d = autograd_tensor_symbol->rnum - 1, \
						.x = idx, \
						.exec_registry = 0, \
						.alias_registry = 0 \
					}; \
					ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + d; \
					if (!tensor_ver->ref_version) \
						tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
					ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
					back_exec->outputs[i] = tensor_ref.d; \
				} else { \
					/* Otherwise, in case that this is an alias, we try to find the existing one (in tensor_ver),
					 * see if can meet the need (thus, for the tensor info / ofs, it fits). */ \
					ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + (alias_ref - 1); \
					if (!tensor_ver->ref_version) \
						tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
					/* If already exists a ref version, check if any of these not-sealed tensors have free space. */ \
					int found = 0; \
					for (j = tensor_ver->c; j < tensor_ver->ref_version->rnum; j++) \
					{ \
						ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, j); \
						if (!_ccv_nnc_tensor_ref_version_involve_alias(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, tensor_symbol_info + d)) \
						{ \
							tensor_sym.alias_ref = tensor_ref->d + 1; \
							ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
							const int ad = autograd_tensor_symbol->rnum - 1; \
							ccv_array_push(tensor_ref->alias_registry, &ad); \
							if (!tensor_ref->exec_registry) \
								tensor_ref->exec_registry = ccv_array_new(sizeof(int), 1, 0); \
							ccv_array_push(tensor_ref->exec_registry, &idx); \
							back_exec->outputs[i] = ad; \
							found = 1; \
							break; \
						} \
					} \
					if (!found) /* Cannot find an tensor ref to insert, create one first */ \
					{ \
						tensor_sym.d = alias_ref - 1; /* Reference back to the non-alias. */ \
						ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
						const ccv_nnc_tensor_ref_t tensor_ref = { \
							.d = autograd_tensor_symbol->rnum - 1, \
							.x = idx, \
							.exec_registry = 0, \
							.alias_registry = ccv_array_new(sizeof(int), 1, 0) \
						}; \
						ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
						tensor_sym.d = d; /* set back */ \
						tensor_sym.alias_ref = tensor_ref.d + 1; \
						ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
						const int ad = autograd_tensor_symbol->rnum - 1; \
						ccv_array_push(tensor_ref.alias_registry, &ad); \
						back_exec->outputs[i] = ad; \
					} \
				} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, backward_info, exec_symbol_size, destinations, destination_size, sources, source_size, visitor);
#undef visitor
	// Find all relevant wrt symbols, generate sum for them if needed.
	for (i = 0; i < wrt_symbol_size; i++)
	{
		const int d = wrt_symbols[i].d;
		ccv_nnc_autograd_tensor_version_t* tensor_ver = (!tensor_symbol_info[d].alias_ref) ? autograd_tensor_version + d : autograd_tensor_version + (tensor_symbol_info[d].alias_ref - 1);
		// If there are more than one tensor in the list, it is possible to sum them up.
		if (tensor_ver->c < tensor_ver->ref_version->rnum - 1)
			_ccv_nnc_graph_sum_autograd_tensor_versions(-1, d, exec_symbol_size, tensor_symbol_info, tensor_ver, autograd_exec, autograd_tensor_symbol, sum_or_set_exec);
		// The tensor version should have ref_version, and only one now (after sum up).
		assert(tensor_ver->c == tensor_ver->ref_version->rnum - 1);
	}
	// Generate required symbols based on the information gathered above.
	for (i = 0; i < autograd_tensor_symbol->rnum; i++)
	{
		ccv_nnc_autograd_tensor_symbol_t* symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, i);
		assert(symbol->d >= 0);
		assert(symbol->d < tensor_symbol_size);
		ccv_nnc_tensor_symbol_info_t* forw_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, symbol->d);
		if (!symbol->alias_ref)
		{
			assert(!forw_symbol->alias_ref);
			symbol->symbol = ccv_nnc_tensor_symbol_new(graph, forw_symbol->info, 0);
			ccv_nnc_tensor_symbol_set_flags(graph, symbol->symbol, symbol->flags);
		} else {
			assert(forw_symbol->alias_ref);
			assert(symbol->flags == 0); // We don't set flags on alias.
			// Due to our generation order, this must be after the original symbol is created.
			ccv_nnc_autograd_tensor_symbol_t* ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, symbol->alias_ref - 1);
			symbol->symbol = ccv_nnc_tensor_symbol_alias_new(graph, ref->symbol, forw_symbol->ofs, forw_symbol->inc, forw_symbol->info, 0);
		}
	}
	ccv_array_t* symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	for (i = 0; i < exec_symbol_size; i++)
	{
		// This is not going to be an interesting node. Skip.
		if (backward_info[i].f_wrt != 0x3)
			continue;
		ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + i;
		ccv_array_clear(symbols);
		// Gradient inputs.
		for (j = 0; j < back_exec->input_size; j++)
			ccv_array_push(symbols, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, back_exec->inputs[j]))->symbol));
		ccv_nnc_graph_exec_symbol_info_t* forw_exec = exec_symbol_info + i;
		// Inputs from forward function.
		for (j = 0; j < forw_exec->input_size; j++)
		{
			ccv_nnc_tensor_symbol_t symbol = {
				.info = tensor_symbol_info[forw_exec->inputs[j]].info,
				.d = forw_exec->inputs[j],
				.graph = graph
			};
			ccv_array_push(symbols, &symbol);
		}
		// Outputs from forward function.
		for (j = 0; j < forw_exec->output_size; j++)
		{
			ccv_nnc_tensor_symbol_t symbol = {
				.info = tensor_symbol_info[forw_exec->outputs[j]].info,
				.d = forw_exec->outputs[j],
				.graph = graph
			};
			ccv_array_push(symbols, &symbol);
		}
		for (j = 0; j < back_exec->output_size; j++)
			ccv_array_push(symbols, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, back_exec->outputs[j]))->symbol));
		back_exec->symbol = ccv_nnc_graph_exec_symbol_new(graph, back_exec->cmd, ccv_array_get(symbols, 0), back_exec->input_size + forw_exec->input_size + forw_exec->output_size, ccv_array_get(symbols, back_exec->input_size + forw_exec->input_size + forw_exec->output_size), back_exec->output_size, 0);
	}
	for (i = 0; i < sum_or_set_exec->rnum; i++)
	{
		ccv_nnc_graph_sum_or_set_exec_t* exec = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, i);
		if (exec->input_size)
		{
			ccv_array_clear(symbols);
			// This is to sum.
			for (j = 0; j < exec->input_size; j++)
				ccv_array_push(symbols, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, exec->inputs[j]))->symbol));
			ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0);
			exec->symbol = ccv_nnc_graph_exec_symbol_new(graph, cmd, ccv_array_get(symbols, 0), exec->input_size, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, exec->output))->symbol), 1, 0);
		} else {
			// This is to zero.
			ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_SET_FORWARD, 0, CMD_BLAS(0), 0);
			exec->symbol = ccv_nnc_graph_exec_symbol_new(graph, cmd, 0, 0, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, exec->output))->symbol), 1, 0);
		}
	}
	ccv_array_free(symbols);
	for (i = 0; i < exec_symbol_size; i++)
	{
		// This is not going to be an interesting node. Skip.
		if (backward_info[i].f_wrt != 0x3)
			continue;
		ccv_nnc_graph_exec_symbol_t forw_exec = {
			.d = i,
			.graph = graph
		};
		ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + i;
		ccv_nnc_graph_exec_symbol_concat(graph, forw_exec, back_exec->symbol);
		if (back_exec->outgoings)
			for (j = 0; j < back_exec->outgoings->rnum; j++)
			{
				int d = *(int*)ccv_array_get(back_exec->outgoings, j);
				if (d < exec_symbol_size)
					ccv_nnc_graph_exec_symbol_concat(graph, back_exec->symbol, autograd_exec[d].symbol);
				else
					ccv_nnc_graph_exec_symbol_concat(graph, back_exec->symbol, ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, d - exec_symbol_size))->symbol);
			}
	}
	for (i = 0; i < sum_or_set_exec->rnum; i++)
	{
		ccv_nnc_graph_sum_or_set_exec_t* exec = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, i);
		if (exec->outgoings)
			for (j = 0; j < exec->outgoings->rnum; j++)
			{
				int d = *(int*)ccv_array_get(exec->outgoings, j);
				if (d < exec_symbol_size)
					ccv_nnc_graph_exec_symbol_concat(graph, exec->symbol, autograd_exec[d].symbol);
				else
					ccv_nnc_graph_exec_symbol_concat(graph, exec->symbol, ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, d - exec_symbol_size))->symbol);
			}
	}
	// Now, everything is done, set the metadata on graph so that we can lookup later for backward symbols
	if (graph->backward_tensor_symbols)
		graph->backward_tensor_symbols = (int*)ccrealloc(graph->backward_tensor_symbols, sizeof(int) * graph->tensor_symbol_info->rnum);
	else
		graph->backward_tensor_symbols = (int*)ccmalloc(sizeof(int) * graph->tensor_symbol_info->rnum);
	graph->backward_exec_symbols = graph->backward_tensor_symbols + tensor_symbol_size;
	graph->forward_symbol_size = tensor_symbol_size;
	graph->backward_symbol_size = graph->tensor_symbol_info->rnum - tensor_symbol_size;
	for (i = 0; i < graph->forward_symbol_size; i++)
		graph->backward_tensor_symbols[i] = -1;
	for (i = 0; i < graph->backward_symbol_size; i++)
		graph->backward_exec_symbols[i] = -1;
	// Assigning for wrt symbols.
	for (i = 0; i < wrt_symbol_size; i++)
	{
		const int d = wrt_symbols[i].d;
		assert(d >= 0);
		assert(d < tensor_symbol_size);
		// If this wrt symbol is an alias, create extra alias for this.
		ccv_nnc_tensor_ref_t* tensor_ref;
		int dd;
		if (tensor_symbol_info[d].alias_ref)
		{
			ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + (tensor_symbol_info[d].alias_ref - 1);
			assert(tensor_ver->ref_version);
			tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c);
			ccv_nnc_autograd_tensor_symbol_t* autograd_symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
			ccv_nnc_tensor_symbol_t alias = ccv_nnc_tensor_symbol_alias_new(graph, autograd_symbol->symbol, tensor_symbol_info[d].ofs, tensor_symbol_info[d].inc, tensor_symbol_info[d].info, 0);
			graph->backward_tensor_symbols[d] = alias.d;
			assert(alias.d >= tensor_symbol_size);
			dd = alias.d - tensor_symbol_size;
		} else {
			ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + d;
			assert(tensor_ver->ref_version);
			tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c);
			ccv_nnc_autograd_tensor_symbol_t* autograd_symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
			graph->backward_tensor_symbols[d] = autograd_symbol->symbol.d;
			assert(autograd_symbol->symbol.d >= tensor_symbol_size);
			dd = autograd_symbol->symbol.d - tensor_symbol_size;
		}
		const int x = tensor_ref->x;
		if (tensor_ref->exec_registry && tensor_ref->exec_registry->rnum) // Create no-op node.
		{
			ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, 0);
			if (x < exec_symbol_size)
				ccv_nnc_graph_exec_symbol_concat(graph, autograd_exec[x].symbol, noop);
			else
				ccv_nnc_graph_exec_symbol_concat(graph, ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size))->symbol, noop);
			for (j = 0; j < tensor_ref->exec_registry->rnum; j++)
			{
				const int x = *(int*)ccv_array_get(tensor_ref->exec_registry, j);
				assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
				assert(x < exec_symbol_size); // exec_registry is only used by alias_registry, it simply cannot reference to a sum operation.
				ccv_nnc_graph_exec_symbol_concat(graph, autograd_exec[x].symbol, noop);
			}
			graph->backward_exec_symbols[dd] = noop.d;
		} else {
			if (x < exec_symbol_size)
				graph->backward_exec_symbols[dd] = autograd_exec[x].symbol.d;
			else
				graph->backward_exec_symbols[dd] = ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size))->symbol.d;
		}
	}
	// Assigning for f symbols.
	for (i = 0; i < f_symbol_size; i++)
	{
		const int d = f_symbols[i].d;
		assert(d >= 0);
		assert(d < tensor_symbol_size);
		ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + d;
		assert(tensor_ver->ref_version);
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c);
		ccv_nnc_autograd_tensor_symbol_t* autograd_symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
		graph->backward_tensor_symbols[d] = autograd_symbol->symbol.d;
		// Cannot find relevant backward exec symbols for f.
	}
	for (i = 0; i < exec_symbol_size; i++)
	{
		if (autograd_exec[i].inputs)
			ccfree(autograd_exec[i].inputs);
		if (autograd_exec[i].outgoings)
			ccv_array_free(autograd_exec[i].outgoings);
	}
	ccfree(autograd_exec);
	for (i = 0; i < tensor_symbol_size; i++)
	{
		if (autograd_tensor_version[i].ref_version)
		{
			for (j = 0; j < autograd_tensor_version[i].ref_version->rnum; j++)
			{
				ccv_nnc_tensor_ref_t* ref_version = (ccv_nnc_tensor_ref_t*)ccv_array_get(autograd_tensor_version[i].ref_version, j);
				if (ref_version->exec_registry)
					ccv_array_free(ref_version->exec_registry);
				if (ref_version->alias_registry)
					ccv_array_free(ref_version->alias_registry);
			}
			ccv_array_free(autograd_tensor_version[i].ref_version);
		}
	}
	ccfree(autograd_tensor_version);
	ccv_array_free(autograd_tensor_symbol);
	for (i = 0; i < sum_or_set_exec->rnum; i++)
	{
		ccv_nnc_graph_sum_or_set_exec_t* sum_or_set = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, i);
		if (sum_or_set->inputs)
			ccfree(sum_or_set->inputs);
		if (sum_or_set->outgoings)
			ccv_array_free(sum_or_set->outgoings);
	}
	ccv_array_free(sum_or_set_exec);
	for (i = 0; i < exec_symbol_size; i++)
		if (backward_info[i].outgoings)
			ccv_array_free(backward_info[i].outgoings);
	ccfree(backward_info);
	ccfree(exec_symbol_info);
	ccfree(tensor_symbol_info);
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_for_backward(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_symbol_t symbol)
{
	assert(symbol.d >= 0);
	assert(symbol.d < graph->forward_symbol_size);
	assert(graph->backward_tensor_symbols[symbol.d] >= 0);
	ccv_nnc_tensor_symbol_t tensor = {
		.d = graph->backward_tensor_symbols[symbol.d],
		.graph = graph,
		.info = symbol.info
	};
	return tensor;
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol_for_backward(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_symbol_t symbol)
{
	assert(symbol.d >= graph->forward_symbol_size);
	assert(symbol.d < graph->forward_symbol_size + graph->backward_symbol_size);
	const int dd = symbol.d - graph->forward_symbol_size;
	assert(graph->backward_exec_symbols[dd] >= 0);
	ccv_nnc_graph_exec_symbol_t exec = {
		.d = graph->backward_exec_symbols[dd],
		.graph = graph
	};
	return exec;
}
