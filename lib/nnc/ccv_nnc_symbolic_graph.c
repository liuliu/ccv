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
	// This is a table of tensor references to real allocated tensors.
	ccv_nnc_tensor_t** vt_tensor;
	// This is the allocated non-continuous buffers.
	int buffer_size;
	void** buffer;
	// Real allocated tensor headers.
	int tensor_size;
	ccv_nnc_tensor_t tensor[1];
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

typedef struct {
	int s;
	int t;
} ccv_nnc_tensor_liveness_t;

static const int CONST_TENSOR = -2;
static const int UNASSIGNED = -1;

static ccv_nnc_tensor_arena_t* _ccv_nnc_tensor_arena_new(const ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info, const int exec_symbol_info_size, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const int tensor_symbol_info_size, const ccv_nnc_tensor_liveness_t* tensor_liveness)
{
	// Compute how many dis-continuous buffers are needed.
	// We prefer to have several dis-continuous buffers instead of one big buffer because
	// in this way, we can rely on system memory allocators (jemalloc, tcmalloc, or CUDA's allocator)
	// to fully utilize memory.
	int i, j;
	// The interference matrix. We can only use 1-bit on the matrix to denote if a tensor interferes with another or not.
	uint32_t* itf = (uint32_t*)ccmalloc((sizeof(uint32_t) * tensor_symbol_info_size * (tensor_symbol_info_size + 1) / 2 + 31) / 32);
#define IDX(x, y) (((x) <= (y)) ? (x) * tensor_symbol_info_size - ((x) - 1) * (x) / 2 + (y) - (x) : (y) * tensor_symbol_info_size - ((y) - 1) * (y) / 2 + (x) - (y))
#define GET_ITF(x, y) { \
		int idx = IDX(x, y); \
		(itf[(idx >> 4)] & (1u << (idx & 0x31))); \
	}
#define SET_ITF(x, y, z) do { \
		int idx = IDX(x, y); \
		if (z) \
			itf[(idx >> 4)] |= (1u << (idx & 0x31)); \
		else \
			itf[(idx >> 4)] &= (~(1u << (idx & 0x31))); \
	} while(0)
	// Overlap count.
	int avaliable_tensor_size = 0;
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_liveness[i].s != UNASSIGNED)
			++avaliable_tensor_size;
	uint32_t* oc = (uint32_t*)cccalloc(tensor_symbol_info_size, sizeof(uint32_t));
	for (i = 0; i < tensor_symbol_info_size; i++)
		for (j = i + 1; j < tensor_symbol_info_size; j++)
			// If these two tensors are still alive, analyze them.
			if (tensor_liveness[i].s != UNASSIGNED && tensor_liveness[j].s != UNASSIGNED)
			{
				int flag = (ccv_max(tensor_liveness[i].s, tensor_liveness[j].s) <= ccv_min(tensor_liveness[i].t, tensor_liveness[j].t));
				// If their life time overlaps, then set the ITF.
				SET_ITF(i, j, flag);
				// Compute how many tensors it overlap.
				oc[i] += flag;
			}
	// Allocation graph (assuming there is a source node, and a destination node, which is 0, and (tensor_symbol_info_size + 1)
	uint64_t* alloc = (uint64_t*)cccalloc((tensor_symbol_info_size + 2) * (tensor_symbol_info_size + 2), sizeof(uint64_t));
	uint32_t* assigned = (uint32_t*)cccalloc((tensor_symbol_info_size + 31) / 32, sizeof(uint32_t));
	uint32_t* assignment = (uint32_t*)ccmalloc(sizeof(uint32_t) * tensor_symbol_info_size);
	// Extract one symbol at a time.
	for (j = 0; j < avaliable_tensor_size; j++)
	{
		// Find the one with largest overlap.
		int moc = 0, k = -1;
		for (i = 0; i < tensor_symbol_info_size; i++)
			if (oc[i] > moc)
				moc = oc[i], k = i;
		if (k == -1) // Cannot find overlaps for the rest of the tensors, insert them all in this one batch and then exit.
		{
			break;
		} else {
			// Otherwise, select the best insertion location for k.
		}
	}
#undef SET_ITF
#undef GET_ITF
#undef IDX
	ccfree(assignment);
	ccfree(assigned);
	ccfree(alloc);
	ccfree(oc);
	ccfree(itf);
	return 0;
}

void ccv_nnc_symbolic_graph_compile(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_symbol_t* tensor_symbol_bindings, const int tensor_symbol_binding_size, const ccv_nnc_tensor_t** tensor_bindings, const int tensor_binding_size, const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, ccv_nnc_graph_t** graph_ref, ccv_nnc_tensor_arena_t** tensor_arena_ref)
{
	assert(tensor_symbol_binding_size == tensor_binding_size);
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
	ccv_nnc_tensor_liveness_t* tensor_liveness = (ccv_nnc_tensor_liveness_t*)ccmalloc(sizeof(ccv_nnc_tensor_liveness_t) * graph->tensor_symbol_info->rnum);
	for (i = 0; i < graph->tensor_symbol_info->rnum; i++)
	{
		// Check no tensor info is auto now.
		assert(!ccv_nnc_is_tensor_auto(tensor_symbol_info[i].info));
		tensor_liveness[i].s = UNASSIGNED;
	}
	int high = 0;
#define visitor(node, _, level) \
	do { \
		for (i = 0; i < node->input_size; i++) \
		{ \
			if (tensor_liveness[node->inputs[i]].s == UNASSIGNED) \
				tensor_liveness[node->inputs[i]].s = CONST_TENSOR; /* input starts this node first, therefore, its liveness starts at -2. */ \
			else if (tensor_liveness[node->inputs[i]].s >= 0) \
				tensor_liveness[node->inputs[i]].t = level; \
		} \
		for (i = 0; i < node->output_size; i++) \
		{ \
			if (tensor_liveness[node->outputs[i]].s == UNASSIGNED) /* Only deal with ordinary start point. */ \
				tensor_liveness[node->outputs[i]].s = level; \
			tensor_liveness[node->outputs[i]].t = level; \
		} \
		high = ccv_max(high, level); \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// For symbols starts with input (taking it as constant), the start point is 0 and end point is highest.
	for (i = 0; i < graph->tensor_symbol_info->rnum; i++)
		if (tensor_liveness[i].s == CONST_TENSOR)
			tensor_liveness[i].s = 0, tensor_liveness[i].t = high;

	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
		// Remove tensor symbols that is for in-place operations (and it matches the start, end tensor).
		if (ccv_nnc_cmd_support(exec_symbol_info[i].cmd, CCV_NNC_COMPUTE_SUPPORT_INPLACE))
		{
			const ccv_nnc_graph_exec_symbol_info_t exec_symbol = exec_symbol_info[i];
			int x, y;
			for (x = 0; x < exec_symbol.input_size; x++)
			{
				const ccv_nnc_tensor_symbol_info_t x_symbol = tensor_symbol_info[exec_symbol.inputs[x]];
				for (y = 0; y < exec_symbol.output_size; y++)
					// Only proceed if the input symbol is different from the output symbol,
					// and the input symbol meets the output symbol exactly at the same spot.
					if (exec_symbol.inputs[x] != exec_symbol.outputs[y] &&
						tensor_liveness[exec_symbol.inputs[x]].t == tensor_liveness[exec_symbol.outputs[y]].s)
					{
						const ccv_nnc_tensor_symbol_info_t y_symbol = tensor_symbol_info[exec_symbol.outputs[y]];
						// If dimension matches perfectly, then we can assign y_symbol to x.
						if (memcmp(x_symbol.info.dim, y_symbol.info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
						{
							// Make the exec_symbol_info[i]'s output reference to the same tensor symbol, extends the liveness.
							tensor_liveness[exec_symbol.inputs[x]].t = tensor_liveness[exec_symbol.outputs[y]].t;
							// Mark the original as unassigned.
							tensor_liveness[exec_symbol.outputs[y]].s = UNASSIGNED;
							exec_symbol_info[i].outputs[y] = exec_symbol.inputs[x];
						}
					}
			}
		}
	// Ignore tensors that are already binded.
	for (i = 0; i < tensor_symbol_binding_size; i++)
		tensor_liveness[tensor_symbol_bindings[i].d].s = UNASSIGNED;

	// Now, everything is prepared, tensor liveness is analyzed, inplace operations are collapsed, all tensor symbols and hints
	// are automatically filled in. It is time to guess what's the best tensor placement and create the opaque tensor arena.
	ccv_nnc_tensor_arena_t* tensor_arena = _ccv_nnc_tensor_arena_new(exec_symbol_info, graph->exec_symbol_info->rnum, tensor_symbol_info, graph->tensor_symbol_info->rnum, tensor_liveness);
	if (tensor_arena_ref)
		*tensor_arena_ref = tensor_arena;

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
