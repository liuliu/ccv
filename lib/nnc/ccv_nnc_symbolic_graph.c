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

static ccv_nnc_tensor_arena_t* _ccv_nnc_tensor_arena_new(const ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info, const int exec_symbol_info_size, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const int tensor_symbol_info_size, const ccv_nnc_tensor_liveness_t* tensor_liveness, const int high)
{
	// Compute how many dis-continuous buffers are needed.
	// We prefer to have several dis-continuous buffers instead of one big buffer because
	// in this way, we can rely on system memory allocators (jemalloc, tcmalloc, or CUDA's allocator)
	// to fully utilize memory.
	int i, j;
	// Overlap count.
	int avaliable_tensor_size = 0;
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_liveness[i].s != UNASSIGNED)
			++avaliable_tensor_size;
	// Allocate workspace memory.
	uint32_t* oc = (uint32_t*)cccalloc(tensor_symbol_info_size + ((tensor_symbol_info_size + 31) / 32), sizeof(uint32_t));
	uint32_t* assigned = oc + tensor_symbol_info_size;
	for (i = 0; i < tensor_symbol_info_size; i++)
		for (j = 0; j < tensor_symbol_info_size; j++)
			// If these two tensors are still alive, analyze them.
			if (i != j && tensor_liveness[i].s != UNASSIGNED && tensor_liveness[j].s != UNASSIGNED &&
			// If their life time overlaps, compute how many tensors it overlap.
				ccv_max(tensor_liveness[i].s, tensor_liveness[j].s) <= ccv_min(tensor_liveness[i].t, tensor_liveness[j].t))
					++oc[i];
	// Allocation graph (assuming there is a source node, and a destination node, which is 0, and (tensor_symbol_info_size + 1)
	ccv_sparse_matrix_t* alloc = ccv_sparse_matrix_new(tensor_symbol_info_size + 2, tensor_symbol_info_size + 2, CCV_64S | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	// Extract one symbol at a time.
	for (j = 0; j < avaliable_tensor_size; j++)
	{
		// Find the one with largest overlap, and it is not assigned.
		int max_oc = 0, k = -1;
		uint64_t max_size = 0;
		for (i = 0; i < tensor_symbol_info_size; i++)
			if (oc[i] >= max_oc && tensor_liveness[i].s != UNASSIGNED && !(assigned[i / 32] & (1 << (i & 31))))
			{
				// In case we have a tie, break the tie by prefer larger allocation first.
				uint64_t size = ccv_nnc_tensor_count(tensor_symbol_info[i].info);
				if (oc[i] > max_oc || size > max_size)
					max_size = size, max_oc = oc[i], k = i;
			}
		assert(k >= 0);
		// Select the best insertion location for k.
		for (i = 0; i < tensor_symbol_info_size; i++)
			// If this one is not k, and is alive, and not assigned out yet, and interfere with k
			// Decrease the oc count on that symbol.
			if (i != k && tensor_liveness[i].s != UNASSIGNED &&
				ccv_max(tensor_liveness[i].s, tensor_liveness[k].s) <= ccv_min(tensor_liveness[i].t, tensor_liveness[k].t))
				--oc[i];
		// Assign out k.
		assigned[k / 32] |= (1 << (k & 31));
		// Assuming all tensors has the same data format (32F), therefore, we only need to consider the dimensional size.
		uint64_t size = ccv_nnc_tensor_count(tensor_symbol_info[k].info);
		const ccv_nnc_tensor_liveness_t k_liveness = tensor_liveness[k];
		int min_y = 0, min_x = tensor_symbol_info_size + 1, min_g = high + 2;
#define for_block(y, x, val) do { \
			/* Get liveness, including phony source and destination one */ \
			ccv_nnc_tensor_liveness_t y_liveness, x_liveness; \
			if (y > 0 && y < tensor_symbol_info_size + 1) \
				y_liveness = tensor_liveness[y - 1]; \
			else if (y == 0) \
				y_liveness.s = y_liveness.t = -1; \
			assert(y != tensor_symbol_info_size + 1); \
			if (x > 0 && x < tensor_symbol_info_size + 1) \
				x_liveness = tensor_liveness[x - 1]; \
			else if (x == tensor_symbol_info_size + 1) \
				x_liveness.s = x_liveness.t = high + 1; \
			assert(x != 0); \
			/* y is always earlier than x. */ \
			assert(y_liveness.s < x_liveness.s); \
			/* If this edge satisfy the requirement, now we need to find the ones with tightest possible bounds. */ \
			/* Thus, the gap between y and x (in terms of its life time) should be smallest ones. */ \
			if (((uint64_t*)val)[0] >= size && \
				/* k doesn't overlap with y */ \
				ccv_max(y_liveness.s, k_liveness.s) > ccv_min(y_liveness.t, k_liveness.t) && \
				/* k doesn't overlap with x */ \
				ccv_max(x_liveness.s, k_liveness.s) > ccv_min(x_liveness.t, k_liveness.t) && \
				/* k is after y, and before x or (no overlapping, we can just compare s). */ \
				k_liveness.s > y_liveness.s && k_liveness.s < x_liveness.s) \
			{ \
				int g = x_liveness.s - y_liveness.t; \
				if (g < min_g) \
					min_g = g, min_y = y, min_x = x; \
			} \
		} while (0)
		CCV_SPARSE_FOREACH(alloc, for_block);
#undef for_block
		// Now I find greedy y and x, set it! Note that based on the rules, k's life is between min_y and min_x.
		ccv_set_sparse_matrix_cell(alloc, min_y, k + 1, &size);
		ccv_set_sparse_matrix_cell(alloc, k + 1, min_x, &size);
		// If min_y is source and min_x is destination, we don't need to do anything, otherwise, decrease the weight on that edge.
		if (min_y != 0 || min_x != tensor_symbol_info_size + 1)
		{
			uint64_t curr_size = ccv_get_sparse_matrix_cell(alloc, min_y, min_x).i64[0];
			assert(curr_size >= size);
			curr_size -= size;
			ccv_set_sparse_matrix_cell(alloc, min_y, min_x, &curr_size);
		}
		printf("allocated k %d, find node to insert between %d, %d\n", k, min_y - 1, min_x - 1);
	}
	// Now I have the allocation graph, The total size of outgoing edge of source is the total size of our allocation.
	// It is still interesting to find how many distinctive buffers we can have.
	// (it is smaller than min(outgoing edges of source, incoming edges of destination).
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(alloc, 0);
	uint64_t total_size = 0;
#define for_block(i, val) do { \
		total_size += ((uint64_t*)val)[0]; \
	} while (0)
	CCV_SPARSE_VECTOR_FOREACH(alloc, vector, for_block);
	uint64_t total_tensor_size = 0;
	for (i = 0; i < tensor_symbol_info_size; i++)
		total_tensor_size += ccv_nnc_tensor_count(tensor_symbol_info[i].info);
	printf("total allocation size %lu\ntotal tensor size %lu\n", total_size, total_tensor_size);
#undef for_block
	ccv_matrix_free(alloc);
	ccfree(oc);
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
	ccv_nnc_tensor_arena_t* tensor_arena = _ccv_nnc_tensor_arena_new(exec_symbol_info, graph->exec_symbol_info->rnum, tensor_symbol_info, graph->tensor_symbol_info->rnum, tensor_liveness, high);
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
