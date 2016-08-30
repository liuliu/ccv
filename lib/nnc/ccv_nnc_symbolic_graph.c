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
		.alias_ref = 0,
		.info = info
	};
	ccv_array_push(graph->tensor_symbol_info, &symbol_info);
	return symbol;
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_alias(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_tensor_symbol_t tensor_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info)
{
	assert(tensor_symbol.graph == graph);
	assert(tensor_symbol.d >= 0 && tensor_symbol.d < graph->tensor_symbol_info->rnum);
	ccv_nnc_tensor_symbol_t alias = {
		.info = info,
		.d = graph->tensor_symbol_info->rnum,
		.graph = graph
	};
	ccv_nnc_tensor_symbol_info_t alias_info = {
		.alias_ref = tensor_symbol.d + 1,
		.info = info
	};
	ccv_array_push(graph->tensor_symbol_info, &alias_info);
	return alias;
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

int ccv_nnc_graph_exec_symbol_autoconcat(const ccv_nnc_symbolic_graph_t* graph, const ccv_nnc_graph_exec_symbol_t* execs, const int exec_size)
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
				for (y = 0; y < b_symbol_info->output_size && !b_to_a; y++)
					if (a_symbol_info->inputs[x] == b_symbol_info->outputs[y])
						// This two have matching inputs and outputs, thus, you can concat b to a.
						b_to_a = 1;
			if (b_to_a)
				ccv_nnc_graph_exec_symbol_concat(graph, execs[j], execs[i]);
			int a_to_b = 0;
			for (x = 0; x < a_symbol_info->output_size && !a_to_b; x++)
				for (y = 0; y < b_symbol_info->input_size && !a_to_b; y++)
					if (a_symbol_info->outputs[x] == b_symbol_info->inputs[y])
						// This two have matching inputs and outputs, thus, you can concat b to a.
						a_to_b = 1;
			if (a_to_b)
				ccv_nnc_graph_exec_symbol_concat(graph, execs[i], execs[j]);
		}
	}
	return 0;
}

typedef struct {
	int flag;
	int ref;
	ccv_array_t* head; // The head nodes (it could be multiple if from the graph, one cannot determine which is the first).
	ccv_array_t* tail; // The tail nodes (it could be multiple if from the graph, one cannot determine which is the last).
} ccv_nnc_tensor_expect_t;

static const int UNASSIGNED = 0x1;
static const int ALIAS = 0x2;
static const int CONST_TENSOR = 0x3;

#define TENSOR_EXPECT_UNASSIGNED(t) (t.flag == UNASSIGNED)
#define TENSOR_EXPECT_ALIAS(t) (t.flag == ALIAS)
#define TENSOR_EXPECT_CONST(t) (t.flag == CONST_TENSOR)
#define TENSOR_EXPECT_COMPUTABLE(t) (!TENSOR_EXPECT_ALIAS(t) && !TENSOR_EXPECT_UNASSIGNED(t))

typedef struct {
	int index;
	int companion; // The companion node index (the node that doesn't interfere with current one).
	uint64_t size;
} ccv_nnc_tensor_opt_t;

#define more_than(i1, i2, aux) ((i1).size >= (i2).size)
static CCV_IMPLEMENT_QSORT(_ccv_nnc_tensor_opt_sortby_size, ccv_nnc_tensor_opt_t, more_than)
#undef more_than

// If a's head is deterministically after b's tail.
static int _ccv_nnc_tensor_expect_head_after_tail(const ccv_sparse_matrix_t* exec_dep, const ccv_nnc_tensor_expect_t a, const ccv_nnc_tensor_expect_t b)
{
	assert(a.head);
	assert(b.tail);
	int x, y;
	for (x = 0; x < a.head->rnum; x++)
		for (y = 0; y < b.tail->rnum; y++)
		{
			ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, *(int*)ccv_array_get(a.head, x), *(int*)ccv_array_get(b.tail, y));
			if (cell.i32 && cell.i32[0] > 0)
				return cell.i32[0];
		}
	return 0;
}

static ccv_nnc_tensor_arena_t* _ccv_nnc_tensor_arena_new(const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const int tensor_symbol_info_size, const ccv_sparse_matrix_t* exec_dep, const ccv_nnc_tensor_expect_t* tensor_expect)
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
		_ccv_nnc_tensor_opt_sortby_size((ccv_nnc_tensor_opt_t*)opt->data, opt->rnum, 0);
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
			*(ccv_nnc_tensor_t*)(tensor_arena->tensor + j) = ccv_nnc_tensor(tensor_arena->buffer[assigned[i] - 1] + allocated_offset[i], tensor_symbol_info[i].info, 0);
			assert(allocated_offset[i] + (((uint64_t)CCV_GET_DATA_TYPE_SIZE(CCV_32F) * ccv_nnc_tensor_count(tensor_symbol_info[i].info) + 15) / 16 * 16) <= tensor_arena->buffer_size[assigned[i] - 1]);
			++j;
		} else if (!TENSOR_EXPECT_ALIAS(tensor_expect[i]))
			tensor_arena->vt_tensor[i] = 0;
	ccfree(allocated_offset);
	ccfree(assigned);
	for (i = 0; i < tensor_symbol_info_size; i++)
		// It could be binded tensor, in that case, it doesn't have a ref.
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
			// If there is no ofs, we take a shortcut and just init a normal tensor.
			if (memcmp(ccv_nnc_no_ofs, tensor_symbol_info[i].ofs, sizeof(ccv_nnc_no_ofs)) == 0)
				*(ccv_nnc_tensor_t*)(tensor_arena->tensor + j) = ccv_nnc_tensor(tensor_arena->vt_tensor[alias_ref]->data.u8, tensor_symbol_info[i].info, 0);
			else // Otherwise initialize a tensor view.
				tensor_arena->tensor[j] = ccv_nnc_tensor_view(tensor_arena->vt_tensor[alias_ref], tensor_symbol_info[i].ofs, tensor_symbol_info[i].info.dim);
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

void ccv_nnc_symbolic_graph_compile(const ccv_nnc_symbolic_graph_t* symbolic_graph, const ccv_nnc_tensor_symbol_t* tensor_symbol_bindings, const int tensor_symbol_binding_size, ccv_nnc_tensor_t* const* tensor_bindings, const int tensor_binding_size, const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, ccv_nnc_graph_t** graph_ref, ccv_nnc_tensor_arena_t** tensor_arena_ref, ccv_nnc_graph_exec_arena_t** graph_exec_arena_ref)
{
	assert(graph_ref);
	assert(tensor_arena_ref);
	assert(graph_exec_arena_ref);
	assert(tensor_symbol_binding_size == tensor_binding_size);
	// First, fill all the "auto" holes.
	// This is the symbol table that with "auto" info filled up.
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * symbolic_graph->tensor_symbol_info->rnum);
	memcpy(tensor_symbol_info, symbolic_graph->tensor_symbol_info->data, sizeof(ccv_nnc_tensor_symbol_info_t) * symbolic_graph->tensor_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * symbolic_graph->exec_symbol_info->rnum);
	memcpy(exec_symbol_info, symbolic_graph->exec_symbol_info->data, sizeof(ccv_nnc_graph_exec_symbol_info_t) * symbolic_graph->exec_symbol_info->rnum);

	int i, j;
	// Materialize auto hints.
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
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
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor

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
#define visitor(node, idx, _) \
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
			/* If not found, set */ \
			if (!cell.i32 || cell.i32[0] == 0) \
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

	// Now, collect information about the tensor's expected start / end execs.
	ccv_nnc_tensor_expect_t* tensor_expect = (ccv_nnc_tensor_expect_t*)cccalloc(symbolic_graph->tensor_symbol_info->rnum, sizeof(ccv_nnc_tensor_expect_t));
	// Ignore tensors that are already binded.
	for (i = 0; i < tensor_symbol_binding_size; i++)
		tensor_expect[tensor_symbol_bindings[i].d].flag = UNASSIGNED;
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
	// Collet head nodes and tail nodes for each tensor.
#define visitor(node, idx, _) \
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
#define visitor(node, idx, _) \
	do { \
		/* Remove tensor symbols that is for in-place operations (and it matches the start, end tensor). */ \
		if (ccv_nnc_cmd_support(node->cmd, CCV_NNC_COMPUTE_SUPPORT_INPLACE)) \
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
	ccv_nnc_tensor_arena_t* tensor_arena = _ccv_nnc_tensor_arena_new(tensor_symbol_info, symbolic_graph->tensor_symbol_info->rnum, exec_dep, tensor_expect);
	// Handle binded tensors.
	for (i = 0; i < tensor_symbol_binding_size; i++)
	{
		// For binded tensors, it shouldn't be assigned yet.
		assert(tensor_arena->vt_tensor[tensor_symbol_bindings[i].d] == 0);
		// I have to cast this, unfortunately.
		tensor_arena->vt_tensor[tensor_symbol_bindings[i].d] = (ccv_nnc_tensor_t*)tensor_bindings[i];
	}
	*tensor_arena_ref = tensor_arena;

	ccv_matrix_free(exec_dep);
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		if (tensor_expect[i].head)
			ccv_array_free(tensor_expect[i].head);
		if (tensor_expect[i].tail)
			ccv_array_free(tensor_expect[i].tail);
	}
	ccfree(tensor_expect);
	ccfree(tensor_symbol_info);

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
	ccv_nnc_tensor_t** max_inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_input_size);
	ccv_nnc_tensor_t** max_outputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_output_size);
#define visitor(node, idx, _) \
	do { \
		if (CCV_NO_GRAPH_EXEC(graph_exec[idx])) \
		{ \
			for (i = 0; i < node->input_size; i++) \
				max_inputs[i] = tensor_arena->vt_tensor[node->inputs[i]]; \
			for (i = 0; i < node->output_size; i++) \
				max_outputs[i] = tensor_arena->vt_tensor[node->outputs[i]]; \
			graph_exec[idx] = ccv_nnc_graph_exec(graph, node->cmd, node->hint, max_inputs, node->input_size, max_outputs, node->output_size); \
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
				graph_exec[outgoing] = ccv_nnc_graph_exec(graph, outgoing_node->cmd, outgoing_node->hint, max_inputs, outgoing_node->input_size, max_outputs, outgoing_node->output_size); \
			} \
			ccv_nnc_graph_exec_concat(graph, graph_exec[idx], graph_exec[outgoing]); \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	ccfree(max_inputs);
	ccfree(max_outputs);
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
	assert(symbol.d >= 0 && symbol.d < tensor_arena->vt_tensor_rnum);
	return tensor_arena->vt_tensor[symbol.d];
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

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_from_symbol(const ccv_nnc_graph_exec_arena_t* graph_exec_arena, const ccv_nnc_graph_exec_symbol_t symbol)
{
	assert(symbol.d >= 0 && symbol.d < graph_exec_arena->graph_exec_rnum);
	return graph_exec_arena->graph_exec[symbol.d];
}

void ccv_nnc_graph_exec_arena_free(ccv_nnc_graph_exec_arena_t* graph_exec_arena)
{
	ccfree(graph_exec_arena);
}
