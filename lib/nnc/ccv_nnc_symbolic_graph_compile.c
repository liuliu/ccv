#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_symbolic_graph.h"

typedef struct {
	int flag;
	int ref; // Reference to another tensor block. Start with 1.
	int buffer_ref; // Reference to a particular buffer. Start with 1.
	int graph_ref; // Reference to a particular graph. Start with 1.
	uint64_t size; // The size of the tensor expected.
	ccv_array_t* head; // The head nodes (it could be multiple if from the graph, one cannot determine which is the first).
	ccv_array_t* tail; // The tail nodes (it could be multiple if from the graph, one cannot determine which is the last).
} ccv_nnc_tensor_block_t; // Tensor Arena Block

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

typedef struct {
	int index;
	int companion; // The companion node index (the node that doesn't interfere with current one).
	uint64_t size;
} ccv_nnc_tensor_opt_t;

#define more_than(i1, i2, aux) ((i1).size >= (i2).size)
static CCV_IMPLEMENT_QSORT(_ccv_nnc_tensor_opt_sort_by_size, ccv_nnc_tensor_opt_t, more_than)
#undef more_than

// If every a's head is deterministically after b's tail
static int _ccv_nnc_tensor_block_head_after_tail(const ccv_sparse_matrix_t* const exec_dep, const ccv_nnc_tensor_block_t a, const ccv_nnc_tensor_block_t b)
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

typedef struct {
	ccv_array_t** alloc_dep;
	int buffer_size;
	int vt_block_size;
	int* vt_blocks; // A reference to the block, because blocks only contains available block (thus, doesn't consider alias etc.). -1 means no block pointed to. Starts at 0.
	uint64_t* buffers; // The buffer size for each buffer allocated.
	int block_size;
	struct {
		int buffer_ref; // A reference for block to which buffer to use. Starts at 0.
		int block_ref; // A reference to which block in the given tensor_block to use.
		uint64_t offset; // The offset of this block.
	}* blocks;
} ccv_nnc_tensor_alloc_prep_t;

static ccv_nnc_tensor_alloc_prep_t* _ccv_nnc_tensor_alloc_prep_new(const ccv_sparse_matrix_t* const exec_dep, const ccv_nnc_tensor_block_t* const tensor_blocks, const int tensor_block_size)
{
	// Compute how many dis-continuous buffers are needed.
	// We prefer to have several dis-continuous buffers instead of one big buffer because
	// in this way, we can rely on system memory allocators (jemalloc, tcmalloc, or CUDA's allocator)
	// to fully utilize memory.
	int i, j, k;
	ccv_array_t** alloc_dep = (ccv_array_t**)cccalloc(tensor_block_size, sizeof(ccv_array_t*));
	int computable_tensor_size = 0, available_tensor_size = 0;
	for (i = 0; i < tensor_block_size; i++)
		if (!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]))
		{
			// Tensors that we need the header info.
			++available_tensor_size;
			if (!TENSOR_EXPECT_ALIAS(tensor_blocks[i]))
				// Tensors that we actually need to compute (exclude the alias).
				++computable_tensor_size;
		}
	ccv_sparse_matrix_t* tensor_itf = ccv_sparse_matrix_new(tensor_block_size, tensor_block_size, CCV_8U | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	// Overlap count.
	for (i = 0; i < tensor_block_size; i++)
		for (j = i + 1; j < tensor_block_size; j++)
			if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]) && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[j]))
			{
				// If either of the tensor is const, it must interfere with each other.
				const uint8_t one = 1;
				if (TENSOR_EXPECT_CONST(tensor_blocks[i]) || TENSOR_EXPECT_CONST(tensor_blocks[j]))
					ccv_set_sparse_matrix_cell(tensor_itf, i, j, &one);
				else {
					// Otherwise, check to see if they interfere (default to yes).
					// If any of the i's head is deterministically later than j's tail
					// or any of the i's tail is deterministically earlier than j's head, they don't interfere.
					int i_hop_j = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[i], tensor_blocks[j]);
					int j_hop_i = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[j], tensor_blocks[i]);
					// It cannot be that both i can hop to j can j can hop to i.
					assert(!(i_hop_j > 0 && j_hop_i > 0));
					if (!i_hop_j && !j_hop_i)
						ccv_set_sparse_matrix_cell(tensor_itf, i, j, &one);
				}
			}
	int* oc = (int*)cccalloc(tensor_block_size, sizeof(int));
	for (i = 0; i < tensor_block_size; i++)
		for (j = 0; j < tensor_block_size; j++)
			// If these two tensors are still alive, analyze them.
			if (i != j && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]) && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[j]))
			{
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(i, j), ccv_max(i, j));
				// If their life time overlaps, compute how many tensors it overlap.
				if (cell.u8 && cell.u8[0] == 1)
					++oc[i];
			}
	int* assigned = (int*)cccalloc(tensor_block_size, sizeof(int));
	uint64_t* allocated_offset = (uint64_t*)cccalloc(tensor_block_size, sizeof(uint64_t));
	uint64_t* allocated_size = (uint64_t*)cccalloc(tensor_block_size, sizeof(uint64_t));
	int num_assigned = 0; 
	// I can do a bit optimization here to assign out const tensor first, but heck, this just works for now.
	// Allocation graph (assuming there is a source node, and a destination node, which is 0, and (tensor_block_size + 1)
	// The first channel denotes the bytes available for allocation,
	// the second channel denotes the offset available for the allocation,
	ccv_sparse_matrix_t* alloc = ccv_sparse_matrix_new(tensor_block_size + 2, tensor_block_size + 2, CCV_64S | CCV_C2, CCV_SPARSE_ROW_MAJOR, 0);
	ccv_array_t* opt = ccv_array_new(sizeof(ccv_nnc_tensor_opt_t), 1, 0);
	for (j = 0; j < computable_tensor_size;)
	{
		// Find the one with largest overlap, and it is not assigned.
		int max_oc = 0;
		ccv_array_clear(opt);
		for (i = 0; i < tensor_block_size; i++)
			if (oc[i] >= max_oc && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]) && !assigned[i])
			{
				ccv_nnc_tensor_opt_t a = {
					.size = tensor_blocks[i].size,
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
			for (k = 0; k < tensor_block_size; k++)
				// Find non-overlapping tensor that has larger size (of course, is unassigned).
				if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[k]) && !assigned[k] && tensor_blocks[k].size > a.size)
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(a.index, k), ccv_max(a.index, k));
					// Good, push to opt array.
					if (!cell.u8 || cell.u8[0] == 0)
					{
						ccv_nnc_tensor_opt_t b = a;
						b.companion = k;
						b.size = tensor_blocks[k].size;
						ccv_array_push(opt, &b);
					}
				}
		}
		// Order opt array by the size.
		_ccv_nnc_tensor_opt_sort_by_size((ccv_nnc_tensor_opt_t*)opt->data, opt->rnum, 0);
		// Assuming all tensors has the same data format (32F), therefore, we only need to consider the dimensional size.
		// Go through opt array again, this time, it is ordered by size, therefore, if we found a place to insert, we are good.
		int min_y = 0, min_x = tensor_block_size + 1, min_i = -1, min_hop = exec_dep->rows * 3;
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
				a_hop_c = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.companion], tensor_blocks[a.index]);
				c_hop_a = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.index], tensor_blocks[a.companion]);
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
						int y_hop_a = (y == 0) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.index], tensor_blocks[y - 1]); \
						int a_hop_x = (x == tensor_block_size + 1) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[x - 1], tensor_blocks[a.index]); \
						int hop = y_hop_a + a_hop_x; \
						/* a.index doesn't overlap with y and x (in between) */ \
						if ((y == 0 || y_hop_a) && (x == tensor_block_size + 1 || a_hop_x) && hop < min_hop) \
							min_y = y, min_x = x, min_hop = hop, \
							min_val[0] = ((uint64_t*)val)[0], min_val[1] = ((uint64_t*)val)[1]; \
					} else { \
						/* a.index doesn't overlap with y and x (in between) */ \
						/* a.companion doesn't overlap with y and x (in between) as well */ \
						/* because we know a.index is before a.companion (a can hop to c), */ \
						/* we can check if y can hop to a and then c can hop to x to determine. */ \
						if (a_hop_c > 0) \
						{ \
							int y_hop_a = (y == 0) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.index], tensor_blocks[y - 1]); \
							int c_hop_x = (x == tensor_block_size + 1) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[x - 1], tensor_blocks[a.companion]); \
							int hop = y_hop_a + c_hop_x; \
							if ((y == 0 || y_hop_a) && (x == tensor_block_size + 1 || c_hop_x) && hop < min_hop) \
								min_y = y, min_x = x, min_hop = hop, \
								min_val[0] = ((uint64_t*)val)[0], min_val[1] = ((uint64_t*)val)[1]; \
						} else { \
							int y_hop_c = (y == 0) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.companion], tensor_blocks[y - 1]); \
							int a_hop_x = (x == tensor_block_size + 1) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[x - 1], tensor_blocks[a.index]); \
							int hop = y_hop_c + a_hop_x; \
							if ((y == 0 || y_hop_c) && (x == tensor_block_size + 1 || a_hop_x) && hop < min_hop) \
								min_y = y, min_x = x, min_hop = hop, \
								min_val[0] = ((uint64_t*)val)[0], min_val[1] = ((uint64_t*)val)[1]; \
						} \
					} \
				} \
			} while (0)
			CCV_SPARSE_FOREACH(alloc, for_block);
#undef for_block
			// If I found a place, stop, and exit.
			if (min_y > 0 || min_x < tensor_block_size + 1)
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
			assert(min_x == tensor_block_size + 1 || assigned[min_x - 1] == assign_group);
		} else if (min_x < tensor_block_size + 1)
			assign_group = assigned[min_x - 1];
		// Assign out the selected one.
		assigned[a.index] = assign_group;
		// The offset for this one, should be either 0 (started a new group, when min_i == -1), or the offset on this edge.
		allocated_offset[a.index] = min_val[1];
		for (i = 0; i < tensor_block_size; i++)
			if (!assigned[i] && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]))
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
			for (i = 0; i < tensor_block_size; i++)
				if (!assigned[i] && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]))
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(i, a.companion), ccv_max(i, a.companion));
					if (cell.u8 && cell.u8[0] == 1)
						--oc[i];
				}
		}
		// If min_y is source and min_x is destination, we don't need to do anything, otherwise, decrease the weight on that edge.
		if (min_y != 0 || min_x != tensor_block_size + 1)
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
			int a_hop_c = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.companion], tensor_blocks[a.index]);
			int c_hop_a = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.index], tensor_blocks[a.companion]);
			// You can only hop from one direction, otherwise we have a loop.
			assert((a_hop_c > 0 && c_hop_a == 0) || (a_hop_c == 0 && c_hop_a > 0));
			if (a_hop_c > 0)
			{
				uint64_t val[2] = {
					tensor_blocks[a.index].size, min_val[1] // keep the offset
				};
				ccv_set_sparse_matrix_cell(alloc, min_y, a.index + 1, val);
				val[0] = a.size;
				assert(a.size == tensor_blocks[a.companion].size);
				ccv_set_sparse_matrix_cell(alloc, a.index + 1, a.companion + 1, val);
				ccv_set_sparse_matrix_cell(alloc, a.companion + 1, min_x, val);
				if (a.size > tensor_blocks[a.index].size)
				{
					// residual size connection between min_y and companion.
					val[0] = a.size - tensor_blocks[a.index].size;
					// offset need to be updated as well.
					val[1] = min_val[1] + tensor_blocks[a.index].size;
					ccv_set_sparse_matrix_cell(alloc, min_y, a.companion + 1, val);
				}
			} else {
				uint64_t val[2] = {
					a.size, min_val[1] // keep the offset
				};
				assert(a.size == tensor_blocks[a.companion].size);
				ccv_set_sparse_matrix_cell(alloc, min_y, a.companion + 1, val);
				val[0] = tensor_blocks[a.index].size;
				ccv_set_sparse_matrix_cell(alloc, a.companion + 1, a.index + 1, val);
				ccv_set_sparse_matrix_cell(alloc, a.index + 1, min_x, val);
				if (a.size > tensor_blocks[a.index].size)
				{
					// residual size connection between min_y and companion.
					val[0] = a.size - tensor_blocks[a.index].size;
					// offset need to be updated as well.
					val[1] = min_val[1] + tensor_blocks[a.index].size;
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
		if (((uint64_t*)val)[0] > 0 && y > 0 && x < tensor_block_size + 1) \
		{ \
			if (!alloc_dep[x - 1]) \
				alloc_dep[x - 1] = ccv_array_new(sizeof(int), 1, 0); \
			ccv_array_replace_int(alloc_dep[x - 1], y - 1, y - 1); \
		} \
	} while (0)
	CCV_SPARSE_FOREACH(alloc, for_block);
#undef for_block
	ccv_matrix_free(alloc);
	ccfree(oc);
	ccv_nnc_tensor_alloc_prep_t* alloc_prep = (ccv_nnc_tensor_alloc_prep_t*)ccmalloc(sizeof(ccv_nnc_tensor_alloc_prep_t) + sizeof(int) * tensor_block_size + sizeof(uint64_t) * num_assigned + sizeof(alloc_prep->blocks[0]) * available_tensor_size);
	alloc_prep->alloc_dep = alloc_dep;
	alloc_prep->vt_block_size = tensor_block_size;
	alloc_prep->vt_blocks = (int*)(alloc_prep + 1);
	alloc_prep->buffer_size = num_assigned;
	alloc_prep->buffers = (uint64_t*)(alloc_prep->vt_blocks + tensor_block_size);
	alloc_prep->block_size = available_tensor_size;
	alloc_prep->blocks = (void*)(alloc_prep->buffers + num_assigned);
	memcpy(alloc_prep->buffers, allocated_size, sizeof(uint64_t) * num_assigned);
	ccfree(allocated_size);
	j = 0;
	// Assigning out the tensors (in case of sharing tensors / in-place ops).
	for (i = 0; i < tensor_block_size; i++)
		if (!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]))
		{
			alloc_prep->blocks[j].block_ref = i;
			if (!TENSOR_EXPECT_ALIAS(tensor_blocks[i]))
			{
				alloc_prep->vt_blocks[i] = j;
				// Also, set its allocations.
				assert(assigned[i] > 0);
				alloc_prep->blocks[j].buffer_ref = assigned[i] - 1;
				alloc_prep->blocks[j].offset = allocated_offset[i];
				assert(allocated_offset[i] + tensor_blocks[i].size <= alloc_prep->buffers[assigned[i] - 1]);
			} else {
				alloc_prep->vt_blocks[i] = -1;
				alloc_prep->blocks[j].buffer_ref = -1;
				alloc_prep->blocks[j].offset = 0;
			}
			++j;
		}
	ccfree(allocated_offset);
	ccfree(assigned);
	return alloc_prep;
}

static void _ccv_nnc_tensor_alloc_prep_free(ccv_nnc_tensor_alloc_prep_t* alloc_prep)
{
	int i;
	for (i = 0; i < alloc_prep->vt_block_size; i++)
		if (alloc_prep->alloc_dep[i])
			ccv_array_free(alloc_prep->alloc_dep[i]);
	ccfree(alloc_prep->alloc_dep);
	ccfree(alloc_prep);
}

static ccv_nnc_tensor_arena_t* _ccv_nnc_tensor_arena_new(const ccv_nnc_tensor_alloc_prep_t* const alloc_prep, const ccv_nnc_tensor_block_t* const tensor_blocks, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const int tensor_symbol_info_size)
{
	// All tensors assigned out, now, the num_assigned is the number of dis-continuous buffers,
	// Each tensor have the designation in assigned array, and offset in allocated_offset.
	ccv_nnc_tensor_arena_t* tensor_arena = (ccv_nnc_tensor_arena_t*)ccmalloc(sizeof(ccv_nnc_tensor_arena_t) + sizeof(ccv_nnc_tensor_t*) * tensor_symbol_info_size + sizeof(tensor_arena->buffers[0]) * alloc_prep->buffer_size + sizeof(ccv_nnc_tensor_view_t) * (alloc_prep->block_size - 1));
	tensor_arena->vt_tensor_size = tensor_symbol_info_size;
	tensor_arena->vt_tensors = (ccv_nnc_tensor_t**)(tensor_arena->tensors + alloc_prep->block_size);
	tensor_arena->buffers = (void*)(tensor_arena->vt_tensors + tensor_symbol_info_size);
	tensor_arena->buffer_size = alloc_prep->buffer_size;
	int i, j;
	for (i = 0; i < alloc_prep->buffer_size; i++)
		tensor_arena->buffers[i].size = alloc_prep->buffers[i];
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
		for (i = 0; i < tensor_arena->buffer_size; i++)
			tensor_arena->buffers[i].ptr = (uint8_t*)cumalloc(device_id, tensor_arena->buffers[i].size);
	} else {
		assert(memory_type == CCV_TENSOR_CPU_MEMORY);
		for (i = 0; i < tensor_arena->buffer_size; i++)
			ccmemalign((void **)&tensor_arena->buffers[i].ptr, 16, tensor_arena->buffers[i].size);
	}
#else
	assert(memory_type == CCV_TENSOR_CPU_MEMORY);
	for (i = 0; i < tensor_arena->buffer_size; i++)
		ccmemalign((void **)&tensor_arena->buffers[i].ptr, 16, tensor_arena->buffers[i].size);
#endif
	j = 0;
	// Assigning out the tensors (in case of sharing tensors / in-place ops).
	memset(tensor_arena->vt_tensors, 0, sizeof(ccv_nnc_tensor_t*) * tensor_symbol_info_size);
	for (i = 0; i < alloc_prep->block_size; i++)
		if (alloc_prep->blocks[i].block_ref < tensor_symbol_info_size)
		{
			const int block_ref = alloc_prep->blocks[i].block_ref;
			const int buffer_ref = alloc_prep->blocks[i].buffer_ref;
			const uint64_t offset = alloc_prep->blocks[i].offset;
			if (!TENSOR_EXPECT_ALIAS(tensor_blocks[block_ref]))
			{
				tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)&tensor_arena->tensors[j];
				// Also, set its allocations.
				// Since tensor view is bit compatible with tensor, we can just cast.
				ccv_nnc_tensor_t tensor = ccv_nnc_tensor(tensor_arena->buffers[buffer_ref].ptr + offset, tensor_symbol_info[block_ref].info, 0);
				memset(tensor_arena->tensors + j, 0, sizeof(ccv_nnc_tensor_view_t));
				memcpy(tensor_arena->tensors + j, &tensor, sizeof(ccv_nnc_tensor_t));
				assert(offset + tensor_blocks[block_ref].size <= tensor_arena->buffers[buffer_ref].size);
				++j;
			}
		}
	for (i = 0; i < alloc_prep->block_size; i++)
		if (alloc_prep->blocks[i].block_ref < tensor_symbol_info_size)
		{
			const int block_ref = alloc_prep->blocks[i].block_ref;
			if (TENSOR_EXPECT_ALIAS(tensor_blocks[block_ref]))
			{
				// Assigning out the tensor aliases.
				assert(tensor_symbol_info[block_ref].alias_ref);
				int alias_ref = tensor_symbol_info[block_ref].alias_ref - 1;
				// It referenced to is not an alias.
				assert(tensor_arena->vt_tensors[alias_ref]);
				assert(!CCV_IS_TENSOR_VIEW(tensor_arena->vt_tensors[alias_ref]));
				tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)&tensor_arena->tensors[j];
				// If there is no ofs, and inc is the same as dim, we take a shortcut and just init as normal tensor.
				if (memcmp(ccv_nnc_no_ofs, tensor_symbol_info[block_ref].ofs, sizeof(ccv_nnc_no_ofs)) == 0 &&
					memcmp(tensor_symbol_info[block_ref].inc, tensor_symbol_info[block_ref].info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
				{
					ccv_nnc_tensor_t tensor = ccv_nnc_tensor(tensor_arena->vt_tensors[alias_ref]->data.u8, tensor_symbol_info[block_ref].info, 0);
					memset(tensor_arena->tensors + j, 0, sizeof(ccv_nnc_tensor_view_t));
					memcpy(tensor_arena->tensors + j, &tensor, sizeof(ccv_nnc_tensor_t));
				} else {
					// Otherwise initialize a tensor view
					// 1). Simple case, if the inc is equal to original tensor, just init a tensor view.
					if (memcmp(tensor_arena->vt_tensors[alias_ref]->info.dim, tensor_symbol_info[block_ref].inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
						tensor_arena->tensors[j] = ccv_nnc_tensor_view(tensor_arena->vt_tensors[alias_ref], tensor_symbol_info[block_ref].ofs, tensor_symbol_info[block_ref].info.dim);
					else {
						// Otherwise, create the tensor first, and then create the tensor view off the new tensor.
						ccv_nnc_tensor_param_t info = tensor_symbol_info[block_ref].info;
						memcpy(info.dim, tensor_symbol_info[block_ref].inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
						assert(ccv_nnc_tensor_count(info) <= ccv_nnc_tensor_count(tensor_arena->vt_tensors[alias_ref]->info));
						ccv_nnc_tensor_t tensor = ccv_nnc_tensor(tensor_arena->vt_tensors[alias_ref]->data.u8, info, 0);
						tensor_arena->tensors[j] = ccv_nnc_tensor_view(&tensor, tensor_symbol_info[block_ref].ofs, tensor_symbol_info[block_ref].info.dim);
					}
				}
				++j;
			}
		}
	assert(j == alloc_prep->block_size);
	for (i = 0; i < tensor_symbol_info_size; i++)
		// It could be binded tensor (or unused), in that case, it doesn't have a ref.
		if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]) && tensor_blocks[i].ref)
		{
			// It must be available.
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[tensor_blocks[i].ref - 1]));
			assert(tensor_arena->vt_tensors[tensor_blocks[i].ref - 1]);
			tensor_arena->vt_tensors[i] = tensor_arena->vt_tensors[tensor_blocks[i].ref - 1];
		}
	return tensor_arena;
}

static void _ccv_nnc_tensor_block_add_exec(const ccv_sparse_matrix_t* const exec_dep, const int idx, ccv_nnc_tensor_block_t tensor_blocks)
{
	int i, found = 0;
	// Try to insert head.
	ccv_array_t* head = tensor_blocks.head;
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
	ccv_array_t* tail = tensor_blocks.tail;
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

ccv_nnc_tensor_t* ccv_nnc_tensor_from_symbol(const ccv_nnc_tensor_arena_t* const tensor_arena, const ccv_nnc_tensor_symbol_t symbol)
{
	assert(symbol.d >= 0 && symbol.d < tensor_arena->vt_tensor_size);
	return tensor_arena->vt_tensors[symbol.d];
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_from_symbol(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena, const ccv_nnc_graph_exec_symbol_t symbol)
{
	assert(symbol.d >= 0 && symbol.d < graph_exec_arena->graph_exec_rnum);
	return graph_exec_arena->graph_exec[symbol.d];
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_source(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	return graph_exec_arena->source;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_destination(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	return graph_exec_arena->destination;
}

typedef struct ccv_nnc_symbolic_graph_prep_s {
	int tensor_block_size;
	ccv_nnc_tensor_block_t* tensor_blocks;
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info;
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info;
	ccv_nnc_tensor_alloc_prep_t* alloc_prep;
	int sub_prep_size;
	struct ccv_nnc_symbolic_graph_prep_s** sub_preps; // The preps of its sub-graphs.
} ccv_nnc_symbolic_graph_prep_t;

// Plan out how we allocate tensor (should I do optimizations on graph here or not at all?).
static ccv_nnc_symbolic_graph_prep_t* _ccv_nnc_symbolic_graph_prep_new(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, const ccv_nnc_graph_exec_symbol_info_t* const p_exec_symbol_info, const int p_exec_symbol_info_size)
{
	assert(source_size > 0);
	assert(destination_size > 0);
	// First, fill all the "auto" holes.
	// This is the symbol table that with "auto" info filled up.
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * symbolic_graph->tensor_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * symbolic_graph->exec_symbol_info->rnum);
	ccv_nnc_symbolic_graph_symbol_infer(symbolic_graph, sources, source_size, destinations, destination_size, p_tensor_symbol_info, p_tensor_symbol_info_size, tensor_symbol_info, exec_symbol_info);
	int i, j;
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
	int tensor_block_size = symbolic_graph->tensor_symbol_info->rnum;
	ccv_nnc_tensor_block_t* tensor_blocks = (ccv_nnc_tensor_block_t*)cccalloc(tensor_block_size, sizeof(ccv_nnc_tensor_block_t));
	// The reason is that I need to make everyone of them to be unassigned unless it is used somewhere. It
	// happens that I have to loop through all relevant node to find out if one is used or not.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		tensor_blocks[i].flag = UNASSIGNED;
#define visitor(node, idx, ...) \
	do { \
		for (i = 0; i < node->input_size; i++) \
			if (node->inputs[i] >= 0) \
				tensor_blocks[node->inputs[i]].flag = 0; \
		for (i = 0; i < node->output_size; i++) \
			if (node->outputs[i] >= 0) \
				tensor_blocks[node->outputs[i]].flag = 0; \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Ignore tensors that are already binded, no matter if it is used or not.
	for (i = 0; i < tensor_bind_size; i++)
		// If there is a tensor binded, then it is unassigned, otherwise, we will allocate as constant.
		tensor_blocks[tensor_binds[i].symbol.d].flag = tensor_binds[i].tensor ? UNASSIGNED : CONST_TENSOR;
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		// Check no tensor info is auto now.
		assert(!ccv_nnc_is_tensor_auto(tensor_symbol_info[i].info));
		if (tensor_symbol_info[i].alias_ref)
		{
			// An alias cannot ref to another alias.
			assert(!tensor_symbol_info[tensor_symbol_info[i].alias_ref - 1].alias_ref);
			tensor_blocks[i].flag = ALIAS;
		}
		// If this tensor is not expected to be unassigned, allocate the arrays for s and t.
		if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]))
		{
			tensor_blocks[i].head = ccv_array_new(sizeof(int), 0, 0);
			tensor_blocks[i].tail = ccv_array_new(sizeof(int), 0, 0);
			// Cache tensor size (align to 16 bytes).
			tensor_blocks[i].size = (uint64_t)ccv_nnc_tensor_data_size(tensor_symbol_info[i].info);
		}
	}
	// Collect head nodes and tail nodes for each tensor.
#define visitor(node, idx, ...) \
	do { \
		for (i = 0; i < node->input_size; i++) \
		{ \
			int d = node->inputs[i]; \
			if (d < 0) \
				continue; \
			if (TENSOR_EXPECT_ALIAS(tensor_blocks[d])) \
				d = tensor_symbol_info[d].alias_ref - 1; \
			if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[d])) \
				continue; \
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[d])); \
			/* If this is first encounter, its head starts (this tensor is init'ed outside of the graph)
			 * from the very beginning of the graph life-cycle and ends here. */ \
			if (tensor_blocks[d].head->rnum == 0) \
				for (j = 0; j < source_size; j++) \
					_ccv_nnc_tensor_block_add_exec(exec_dep, sources[j].d, tensor_blocks[d]); \
			_ccv_nnc_tensor_block_add_exec(exec_dep, idx, tensor_blocks[d]); \
		} \
		for (i = 0; i < node->output_size; i++) \
		{ \
			int d = node->outputs[i]; \
			if (d < 0) \
				continue; \
			if (TENSOR_EXPECT_ALIAS(tensor_blocks[d])) \
				d = tensor_symbol_info[d].alias_ref - 1; \
			if (TENSOR_EXPECT_CONST(tensor_blocks[d]) || \
				TENSOR_EXPECT_UNASSIGNED(tensor_blocks[d])) \
				continue; \
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[d])); \
			_ccv_nnc_tensor_block_add_exec(exec_dep, idx, tensor_blocks[d]); \
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
				if (ref < 0) \
					continue; \
				while (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) && tensor_blocks[ref].ref) \
					ref = tensor_blocks[ref].ref - 1; \
				const ccv_nnc_tensor_symbol_info_t x_symbol = tensor_symbol_info[ref]; \
				if (!TENSOR_EXPECT_CONST(tensor_blocks[ref]) && \
					TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) && \
					tensor_blocks[ref].tail->rnum == 1) \
					for (y = 0; y < node->output_size; y++) \
						/* Only proceed if the input symbol is different from the output symbol, */ \
						/* and the input symbol meets the output symbol exactly at the same spot. */ \
						if (node->outputs[y] >= 0 && \
							ref != node->outputs[y] && \
							!TENSOR_EXPECT_CONST(tensor_blocks[node->outputs[y]]) && \
							TENSOR_EXPECT_COMPUTABLE(tensor_blocks[node->outputs[y]]) && \
							tensor_blocks[node->outputs[y]].head->rnum == 1 && \
							*(int*)ccv_array_get(tensor_blocks[ref].tail, 0) == *(int*)ccv_array_get(tensor_blocks[node->outputs[y]].head, 0)) \
						{ \
							const ccv_nnc_tensor_symbol_info_t y_symbol = tensor_symbol_info[node->outputs[y]]; \
							/* If dimension matches perfectly, then we can assign y_symbol to x. */ \
							if (memcmp(x_symbol.info.dim, y_symbol.info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0) \
							{ \
								ccv_array_free(tensor_blocks[ref].tail); \
								tensor_blocks[ref].tail = tensor_blocks[node->outputs[y]].tail; \
								/* Mark the original as unassigned, set its reference to the head of the current node. */ \
								ccv_array_free(tensor_blocks[node->outputs[y]].head); \
								tensor_blocks[node->outputs[y]].flag = UNASSIGNED; \
								tensor_blocks[node->outputs[y]].ref = ref + 1; \
								tensor_blocks[node->outputs[y]].size = 0; \
								tensor_blocks[node->outputs[y]].head = 0; \
								tensor_blocks[node->outputs[y]].tail = 0; \
							} \
						} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor

	ccv_nnc_symbolic_graph_prep_t** sub_preps = symbolic_graph->sub_graphs && symbolic_graph->sub_graphs->rnum ? (ccv_nnc_symbolic_graph_prep_t**)cccalloc(symbolic_graph->sub_graphs->rnum, sizeof(ccv_nnc_symbolic_graph_prep_t*)) : 0;
	// Now, everything is prepared, tensor life is analyzed, inplace operations are collapsed, all tensor symbols and hints
	// are automatically filled in.
	// In true recursive fashion, I need to call all the sub graphs and do the pre compilation for them one by one.
	int* buffer_assigned = 0;
#define visitor(node, idx, ...) \
	do { \
		if (node->graph_ref) \
		{ \
			ccv_nnc_symbolic_graph_t* while_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, node->graph_ref - 1); \
			ccv_nnc_symbolic_graph_prep_t* const prep = _ccv_nnc_symbolic_graph_prep_new(while_graph, tensor_binds, tensor_bind_size, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(while_graph->sources, 0), while_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(while_graph->destinations, 0), while_graph->destinations->rnum, tensor_symbol_info, symbolic_graph->tensor_symbol_info->rnum, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum); \
			sub_preps[node->graph_ref - 1] = prep; \
			const ccv_nnc_tensor_alloc_prep_t* const alloc_prep = prep->alloc_prep; \
			const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = prep->tensor_symbol_info; \
			if (!buffer_assigned) \
				buffer_assigned = (int*)cccalloc(alloc_prep->buffer_size, sizeof(int)); \
			else { \
				buffer_assigned = (int*)ccrealloc(buffer_assigned, sizeof(int) * alloc_prep->buffer_size); \
				memset(buffer_assigned, 0, sizeof(int) * alloc_prep->buffer_size); \
			} \
			for (i = 0; i < alloc_prep->block_size; i++) \
			{ \
				const int block_ref = alloc_prep->blocks[i].block_ref; \
				const int buffer_ref = alloc_prep->blocks[i].buffer_ref; \
				if (block_ref < while_graph->tensor_symbol_info->rnum) \
				{ \
					if (tensor_symbol_info[block_ref].p_ref) \
					{ \
						int p_ref = tensor_symbol_info[block_ref].p_ref - 1; \
						/* Need to go through refs. Since we reuse the tensor block for this input, it now has to have allocate at least this much space. */ \
						while (tensor_blocks[p_ref].ref) \
							p_ref = tensor_blocks[p_ref].ref - 1; \
						tensor_blocks[p_ref].size = ccv_max(alloc_prep->buffers[buffer_ref], tensor_blocks[p_ref].size); \
						/* We are good, mark this buffer as assigned out. */ \
						buffer_assigned[buffer_ref] = 1; \
					} \
				} \
			} \
			int unassigned_buffer_size = 0; \
			for (i = 0; i < alloc_prep->buffer_size; i++) \
				if (!buffer_assigned[i]) \
					++unassigned_buffer_size; \
			if (unassigned_buffer_size) \
			{ \
				/* Anonymous block, allocate additional tensor blocks for this. */ \
				/* This is either because this is an internal tensor (don't have p_ref) */ \
				/* or it is an anonymous block itself within the sub graphs of this while graph. */ \
				tensor_blocks = (ccv_nnc_tensor_block_t*)ccrealloc(tensor_blocks, sizeof(ccv_nnc_tensor_block_t) * (tensor_block_size + unassigned_buffer_size)); \
				memset(tensor_blocks + tensor_block_size, 0, sizeof(ccv_nnc_tensor_block_t) * unassigned_buffer_size); \
				for (i = 0; i < alloc_prep->buffer_size; i++) \
					if (!buffer_assigned[i]) \
					{ \
						tensor_blocks[tensor_block_size].size = alloc_prep->buffers[i]; \
						tensor_blocks[tensor_block_size].graph_ref = node->graph_ref; \
						tensor_blocks[tensor_block_size].buffer_ref = i + 1; \
						tensor_blocks[tensor_block_size].head = ccv_array_new(sizeof(int), 1, 0); \
						ccv_array_push(tensor_blocks[tensor_block_size].head, &idx); \
						tensor_blocks[tensor_block_size].tail = ccv_array_new(sizeof(int), 1, 0); \
						ccv_array_push(tensor_blocks[tensor_block_size].tail, &idx); \
						/* ref and flags are both 0. */ \
					} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	if (buffer_assigned)
		ccfree(buffer_assigned);

	// It is time to guess what's the best tensor placement and create the opaque tensor arena. The alloc_dep will return
	// the allocation dependencies, thus, which tensor is reused to the existing tensor.
	ccv_nnc_symbolic_graph_prep_t* prep = (ccv_nnc_symbolic_graph_prep_t*)ccmalloc(sizeof(ccv_nnc_symbolic_graph_prep_t));
	ccv_nnc_tensor_alloc_prep_t* alloc_prep = _ccv_nnc_tensor_alloc_prep_new(exec_dep, tensor_blocks, tensor_block_size);
	ccv_matrix_free(exec_dep);
	prep->sub_prep_size = symbolic_graph->sub_graphs ? symbolic_graph->sub_graphs->rnum : 0;
	prep->sub_preps = sub_preps;
	prep->exec_symbol_info = exec_symbol_info;
	prep->tensor_symbol_info = tensor_symbol_info;
	prep->tensor_blocks = tensor_blocks;
	prep->tensor_block_size = tensor_block_size;
	prep->alloc_prep = alloc_prep;
	return prep;
}

static void _ccv_nnc_symbolic_graph_prep_free(ccv_nnc_symbolic_graph_prep_t* prep)
{
	int i;
	for (i = 0; i < prep->tensor_block_size; i++)
	{
		if (prep->tensor_blocks[i].head)
			ccv_array_free(prep->tensor_blocks[i].head);
		if (prep->tensor_blocks[i].tail)
			ccv_array_free(prep->tensor_blocks[i].tail);
	}
	for (i = 0; i < prep->sub_prep_size; i++)
		if (prep->sub_preps[i])
			_ccv_nnc_symbolic_graph_prep_free(prep->sub_preps[i]);
	if (prep->sub_preps)
		ccfree(prep->sub_preps);
	ccfree(prep->tensor_blocks);
	ccfree(prep->tensor_symbol_info);
	ccfree(prep->exec_symbol_info);
	_ccv_nnc_tensor_alloc_prep_free(prep->alloc_prep);
	ccfree(prep);
}

void ccv_nnc_symbolic_graph_compile(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, ccv_nnc_graph_t** const graph_ref, ccv_nnc_tensor_arena_t** const tensor_arena_ref, ccv_nnc_graph_exec_arena_t** const graph_exec_arena_ref)
{
	assert(graph_ref);
	assert(tensor_arena_ref);
	assert(graph_exec_arena_ref);
	ccv_nnc_symbolic_graph_prep_t* graph_prep = _ccv_nnc_symbolic_graph_prep_new(symbolic_graph, tensor_binds, tensor_bind_size, sources, source_size, destinations, destination_size, 0, 0, 0, 0);
	ccv_nnc_tensor_arena_t* tensor_arena = _ccv_nnc_tensor_arena_new(graph_prep->alloc_prep, graph_prep->tensor_blocks, graph_prep->tensor_symbol_info, symbolic_graph->tensor_symbol_info->rnum);
	int i, j, k;
	// Handle binded tensors.
	for (i = 0; i < tensor_bind_size; i++)
	{
		if (!tensor_binds[i].tensor) // If there is no tensor binded, it is a constant, we allocated in arena.
			continue;
		// For binded tensors, it shouldn't be assigned yet.
		assert(tensor_arena->vt_tensors[tensor_binds[i].symbol.d] == 0);
		// I have to cast this, unfortunately.
		tensor_arena->vt_tensors[tensor_binds[i].symbol.d] = (ccv_nnc_tensor_t*)tensor_binds[i].tensor;
	}
	*tensor_arena_ref = tensor_arena;

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
		max_input_size = ccv_max(max_input_size, graph_prep->exec_symbol_info[i].input_size);
		max_output_size = ccv_max(max_input_size, graph_prep->exec_symbol_info[i].output_size);
		graph_exec[i].graph = 0;
	}
	ccv_nnc_tensor_t** max_inputs = max_input_size > 0 ? (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_input_size) : 0;
	ccv_nnc_tensor_t** max_outputs = max_output_size > 0 ? (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_output_size) : 0;
	const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = graph_prep->exec_symbol_info;
#define visitor(node, idx, ...) \
	do { \
		if (CCV_NO_GRAPH_EXEC(graph_exec[idx])) \
		{ \
			for (i = 0; i < node->input_size; i++) \
				max_inputs[i] = node->inputs[i] >= 0 ? tensor_arena->vt_tensors[node->inputs[i]] : 0; \
			for (i = 0; i < node->output_size; i++) \
				max_outputs[i] = node->outputs[i] >= 0 ? tensor_arena->vt_tensors[node->outputs[i]] : 0; \
			graph_exec[idx] = ccv_nnc_graph_exec_new(graph, node->cmd, node->hint, max_inputs, node->input_size, max_outputs, node->output_size); \
		} \
		if (!node->outgoings) \
			break; \
		for (i = 0; i < node->outgoings->rnum; i++) \
		{ \
			int outgoing = *(int*)ccv_array_get(node->outgoings, i); \
			if (CCV_NO_GRAPH_EXEC(graph_exec[outgoing])) \
			{ \
				const ccv_nnc_graph_exec_symbol_info_t* const outgoing_node = exec_symbol_info + outgoing; \
				for (j = 0; j < outgoing_node->input_size; j++) \
					max_inputs[j] = outgoing_node->inputs[j] >= 0 ? tensor_arena->vt_tensors[outgoing_node->inputs[j]] : 0; \
				for (j = 0; j < outgoing_node->output_size; j++) \
					max_outputs[j] = outgoing_node->outputs[j] >= 0 ? tensor_arena->vt_tensors[outgoing_node->outputs[j]] : 0; \
				graph_exec[outgoing] = ccv_nnc_graph_exec_new(graph, outgoing_node->cmd, outgoing_node->hint, max_inputs, outgoing_node->input_size, max_outputs, outgoing_node->output_size); \
			} \
			ccv_nnc_graph_exec_concat(graph, graph_exec[idx], graph_exec[outgoing]); \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	int source_exec_created = 0;
	const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = graph_prep->tensor_symbol_info;
	const ccv_nnc_tensor_block_t* const tensor_blocks = graph_prep->tensor_blocks;
	ccv_array_t* const* const alloc_dep = graph_prep->alloc_prep->alloc_dep;
	// After the graph is materialized, we need to handle the case that some of these tensors require to be initialized to zero before use.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		if (tensor_symbol_info[i].flags & CCV_NNC_SYM_TENSOR_INIT_ZEROS)
		{
			int ref = i;
			while (tensor_symbol_info[ref].alias_ref)
				ref = tensor_symbol_info[ref].alias_ref - 1;
			while (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) && tensor_blocks[ref].ref)
				ref = tensor_blocks[ref].ref - 1;
			// This is not computable. It could be that we marked a const tensor as init zero.
			if (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]))
				continue;
			// If this tensor is not used by any exec, we don't need to init at all. Skip.
			if (!tensor_blocks[ref].head || tensor_blocks[ref].head->rnum == 0)
				continue;
			ccv_nnc_tensor_t* tensor = tensor_arena->vt_tensors[ref];
			// Now, we have the original tensor, we can get the actual tensor, and construct the set command.
			ccv_nnc_graph_exec_t set_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_SET_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, &tensor, 1);
			for (j = 0; j < tensor_blocks[ref].head->rnum; j++)
			{
				const int outgoing = *(int*)ccv_array_get(tensor_blocks[ref].head, j);
				ccv_nnc_graph_exec_concat(graph, set_exec, graph_exec[outgoing]);
			}
			int flag = 0;
			if (alloc_dep[ref])
				for (j = 0; j < alloc_dep[ref]->rnum; j++)
				{
					const int d = *(int*)ccv_array_get(alloc_dep[ref], j);
					// This is from alloc_dep, it should be computable.
					assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[d]));
					if (tensor_blocks[d].tail)
						for (k = 0; k < tensor_blocks[d].tail->rnum; k++)
						{
							const int incoming = *(int*)ccv_array_get(tensor_blocks[d].tail, j);
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
	_ccv_nnc_symbolic_graph_prep_free(graph_prep);
	if (max_inputs)
		ccfree(max_inputs);
	if (max_outputs)
		ccfree(max_outputs);
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

void ccv_nnc_tensor_arena_free(ccv_nnc_tensor_arena_t* const tensor_arena)
{
	int i;
#ifdef HAVE_CUDA
	if (tensor_arena->memory_type == CCV_TENSOR_GPU_MEMORY)
	{
		for (i = 0; i < tensor_arena->buffer_size; i++)
			cufree(tensor_arena->device_id, tensor_arena->buffers[i].ptr);
	} else {
		assert(tensor_arena->memory_type == CCV_TENSOR_CPU_MEMORY);
		for (i = 0; i < tensor_arena->buffer_size; i++)
			ccfree(tensor_arena->buffers[i].ptr);
	}
#else
	assert(tensor_arena->memory_type == CCV_TENSOR_CPU_MEMORY);
	for (i = 0; i < tensor_arena->buffer_size; i++)
		ccfree(tensor_arena->buffers[i].ptr);
#endif
	ccfree(tensor_arena);
}

void ccv_nnc_graph_exec_arena_free(ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	ccfree(graph_exec_arena);
}
