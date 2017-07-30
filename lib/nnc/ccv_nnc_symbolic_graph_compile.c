#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_graph.h"
#include "_ccv_nnc_symbolic_graph.h"

typedef struct {
	int flag;
	int ref; // Reference to another tensor block. Start with 1.
	int graph_ref; // Reference to a particular graph. Start with 1.
	int companion_ref; // Reference to another block that they two share the same memory region. Start with 1. the current crude implementation requires the two mutually be companion. Because there are two, we took the one that companion_ref <= i as the primary and companion_ref > i is the secondary. For allocation algorithm, we use the primary throughout.
	ccv_array_t* r_refs; // If this is referenced by another block, the array point back to these blocks. Start with 1.
	uint64_t size; // The size of the tensor expected.
	int p_refs[2]; // Reference to the parent tensor block, at max there will be only two. Start with 1.
	ccv_array_t* head; // The head nodes (it could be multiple if from the graph, one cannot determine which is the first).
	ccv_array_t* tail; // The tail nodes (it could be multiple if from the graph, one cannot determine which is the last).
} ccv_nnc_tensor_block_t; // Tensor Arena Block

enum {
	UNASSIGNED = 0x1,
	ALIAS = 0x2,
	CONST_TENSOR = 0x3,
	READ_ONLY = 0x4,
	WRITE_ONLY = 0x8,
	READ_WRITE = 0xc,
	ANONYMOUS = 0x10, // Mark this block as anonymous (thus, not reference to any specific tensor).
};

#define TENSOR_EXPECT_ORDINARY(t) ((t.flag & 0x3) == 0)
#define TENSOR_EXPECT_SET_ORDINARY(t) (t.flag = (t.flag & ~0x3))
#define TENSOR_EXPECT_UNASSIGNED(t) ((t.flag & 0x3) == UNASSIGNED)
#define TENSOR_EXPECT_SET_UNASSIGNED(t) (t.flag = ((t.flag & ~0x3) | UNASSIGNED))
#define TENSOR_EXPECT_ALIAS(t) ((t.flag & 0x3) == ALIAS)
#define TENSOR_EXPECT_CONST(t) ((t.flag & 0x3) == CONST_TENSOR)
#define TENSOR_EXPECT_COMPUTABLE(t) (!TENSOR_EXPECT_ALIAS(t) && !TENSOR_EXPECT_UNASSIGNED(t))
#define TENSOR_READ_WRITE(t) (t.flag & 0xc)
#define TENSOR_SET_READ_WRITE(t, rw) (t.flag = ((t.flag & ~0xc) | rw))
#define TENSOR_SET_ANONYMOUS(t) (t.flag = (t.flag & ~0x10 | ANONYMOUS))
#define TENSOR_IS_ANONYMOUS(t) (t.flag & ANONYMOUS)

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
	struct {
		int p_ref; // Reference to the upper level block, Starts at 1.
		uint64_t size; // The size of the buffer allocated.
	}* buffers;
	int block_size;
	struct {
		int buffer_ref; // A reference for block to which buffer to use. Starts at 0.
		int block_ref; // A reference to which block in the given tensor_block to use.
		uint64_t offset; // The offset of this block.
	}* blocks;
} ccv_nnc_tensor_alloc_prep_t;

typedef struct ccv_nnc_symbolic_graph_prep_s {
	intptr_t graph_ref;
	int dup_tensor_block_size;
	ccv_nnc_tensor_block_t* tensor_blocks;
	int tensor_symbol_info_size;
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info;
	int exec_symbol_info_size;
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info;
	int tensor_block_size;
	int* dup_tensor_block_ref;
	ccv_nnc_tensor_alloc_prep_t* alloc_prep;
	struct ccv_nnc_symbolic_graph_prep_s* p;
	int sub_prep_size;
	struct ccv_nnc_symbolic_graph_prep_s** sub_preps; // The preps of its sub-graphs.
	// Structures that don't require to be freed after deallocation.
	ccv_nnc_graph_t* graph; // materialized graph.
} ccv_nnc_symbolic_graph_prep_t;

static ccv_nnc_tensor_alloc_prep_t* _ccv_nnc_tensor_alloc_prep_new(const ccv_sparse_matrix_t* const exec_dep, const ccv_nnc_tensor_block_t* const tensor_blocks, const int tensor_block_size)
{
	// Compute how many dis-continuous buffers are needed.
	// We prefer to have several dis-continuous buffers instead of one big buffer because
	// in this way, we can rely on system memory allocators (jemalloc, tcmalloc, or CUDA's allocator)
	// to fully utilize memory.
	int i, j, k;
	ccv_array_t** alloc_dep = (ccv_array_t**)cccalloc(tensor_block_size, sizeof(ccv_array_t*));
	int allocable_tensor_size = 0, available_tensor_size = 0;
	for (i = 0; i < tensor_block_size; i++)
		if (!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]))
		{
			// Tensors that we need the header info.
			++available_tensor_size;
			if (!TENSOR_EXPECT_ALIAS(tensor_blocks[i]))
				// Tensors that we actually need to allocate (exclude the alias).
				++allocable_tensor_size;
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
	for (j = 0; j < allocable_tensor_size;)
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
					.companion = -1, // If already have a designated companion, use that.
				};
				if (tensor_blocks[i].companion_ref)
					a.size = ccv_max(a.size, tensor_blocks[tensor_blocks[i].companion_ref - 1].size);
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
			// Already have a designated companion.
			if (a.companion >= 0)
				continue;
			const int companion_ref = tensor_blocks[i].companion_ref - 1;
			for (k = 0; k < tensor_block_size; k++)
				// Find non-overlapping tensor that has larger size (of course, is unassigned and is not one with designated companion).
				if (k != a.index && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[k]) && !assigned[k] && tensor_blocks[k].size > a.size)
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(a.index, k), ccv_max(a.index, k));
					// Good, push to opt array.
					if (cell.u8 && cell.u8[0] == 1)
						continue;
					if (companion_ref >= 0 && companion_ref != k)
					{
						// Have to make sure k doesn't interfere with the designated companion as well.
						ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(companion_ref, k), ccv_max(companion_ref, k));
						if (cell.u8 && cell.u8[0] == 1)
							continue;
					}
					ccv_nnc_tensor_opt_t b = a;
					b.companion = k;
					b.size = tensor_blocks[k].size;
					ccv_array_push(opt, &b);
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
			// Now, determine the order between a and c. After this, we can always check whether y
			// can hop to the earliest one and if the latest one can hop to x.
			// The earliest one will be called p and the latest one will be called q.
			int p = a.index;
			int q = a.index;
			if (a.companion >= 0)
			{
				const int a_hop_c = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.companion], tensor_blocks[a.index]);
				const int c_hop_a = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.index], tensor_blocks[a.companion]);
				assert((a_hop_c > 0 && c_hop_a == 0) || (a_hop_c == 0 && c_hop_a > 0));
				if (a_hop_c > 0)
					q = a.companion;
				else
					p = a.companion;
			}
			if (tensor_blocks[a.index].companion_ref && a.companion != tensor_blocks[a.index].companion_ref - 1)
			{
				const int companion_ref = tensor_blocks[a.index].companion_ref - 1;
				const int b_hop_p = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[p], tensor_blocks[companion_ref]);
				if (b_hop_p > 0)
					p = companion_ref;
				else {
					const int q_hop_b = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[companion_ref], tensor_blocks[q]);
					if (q_hop_b > 0)
						q = companion_ref;
					else { // Otherwise, b is in between p and q.
						const int p_hop_b = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[companion_ref], tensor_blocks[p]);
						const int b_hop_q = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[q], tensor_blocks[companion_ref]);
						assert(p_hop_b > 0 && b_hop_q > 0);
					}
				}
			}
#define for_block(y, x, val) do { \
				/* y is always earlier than x, but this is hard to assert now. */ \
				/* If this edge satisfy the requirement, now we need to find the ones with tightest possible bounds. */ \
				/* Thus, the hop between y and x (through a) should be smallest ones. */ \
				if (((uint64_t*)val)[0] >= a.size) \
				{ \
					int y_hop_p = (y == 0) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[p], tensor_blocks[y - 1]); \
					int q_hop_x = (x == tensor_block_size + 1) ? exec_dep->rows : _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[x - 1], tensor_blocks[q]); \
					int hop = y_hop_p + q_hop_x; \
					/* a.index doesn't overlap with y and x (in between) */ \
					if ((y == 0 || y_hop_p) && (x == tensor_block_size + 1 || q_hop_x) && hop < min_hop) \
						min_y = y, min_x = x, min_hop = hop, \
						min_val[0] = ((uint64_t*)val)[0], min_val[1] = ((uint64_t*)val)[1]; \
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
		int strings[3] = {
			a.index + 1
		};
		int string_size = 1;
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
			const int a_hop_c = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a.companion], tensor_blocks[a.index]);
			if (a_hop_c > 0)
				strings[1] = a.companion + 1;
			else {
				strings[1] = strings[0];
				strings[0] = a.companion + 1;
			}
			++string_size;
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
		// Assign out designated companion if it exist.
		if (tensor_blocks[a.index].companion_ref && a.companion != tensor_blocks[a.index].companion_ref - 1)
		{
			const int companion_ref = tensor_blocks[a.index].companion_ref - 1;
			const int b_hop_p = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[strings[0] - 1], tensor_blocks[companion_ref]);
			if (b_hop_p > 0)
			{
				for (i = 0; i < string_size; i++)
					strings[i + 1] = strings[i];
				strings[0] = companion_ref + 1;
			} else {
				const int q_hop_b = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[companion_ref], tensor_blocks[strings[string_size - 1] - 1]);
				if (q_hop_b > 0)
					strings[string_size] = companion_ref + 1;
				else {
					// Because b_hop_p is 0, q_hop_b is nil, p != q, and b must in between p and q. Therefore, I must have 2 allocations.
					assert(string_size == 2);
					strings[2] = strings[1];
					strings[1] = companion_ref + 1;
				}
			}
			++string_size;
			assigned[companion_ref] = assign_group;
			// The offset for this one, should be either 0 (started a new group, when min_i == -1), or the offset on this edge.
			allocated_offset[companion_ref] = min_val[1];
			for (i = 0; i < tensor_block_size; i++)
				if (!assigned[i] && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]))
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(i, companion_ref), ccv_max(i, companion_ref));
					if (cell.u8 && cell.u8[0] == 1)
						--oc[i];
				}
		}
		uint64_t val[2] = {
			a.size, min_val[1]
		};
		uint64_t consumed_size = 0;
		// Go over from min_y to string_size (excluding min_x).
		for (i = 0; i < string_size; i++)
		{
			const uint64_t size = tensor_blocks[strings[i] - 1].size;
			assert(size <= a.size);
			// Update consumed size if it is bigger than "size".
			if (size > consumed_size)
			{
				val[0] = size - consumed_size;
				ccv_set_sparse_matrix_cell(alloc, min_y, strings[i], val);
				consumed_size = size;
				val[1] = min_val[1] + consumed_size;
			}
			// If it consumed all the flow, break out.
			if (consumed_size == a.size)
				break;
		}
		for (i = 0; i < string_size; i++)
		{
			const uint64_t i_size = tensor_blocks[strings[i] - 1].size;
			uint64_t val[2] = {
				i_size, min_val[1]
			};
			uint64_t consumed_size = 0;
			for (k = i + 1; k < string_size; k++)
			{
				const uint64_t size = ccv_min(i_size, tensor_blocks[strings[k] - 1].size);
				// Update consumed size if it is bigger than "size".
				if (size > consumed_size)
				{
					val[0] = size - consumed_size;
					ccv_set_sparse_matrix_cell(alloc, strings[i], strings[k], val);
					consumed_size = size;
					val[1] = min_val[1] + consumed_size;
				}
				// If it consumed all the flow, break out.
				if (consumed_size == i_size)
					break;
			}
			val[0] = i_size - consumed_size;
			// Still have residual, flow it to min_x.
			if (val[0] > 0)
				ccv_set_sparse_matrix_cell(alloc, strings[i], min_x, val);
		}
		j += string_size;
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
	ccv_nnc_tensor_alloc_prep_t* alloc_prep = (ccv_nnc_tensor_alloc_prep_t*)ccmalloc(sizeof(ccv_nnc_tensor_alloc_prep_t) + sizeof(alloc_prep->blocks[0]) * available_tensor_size + sizeof(alloc_prep->buffers[0]) * num_assigned + sizeof(int) * tensor_block_size);
	alloc_prep->alloc_dep = alloc_dep;
	alloc_prep->vt_block_size = tensor_block_size;
	alloc_prep->buffer_size = num_assigned;
	alloc_prep->block_size = available_tensor_size;
	alloc_prep->blocks = (void*)(alloc_prep + 1); // From the biggest structs to smaller ones.
	alloc_prep->buffers = (void*)(alloc_prep->blocks + available_tensor_size);
	alloc_prep->vt_blocks = (int*)(alloc_prep->buffers + num_assigned);
	for (i = 0; i < num_assigned; i++)
	{
		alloc_prep->buffers[i].p_ref = 0;
		alloc_prep->buffers[i].size = allocated_size[i];
	}
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
				assert(allocated_offset[i] + tensor_blocks[i].size <= alloc_prep->buffers[assigned[i] - 1].size);
			} else {
				alloc_prep->vt_blocks[i] = -1;
				alloc_prep->blocks[j].buffer_ref = -1;
				alloc_prep->blocks[j].offset = 0;
			}
			++j;
		} else
			alloc_prep->vt_blocks[i] = -1;
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

// Simple allocator from ccv_array_t.
static int _ccv_nnc_tensor_metadata_pos_new(ccv_array_t* const tensor_metadata, const size_t size)
{
	int pos = tensor_metadata->rnum + 1; // Appending 1 so it starts from non-zero.
	int rsize = (size + 15) / 16;
	ccv_array_resize(tensor_metadata, pos + rsize - 1);
	return pos;
}

static ccv_nnc_tensor_t* _ccv_nnc_tensor_metadata_get(const ccv_array_t* const tensor_metadata, const int pos)
{
	assert(pos <= tensor_metadata->rnum);
	return (ccv_nnc_tensor_t*)ccv_array_get(tensor_metadata, pos - 1);
}

static ccv_nnc_tensor_t* _ccv_nnc_tensor_metadata_rewire(const ccv_array_t* const tensor_metadata, ccv_nnc_tensor_t* const vt_tensor)
{
	ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)vt_tensor);
	if (CCV_IS_TENSOR_MULTIVIEW(tensor))
	{
		ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
		int i;
		if (mv->tv)
			mv->tv = _ccv_nnc_tensor_metadata_rewire(tensor_metadata, mv->tv);
		else
			for (i = 0; i < (mv->kind == CCV_NNC_MULTIVIEW_K12) ? 3 : 2; i++)
				mv->data[i].ptr = _ccv_nnc_tensor_metadata_rewire(tensor_metadata, mv->data[i].ptr);
		// No need to recursively do parent pointer, otherwise we are in deep rewire.
		if (mv->p)
			mv->p = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)vt_tensor);
		if (mv->rtvs)
			for (i = 0; i < mv->rtvs->rnum; i++)
			{
				ccv_nnc_tensor_reference_t* reference = (ccv_nnc_tensor_reference_t*)ccv_array_get(mv->rtvs, i);
				reference->tensor = _ccv_nnc_tensor_metadata_rewire(tensor_metadata, reference->tensor);
				assert(!CCV_IS_TENSOR_MULTIVIEW(reference->tensor));
			}
	}
	return tensor;
}

static ccv_nnc_tensor_arena_t* _ccv_nnc_tensor_arena_new(const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const ccv_nnc_tensor_arena_t* const p_arena, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size)
{
	// All tensors assigned out, now, the num_assigned is the number of dis-continuous buffers,
	// Each tensor have the designation in assigned array, and offset in allocated_offset.
	const ccv_nnc_tensor_alloc_prep_t* const alloc_prep = graph_prep->alloc_prep;
	const ccv_nnc_tensor_block_t* const tensor_blocks = graph_prep->tensor_blocks;
	const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = graph_prep->tensor_symbol_info;
	const int tensor_symbol_info_size = graph_prep->tensor_symbol_info_size;
	const ccv_nnc_symbolic_graph_prep_t* const p_graph_prep = graph_prep->p;
	const ccv_nnc_tensor_alloc_prep_t* const p_alloc_prep = p_graph_prep ? p_graph_prep->alloc_prep : 0;
	ccv_nnc_tensor_arena_t* tensor_arena = (ccv_nnc_tensor_arena_t*)ccmalloc(sizeof(ccv_nnc_tensor_arena_t) + sizeof(tensor_arena->buffers[0]) * alloc_prep->buffer_size + sizeof(ccv_nnc_tensor_t*) * tensor_symbol_info_size + sizeof(ccv_nnc_tensor_arena_t*) * graph_prep->sub_prep_size);
	tensor_arena->graph_ref = graph_prep->graph_ref;
	tensor_arena->buffers = (void*)(tensor_arena + 1);
	tensor_arena->buffer_size = alloc_prep->buffer_size;
	tensor_arena->vt_tensor_size = tensor_symbol_info_size;
	tensor_arena->vt_tensors = (ccv_nnc_tensor_t**)(tensor_arena->buffers + alloc_prep->buffer_size);
	tensor_arena->sub_arenas = (ccv_nnc_tensor_arena_t**)(tensor_arena->vt_tensors + tensor_symbol_info_size);
	tensor_arena->sub_arena_size = graph_prep->sub_prep_size;
	tensor_arena->tensor_metadata = ccv_array_new(16 /* align to 16 bytes */, 0, 0);
	int i;
	for (i = 0; i < alloc_prep->buffer_size; i++)
		tensor_arena->buffers[i].size = alloc_prep->buffers[i].size;
	int memory_type = CCV_TENSOR_GET_MEMORY(tensor_symbol_info[0].info.type);
	int device_id = CCV_TENSOR_GET_DEVICE_ID(tensor_symbol_info[0].info.type);
	for (i = 1; i < tensor_symbol_info_size; i++)
	{
		assert(CCV_TENSOR_GET_MEMORY(tensor_symbol_info[i].info.type) == memory_type);
		assert(CCV_TENSOR_GET_DEVICE_ID(tensor_symbol_info[i].info.type) == device_id);
	}
	tensor_arena->memory_type = memory_type;
	tensor_arena->device_id = device_id;
	assert((p_arena && p_graph_prep) || (!p_arena && !p_graph_prep));
	if (p_arena && p_graph_prep)
	{
		// Don't need to allocate the actual buffer, just use the pointer from the above.
		PRINT(CCV_CLI_VERBOSE, "Buffer assignment for sub arena %p (parent %p)\n", tensor_arena, p_arena);
		for (i = 0; i < tensor_arena->buffer_size; i++)
		{
			const int p_ref = alloc_prep->buffers[i].p_ref - 1;
			assert(p_ref >= 0);
			if (p_graph_prep->dup_tensor_block_ref && p_graph_prep->dup_tensor_block_ref[p_ref] >= 0)
			{
				// This condition means in the parent graph, we point to multiple tensor blocks for the same
				// buffer, therefore, we cannot have one single pointer assigned in this case.
				// Later we will handle this by generate ccv_tensor_multiview_t structure.
				tensor_arena->buffers[i].ptr = 0;
				PRINT(CCV_CLI_VERBOSE, "|-Cannot assign buffer %d, it points to multiple blocks (multi view tensor required)\n", i);
				continue;
			}
			// Otherwise, find the actual buffer pointer.
			const int vt_ref = p_alloc_prep->vt_blocks[p_ref];
			assert(vt_ref >= 0);
			const int buffer_ref = p_alloc_prep->blocks[vt_ref].buffer_ref;
			if (!p_arena->buffers[buffer_ref].ptr)
			{
				// Pass it down as 0 ptr.
				tensor_arena->buffers[i].ptr = 0;
				PRINT(CCV_CLI_VERBOSE, "|-Cannot assign buffer %d, it points to multiple blocks (multi view tensor required)\n", i);
				continue;
			}
			const uint64_t offset = p_alloc_prep->blocks[vt_ref].offset;
			tensor_arena->buffers[i].ptr = p_arena->buffers[buffer_ref].ptr + offset;
			PRINT(CCV_CLI_VERBOSE, "|-Assign block %d in parent arena to buffer %d with offset %lu\n", vt_ref, i, (unsigned long)offset);
		}
	} else {
		// Now, allocate actual buffers.
		PRINT(CCV_CLI_VERBOSE, "Buffer allocation for arena %p\n", tensor_arena);
#ifdef HAVE_CUDA
		if (memory_type == CCV_TENSOR_GPU_MEMORY)
		{
			for (i = 0; i < tensor_arena->buffer_size; i++)
			{
				tensor_arena->buffers[i].ptr = (uint8_t*)cumalloc(device_id, tensor_arena->buffers[i].size);
				PRINT(CCV_CLI_VERBOSE, "|-Allocate buffer %d with ptr %p, size %lu\n", i, tensor_arena->buffers[i].ptr, (unsigned long)tensor_arena->buffers[i].size);
			}
		} else {
			assert(memory_type == CCV_TENSOR_CPU_MEMORY);
			for (i = 0; i < tensor_arena->buffer_size; i++)
			{
				ccmemalign((void**)&tensor_arena->buffers[i].ptr, 16, tensor_arena->buffers[i].size);
				PRINT(CCV_CLI_VERBOSE, "|-Allocate buffer %d with ptr %p, size %lu\n", i, tensor_arena->buffers[i].ptr, (unsigned long)tensor_arena->buffers[i].size);
			}
		}
#else
		assert(memory_type == CCV_TENSOR_CPU_MEMORY);
		for (i = 0; i < tensor_arena->buffer_size; i++)
		{
			ccmemalign((void**)&tensor_arena->buffers[i].ptr, 16, tensor_arena->buffers[i].size);
			PRINT(CCV_CLI_VERBOSE, "|-Allocate buffer %d with ptr %p, size %lu\n", i, tensor_arena->buffers[i].ptr, (unsigned long)tensor_arena->buffers[i].size);
		}
#endif
	}
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
				// Either we have dup_tensor_block_ref in current layer, or we have that in
				// previous layer, therefore, cannot really find the buffer ptr.
				if ((graph_prep->dup_tensor_block_ref && graph_prep->dup_tensor_block_ref[block_ref] >= 0) ||
					!tensor_arena->buffers[buffer_ref].ptr)
				{
				} else {
					// Having ptr.
					const int pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
					tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)(intptr_t)pos; // Cast into vt_tensors for now, and later will rewire it.
					ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
					// Also, set its allocations.
					// Since tensor view is bit compatible with tensor, we can just cast.
					*tensor = ccv_nnc_tensor(tensor_arena->buffers[buffer_ref].ptr + offset, tensor_symbol_info[block_ref].info, 0);
					assert(offset + tensor_blocks[block_ref].size <= tensor_arena->buffers[buffer_ref].size);
				}
			}
		}
	// Assign out refs, refs are simple ones, we should handle it first. (because they point to exactly the same metadata and same region).
	for (i = 0; i < tensor_symbol_info_size; i++)
		// It could be binded tensor (or unused), in that case, it doesn't have a ref.
		if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]) && tensor_blocks[i].ref)
		{
			// It must be available.
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[tensor_blocks[i].ref - 1]));
			assert(tensor_arena->vt_tensors[tensor_blocks[i].ref - 1]);
			tensor_arena->vt_tensors[i] = tensor_arena->vt_tensors[tensor_blocks[i].ref - 1];
		}
	for (i = 0; i < alloc_prep->block_size; i++)
		if (alloc_prep->blocks[i].block_ref < tensor_symbol_info_size)
		{
			const int block_ref = alloc_prep->blocks[i].block_ref;
			if (TENSOR_EXPECT_ALIAS(tensor_blocks[block_ref]))
			{
				// Assigning out the tensor aliases.
				assert(tensor_symbol_info[block_ref].alias_ref);
				const int alias_ref = tensor_symbol_info[block_ref].alias_ref - 1;
				// It referenced to is not an alias.
				assert(tensor_arena->vt_tensors[alias_ref]);
				const ccv_nnc_tensor_t alias_tensor = *_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, (int)(intptr_t)tensor_arena->vt_tensors[alias_ref]);
				assert(!CCV_IS_TENSOR_VIEW(&alias_tensor));
				// If there is no ofs, and inc is the same as dim, we take a shortcut and just init as normal tensor.
				if (memcmp(ccv_nnc_no_ofs, tensor_symbol_info[block_ref].ofs, sizeof(ccv_nnc_no_ofs)) == 0 &&
					memcmp(tensor_symbol_info[block_ref].inc, tensor_symbol_info[block_ref].info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
				{
					const int pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
					ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
					*tensor = ccv_nnc_tensor(alias_tensor.data.u8, tensor_symbol_info[block_ref].info, 0);
					tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)(intptr_t)pos;
				} else {
					const int pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_view_t));
					ccv_nnc_tensor_view_t* const tensor_view = (ccv_nnc_tensor_view_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
					// Otherwise initialize a tensor view
					// 1). Simple case, if the inc is equal to original tensor, just init a tensor view.
					if (memcmp(alias_tensor.info.dim, tensor_symbol_info[block_ref].inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
						*tensor_view = ccv_nnc_tensor_view(&alias_tensor, tensor_symbol_info[block_ref].ofs, tensor_symbol_info[block_ref].info.dim);
					else {
						// Otherwise, create the tensor first, and then create the tensor view off the new tensor.
						ccv_nnc_tensor_param_t info = tensor_symbol_info[block_ref].info;
						memcpy(info.dim, tensor_symbol_info[block_ref].inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
						assert(ccv_nnc_tensor_count(info) <= ccv_nnc_tensor_count(alias_tensor.info));
						ccv_nnc_tensor_t tensor = ccv_nnc_tensor(alias_tensor.data.u8, info, 0);
						*tensor_view = ccv_nnc_tensor_view(&tensor, tensor_symbol_info[block_ref].ofs, tensor_symbol_info[block_ref].info.dim);
					}
					tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)(intptr_t)pos;
				}
			}
		}
	// Everything is done, rewire vt_tensor to real locations. From now on, no push to tensor_metadata is possible.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_arena->vt_tensors[i])
			tensor_arena->vt_tensors[i] = _ccv_nnc_tensor_metadata_rewire(tensor_arena->tensor_metadata, tensor_arena->vt_tensors[i]);
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
	// Everything is allocated, I can go over sub_preps and allocate arenas for them.
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		// TODO: I also need to pass binded tensor properly to the lower level.
		if (graph_prep->sub_preps[i])
			tensor_arena->sub_arenas[i] = _ccv_nnc_tensor_arena_new(graph_prep->sub_preps[i], tensor_arena, 0, 0);
		else
			tensor_arena->sub_arenas[i] = 0;
	return tensor_arena;
}

static void _ccv_nnc_tensor_block_add_exec(const ccv_sparse_matrix_t* const exec_dep, const int idx, ccv_nnc_tensor_block_t tensor_blocks)
{
	int i, found = 0;
	// Try to insert head.
	ccv_array_t* head = tensor_blocks.head;
	for (i = 0; i < head->rnum;)
	{
		const int head_idx = *(int*)ccv_array_get(head, i);
		if (head_idx == idx)
		{
			found = 1;
			break;
		}
		ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, head_idx, idx);
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
			cell = ccv_get_sparse_matrix_cell(exec_dep, idx, head_idx);
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
		const int tail_idx = *(int*)ccv_array_get(tail, i);
		if (tail_idx == idx)
		{
			found = 1;
			break;
		}
		ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, idx, tail_idx);
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
			cell = ccv_get_sparse_matrix_cell(exec_dep, tail_idx, idx);
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
	if ((intptr_t)symbol.graph == tensor_arena->graph_ref)
	{
		assert(symbol.d >= 0 && symbol.d < tensor_arena->vt_tensor_size);
		return tensor_arena->vt_tensors[symbol.d];
	}
	int i;
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
	{
		ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_from_symbol(tensor_arena->sub_arenas[i], symbol);
		if (tensor)
			return tensor;
	}
	return 0;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_from_symbol(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena, const ccv_nnc_graph_exec_symbol_t symbol)
{
	assert(symbol.d >= 0 && symbol.d < graph_exec_arena->graph_exec_size);
	return graph_exec_arena->graph_execs[symbol.d];
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_source(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	return graph_exec_arena->source;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_destination(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	return graph_exec_arena->destination;
}

// Make two tensor blocks one.
static void _ccv_nnc_tensor_blocks_fold(ccv_nnc_tensor_block_t* const tensor_blocks, const int p_ref_0, const int p_ref_1)
{
	// Now we are sure p_ref_0 points to the input, p_ref_1 points to the output.
	if (!TENSOR_EXPECT_CONST(tensor_blocks[p_ref_0]) &&
		!TENSOR_EXPECT_CONST(tensor_blocks[p_ref_1]) &&
		tensor_blocks[p_ref_0].tail->rnum == 1 &&
		tensor_blocks[p_ref_1].head->rnum == 1 &&
		*(int*)ccv_array_get(tensor_blocks[p_ref_0].tail, 0) == *(int*)ccv_array_get(tensor_blocks[p_ref_1].head, 0))
	{
		// If the two parent refs matches (thus, they meet at the same node), we can concatenate with each other and mark one as a ref. This is very similar to in-place operation combining.
		assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[p_ref_0]));
		assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[p_ref_1]));
		ccv_array_free(tensor_blocks[p_ref_0].tail);
		tensor_blocks[p_ref_0].tail= tensor_blocks[p_ref_1].tail;
		if (tensor_blocks[p_ref_1].p_refs[0])
		{
			assert(tensor_blocks[p_ref_1].p_refs[1] == 0); // It simply cannot have more than one p_refs, otherwise we cannot merge.
			if (!tensor_blocks[p_ref_0].p_refs[0])
				tensor_blocks[p_ref_0].p_refs[0] = tensor_blocks[p_ref_1].p_refs[0];
			else
				tensor_blocks[p_ref_0].p_refs[1] = tensor_blocks[p_ref_1].p_refs[0];
		}
		TENSOR_SET_READ_WRITE(tensor_blocks[p_ref_0], TENSOR_READ_WRITE(tensor_blocks[p_ref_0]) | TENSOR_READ_WRITE(tensor_blocks[p_ref_1]));
		ccv_array_free(tensor_blocks[p_ref_1].head);
		TENSOR_EXPECT_SET_UNASSIGNED(tensor_blocks[p_ref_1]);
		tensor_blocks[p_ref_1].ref = p_ref_0 + 1;
		if (!tensor_blocks[p_ref_0].r_refs)
			tensor_blocks[p_ref_0].r_refs = ccv_array_new(sizeof(int), 0, 0);
		ccv_array_replace_int(tensor_blocks[p_ref_0].r_refs, p_ref_1 + 1, p_ref_1 + 1);
		tensor_blocks[p_ref_1].size = 0;
		tensor_blocks[p_ref_1].head = 0;
		tensor_blocks[p_ref_1].tail = 0;
	}
}

static void _ccv_nnc_exec_dep_and_tensor_blocks_prep(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_sparse_matrix_t** r_exec_dep, ccv_nnc_tensor_block_t** r_tensor_blocks)
{
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
		ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx); \
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
	const int tensor_block_size = symbolic_graph->tensor_symbol_info->rnum;
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
			tensor_blocks[i].ref = tensor_symbol_info[i].alias_ref; // Assign the ref.
			const int ref = tensor_blocks[i].ref - 1;
			if (!tensor_blocks[ref].r_refs)
				tensor_blocks[ref].r_refs = ccv_array_new(sizeof(int), 0, 0);
			ccv_array_replace_int(tensor_blocks[ref].r_refs, i + 1, i + 1);
		}
		// If this tensor is not expected to be unassigned, allocate the arrays for s and t.
		if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]))
		{
			tensor_blocks[i].head = ccv_array_new(sizeof(int), 0, 0);
			tensor_blocks[i].tail = ccv_array_new(sizeof(int), 0, 0);
			// Cache tensor size (align to 16 bytes).
			tensor_blocks[i].size = (uint64_t)ccv_nnc_tensor_data_size(tensor_symbol_info[i].info);
		}
		// If there is a p_ref, add the one to the p_refs list.
		if (tensor_symbol_info[i].p_ref)
			tensor_blocks[i].p_refs[0] = tensor_symbol_info[i].p_ref;
	}
	// Collect head nodes and tail nodes for each tensor.
#define visitor(node, idx, ...) \
	do { \
		for (i = 0; i < node->input_size; i++) \
		{ \
			int d = node->inputs[i]; \
			if (d < 0) \
				continue; \
			tensor_blocks[d].flag |= READ_ONLY; \
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
			tensor_blocks[d].flag |= WRITE_ONLY; \
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
				assert(tensor_blocks[ref].ref == 0); \
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
							TENSOR_EXPECT_COMPUTABLE(tensor_blocks[node->outputs[y]])) \
						{ \
							const int node_output_y = node->outputs[y]; \
							const ccv_nnc_tensor_symbol_info_t y_symbol = tensor_symbol_info[node_output_y]; \
							/* If dimension matches perfectly, then we can assign y_symbol to x. */ \
							if (memcmp(x_symbol.info.dim, y_symbol.info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0) \
								_ccv_nnc_tensor_blocks_fold(tensor_blocks, ref, node_output_y); \
						} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	*r_exec_dep = exec_dep;
	*r_tensor_blocks = tensor_blocks;
}

static ccv_nnc_cmd_t _ccv_nnc_subst_sub_graph_with_noop(const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd)
{
	if (cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
	{
		ccv_nnc_cmd_t retval = cmd;
		retval.cmd = CCV_NNC_NOOP;
		return retval;
	}
	return cmd;
}

static ccv_nnc_tensor_symbol_t _ccv_nnc_dup_tensor_symbol(ccv_nnc_symbolic_graph_t* const dup_graph, int* const dup_tensor_block_ref, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const int input)
{
	if (dup_tensor_block_ref[input] < 0) // No tensor ref, create one.
	{
		if (tensor_symbol_info[input].alias_ref)
		{
			const int alias_ref = tensor_symbol_info[input].alias_ref - 1;
			assert(tensor_symbol_info[alias_ref].alias_ref == 0);
			ccv_nnc_tensor_symbol_t tensor_symbol = ccv_nnc_tensor_symbol_new(dup_graph, tensor_symbol_info[alias_ref].info, 0);
			dup_tensor_block_ref[alias_ref] = tensor_symbol.d;
			ccv_nnc_tensor_symbol_t alias_symbol = ccv_nnc_tensor_symbol_alias_new(dup_graph, tensor_symbol, tensor_symbol_info[input].ofs, tensor_symbol_info[input].inc, tensor_symbol_info[input].info, 0);
			dup_tensor_block_ref[input] = alias_symbol.d;
		} else {
			ccv_nnc_tensor_symbol_t tensor_symbol = ccv_nnc_tensor_symbol_new(dup_graph, tensor_symbol_info[input].info, 0);
			dup_tensor_block_ref[input] = tensor_symbol.d;
		}
	}
	return (ccv_nnc_tensor_symbol_t) {
		.d = dup_tensor_block_ref[input],
		.graph = dup_graph,
	};
}

static ccv_nnc_graph_exec_symbol_t _ccv_nnc_dup_graph_exec_symbol(ccv_nnc_symbolic_graph_t* const dup_graph, int* const dup_exec_ref, int* const dup_tensor_block_ref, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const ccv_nnc_graph_exec_symbol_info_t* const node, const int idx, ccv_nnc_tensor_symbol_t* const max_inputs, ccv_nnc_tensor_symbol_t* const max_outputs)
{
	int i;
	if (dup_exec_ref[idx] < 0)
	{
		for (i = 0; i < node->input_size; i++)
			max_inputs[i] = _ccv_nnc_dup_tensor_symbol(dup_graph, dup_tensor_block_ref, tensor_symbol_info, node->inputs[i]);
		for (i = 0; i < node->output_size; i++)
			max_outputs[i] = _ccv_nnc_dup_tensor_symbol(dup_graph, dup_tensor_block_ref, tensor_symbol_info, node->outputs[i]);
		ccv_nnc_graph_exec_symbol_t exec_symbol = ccv_nnc_graph_exec_symbol_new(dup_graph, node->cmd, max_inputs, node->input_size, max_outputs, node->output_size, 0);
		dup_exec_ref[idx] = exec_symbol.d;
	}
	return (ccv_nnc_graph_exec_symbol_t) {
		.d = dup_exec_ref[idx],
		.graph = dup_graph,
	};
}

static void _ccv_nnc_redo_exec_dep_and_tensor_blocks_if_expanse(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, ccv_nnc_graph_exec_symbol_info_t** r_exec_symbol_info, ccv_nnc_tensor_symbol_info_t** r_tensor_symbol_info, ccv_sparse_matrix_t** r_exec_dep, ccv_nnc_tensor_block_t** r_tensor_blocks, int* r_tensor_block_size, ccv_nnc_symbolic_graph_t** r_dup_graph, int** r_dup_exec_ref, int** r_dup_tensor_block_ref)
{
	int i, j;
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = *r_exec_symbol_info;
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = *r_tensor_symbol_info;
	ccv_sparse_matrix_t* exec_dep = *r_exec_dep;
	ccv_nnc_tensor_block_t* tensor_blocks = *r_tensor_blocks;
	// blocks that cannot be simply solved with either in-place operation tensor block folding or using the same memory region.
	// Unfortunately, I cannot do this analysis to the block folding done for sub-graphs, because we do sub-graph placement later.
	ccv_array_t* hard = 0;
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		if (!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]) && tensor_symbol_info[i].assign_ref)
		{
			// This is is a parameter, thus, it has to be either an alias or used.
			assert(tensor_blocks[i].ref || TENSOR_EXPECT_ORDINARY(tensor_blocks[i]));
			const int assign_ref = tensor_symbol_info[i].assign_ref - 1; // Starts at 1.
			// The parameter it assign to has to be either an alias or used.
			assert(tensor_blocks[assign_ref].ref || TENSOR_EXPECT_ORDINARY(tensor_blocks[assign_ref]));
			// If any of this two (assigner and assignee) is an alias, check to see if they are the same.
			// If it is the same, we are good, no need to extend.
			int a_ref = i;
			while (tensor_blocks[a_ref].ref)
				a_ref = tensor_blocks[a_ref].ref - 1;
			int b_ref = assign_ref;
			while (tensor_blocks[b_ref].ref)
				b_ref = tensor_blocks[b_ref].ref - 1;
			if (a_ref != b_ref)
			{
				// If any of the b's head is deterministically later than a's tail
				// or any of the b's tail is deterministically earlier than a's head, they don't interfere.
				int a_hop_b = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[b_ref], tensor_blocks[a_ref]);
				int b_hop_a = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[a_ref], tensor_blocks[b_ref]);
				// It cannot be that both i can hop to j can j can hop to i.
				assert(!(a_hop_b > 0 && b_hop_a > 0));
				// These two can be assigned to the same region of memory without issue (because their life-time doesn't interfere).
				if (a_hop_b || b_hop_a)
				{
					tensor_blocks[a_ref].companion_ref = b_ref + 1;
					tensor_blocks[b_ref].companion_ref = a_ref + 1;
				} else {
					if (!hard)
						hard = ccv_array_new(sizeof(int), 1, 0);
					ccv_array_push(hard, &i);
				}
			}
		}
	// No need to change anything, we are good.
	if (!hard)
		return;
	// Below logic is correct, just make the current build pass.
	ccv_array_free(hard);
	return;
	// Have conditions that cannot be satisfied with simple solution (allocate to the same memory region).
	// The visited exec nodes, these are the nodes we are going to extend.
	uint8_t* visited = (uint8_t*)cccalloc(symbolic_graph->exec_symbol_info->rnum, sizeof(uint8_t));
	for (i = 0; i < hard->rnum; i++)
	{
		int ref = *(int*)ccv_array_get(hard, i);
		int assign_ref = tensor_symbol_info[ref].assign_ref - 1;
		assert(assign_ref >= 0);
		while (tensor_blocks[ref].ref)
			ref = tensor_blocks[ref].ref - 1;
		while (tensor_blocks[assign_ref].ref)
			assign_ref = tensor_blocks[assign_ref].ref - 1;
		assert(tensor_blocks[ref].ref == 0);
		assert(tensor_blocks[assign_ref].ref == 0);
		// This covered all the cases for this block, because if it has aliases, all aliases have
		// its relevant exec attributed back, if it is normal ref (for in-place exec), it is attributed
		// back as well.
		// Now loop over all the tails, and mark all its dependencies. Thus, when we expand the graph,
		// we can only expand the relevant ones.
#define for_block(x, val) \
		do { \
			if (((int32_t*)val)[0] > 0) \
				visited[x] = 1; \
		} while (0)
		if (tensor_blocks[ref].tail)
			for (j = 0; j < tensor_blocks[ref].tail->rnum; j++)
			{
				const int idx = *(int*)ccv_array_get(tensor_blocks[ref].tail, j);
				visited[idx] = 1;
				ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx);
				CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
			}
		if (tensor_blocks[assign_ref].tail)
			for (j = 0; j < tensor_blocks[assign_ref].tail->rnum; j++)
			{
				const int idx = *(int*)ccv_array_get(tensor_blocks[assign_ref].tail, j);
				visited[idx] = 1;
				ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx);
				CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
			}
#undef for_block
		// Verify that we did everything alright (by loop through exec_dep, we should already covered all head).
		if (tensor_blocks[ref].head)
			for (j = 0; j < tensor_blocks[ref].head->rnum; j++)
				{ assert(visited[*(int*)ccv_array_get(tensor_blocks[ref].head, j)] == 1); }
		if (tensor_blocks[assign_ref].head)
			for (j = 0; j < tensor_blocks[assign_ref].head->rnum; j++)
				{ assert(visited[*(int*)ccv_array_get(tensor_blocks[assign_ref].head, j)] == 1); }
	}
	ccv_array_free(hard);
	int max_input_size = 0;
	int max_output_size = 0;
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
	{
		max_input_size = ccv_max(exec_symbol_info[i].input_size, max_input_size);
		max_output_size = ccv_max(exec_symbol_info[i].output_size, max_output_size);
	}
	ccv_nnc_tensor_symbol_t* max_inputs = max_input_size > 0 ? (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * max_input_size) : 0;
	ccv_nnc_tensor_symbol_t* max_outputs = max_output_size > 0 ? (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * max_output_size) : 0;
	// Doing graph expansion, first duplicate the old graph, but replace all sub graphs with noop.
	ccv_nnc_symbolic_graph_t* dup_graph = ccv_nnc_symbolic_graph_dup(symbolic_graph, _ccv_nnc_subst_sub_graph_with_noop);
	// It goes without saying, we must have more than one tensors / execs (otherwise I cannot use 0 as no exec ref).
	assert(dup_graph->exec_symbol_info->rnum > 0);
	assert(dup_graph->tensor_symbol_info->rnum > 0);
	int* dup_exec_ref = (int*)ccmalloc(sizeof(int) * symbolic_graph->exec_symbol_info->rnum);
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
		dup_exec_ref[i] = -1;
	int* dup_tensor_block_ref = (int*)ccmalloc(sizeof(int) * symbolic_graph->tensor_symbol_info->rnum);
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		// If there is a assign_ref, that means I don't need to dup the tensor.
		if (tensor_symbol_info[i].assign_ref)
			dup_tensor_block_ref[i] = tensor_symbol_info[i].assign_ref - 1;
		else if (TENSOR_READ_WRITE(tensor_blocks[i]) == READ_ONLY)
		// If this is a read-only tensor block, no need to duplicate because the value never changes
		// (note we handled assign_ref first), therefore, no need to generate duplicate.
			dup_tensor_block_ref[i] = i;
		else
			dup_tensor_block_ref[i] = -1;
	}
	// Go through the original graph, make copies of the node if it is visited.
#define INCOMING_NODE (2)
#define OUTGOING_NODE (4)
#define visitor(node, idx, ...) \
	do { \
		if (visited[idx]) \
		{ \
			ccv_nnc_graph_exec_symbol_t exec_symbol = _ccv_nnc_dup_graph_exec_symbol(dup_graph, dup_exec_ref, dup_tensor_block_ref, tensor_symbol_info, node, idx, max_inputs, max_outputs); \
			visited[idx] |= INCOMING_NODE; /* Mark this node as incoming. */ \
			if (!node->outgoings) \
				break; \
			for (i = 0; i < node->outgoings->rnum; i++) \
			{ \
				const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, i); \
				if (visited[outgoing_idx]) \
					visited[outgoing_idx] |= OUTGOING_NODE; /* Mark this node as outgoing. */ \
				ccv_nnc_graph_exec_symbol_t outgoing_symbol = _ccv_nnc_dup_graph_exec_symbol(dup_graph, dup_exec_ref, dup_tensor_block_ref, tensor_symbol_info, exec_symbol_info + outgoing_idx, outgoing_idx, max_inputs, max_outputs); \
				ccv_nnc_graph_exec_symbol_concat(dup_graph, exec_symbol, outgoing_symbol); \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Check the visitor are all marked as either incoming or outgoing.
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
		if (visited[i])
		{
			assert((visited[i] & INCOMING_NODE) || (visited[i] & OUTGOING_NODE));
			// If this is pure incoming nodes, then I need to concat this one with all original destination node
			if (visited[i] == (INCOMING_NODE | 1))
				for (j = 0; j < destination_size; j++)
				{
					ccv_nnc_graph_exec_symbol_concat(dup_graph, (ccv_nnc_graph_exec_symbol_t) {
						.d = destinations[j].d,
						.graph = dup_graph,
					}, (ccv_nnc_graph_exec_symbol_t) {
						.d = dup_exec_ref[i],
						.graph = dup_graph,
					});
				}
		}
	if (dup_graph->sources)
		ccv_array_clear(dup_graph->sources);
	for (i = 0; i < source_size; i++)
	{
		ccv_nnc_symbolic_graph_add_source(dup_graph, (ccv_nnc_graph_exec_symbol_t) {
			.graph = dup_graph,
			.d = sources[i].d,
		});
	}
	if (dup_graph->destinations)
		ccv_array_clear(dup_graph->destinations);
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
		if (visited[i] & 1) // If this is a visited node that we expanded.
		{
			const int d = dup_exec_ref[i];
			ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(dup_graph->exec_symbol_info, d);
			// If this has no outgoing node, add to the destination.
			if (!exec_symbol_info->outgoings || exec_symbol_info->outgoings->rnum == 0)
				ccv_nnc_symbolic_graph_add_destination(dup_graph, (ccv_nnc_graph_exec_symbol_t) {
					.graph = dup_graph,
					.d = d,
				});
		}
	tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccrealloc(tensor_symbol_info, sizeof(ccv_nnc_tensor_symbol_info_t) * dup_graph->tensor_symbol_info->rnum);
	exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccrealloc(exec_symbol_info, sizeof(ccv_nnc_graph_exec_symbol_info_t) * dup_graph->exec_symbol_info->rnum);
	ccv_nnc_symbolic_graph_symbol_infer(dup_graph, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->sources, 0), dup_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->destinations, 0), dup_graph->destinations->rnum, p_tensor_symbol_info, p_tensor_symbol_info_size, tensor_symbol_info, exec_symbol_info);
	// Free out the old exec_dep
	ccv_matrix_free(exec_dep);
	// and the tensor blocks, prepare for the new.
	ccfree(tensor_blocks);
	_ccv_nnc_exec_dep_and_tensor_blocks_prep(dup_graph, tensor_binds, tensor_bind_size, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->sources, 0), dup_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->destinations, 0), dup_graph->destinations->rnum, exec_symbol_info, tensor_symbol_info, &exec_dep, &tensor_blocks);
#undef INCOMING_NODE
#undef OUTGOING_NODE
	ccfree(visited);
	ccfree(max_inputs);
	ccfree(max_outputs);
	// Extend the dup tensor block ref, prepare for future extensions.
	dup_tensor_block_ref = (int*)ccrealloc(dup_tensor_block_ref, sizeof(int) * dup_graph->tensor_symbol_info->rnum);
	for (i = symbolic_graph->tensor_symbol_info->rnum; i < dup_graph->tensor_symbol_info->rnum; i++)
		dup_tensor_block_ref[i] = -1;
	// Assign out changed properties.
	*r_exec_symbol_info = exec_symbol_info;
	*r_tensor_symbol_info = tensor_symbol_info;
	*r_exec_dep = exec_dep;
	*r_tensor_blocks = tensor_blocks;
	*r_tensor_block_size = dup_graph->tensor_symbol_info->rnum;
	*r_dup_graph = dup_graph;
	*r_dup_exec_ref = dup_exec_ref;
	*r_dup_tensor_block_ref = dup_tensor_block_ref;
}

static int _ccv_nnc_is_symbolic_graph_exec_input_or_output(const int p_ref, const ccv_nnc_graph_exec_symbol_info_t *const node)
{
	int i;
	int is_input = 0;
	for (i = 0; i < node->input_size && !is_input; i++)
		if (p_ref == node->inputs[i])
			is_input = 1;
	int is_output = 0;
	for (i = 0; i < node->output_size && !is_output; i++)
		if (p_ref == node->outputs[i])
			is_output = 1;
	assert(!(is_input && is_output));
	if (is_input)
		return -1;
	if (is_output)
		return 1;
	return 0;
}

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
	int i;
	ccv_sparse_matrix_t* exec_dep;
	ccv_nnc_tensor_block_t* tensor_blocks;
	_ccv_nnc_exec_dep_and_tensor_blocks_prep(symbolic_graph, tensor_binds, tensor_bind_size, sources, source_size, destinations, destination_size, exec_symbol_info, tensor_symbol_info, &exec_dep, &tensor_blocks);
	int tensor_block_size = symbolic_graph->tensor_symbol_info->rnum;
	// Now, everything is prepared, tensor life is analyzed, inplace operations are collapsed, all tensor symbols and hints
	// are automatically filled in, and all the sub-graphs are processed.
	// There is a last step though, for a while loop, it is parameterized:
	// while (x > 5) {
	//     y = x + 1;
	// } (y => x) // This means after this loop is done, y's value will be copied over to x.
	// we will do our best to avoid to do the actual data copy, what we do here is to check whether y can be x's alias.
	// If y can be x's alias, this is good, no other changes required. In above case, y can be x's alias because
	// it is a inplace operation.
	// But if y cannot be x's alias, for example, this while loop looks like this:
	// while (x > 5) {
	//     y = x + a
	//     b = x + y
	// } (y => x, b => a) // This means after this loop is done, y's value copied to x and b's value copied to a.
	// For this example, y cannot be x's alias because x is used later to compute b (and that computation
	// has dependency on y as well).
	// For this case, we need to modify the computation graph. Previously, the graph looks like this:
	// y = x + a -> b = x + y
	// This graph will be extended to look like this:
	// y0 = x0 + a0 -> b0 = x0 + y0 -> y1 = y0 + b0 -> b1 = y0 + y1, or:
	// while (x0 > 5) {
	//     y0 = x0 + a0
	//     b0 = x0 + y0
	//     if (y0 > 5) break
	//     y1 = y0 + b0
	//     b1 = y0 + y1
	// } (y1 => x0, b1 => a0)
	// After this expansion, y1 now can be the alias of x0, as well as b1 can be alias of a0 (they don't interfere
	// with each other now).
	// With this algorithm, we don't need to insert any data copy logic, the only thing need is to switch pointers
	// which is covered by the tensor_multiview_t construct (thus, y (y0, y1), x (y1, y0), b (b0, b1), a (b1, b0))
	ccv_nnc_symbolic_graph_t* dup_graph = 0;
	int* dup_exec_ref = 0;
	int* dup_tensor_block_ref = 0;
	// Cannot handle dup a node that is a graph as well.
	_ccv_nnc_redo_exec_dep_and_tensor_blocks_if_expanse(symbolic_graph, tensor_binds, tensor_bind_size, sources, source_size, destinations, destination_size, p_tensor_symbol_info, p_tensor_symbol_info_size, &exec_symbol_info, &tensor_symbol_info, &exec_dep, &tensor_blocks, &tensor_block_size, &dup_graph, &dup_exec_ref, &dup_tensor_block_ref);
	// In true recursive fashion, I need to call all the sub graphs and do the pre compilation for them one by one.
	ccv_nnc_symbolic_graph_prep_t* prep = (ccv_nnc_symbolic_graph_prep_t*)ccmalloc(sizeof(ccv_nnc_symbolic_graph_prep_t));
	prep->graph = ccv_nnc_graph_new(); // Just allocate the graph right now.
	// TODO: I should be able to express that you cannot fold / or reuse certain tensor blocks.
	ccv_nnc_symbolic_graph_prep_t** sub_preps = symbolic_graph->sub_graphs && symbolic_graph->sub_graphs->rnum ? (ccv_nnc_symbolic_graph_prep_t**)cccalloc(symbolic_graph->sub_graphs->rnum, sizeof(ccv_nnc_symbolic_graph_prep_t*)) : 0;
#define visitor(node, idx, ...) \
	do { \
		if (node->graph_ref) \
		{ \
			ccv_nnc_symbolic_graph_t* while_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, node->graph_ref - 1); \
			ccv_nnc_symbolic_graph_prep_t* const sub_prep = _ccv_nnc_symbolic_graph_prep_new(while_graph, tensor_binds, tensor_bind_size, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(while_graph->sources, 0), while_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(while_graph->destinations, 0), while_graph->destinations->rnum, tensor_symbol_info, symbolic_graph->tensor_symbol_info->rnum, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum); \
			sub_prep->p = prep; \
			sub_preps[node->graph_ref - 1] = sub_prep; \
			const ccv_nnc_tensor_alloc_prep_t* const s_alloc_prep = sub_prep->alloc_prep; \
			const ccv_nnc_tensor_block_t* const s_tensor_blocks = sub_prep->tensor_blocks; \
			for (i = 0; i < s_alloc_prep->block_size; i++) \
			{ \
				const int block_ref = s_alloc_prep->blocks[i].block_ref; \
				const int buffer_ref = s_alloc_prep->blocks[i].buffer_ref; \
				if (block_ref < sub_prep->tensor_symbol_info_size) \
				{ \
					if (s_tensor_blocks[block_ref].p_refs[0]) \
					{ \
						int p_ref_0 = s_tensor_blocks[block_ref].p_refs[0] - 1; \
						/* Need to go through refs. Since we reuse the tensor block for this input, it now has to have allocate at least this much space. */ \
						const int p_ref_0_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_0, node); \
						assert(p_ref_0_is_in_or_out != 0); \
						while (tensor_blocks[p_ref_0].ref) \
							p_ref_0 = tensor_blocks[p_ref_0].ref - 1; \
						/* This parent tensor block cannot be unassigned because it is either input / output of this sub-graph node. */ \
						assert(!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[p_ref_0])); \
						if (s_tensor_blocks[block_ref].p_refs[1]) \
						{ \
							int p_ref_1 = s_tensor_blocks[block_ref].p_refs[1] - 1; \
							const int p_ref_1_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_1, node); \
							assert(p_ref_1_is_in_or_out != 0); \
							while (tensor_blocks[p_ref_1].ref) \
								p_ref_1 = tensor_blocks[p_ref_1].ref - 1; \
							/* See above comment for the similar p_ref_0 check. */ \
							assert(!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[p_ref_1])); \
							assert(p_ref_0_is_in_or_out != p_ref_1_is_in_or_out); \
							int p_ref_t; \
							if (p_ref_0_is_in_or_out > p_ref_1_is_in_or_out) /* if p_ref_0 is output and p_ref_1 is input, switch. */ \
								CCV_SWAP(p_ref_0, p_ref_1, p_ref_t); \
							_ccv_nnc_tensor_blocks_fold(tensor_blocks, p_ref_0, p_ref_1); \
							if (dup_tensor_block_ref) /* Fold its duplicates as well. */ \
								_ccv_nnc_tensor_blocks_fold(tensor_blocks, dup_tensor_block_ref[p_ref_0], dup_tensor_block_ref[p_ref_1]); \
							/* We are good, mark this buffer as assigned out for p_ref_1 (the output tensor). */ \
							s_alloc_prep->buffers[buffer_ref].p_ref = p_ref_1 + 1; \
						} else { \
							/* We are good, mark this buffer as assigned out for p_ref_0. */ \
							s_alloc_prep->buffers[buffer_ref].p_ref = p_ref_0 + 1; \
						} \
						/* This parent tensor block cannot be unassigned because it is either input / output of this sub-graph node. */ \
						assert(!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[p_ref_0])); \
						tensor_blocks[p_ref_0].size = ccv_max(s_alloc_prep->buffers[buffer_ref].size, tensor_blocks[p_ref_0].size); \
						if (dup_tensor_block_ref) /* Change the size of its duplicates as well. */ \
							tensor_blocks[dup_tensor_block_ref[p_ref_0]].size = tensor_blocks[p_ref_0].size; \
					} \
				} \
			} \
			int anonymous_buffer_size = 0; \
			for (i = 0; i < s_alloc_prep->buffer_size; i++) \
				if (!s_alloc_prep->buffers[i].p_ref) \
					++anonymous_buffer_size; \
			if (anonymous_buffer_size) \
			{ \
				/* Anonymous block, allocate additional tensor blocks for this. */ \
				/* This is either because this is an internal tensor (don't have p_ref) */ \
				/* or it is an anonymous block itself within the sub graphs of this while graph. */ \
				if (dup_tensor_block_ref) \
				{ \
					tensor_blocks = (ccv_nnc_tensor_block_t*)ccrealloc(tensor_blocks, sizeof(ccv_nnc_tensor_block_t) * (tensor_block_size + 2 * anonymous_buffer_size)); \
					memset(tensor_blocks + tensor_block_size, 0, sizeof(ccv_nnc_tensor_block_t) * 2 * anonymous_buffer_size); \
					dup_tensor_block_ref = (int*)ccrealloc(dup_tensor_block_ref, sizeof(int) * (tensor_block_size + 2 * anonymous_buffer_size)); \
				} else { \
					tensor_blocks = (ccv_nnc_tensor_block_t*)ccrealloc(tensor_blocks, sizeof(ccv_nnc_tensor_block_t) * (tensor_block_size + anonymous_buffer_size)); \
					memset(tensor_blocks + tensor_block_size, 0, sizeof(ccv_nnc_tensor_block_t) * anonymous_buffer_size); \
				} \
				for (i = 0; i < s_alloc_prep->buffer_size; i++) \
					if (!s_alloc_prep->buffers[i].p_ref) \
					{ \
						TENSOR_SET_ANONYMOUS(tensor_blocks[tensor_block_size]); \
						tensor_blocks[tensor_block_size].size = s_alloc_prep->buffers[i].size; \
						s_alloc_prep->buffers[i].p_ref = tensor_block_size + 1; \
						tensor_blocks[tensor_block_size].graph_ref = node->graph_ref; \
						tensor_blocks[tensor_block_size].head = ccv_array_new(sizeof(int), 1, 0); \
						tensor_blocks[tensor_block_size].tail = ccv_array_new(sizeof(int), 1, 0); \
						ccv_array_push(tensor_blocks[tensor_block_size].head, &idx); \
						ccv_array_push(tensor_blocks[tensor_block_size].tail, &idx); \
						++tensor_block_size; \
						/* ref and flags are both 0. */ \
						if (dup_tensor_block_ref) \
						{ \
							dup_tensor_block_ref[tensor_block_size - 1] = tensor_block_size; \
							dup_tensor_block_ref[tensor_block_size] = -1; \
							TENSOR_SET_ANONYMOUS(tensor_blocks[tensor_block_size]); \
							tensor_blocks[tensor_block_size].size = s_alloc_prep->buffers[i].size; \
							tensor_blocks[tensor_block_size].head = ccv_array_new(sizeof(int), 1, 0); \
							tensor_blocks[tensor_block_size].tail = ccv_array_new(sizeof(int), 1, 0); \
							/* Attach to duplicated exec for this tensor block. */ \
							ccv_array_push(tensor_blocks[tensor_block_size].head, &dup_exec_ref[idx]); \
							ccv_array_push(tensor_blocks[tensor_block_size].tail, &dup_exec_ref[idx]); \
							++tensor_block_size; \
						} \
					} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// It is time to guess what's the best tensor placement and create the opaque tensor arena. The alloc_dep will return
	// the allocation dependencies, thus, which tensor is reused to the existing tensor.
	ccv_nnc_tensor_alloc_prep_t* alloc_prep = _ccv_nnc_tensor_alloc_prep_new(exec_dep, tensor_blocks, tensor_block_size);
	ccv_matrix_free(exec_dep);
	prep->p = 0;
	prep->graph_ref = (intptr_t)symbolic_graph;
	prep->sub_prep_size = symbolic_graph->sub_graphs ? symbolic_graph->sub_graphs->rnum : 0;
	prep->sub_preps = sub_preps;
	prep->exec_symbol_info_size = symbolic_graph->exec_symbol_info->rnum;
	prep->exec_symbol_info = exec_symbol_info;
	prep->tensor_symbol_info_size = symbolic_graph->tensor_symbol_info->rnum;
	prep->tensor_symbol_info = tensor_symbol_info;
	prep->dup_tensor_block_ref = dup_tensor_block_ref;
	prep->tensor_block_size = tensor_block_size;
	prep->tensor_blocks = tensor_blocks;
	prep->alloc_prep = alloc_prep;
	if (dup_graph)
		ccv_nnc_symbolic_graph_free(dup_graph);
	if (dup_exec_ref)
		ccfree(dup_exec_ref);
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
		if (prep->tensor_blocks[i].r_refs)
			ccv_array_free(prep->tensor_blocks[i].r_refs);
	}
	for (i = 0; i < prep->sub_prep_size; i++)
		if (prep->sub_preps[i])
			_ccv_nnc_symbolic_graph_prep_free(prep->sub_preps[i]);
	if (prep->sub_preps)
		ccfree(prep->sub_preps);
	ccfree(prep->tensor_blocks);
	ccfree(prep->tensor_symbol_info);
	ccfree(prep->exec_symbol_info);
	if (prep->dup_tensor_block_ref)
		ccfree(prep->dup_tensor_block_ref);
	_ccv_nnc_tensor_alloc_prep_free(prep->alloc_prep);
	ccfree(prep);
}

static ccv_nnc_graph_exec_arena_t* _ccv_nnc_graph_exec_arena_new(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const ccv_nnc_tensor_arena_t* const tensor_arena)
{
	int i, j, k;
	ccv_nnc_graph_t* const graph = graph_prep->graph;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = (ccv_nnc_graph_exec_arena_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_arena_t) + sizeof(ccv_nnc_graph_exec_arena_t*) * graph_prep->sub_prep_size + sizeof(ccv_nnc_graph_exec_t) * (symbolic_graph->exec_symbol_info->rnum - 1));
	graph_exec_arena->graph_exec_size = symbolic_graph->exec_symbol_info->rnum;
	graph_exec_arena->sub_arena_size = graph_prep->sub_prep_size;
	graph_exec_arena->sub_arenas = (ccv_nnc_graph_exec_arena_t**)(graph_exec_arena->graph_execs + symbolic_graph->exec_symbol_info->rnum);
	memset(graph_exec_arena->sub_arenas, 0, sizeof(ccv_nnc_graph_exec_arena_t*) * graph_exec_arena->sub_arena_size);
	ccv_nnc_graph_exec_t* graph_execs = graph_exec_arena->graph_execs;
	int max_input_size = 0, max_output_size = 0, max_cond_eval_size = 0;
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
	{
		max_input_size = ccv_max(max_input_size, graph_prep->exec_symbol_info[i].input_size);
		max_output_size = ccv_max(max_input_size, graph_prep->exec_symbol_info[i].output_size);
		graph_execs[i].graph = 0;
	}
	for (i = 0; i < graph_prep->sub_prep_size; i++)
		max_cond_eval_size = ccv_max(max_cond_eval_size, (*(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, i))->cond_eval_size);
	ccv_nnc_tensor_t** max_inputs = max_input_size > 0 ? (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_input_size) : 0;
	ccv_nnc_tensor_t** max_outputs = max_output_size > 0 ? (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * max_output_size) : 0;
	ccv_nnc_graph_exec_t* max_cond_evals = max_cond_eval_size > 0 ? (ccv_nnc_graph_exec_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_t) * max_cond_eval_size) : 0;
	const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = graph_prep->exec_symbol_info;
#define visitor(node, idx, ...) \
	do { \
		if (CCV_NO_GRAPH_EXEC(graph_execs[idx])) \
		{ \
			for (i = 0; i < node->input_size; i++) \
				max_inputs[i] = node->inputs[i] >= 0 ? tensor_arena->vt_tensors[node->inputs[i]] : 0; \
			for (i = 0; i < node->output_size; i++) \
				max_outputs[i] = node->outputs[i] >= 0 ? tensor_arena->vt_tensors[node->outputs[i]] : 0; \
			if (node->graph_ref) \
			{ \
				const int graph_ref = node->graph_ref - 1; \
				ccv_nnc_graph_t* const sub_graph = graph_prep->sub_preps[graph_ref]->graph; \
				const ccv_nnc_symbolic_graph_t* const sub_symbolic_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, graph_ref); \
				const ccv_nnc_graph_exec_arena_t* const sub_arena = graph_exec_arena->sub_arenas[graph_ref] = _ccv_nnc_graph_exec_arena_new(sub_symbolic_graph, ccv_nnc_symbolic_graph_sources(sub_symbolic_graph), ccv_nnc_symbolic_graph_source_size(sub_symbolic_graph), ccv_nnc_symbolic_graph_destinations(sub_symbolic_graph), ccv_nnc_symbolic_graph_destination_size(sub_symbolic_graph), graph_prep->sub_preps[graph_ref], tensor_arena->sub_arenas[graph_ref]); \
				for (i = 0; i < sub_symbolic_graph->cond_eval_size; i++) \
					max_cond_evals[i] = ccv_nnc_graph_exec_from_symbol(sub_arena, sub_symbolic_graph->cond_evals[i]); \
				ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(sub_arena); \
				ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(sub_arena); \
				graph_execs[idx] = ccv_nnc_graph_while(graph, node->cmd.cmd, sub_graph); \
				ccv_nnc_graph_set_sources(sub_graph, &source, 1); \
				ccv_nnc_graph_set_destinations(sub_graph, &destination, 1); \
				ccv_nnc_graph_set_while_expr(sub_graph, sub_symbolic_graph->while_expr, sub_symbolic_graph->while_data, max_cond_evals, sub_symbolic_graph->cond_eval_size); \
				ccv_nnc_graph_exec_set_io(graph, graph_execs[idx], max_inputs, node->input_size, max_outputs, node->output_size); \
			} else { \
				graph_execs[idx] = ccv_nnc_graph_exec_new(graph, node->cmd, node->hint, max_inputs, node->input_size, max_outputs, node->output_size); \
			} \
		} \
		if (!node->outgoings) \
			break; \
		for (i = 0; i < node->outgoings->rnum; i++) \
		{ \
			int outgoing = *(int*)ccv_array_get(node->outgoings, i); \
			if (CCV_NO_GRAPH_EXEC(graph_execs[outgoing])) \
			{ \
				const ccv_nnc_graph_exec_symbol_info_t* const outgoing_node = exec_symbol_info + outgoing; \
				for (j = 0; j < outgoing_node->input_size; j++) \
					max_inputs[j] = outgoing_node->inputs[j] >= 0 ? tensor_arena->vt_tensors[outgoing_node->inputs[j]] : 0; \
				for (j = 0; j < outgoing_node->output_size; j++) \
					max_outputs[j] = outgoing_node->outputs[j] >= 0 ? tensor_arena->vt_tensors[outgoing_node->outputs[j]] : 0; \
				if (outgoing_node->graph_ref) \
				{ \
					const int graph_ref = outgoing_node->graph_ref - 1; \
					ccv_nnc_graph_t* const sub_graph = graph_prep->sub_preps[graph_ref]->graph; \
					const ccv_nnc_symbolic_graph_t* const sub_symbolic_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, graph_ref); \
					const ccv_nnc_graph_exec_arena_t* const sub_arena = graph_exec_arena->sub_arenas[graph_ref] = _ccv_nnc_graph_exec_arena_new(sub_symbolic_graph, ccv_nnc_symbolic_graph_sources(sub_symbolic_graph), ccv_nnc_symbolic_graph_source_size(sub_symbolic_graph), ccv_nnc_symbolic_graph_destinations(sub_symbolic_graph), ccv_nnc_symbolic_graph_destination_size(sub_symbolic_graph), graph_prep->sub_preps[graph_ref], tensor_arena->sub_arenas[graph_ref]); \
					for (j = 0; j < sub_symbolic_graph->cond_eval_size; j++) \
						max_cond_evals[j] = ccv_nnc_graph_exec_from_symbol(sub_arena, sub_symbolic_graph->cond_evals[j]); \
					ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(sub_arena); \
					ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(sub_arena); \
					graph_execs[outgoing] = ccv_nnc_graph_while(graph, node->cmd.cmd, sub_graph); \
					ccv_nnc_graph_set_sources(sub_graph, &source, 1); \
					ccv_nnc_graph_set_destinations(sub_graph, &destination, 1); \
					ccv_nnc_graph_set_while_expr(sub_graph, sub_symbolic_graph->while_expr, sub_symbolic_graph->while_data, max_cond_evals, sub_symbolic_graph->cond_eval_size); \
					ccv_nnc_graph_exec_set_io(graph, graph_execs[outgoing], max_inputs, outgoing_node->input_size, max_outputs, outgoing_node->output_size); \
				} else { \
					graph_execs[outgoing] = ccv_nnc_graph_exec_new(graph, outgoing_node->cmd, outgoing_node->hint, max_inputs, outgoing_node->input_size, max_outputs, outgoing_node->output_size); \
				} \
			} \
			ccv_nnc_graph_exec_concat(graph, graph_execs[idx], graph_execs[outgoing]); \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(symbolic_graph, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	if (max_inputs)
		ccfree(max_inputs);
	if (max_outputs)
		ccfree(max_outputs);
	if (max_cond_evals)
		ccfree(max_cond_evals);
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
				ccv_nnc_graph_exec_concat(graph, set_exec, graph_execs[outgoing]);
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
							ccv_nnc_graph_exec_concat(graph, graph_execs[incoming], set_exec);
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
	// Create source / destination phony node. This is to facilitate use of compiled graph.
	// Also, this is needed if you have init zero execs.
	if (source_exec_created || source_size > 1)
	{
		if (!source_exec_created)
			graph_exec_arena->source = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
		for (i = 0; i < source_size; i++)
			ccv_nnc_graph_exec_concat(graph, graph_exec_arena->source, graph_execs[sources[i].d]);
	} else {
		assert(!source_exec_created);
		assert(source_size == 1);
		graph_exec_arena->source = graph_execs[sources[0].d];
	}
	if (destination_size == 1)
		graph_exec_arena->destination = graph_execs[destinations[0].d];
	else {
		graph_exec_arena->destination = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
		for (i = 0; i < destination_size; i++)
			ccv_nnc_graph_exec_concat(graph, graph_execs[destinations[i].d], graph_exec_arena->destination);
	}
	return graph_exec_arena;
}

void ccv_nnc_symbolic_graph_compile(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, ccv_nnc_graph_t** const graph_ref, ccv_nnc_tensor_arena_t** const tensor_arena_ref, ccv_nnc_graph_exec_arena_t** const graph_exec_arena_ref)
{
	assert(graph_ref);
	assert(tensor_arena_ref);
	assert(graph_exec_arena_ref);
	ccv_nnc_symbolic_graph_prep_t* graph_prep = _ccv_nnc_symbolic_graph_prep_new(symbolic_graph, tensor_binds, tensor_bind_size, sources, source_size, destinations, destination_size, 0, 0, 0, 0);
	ccv_nnc_tensor_arena_t* tensor_arena = _ccv_nnc_tensor_arena_new(graph_prep, 0, tensor_binds, tensor_bind_size);
	*tensor_arena_ref = tensor_arena;
	// The above handled tensor allocation, now we need to materialize the graph from symbolic to real.
	*graph_ref = graph_prep->graph;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = _ccv_nnc_graph_exec_arena_new(symbolic_graph, sources, source_size, destinations, destination_size, graph_prep, tensor_arena);
	*graph_exec_arena_ref = graph_exec_arena;
	_ccv_nnc_symbolic_graph_prep_free(graph_prep);
}

static void _ccv_nnc_tensor_arena_free(ccv_nnc_tensor_arena_t* const tensor_arena)
{
	// Buffers are inherited from above, no need to dealloc.
	int i;
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (tensor_arena->sub_arenas[i])
			_ccv_nnc_tensor_arena_free(tensor_arena->sub_arenas[i]);
	ccv_array_free(tensor_arena->tensor_metadata);
	ccfree(tensor_arena);
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
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (tensor_arena->sub_arenas[i])
			_ccv_nnc_tensor_arena_free(tensor_arena->sub_arenas[i]);
	ccv_array_free(tensor_arena->tensor_metadata);
	ccfree(tensor_arena);
}

void ccv_nnc_graph_exec_arena_free(ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	int i;
	for (i = 0; i < graph_exec_arena->sub_arena_size; i++)
		if (graph_exec_arena->sub_arenas[i])
			ccv_nnc_graph_exec_arena_free(graph_exec_arena->sub_arenas[i]);
	ccfree(graph_exec_arena);
}
