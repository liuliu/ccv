#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_graph.h"
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-3 API
 */

typedef struct {
	int flags;
	int type;
	int ref; // Reference to another tensor block. Start with 1.
	int bypass_ref; // Copy over the bypass_ref from tensor symbol underneath. Start with 1.
	int companion_ref; // Reference to another block that they two share the same memory region. Start with 1. the current crude implementation requires the two mutually be companion. Because there are two, we took the one that companion_ref <= i as the primary and companion_ref > i is the secondary. For allocation algorithm, we use the primary throughout.
	int unfoldable_except_ref; // Reference to a tensor block that can be the exception to unfoldable (as output). Start with 1.
	ccv_array_t* r_refs; // If this is referenced by another block, the array point back to these blocks. Start with 1.
	uint64_t size; // The size of the tensor expected.
	int p_refs[2]; // Reference to the parent tensor block, at max there will be only two. Start with 1.
	ccv_array_t* dup_p_refs; // Reference to the parent tensor block from the duplicated tensor blocks. It could be many. Start with 0.
	ccv_array_t* head; // The head nodes (it could be multiple if from the graph, one cannot determine which is the first).
	ccv_array_t* tail; // The tail nodes (it could be multiple if from the graph, one cannot determine which is the last).
} ccv_nnc_tensor_block_t; // Tensor Arena Block

#define IS_PRIMARY_COMPANION(idx, block) ((idx) < (uint32_t)((block).companion_ref - 1))

enum {
	UNASSIGNED = 0x1,
	ALIAS = 0x2,
	READ_ONLY = 0x4,
	WRITE_ONLY = 0x8,
	READ_WRITE = 0xc,
	ANONYMOUS = 0x10, // Mark this block as anonymous (thus, not reference to any specific tensor).
	UNFOLDABLE_AS_INPUT = 0x20, // If this block is used as input, it cannot be folded into any output blocks.
	UNFOLDABLE_AS_OUTPUT = 0x40, // If this block is used as output, it cannot be folded into any input blocks.
};

#define TENSOR_EXPECT_ORDINARY(t) ((t.flags & 0x3) == 0)
#define TENSOR_EXPECT_SET_ORDINARY(t) (t.flags = (t.flags & ~0x3))
#define TENSOR_EXPECT_UNASSIGNED(t) ((t.flags & 0x3) == UNASSIGNED)
#define TENSOR_EXPECT_SET_UNASSIGNED(t) (t.flags = ((t.flags & ~0x3) | UNASSIGNED))
#define TENSOR_EXPECT_UNSET_UNASSIGNED(t) (t.flags = (t.flags & ~0x1))
#define TENSOR_EXPECT_ALIAS(t) ((t.flags & 0x3) == ALIAS)
#define TENSOR_EXPECT_COMPUTABLE(t) (!TENSOR_EXPECT_ALIAS(t) && !TENSOR_EXPECT_UNASSIGNED(t))
#define TENSOR_READ_WRITE(t) (t.flags & 0xc)
#define TENSOR_SET_READ_WRITE(t, rw) (t.flags = ((t.flags & ~0xc) | rw))
#define TENSOR_SET_ANONYMOUS(t) (t.flags = (t.flags & ~0x10 | ANONYMOUS))
#define TENSOR_IS_ANONYMOUS(t) (t.flags & ANONYMOUS)
#define TENSOR_SET_UNFOLDABLE_AS_INPUT(t) (t.flags = (t.flags | UNFOLDABLE_AS_INPUT))
#define TENSOR_IS_UNFOLDABLE_AS_INPUT(t) (t.flags & UNFOLDABLE_AS_INPUT)
#define TENSOR_SET_UNFOLDABLE_AS_OUTPUT(t) (t.flags = (t.flags | UNFOLDABLE_AS_OUTPUT))
#define TENSOR_IS_UNFOLDABLE_AS_OUTPUT(t) (t.flags & UNFOLDABLE_AS_OUTPUT)

// Holds additional information about the exe nodes.
typedef struct {
	int flags;
} ccv_nnc_graph_exec_flag_t;

enum {
	CCV_NNC_GRAPH_EXEC_ATTR_CASE_OF_NO_BYPASS_IO = 0x1, // Need to insert additional IO transfer for case..of statement.
};

typedef struct {
	int index;
	int companion; // The companion node index (the node that doesn't interfere with current one).
	int oc;
	int type;
	uint64_t size;
} ccv_nnc_tensor_opt_t;

#define more_than(i1, i2, aux) (((i1).size > (i2).size) || ((i1).size == (i2).size && (i1).oc >= (i2).oc))
static CCV_IMPLEMENT_QSORT(_ccv_nnc_tensor_opt_sort_by_size_and_oc, ccv_nnc_tensor_opt_t, more_than)
#undef more_than

// If b has items overlap with a, a is still after b (inclusive).
static int _ccv_nnc_tensor_block_a_after_b_inclusively(const ccv_sparse_matrix_t* const exec_dep, const ccv_array_t* const a, const ccv_array_t* const b)
{
	assert(a);
	assert(b);
	int x, y;
	for (x = 0; x < b->rnum; x++)
	{
		const int p = *(int*)ccv_array_get(b, x);
		int flag = 0;
		// In extreme cases where a is a superset of b, then a is still after b, we are good.
		for (y = 0; !flag && y < a->rnum; y++)
		{
			const int q = *(int*)ccv_array_get(a, y);
			flag = (p == q);
		}
		if (!flag)
			for (y = 0; y < a->rnum; y++)
			{
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, *(int*)ccv_array_get(a, y), p);
				if (!cell.i32 || cell.i32[0] == 0)
					return 0;
			}
	}
	// If b->rnum == 0, a is after b for sure.
	// Otherwise, if a->rnum == 0, we don't check any, buf if b->rnum > 0, then we cannot say a is after b.
	// if both a->rnum > 0 and b->rnum > 0, above logic should checked all.
	return (a->rnum > 0 || b->rnum == 0);
}

static int _ccv_nnc_tensor_block_a_after_b_exclusively(const ccv_sparse_matrix_t* const exec_dep, const ccv_array_t* const a, const ccv_array_t* const b)
{
	assert(a);
	assert(b);
	int x, y;
	for (x = 0; x < a->rnum; x++)
		for (y = 0; y < b->rnum; y++)
		{
			ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, *(int*)ccv_array_get(a, x), *(int*)ccv_array_get(b, y));
			if (!cell.i32 || cell.i32[0] == 0)
				return 0;
		}
	// We've entered this nested-for loop, therefore, it must be verifiably, deterministically after b now.
	return (a->rnum > 0 && b->rnum > 0);
}

// If every a's head is deterministically after b's tail
static int _ccv_nnc_tensor_block_head_after_tail(const ccv_sparse_matrix_t* const exec_dep, const ccv_nnc_tensor_block_t a, const ccv_nnc_tensor_block_t b)
{
	return _ccv_nnc_tensor_block_a_after_b_exclusively(exec_dep, a.head, b.tail);
}

typedef struct {
	ccv_array_t** alloc_dep;
	int vt_block_size;
	int buffer_size;
	int block_size;
	int* vt_blocks; // A reference to the block, because blocks only contains available block (thus, doesn't consider alias etc.). -1 means no block pointed to. Starts at 0.
	struct {
		int type; // The type from tensor blocks.
		int flags; // The flags (currently for READ_ONLY or not).
		uint64_t size; // The size of the buffer allocated.
		int p_refs[2]; // Reference to the upper level block, Starts at 1. Only index 0 is valid throughout, I do use two in the code as a temporary placeholder.
		ccv_array_t* dup_p_refs; // Reference to the parent tensor block from the duplicated tensor blocks. From buffer, it can point to multiple because it can be associated with multiple tensor blocks that points to different outputs (for example, in 1st unroll, pointing to one block while in 2nd unroll, pointing to another). Start with 0.
	}* buffers;
	struct {
		int buffer_ref; // A reference for block to which buffer to use. Starts at 0.
		int block_ref; // A reference to which block in the given tensor_block to use.
		uint64_t offset; // The offset of this block.
	}* blocks;
} ccv_nnc_tensor_alloc_prep_t;

typedef struct ccv_nnc_symbolic_graph_prep_s {
	int flags;
	int while_count_tensor; // This graph will generate a while count tensor. If this is set to 1, we reserve tensor_metadata at 0 for this.
	int p_idx; // Reference to the index in its parent graph's sub-graph array, Starts at 1.
	int exec_idx;
	int unroll_count; // How many times this graph is unrolled before we can have proper assignment.
	int tensor_symbol_info_size;
	int exec_symbol_info_size;
	int tensor_block_size;
	int sub_prep_size;
	ccv_nnc_tensor_block_t* tensor_blocks;
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info;
	ccv_nnc_graph_exec_flag_t* exec_flags;
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info;
	int* dup_tensor_block_ref;
	ccv_nnc_graph_visit_t* visit;
	ccv_nnc_tensor_alloc_prep_t* alloc_prep;
	struct ccv_nnc_symbolic_graph_prep_s* p;
	struct ccv_nnc_symbolic_graph_prep_s** sub_preps; // The preps of its sub-graphs.
	// Structures that don't require to be freed after deallocation.
	const ccv_nnc_symbolic_graph_t* symbolic_graph; // Constant because I cannot modify it.
	ccv_nnc_graph_t* graph; // Materialized graph, not managed by prep after created.
	ccv_nnc_tensor_arena_t* tensor_arena; // Tensor arena, not managed by prep as well.
	ccv_array_t* dup_breakpoints; // The noop breakpoints, used to extend the inputs life-cycle for while expr.
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
				// Check to see if they interfere (default to yes).
				// If any of the i's head is deterministically later than j's tail
				// or any of the i's tail is deterministically earlier than j's head, they don't interfere.
				const uint8_t one = 1;
				int i_hop_j = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[i], tensor_blocks[j]);
				int j_hop_i = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[j], tensor_blocks[i]);
				// It cannot be that both i can hop to j can j can hop to i.
				assert(!(i_hop_j > 0 && j_hop_i > 0));
				if (!i_hop_j && !j_hop_i)
					ccv_set_sparse_matrix_cell(tensor_itf, i, j, &one);
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
			if (oc[i] >= max_oc && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]) && !assigned[i] && IS_PRIMARY_COMPANION(i, tensor_blocks[i]))
			{
				ccv_nnc_tensor_opt_t a = {
					.size = tensor_blocks[i].size,
					.index = i,
					.companion = -1, // If already have a designated companion, use that.
					.oc = oc[i],
					.type = tensor_blocks[i].type,
				};
				if (tensor_blocks[i].companion_ref)
				{
					const int companion_ref = tensor_blocks[i].companion_ref - 1;
					a.size = ccv_max(a.size, tensor_blocks[companion_ref].size);
					a.oc += oc[companion_ref];
				}
				// In case we have a tie, take them all in the array.
				if (a.oc > max_oc)
					ccv_array_clear(opt), max_oc = a.oc;
				ccv_array_push(opt, &a);
			}
		assert(opt->rnum > 0);
		// Go through opt array, find all tensors that doesn't interfere with it, and have tensor size larger than it.
		// Push them with the "companion" into the opt array as well.
		const int rnum = opt->rnum;
		for (i = 0; i < rnum; i++)
		{
			// Copy it out, because after insertion, it may hold invalid pointer.
			ccv_nnc_tensor_opt_t a = *(ccv_nnc_tensor_opt_t*)ccv_array_get(opt, i);
			assert(a.companion == -1);
			const int companion_ref = tensor_blocks[i].companion_ref - 1;
			for (k = 0; k < tensor_block_size; k++)
				// Find non-overlapping tensor that has larger size (of course, is unassigned and is not one with designated companion).
				if (k != a.index && !tensor_blocks[k].companion_ref && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[k]) && !assigned[k] && tensor_blocks[k].size > a.size && tensor_blocks[k].type == a.type)
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(a.index, k), ccv_max(a.index, k));
					// Good, push to opt array.
					if (cell.u8 && cell.u8[0] == 1)
						continue;
					if (companion_ref >= 0)
					{
						assert(companion_ref != k);
						// Have to make sure k doesn't interfere with the designated companion as well.
						ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(companion_ref, k), ccv_max(companion_ref, k));
						if (cell.u8 && cell.u8[0] == 1)
							continue;
					}
					ccv_nnc_tensor_opt_t b = a;
					b.companion = k;
					b.oc = a.oc + oc[k];
					b.size = tensor_blocks[k].size;
					ccv_array_push(opt, &b);
				}
		}
		// Order opt array by the size.
		_ccv_nnc_tensor_opt_sort_by_size_and_oc((ccv_nnc_tensor_opt_t*)opt->data, opt->rnum, 0);
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
			if (tensor_blocks[a.index].companion_ref)
			{
				const int companion_ref = tensor_blocks[a.index].companion_ref - 1;
				assert(a.companion != companion_ref);
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
			assert(tensor_blocks[q].type == tensor_blocks[p].type);
			const int type = tensor_blocks[p].type;
#define for_block(y, x, val) do { \
				/* y is always earlier than x, but this is hard to assert now. */ \
				/* If this edge satisfy the requirement, now we need to find the ones with tightest possible bounds. */ \
				/* Thus, the hop between y and x (through a) should be smallest ones. */ \
				if (((uint64_t*)val)[0] >= a.size && \
					(y == 0 || tensor_blocks[y - 1].type == type) /* check the tensor block type matches. */ && \
					(x == tensor_block_size + 1 || tensor_blocks[x - 1].type == type)) \
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
		int strings[3];
		strings[0] = a.index + 1;
		int string_size = 1;
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
		}
		// Assign out designated companion if it exist.
		if (tensor_blocks[a.index].companion_ref && a.companion != tensor_blocks[a.index].companion_ref - 1)
		{
			const int companion_ref = tensor_blocks[a.index].companion_ref - 1;
			assert(tensor_blocks[a.index].type == tensor_blocks[companion_ref].type);
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
		}
		// Assign out and update oc.
		for (i = 0; i < string_size; i++)
		{
			const int index = strings[i] - 1;
			// Assign out the selected one.
			assigned[index] = assign_group;
			// The offset for this one, should be either 0 (started a new group, when min_i == -1), or the offset on this edge.
			allocated_offset[index] = min_val[1];
			for (k = 0; k < tensor_block_size; k++)
				if (!assigned[k] && TENSOR_EXPECT_COMPUTABLE(tensor_blocks[k]))
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(tensor_itf, ccv_min(k, index), ccv_max(k, index));
					if (cell.u8 && cell.u8[0] == 1)
						--oc[k];
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
			ccv_array_add_unique_int(alloc_dep[x - 1], y - 1); \
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
	memset(alloc_prep->buffers, 0, sizeof(alloc_prep->buffers[0]) * num_assigned);
	for (i = 0; i < num_assigned; i++)
		alloc_prep->buffers[i].size = allocated_size[i];
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
				const int buffer_ref = alloc_prep->blocks[j].buffer_ref = assigned[i] - 1;
				alloc_prep->blocks[j].offset = allocated_offset[i];
				if (!alloc_prep->buffers[buffer_ref].type)
					alloc_prep->buffers[buffer_ref].type = tensor_blocks[i].type;
				alloc_prep->buffers[buffer_ref].flags |= TENSOR_READ_WRITE(tensor_blocks[i]);
				assert(allocated_offset[i] + tensor_blocks[i].size <= alloc_prep->buffers[buffer_ref].size);
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
	for (i = 0; i < alloc_prep->buffer_size; i++)
		if (alloc_prep->buffers[i].dup_p_refs)
			ccv_array_free(alloc_prep->buffers[i].dup_p_refs);
	ccfree(alloc_prep->alloc_dep);
	ccfree(alloc_prep);
}

// Simple allocator from ccv_array_t.
static int _ccv_nnc_tensor_metadata_pos_new(ccv_array_t* const tensor_metadata, const size_t size)
{
	int pos = tensor_metadata->rnum;
	int rsize = (size + 15) / 16;
	ccv_array_resize(tensor_metadata, pos + rsize);
	return (pos << 1) + 1;
}

static ccv_nnc_tensor_t* _ccv_nnc_tensor_metadata_get(const ccv_array_t* const tensor_metadata, const int pos)
{
	assert((pos >> 1) < tensor_metadata->rnum);
	return (ccv_nnc_tensor_t*)ccv_array_get(tensor_metadata, pos >> 1);
}

#define CCV_NNC_IS_METADATA_POS(ptr) ((uintptr_t)(ptr) & 1)

static ccv_nnc_tensor_t* _ccv_nnc_tensor_metadata_rewire(const ccv_array_t* const tensor_metadata, ccv_nnc_tensor_t* const vt_tensor)
{
	// If the low bit is not 1, this is not a position (but a normal tensor pointer), just return directly.
	if (!CCV_NNC_IS_METADATA_POS(vt_tensor))
		return vt_tensor;
	ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)vt_tensor);
	if (tensor->alias_ref && CCV_NNC_IS_METADATA_POS(tensor->alias_ref))
	{
		const int alias_ref = tensor->alias_ref;
		tensor->alias_ref = (uintptr_t)_ccv_nnc_tensor_metadata_get(tensor_metadata, (int)tensor->alias_ref);
		_ccv_nnc_tensor_metadata_rewire(tensor_metadata, (ccv_nnc_tensor_t*)(intptr_t)alias_ref);
	}
	if (CCV_IS_TENSOR_MULTIVIEW(tensor))
	{
		ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
		int i;
		const int count = mv->kind + mv->repeat;
		for (i = 0; i < count; i++)
		{
			if (CCV_NNC_IS_METADATA_POS(CCV_NNC_MULTIVIEW_DATA(mv)[i]))
			{
				const int pos = (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i];
				CCV_NNC_MULTIVIEW_DATA(mv)[i] = _ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i]);
				_ccv_nnc_tensor_metadata_rewire(tensor_metadata, (ccv_nnc_tensor_t*)(intptr_t)pos);
			}
		}
		// No need to recursively do parent pointer, otherwise we are in deep rewire.
		if (mv->p && CCV_NNC_IS_METADATA_POS(mv->p))
			mv->p = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)mv->p);
		if (mv->sp)
			for (i = 0; i < mv->sp->rnum; i++)
			{
				ccv_nnc_tensor_t** const tensor = (ccv_nnc_tensor_t**)ccv_array_get(mv->sp, i);
				if (CCV_NNC_IS_METADATA_POS(*tensor))
				{
					const int pos = (int)(intptr_t)*tensor;
					*tensor = _ccv_nnc_tensor_metadata_get(tensor_metadata, pos);
					assert(!CCV_IS_TENSOR_MULTIVIEW(*tensor));
					_ccv_nnc_tensor_metadata_rewire(tensor_metadata, (ccv_nnc_tensor_t*)(intptr_t)pos);
				}
			}
	}
	return tensor;
}

typedef struct {
	const uint8_t* ptr;
	int pos;
} ccv_nnc_tensor_block_pos_t;

static int _ccv_nnc_tensor_multiview_find_pos(ccv_array_t* const tensor_metadata, const ccv_nnc_tensor_param_t params, const ccv_nnc_symbolic_graph_prep_t* const *const preps, const int block_ref, const int* const ch, const int idx, const ccv_nnc_symbolic_graph_prep_t* prep)
{
	int i;
	int unref_block_ref = block_ref;
	while (prep->tensor_blocks[unref_block_ref].ref)
		unref_block_ref = prep->tensor_blocks[unref_block_ref].ref - 1;
	int vt_ref = prep->alloc_prep->vt_blocks[unref_block_ref];
	assert(vt_ref >= 0);
	assert(unref_block_ref == prep->alloc_prep->blocks[vt_ref].block_ref);
	const int buffer_ref = prep->alloc_prep->blocks[vt_ref].buffer_ref;
	uint64_t offset = prep->alloc_prep->blocks[vt_ref].offset;
	int p_ref = prep->alloc_prep->buffers[buffer_ref].p_refs[0] - 1;
	for (i = idx - 1; i >= 0; i--)
	{
		assert(p_ref >= 0);
		const ccv_nnc_symbolic_graph_prep_t* const graph_prep = preps[i];
		const int unroll_count = graph_prep->unroll_count;
		if (ch[i]) // Prefer the dup side of things.
			p_ref = graph_prep->dup_tensor_block_ref[p_ref * unroll_count + ch[i] - 1];
		int unref_p_ref = p_ref;
		while (graph_prep->tensor_blocks[unref_p_ref].ref)
			unref_p_ref = graph_prep->tensor_blocks[unref_p_ref].ref - 1;
		vt_ref = graph_prep->alloc_prep->vt_blocks[unref_p_ref];
		const int buffer_ref = graph_prep->alloc_prep->blocks[vt_ref].buffer_ref;
		offset += graph_prep->alloc_prep->blocks[vt_ref].offset;
		// If the buffer already exists, prefer that.
		const uint8_t* ptr = graph_prep->tensor_arena->buffers[buffer_ref].ptr;
		if (ptr)
		{
			// If I have any remaining path that is not covered from 0, I cannot possibly
			// have any pointer from buffer (that can only happen if it is not dup).
			for (--i; i >= 0; i--)
				if (ch[i] != 0)
					return 0;
			// Try to find the created tensor block pos in the array, just linear scan.
			const int tv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_metadata, sizeof(ccv_nnc_tensor_t));
			ccv_nnc_tensor_t* const tv = (ccv_nnc_tensor_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, tv_pos);
			*tv = ccv_nnc_tensor(graph_prep->tensor_arena->buffers[buffer_ref].ptr + offset, params, 0);
			return tv_pos;
		}
		p_ref = graph_prep->alloc_prep->buffers[buffer_ref].p_refs[0] - 1;
	}
	return 0;
}

// Descent from root to the prep level, and compose multiview from there.
static int _ccv_nnc_tensor_multiview_down_find_pos(ccv_array_t* const tensor_metadata, const ccv_nnc_tensor_param_t params, const int preserve, const int assign_update, const ccv_nnc_symbolic_graph_prep_t* const *const preps, const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const int block_ref, int* ch, const int idx, int* const pos_ref)
{
	assert(pos_ref);
	int i;
	const ccv_nnc_symbolic_graph_prep_t* const prep = preps[idx];
	const int unroll_count = prep->unroll_count;
	if (prep == graph_prep)
	{
		const int data_pos = _ccv_nnc_tensor_multiview_find_pos(tensor_metadata, params, preps, block_ref, ch, idx, prep);
		if (!data_pos)
			return -1;
		// Based on ch, go all the way back to find the exact pointer to compose.
		if (// !assign_update && // If I plan to receive assign update, we don't need to have multiple receiver. Just one tensor to receive update is enough.
			prep->dup_tensor_block_ref &&
			prep->dup_tensor_block_ref[block_ref * unroll_count] >= 0 &&
			prep->dup_tensor_block_ref[block_ref * unroll_count] != block_ref)
		{
			int pos[unroll_count + 1];
			pos[0] = data_pos;
			for (i = 0; i < unroll_count; i++)
				pos[i + 1] = _ccv_nnc_tensor_multiview_find_pos(tensor_metadata, params, preps, prep->dup_tensor_block_ref[block_ref * unroll_count + i], ch, idx, prep);
			const int mv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_metadata, sizeof(ccv_nnc_tensor_multiview_t));
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, mv_pos);
			ccv_nnc_tensor_t* data[unroll_count + 1];
			for (i = 0; i < unroll_count + 1; i++)
				data[i] = _ccv_nnc_tensor_metadata_get(tensor_metadata, pos[i]);
			ccv_nnc_tensor_multiview(data, CCV_NNC_MULTIVIEW_K0N, unroll_count + 1, prep->graph, mv);
			for (i = 0; i < unroll_count + 1; i++)
				CCV_NNC_MULTIVIEW_DATA(mv)[i] = (ccv_nnc_tensor_t*)(intptr_t)pos[i];
			*pos_ref = mv_pos;
		} else {
			*pos_ref = data_pos;
		}
		if (preserve)
		{
			// If need to preserve, this need to be more complicated. At loop 0, I need to access the new assigned tv.
			// at any other loops, it should be the same. Thus, for this case, I will create a mv tensor as following:
			// mv of K11, thus, when loop is 0, it unwrap to mv->data[0], otherwise, unwrap to mv->data[1].
			// mv->data[0] (thin_mv) is a K01, which points to the assigned tv (using 1 as a placeholder here until parent
			// arena allocated).
			// mv->data[1] (prev_mv_pos_ is a K01 or K02, depending on whether above we passed raw pointer directly or
			// a mv structure. If we pass a mv structure, we just pass it here. If we pass a raw pointer, we need to wrap
			// it to a K01 structure.
			// Why we didn't wrap it directly as mv->data[0] pointing to a assigned tv pointer and the mv->data[1] pointing
			// to the raw pointer (as ptr_ref) with K11? The reason is we don't know the assigned tv is pointing to one
			// memory region, or is a managed by multi-view tensor, which could pointing to different memory regions.
			int prev_mv_pos = *pos_ref;
			if (prev_mv_pos == -1)
			{
				prev_mv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_metadata, sizeof(ccv_nnc_tensor_multiview_t));
				ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, prev_mv_pos);
				ccv_nnc_tensor_t* const tv = _ccv_nnc_tensor_metadata_get(tensor_metadata, data_pos);
				ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
					tv,
				}, CCV_NNC_MULTIVIEW_K0N, 1, prep->graph, mv);
				CCV_NNC_MULTIVIEW_DATA(mv)[0] = (ccv_nnc_tensor_t*)(intptr_t)data_pos;
			}
			const int mv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_metadata, sizeof(ccv_nnc_tensor_multiview_t));
			ccv_nnc_tensor_multiview_t* const prev_mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, prev_mv_pos);
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, mv_pos);
			ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
				CCV_NNC_TENSOR_PLACEHOLDER,
				(ccv_nnc_tensor_t*)prev_mv,
			}, CCV_NNC_MULTIVIEW_K1N, 1, prep->graph, mv);
			prev_mv->p = (void*)(intptr_t)mv_pos;
			CCV_NNC_MULTIVIEW_DATA(mv)[0] = CCV_NNC_TENSOR_PLACEHOLDER;
			CCV_NNC_MULTIVIEW_DATA(mv)[1] = (ccv_nnc_tensor_t*)(intptr_t)prev_mv_pos;
			*pos_ref = mv_pos;
		}
		return 0;
	}
	ch[idx] = 0;
	int pos[unroll_count + 1];
	pos[0] = 0;
	const int retval = _ccv_nnc_tensor_multiview_down_find_pos(tensor_metadata, params, preserve, assign_update, preps, graph_prep, block_ref, ch, idx + 1, pos);
	assert(retval == 0);
	for (i = 0; i < unroll_count; i++)
	{
		ch[idx] = i + 1;
		pos[i + 1] = 0;
		const int dup_retval = _ccv_nnc_tensor_multiview_down_find_pos(tensor_metadata, params, preserve, assign_update, preps, graph_prep, block_ref, ch, idx + 1, pos + i + 1);
		if (dup_retval < 0)
		{
			assert(i == 0);
			break;
		}
	}
	// If current prep has no dup.
	if (i == 0)
	{
		*pos_ref = pos[0];
		return 0;
	}
	ccv_nnc_tensor_t* data[unroll_count + 1];
	// Compose to a new multiview.
	for (i = 0; i < unroll_count + 1; i++)
		{ assert(pos[i] > 0); }
	const int mv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_metadata, sizeof(ccv_nnc_tensor_multiview_t));
	for (i = 0; i < unroll_count + 1; i++)
		data[i] = _ccv_nnc_tensor_metadata_get(tensor_metadata, pos[i]);
	ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, mv_pos);
	ccv_nnc_tensor_multiview(data, CCV_NNC_MULTIVIEW_K0N, unroll_count + 1, prep->graph, mv);
	for (i = 0; i < unroll_count + 1; i++)
		if (data[i] != CCV_NNC_TENSOR_PLACEHOLDER && CCV_IS_TENSOR_MULTIVIEW(data[i]))
			((ccv_nnc_tensor_multiview_t*)data[i])->p = (void*)(intptr_t)mv_pos;
	for (i = 0; i < unroll_count + 1; i++)
		CCV_NNC_MULTIVIEW_DATA(mv)[i] = (ccv_nnc_tensor_t*)(intptr_t)pos[i];
	*pos_ref = mv_pos;
	return 0;
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
	// Prefer it is an output if it is both the input and the output.
	if (is_output)
		return 1;
	if (is_input)
		return -1;
	return 0;
}

static int _ccv_nnc_tensor_block_check_preserve(const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const int block_ref)
{
	// No need to check whether to preserve if this is not a while loop.
	if (!(graph_prep->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
		return 0;
	assert(block_ref >= 0 && block_ref < graph_prep->tensor_symbol_info_size);
	// If it is unassigned, no need to preserve.
	if (TENSOR_EXPECT_UNASSIGNED(graph_prep->tensor_blocks[block_ref]))
		return 0;
	const int p_ref = graph_prep->tensor_blocks[block_ref].p_refs[0] - 1;
	// If p is not input, no need to preserve at all.
	if (-1 != _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref, graph_prep->p->exec_symbol_info + (graph_prep->exec_idx - 1)))
		return 0;
	const int vt_ref = graph_prep->alloc_prep->vt_blocks[block_ref];
	assert(vt_ref >= 0);
	assert(block_ref == graph_prep->alloc_prep->blocks[vt_ref].block_ref);
	const int buffer_ref = graph_prep->alloc_prep->blocks[vt_ref].buffer_ref;
	// If the buffer is a truly read-only one, no need to preserve.
	if (TENSOR_READ_WRITE(graph_prep->alloc_prep->buffers[buffer_ref]) == READ_ONLY)
		return 0;
	/* This needs detailed explanation, what does preserve mean?
	 * For a parameterized loop, such as while { y = x + 1 } (y => x), if tensor x is
	 * also used outside of the while loop, we cannot reuse the memory region of x for
	 * the for loop, otherwise we will destroy x when doing y = x + 1 computation (assuming
	 * y uses the same memory region as x). The way to workaround this is by using a different
	 * memory region for y = x + 1, but for the first iteration, having x pointing to the
	 * original. During the allocation process, the way to identify whether x should preserve
	 * its value or not by looking up its parent tensor. If the symbol (tensor_block)'s input
	 * parent tensor is the same as the memory region it plans to use in the buffer, then we are
	 * good (buffer.p_refs[0] == p_refs[0]). A buffer can only point to one parent tensor, and
	 * it is the input tensor whenever that is possible. A tensor block can point to two parent
	 * tensors, one is input tensor, one is the output tensor. p_refs[0] should be the input
	 * tensor whenever that is possible. */
	if (graph_prep->alloc_prep->buffers[buffer_ref].p_refs[0] - 1 == p_ref)
		return 0;
	// Otherwise, return 1 because we now need to preserve.
	return 1;
}

static int _ccv_nnc_tensor_block_check_force_broadcast(const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const int block_ref)
{
	assert(block_ref >= 0 && block_ref < graph_prep->tensor_symbol_info_size);
	// If it is unassigned, no need to preserve.
	if (TENSOR_EXPECT_UNASSIGNED(graph_prep->tensor_blocks[block_ref]))
		return 0;
	// Only tape var need to force broadcast, otherwise we already share the same memory region.
	if (!(graph_prep->tensor_symbol_info[block_ref].flags & CCV_NNC_TENSOR_SYMBOL_TAPE_VAR))
		return 0;
	const int p_ref = graph_prep->tensor_blocks[block_ref].p_refs[0] - 1;
	// If p is not output, no need to broadcast at all.
	if (1 != _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref, graph_prep->p->exec_symbol_info + (graph_prep->exec_idx - 1)))
		return 0;
	const int vt_ref = graph_prep->alloc_prep->vt_blocks[block_ref];
	assert(vt_ref >= 0);
	assert(block_ref == graph_prep->alloc_prep->blocks[vt_ref].block_ref);
	const int buffer_ref = graph_prep->alloc_prep->blocks[vt_ref].buffer_ref;
	// If the buffer is a truly read-only one, no need to broadcast.
	if (TENSOR_READ_WRITE(graph_prep->alloc_prep->buffers[buffer_ref]) == READ_ONLY)
		return 0;
	// Otherwise, return 1 because we now need to force broadcast for this tape var.
	return 1;
}

static void _ccv_nnc_tensor_multiview_full_pos(ccv_nnc_tensor_multiview_t* const mv, ccv_nnc_tensor_t* const tensor)
{
	assert(CCV_IS_TENSOR_MULTIVIEW(mv));
	int i;
	for (i = 0; i < mv->kind + mv->repeat; i++)
		if (CCV_NNC_MULTIVIEW_DATA(mv)[i] == CCV_NNC_TENSOR_PLACEHOLDER)
			CCV_NNC_MULTIVIEW_DATA(mv)[i] = tensor;
		else if (CCV_IS_TENSOR_MULTIVIEW(CCV_NNC_MULTIVIEW_DATA(mv)[i]))
			_ccv_nnc_tensor_multiview_full_pos((ccv_nnc_tensor_multiview_t*)CCV_NNC_MULTIVIEW_DATA(mv)[i], tensor);
}

static void _ccv_nnc_tensor_multiview_full_pos_rewire(const ccv_array_t* const tensor_metadata, ccv_nnc_tensor_multiview_t* const mv)
{
	assert(CCV_IS_TENSOR_MULTIVIEW(mv));
	int i;
	if (mv->sp)
		for (i = 0; i < mv->sp->rnum; i++)
		{
			ccv_nnc_tensor_t** const tensor = (ccv_nnc_tensor_t**)ccv_array_get(mv->sp, i);
			if (CCV_NNC_IS_METADATA_POS(*tensor))
			{
				const int pos = (int)(intptr_t)*tensor;
				*tensor = _ccv_nnc_tensor_metadata_get(tensor_metadata, pos);
				assert(!CCV_IS_TENSOR_MULTIVIEW(*tensor));
				_ccv_nnc_tensor_metadata_rewire(tensor_metadata, (ccv_nnc_tensor_t*)(intptr_t)pos);
			}
		}
	for (i = 0; i < mv->kind + mv->repeat; i++)
	{
		if (CCV_NNC_IS_METADATA_POS((int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i]))
			CCV_NNC_MULTIVIEW_DATA(mv)[i] = _ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i]);
		if (CCV_NNC_IS_METADATA_POS((int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i]->alias_ref))
			CCV_NNC_MULTIVIEW_DATA(mv)[i]->alias_ref = (uintptr_t)_ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[i]->alias_ref);
		if (CCV_IS_TENSOR_MULTIVIEW(CCV_NNC_MULTIVIEW_DATA(mv)[i]))
			_ccv_nnc_tensor_multiview_full_pos_rewire(tensor_metadata, (ccv_nnc_tensor_multiview_t*)CCV_NNC_MULTIVIEW_DATA(mv)[i]);
	}
}

static int _ccv_nnc_tensor_multiview_gen(ccv_array_t* const tensor_metadata, const int preserve, const int assign_update, const ccv_nnc_tensor_param_t params, const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const ccv_nnc_tensor_arena_t* const tensor_arena, const int block_ref)
{
	// Go to the root of the graph.
	const ccv_nnc_symbolic_graph_prep_t* prep = graph_prep;
	int i;
	for (i = 1; prep->p; i++)
		prep = prep->p;
	// Root graph should have no dup tensor blocks.
	assert(!prep->dup_tensor_block_ref);
	const int c = i;
	const ccv_nnc_symbolic_graph_prep_t* preps[c];
	prep = graph_prep;
	preps[c - 1] = prep;
	for (i = 0; prep->p; i++)
		preps[c - 2 - i] = prep = prep->p;
	int ch[c]; // Use dynamic allocation for array. This is an array to record our selections when recursive from top to bottom.
	memset(ch, 0, sizeof(int) * c);
	int pos = 0;
	_ccv_nnc_tensor_multiview_down_find_pos(tensor_metadata, params, preserve, assign_update, preps, graph_prep, block_ref, ch, 0, &pos);
	assert(ch[c - 1] == 0); // This shouldn't never be modified.
	assert(pos > 0);
	return pos;
}

static int _ccv_nnc_tensor_multiview_preserve_gen(ccv_array_t* const tensor_metadata, const ccv_nnc_tensor_param_t params, const ccv_nnc_symbolic_graph_prep_t* const graph_prep, ccv_nnc_tensor_t* const tensor)
{
	const int mv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_metadata, sizeof(ccv_nnc_tensor_multiview_t));
	ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, mv_pos);
	ccv_nnc_tensor_t* const tv = CCV_NNC_IS_METADATA_POS(tensor) ? _ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)tensor) : tensor;
	ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
		CCV_NNC_TENSOR_PLACEHOLDER,
		tv,
	}, CCV_NNC_MULTIVIEW_K1N, 1, graph_prep->graph, mv);
	CCV_NNC_MULTIVIEW_DATA(mv)[0] = CCV_NNC_TENSOR_PLACEHOLDER;
	CCV_NNC_MULTIVIEW_DATA(mv)[1] = tensor;
	return mv_pos;
}

static int _ccv_nnc_tensor_flat_if_multiview(ccv_array_t* const tensor_metadata, const int pos)
{
	ccv_nnc_tensor_t* tensor_ptr = _ccv_nnc_tensor_metadata_get(tensor_metadata, pos);
	const int is_multiview = CCV_IS_TENSOR_MULTIVIEW(tensor_ptr);
	if (!is_multiview)
		return pos;
	while (CCV_IS_TENSOR_MULTIVIEW(tensor_ptr))
	{
		const ccv_nnc_tensor_multiview_t* const mv = (const ccv_nnc_tensor_multiview_t*)tensor_ptr;
		tensor_ptr = _ccv_nnc_tensor_metadata_get(tensor_metadata, (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[0]);
	}
	const ccv_nnc_tensor_t tensor = *tensor_ptr;
	const int new_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_metadata, sizeof(ccv_nnc_tensor_t));
	ccv_nnc_tensor_t* const new_tensor = _ccv_nnc_tensor_metadata_get(tensor_metadata, new_pos);
	*new_tensor = ccv_nnc_tensor(tensor.data.u8, tensor.info, 0);
	ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_metadata, pos);
	new_tensor->alias_ref = (uintptr_t)pos;
	ccv_nnc_tensor_synchronize_to_multiview(mv, (ccv_nnc_tensor_t*)(intptr_t)new_pos);
	return new_pos;
}

static ccv_nnc_tensor_arena_t* _ccv_nnc_tensor_arena_new(ccv_nnc_symbolic_graph_prep_t* const graph_prep, const ccv_nnc_tensor_arena_t* const p_arena, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size)
{
	// All tensors assigned out, now, the num_assigned is the number of dis-continuous buffers,
	// Each tensor have the designation in assigned array, and offset in allocated_offset.
	const ccv_nnc_tensor_alloc_prep_t* const alloc_prep = graph_prep->alloc_prep;
	ccv_nnc_tensor_block_t* const tensor_blocks = graph_prep->tensor_blocks;
	const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = graph_prep->tensor_symbol_info;
	const int tensor_symbol_info_size = graph_prep->tensor_symbol_info_size;
	const ccv_nnc_symbolic_graph_prep_t* const p_graph_prep = graph_prep->p;
	const ccv_nnc_tensor_alloc_prep_t* const p_alloc_prep = p_graph_prep ? p_graph_prep->alloc_prep : 0;
	const int* const dup_tensor_block_ref = graph_prep->dup_tensor_block_ref;
	const int unroll_count = graph_prep->unroll_count;
	int i, j;
	for (i = 0; i < tensor_symbol_info_size; i++)
		for (j = 0; TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]) && j < unroll_count; j++)
		{
			const int dup_ref = dup_tensor_block_ref[i * unroll_count + j];
			if (dup_ref >= 0 && !TENSOR_EXPECT_UNASSIGNED(tensor_blocks[dup_ref]))
				TENSOR_EXPECT_UNSET_UNASSIGNED(tensor_blocks[i]);
		}
	ccv_nnc_tensor_arena_t* tensor_arena = (ccv_nnc_tensor_arena_t*)ccmalloc(sizeof(ccv_nnc_tensor_arena_t) + sizeof(tensor_arena->buffers[0]) * alloc_prep->buffer_size + sizeof(ccv_nnc_tensor_t*) * tensor_symbol_info_size + sizeof(ccv_nnc_tensor_arena_t*) * graph_prep->sub_prep_size);
	graph_prep->tensor_arena = tensor_arena;
	tensor_arena->graph_ref = (intptr_t)graph_prep->symbolic_graph;
	tensor_arena->buffers = (void*)(tensor_arena + 1);
	tensor_arena->buffer_size = alloc_prep->buffer_size;
	tensor_arena->vt_tensor_size = tensor_symbol_info_size;
	tensor_arena->vt_tensors = (ccv_nnc_tensor_t**)(tensor_arena->buffers + alloc_prep->buffer_size);
	tensor_arena->sub_arenas = (ccv_nnc_tensor_arena_t**)(tensor_arena->vt_tensors + tensor_symbol_info_size);
	tensor_arena->sub_arena_size = graph_prep->sub_prep_size;
	tensor_arena->tensor_metadata = ccv_array_new(16 /* align to 16 bytes */, 0, 0);
	tensor_arena->m_tensor_idx = ccv_array_new(sizeof(int), 0, 0);
	for (i = 0; i < alloc_prep->buffer_size; i++)
		tensor_arena->buffers[i].type = alloc_prep->buffers[i].type, tensor_arena->buffers[i].size = alloc_prep->buffers[i].size;
	if (graph_prep->while_count_tensor)
	{
		// If we need to have a while count tensor, allocate that first, set its pointer to point the while_count variable.
		int pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
		assert((0 << 1) + 1 == pos); // pos must be 0 position.
		ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
		*tensor = ccv_nnc_tensor_for_while_count(graph_prep->graph);
	}
	assert((p_arena && p_graph_prep) || (!p_arena && !p_graph_prep));
	if (p_arena && p_graph_prep)
	{
		// Don't need to allocate the actual buffer, just use the pointer from the above.
		PRINT(CCV_CLI_VERBOSE, "Buffer assignment for sub arena %p (parent %p)\n", tensor_arena, p_arena);
		for (i = 0; i < tensor_arena->buffer_size; i++)
		{
			const int p_ref = alloc_prep->buffers[i].p_refs[0] - 1;
			int unref_p_ref = p_ref;
			while (p_graph_prep->tensor_blocks[unref_p_ref].ref)
				unref_p_ref = p_graph_prep->tensor_blocks[unref_p_ref].ref - 1;
			assert(unref_p_ref >= 0);
			const int p_unroll_count = p_graph_prep->unroll_count;
			if (p_graph_prep->dup_tensor_block_ref &&
				p_graph_prep->dup_tensor_block_ref[p_ref * p_unroll_count] >= 0 &&
				p_graph_prep->dup_tensor_block_ref[p_ref * p_unroll_count] != p_ref)
			{
				// This condition means in the parent graph, we point to multiple tensor blocks for the same
				// buffer, therefore, we cannot have one single pointer assigned in this case.
				// Later we will handle this by generate ccv_tensor_multiview_t structure.
				tensor_arena->buffers[i].ptr = 0;
				PRINT(CCV_CLI_VERBOSE, "|-Cannot assign buffer %d, it points to multiple blocks (multi view tensor required)\n", i);
				continue;
			}
			// Otherwise, find the actual buffer pointer.
			const int vt_ref = p_alloc_prep->vt_blocks[unref_p_ref];
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
		for (i = 0; i < tensor_arena->buffer_size; i++)
		{
			const int memory_type = CCV_TENSOR_GET_MEMORY(tensor_arena->buffers[i].type);
#ifdef HAVE_CUDA
			if (memory_type == CCV_TENSOR_GPU_MEMORY)
			{
				const int device_id = CCV_TENSOR_GET_DEVICE_ID(tensor_arena->buffers[i].type);
				tensor_arena->buffers[i].ptr = (uint8_t*)cumalloc(device_id, tensor_arena->buffers[i].size);
				PRINT(CCV_CLI_VERBOSE, "|-Allocate buffer %d with ptr %p, size %lu\n", i, tensor_arena->buffers[i].ptr, (unsigned long)tensor_arena->buffers[i].size);
			} else {
				assert(memory_type == CCV_TENSOR_CPU_MEMORY);
				ccmemalign((void**)&tensor_arena->buffers[i].ptr, 16, tensor_arena->buffers[i].size);
				PRINT(CCV_CLI_VERBOSE, "|-Allocate buffer %d with ptr %p, size %lu\n", i, tensor_arena->buffers[i].ptr, (unsigned long)tensor_arena->buffers[i].size);
			}
#else
			assert(memory_type == CCV_TENSOR_CPU_MEMORY);
			ccmemalign((void**)&tensor_arena->buffers[i].ptr, 16, tensor_arena->buffers[i].size);
			PRINT(CCV_CLI_VERBOSE, "|-Allocate buffer %d with ptr %p, size %lu\n", i, tensor_arena->buffers[i].ptr, (unsigned long)tensor_arena->buffers[i].size);
#endif
			assert(tensor_arena->buffers[i].ptr);
		}
	}
	// Go over sub_preps and allocate arenas for them. Do it this early because
	// we may reference tensors from sub arenas, the reason why we need to reference
	// tensors from sub arenas is because for output tensors, sub arena's tensor
	// will have automatic reference updates.
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (graph_prep->sub_preps[i])
			tensor_arena->sub_arenas[i] = _ccv_nnc_tensor_arena_new(graph_prep->sub_preps[i], tensor_arena, tensor_binds, tensor_bind_size);
		else
			tensor_arena->sub_arenas[i] = 0;
	memset(tensor_arena->vt_tensors, 0, sizeof(ccv_nnc_tensor_t*) * tensor_symbol_info_size);
	// Now sub-arenas are all assigned, go over its outputs to assign out tensors from its output directly.
	ccv_nnc_tensor_t** sub_arena_out_tensors = tensor_arena->sub_arena_size ? (ccv_nnc_tensor_t**)cccalloc(tensor_symbol_info_size, sizeof(ccv_nnc_tensor_t*)) : 0;
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (tensor_arena->sub_arenas[i])
		{
			const int exec_idx = graph_prep->sub_preps[i]->exec_idx - 1;
			const ccv_nnc_graph_exec_symbol_info_t* const node = graph_prep->exec_symbol_info + exec_idx;
			if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
				for (j = 0; j < node->output_size; j++)
				{
					const int idx = node->outputs[j];
					const int s_idx = *(int*)ccv_array_get(tensor_symbol_info[idx].s_ref, i) - 1;
					assert(s_idx >= 0);
					ccv_nnc_tensor_t* sub_tensor = tensor_arena->sub_arenas[i]->vt_tensors[s_idx];
					assert(sub_arena_out_tensors[idx] == 0);
					ccv_nnc_tensor_t* sub_alias = (ccv_nnc_tensor_t*)sub_tensor->alias_ref;
					// Only assign if it is a multiview tensor.
					if (CCV_IS_TENSOR_MULTIVIEW(sub_tensor) ||
						(sub_alias && CCV_IS_TENSOR_MULTIVIEW(sub_alias)))
						sub_arena_out_tensors[idx] = sub_tensor;
				}
		}
	// Assigning out the tensors (in case of sharing tensors / in-place ops).
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]))
		{
			const int vt_ref = alloc_prep->vt_blocks[i];
			const int buffer_ref = vt_ref >= 0 ? alloc_prep->blocks[vt_ref].buffer_ref : -1;
			// Either we have dup_tensor_block_ref in current layer, or we have that in
			// previous layer, therefore, cannot really find the buffer ptr.
			if ((!sub_arena_out_tensors || !sub_arena_out_tensors[i]) && // If it is already generated by sub arena, it can be ordinary out tensors. (What if out tensor is not even generated by sub graph when running? In this case, the behavior is undefined anyway).
				((graph_prep->dup_tensor_block_ref &&
				  graph_prep->dup_tensor_block_ref[i * unroll_count] >= 0 &&
				  graph_prep->dup_tensor_block_ref[i * unroll_count] != i) ||
				 (buffer_ref >= 0 && !tensor_arena->buffers[buffer_ref].ptr)))
			{
				assert(graph_prep->p); // This must be in a sub-graph.
				// If this is an input tensor, and it need to be preserved, wait until when we go through inputs to preserve.
				if (graph_prep->tensor_blocks[i].p_refs[0] && _ccv_nnc_tensor_block_check_preserve(graph_prep, i))
					continue;
				const int pos = _ccv_nnc_tensor_multiview_gen(tensor_arena->tensor_metadata, 0, tensor_symbol_info[i].assign_ref, tensor_symbol_info[i].info, graph_prep, tensor_arena, i);
				tensor_arena->vt_tensors[i] = (ccv_nnc_tensor_t*)(intptr_t)pos;
				ccv_array_push(tensor_arena->m_tensor_idx, &pos);
			} else if (!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i])) {
				// When we want to allocate, we don't really need to if it need force broadcast, because we will handle that later.
				const uint64_t offset = alloc_prep->blocks[vt_ref].offset;
				// If already created, use the same tensor, and continue.
				// Having ptr.
				int pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
				ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
				// Also, set its allocations.
				// Since tensor view is bit compatible with tensor, we can just cast.
				*tensor = ccv_nnc_tensor(tensor_arena->buffers[buffer_ref].ptr + offset, tensor_symbol_info[i].info, 0);
				assert(offset + tensor_blocks[i].size <= tensor_arena->buffers[buffer_ref].size);
				// If we need to force broadcast, we need to wrap it in a multiview.
				if (graph_prep->tensor_blocks[i].p_refs[0] &&
					_ccv_nnc_tensor_block_check_force_broadcast(graph_prep, i))
				{
					const int mv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_multiview_t));
					ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, mv_pos);
					ccv_nnc_tensor_t* const tv = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
					ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
						tv,
					}, 0, 1, graph_prep->graph, mv);
					CCV_NNC_MULTIVIEW_DATA(mv)[0] = (ccv_nnc_tensor_t*)(intptr_t)pos;
					pos = mv_pos;
					ccv_array_push(tensor_arena->m_tensor_idx, &mv_pos);
				}
				tensor_arena->vt_tensors[i] = (ccv_nnc_tensor_t*)(intptr_t)pos; // Cast into vt_tensors for now, and later will rewire it.
			}
		}
	// Assign out refs, refs are simple ones, we should handle it first. (because they point to exactly the same metadata and same region).
	for (i = 0; i < tensor_symbol_info_size; i++)
		// It could be binded tensor (or unused), in that case, it doesn't have a ref.
		if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]) && tensor_blocks[i].ref && !tensor_arena->vt_tensors[i])
		{
			int ref = tensor_blocks[i].ref - 1;
			while (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) && tensor_blocks[ref].ref)
				ref = tensor_blocks[ref].ref - 1;
			assert(tensor_arena->vt_tensors[ref]);
			tensor_arena->vt_tensors[i] = tensor_arena->vt_tensors[ref];
		}
	// Now after refs assigned out, handle the case I need to preserve because I am a sub graph of while loop.
	if (graph_prep->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
	{
		const ccv_nnc_graph_exec_symbol_info_t* node = graph_prep->p->exec_symbol_info + (graph_prep->exec_idx - 1);
		const int p_idx = graph_prep->p_idx - 1;
		for (i = 0; i < node->input_size; i++)
		{
			const int idx = node->inputs[i];
			int block_ref = *(int*)ccv_array_get(graph_prep->p->tensor_symbol_info[idx].s_ref, p_idx) - 1;
			assert(!tensor_blocks[block_ref].ref);
			const int vt_ref = alloc_prep->vt_blocks[block_ref];
			if (!_ccv_nnc_tensor_block_check_preserve(graph_prep, block_ref))
				continue;
			assert(vt_ref >= 0);
			const int buffer_ref = alloc_prep->blocks[vt_ref].buffer_ref;
			assert(!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[block_ref]));
			assert(!TENSOR_EXPECT_ALIAS(tensor_blocks[block_ref]));
			// Either we have dup_tensor_block_ref in current layer, or we have that in
			// previous layer, therefore, cannot really find the buffer ptr.
			if ((!sub_arena_out_tensors || !sub_arena_out_tensors[block_ref]) && // If it is already generated by sub arena, it can be ordinary out tensors. (What if out tensor is not even generated by sub graph when running? In this case, the behavior is undefined anyway).
				((graph_prep->dup_tensor_block_ref &&
				  graph_prep->dup_tensor_block_ref[block_ref * unroll_count] >= 0 &&
				  graph_prep->dup_tensor_block_ref[block_ref * unroll_count] != block_ref) ||
				 !tensor_arena->buffers[buffer_ref].ptr))
			{
				// We haven't allocated anything for this yet.
				assert(tensor_arena->vt_tensors[block_ref] == 0);
				const int pos = _ccv_nnc_tensor_multiview_gen(tensor_arena->tensor_metadata, 1, tensor_symbol_info[i].assign_ref, tensor_symbol_info[block_ref].info, graph_prep, tensor_arena, block_ref);
				tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)(intptr_t)pos;
				ccv_array_push(tensor_arena->m_tensor_idx, &pos);
			} else {
				const int mv_pos = _ccv_nnc_tensor_multiview_preserve_gen(tensor_arena->tensor_metadata, tensor_symbol_info[block_ref].info, graph_prep, tensor_arena->vt_tensors[block_ref]);
				tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)(intptr_t)mv_pos; // Cast into vt_tensors for now, and later will rewire.
				ccv_array_push(tensor_arena->m_tensor_idx, &mv_pos);
			}
		}
	}
	// For case..of statement, the output is a phi variable, thus, if we take the skip branch, we will select the original input.
	// This created the multi-view tensor to achieve that.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_blocks[i].bypass_ref && tensor_arena->vt_tensors[i])
		{
			const int bypass_ref = tensor_blocks[i].bypass_ref - 1;
			// Create phi multi-view.
			const int mv_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_multiview_t));
			const int intv_pos = _ccv_nnc_tensor_flat_if_multiview(tensor_arena->tensor_metadata, (int)(intptr_t)tensor_arena->vt_tensors[bypass_ref]);
			const int outv_pos = _ccv_nnc_tensor_flat_if_multiview(tensor_arena->tensor_metadata, (int)(intptr_t)tensor_arena->vt_tensors[i]);
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, mv_pos);
			ccv_nnc_tensor_t* const intv = (ccv_nnc_tensor_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, intv_pos);
			ccv_nnc_tensor_t* const outv = (ccv_nnc_tensor_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, outv_pos);
			ccv_nnc_tensor_multiview((ccv_nnc_tensor_t*[]){
				intv,
				outv,
			}, CCV_NNC_MULTIVIEW_K0N, 2, (ccv_nnc_graph_t*)CCV_NNC_MULTIVIEW_PHI, mv);
			CCV_NNC_MULTIVIEW_DATA(mv)[0] = (ccv_nnc_tensor_t*)(intptr_t)intv_pos;
			CCV_NNC_MULTIVIEW_DATA(mv)[1] = (ccv_nnc_tensor_t*)(intptr_t)outv_pos;
			tensor_arena->vt_tensors[i] = (ccv_nnc_tensor_t*)(intptr_t)mv_pos;
			ccv_array_push(tensor_arena->m_tensor_idx, &mv_pos);
		}
	// Handle binded tensors. We handle it here so the alias can reference to binded tensors.
	for (i = 0; i < tensor_bind_size; i++)
	{
		assert(tensor_binds[i].tensor);
		const ccv_nnc_tensor_symbol_t resolved_symbol = ccv_nnc_tensor_symbol_resolve(graph_prep->symbolic_graph, tensor_binds[i].symbol);
		if (resolved_symbol.d >= 0)
		{
			// For binded tensors, it shouldn't be assigned yet.
			assert(tensor_arena->vt_tensors[resolved_symbol.d] == 0);
			if (CCV_IS_TENSOR_VIEW(tensor_binds[i].tensor))
			{
				int pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_view_t));
				ccv_nnc_tensor_view_t* const tv = (ccv_nnc_tensor_view_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
				memcpy(tv, tensor_binds[i].tensor, sizeof(ccv_nnc_tensor_view_t));
				tensor_arena->vt_tensors[resolved_symbol.d] = (ccv_nnc_tensor_t*)(intptr_t)pos;
			} else {
				int pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
				ccv_nnc_tensor_t* const tv = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
				*tv = ccv_nnc_tensor(tensor_binds[i].tensor->data.ptr, tensor_binds[i].tensor->info, 0);
				tensor_arena->vt_tensors[resolved_symbol.d] = (ccv_nnc_tensor_t*)(intptr_t)pos;
			}
		}
	}
	// Now it is time to handle alias.
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
				// If this is not alias (it is binded then).
				if (!CCV_NNC_IS_METADATA_POS(tensor_arena->vt_tensors[alias_ref]))
				{
					int pos;
					if (memcmp(ccv_nnc_no_ofs, tensor_symbol_info[block_ref].ofs, sizeof(ccv_nnc_no_ofs)) == 0 &&
						memcmp(tensor_symbol_info[block_ref].inc, tensor_symbol_info[block_ref].info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
					{
						pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
						ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
						*tensor = ccv_nnc_tensor(tensor_arena->vt_tensors[alias_ref]->data.u8, tensor_symbol_info[block_ref].info, 0);
					} else {
						pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_view_t));
						ccv_nnc_tensor_view_t* const tensor_view = (ccv_nnc_tensor_view_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
						// Otherwise initialize a tensor view
						*tensor_view = ccv_nnc_tensor_view(tensor_arena->vt_tensors[alias_ref], tensor_symbol_info[block_ref].info.dim, tensor_symbol_info[block_ref].ofs, tensor_symbol_info[block_ref].inc);
						tensor_view->alias_ref = (uintptr_t)tensor_arena->vt_tensors[alias_ref];
					}
					tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)(intptr_t)pos;
					continue;
				}
				const int alias_pos = (int)(intptr_t)tensor_arena->vt_tensors[alias_ref];
				const ccv_nnc_tensor_t* alias_tensor_ptr = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, alias_pos);
				assert(!CCV_IS_TENSOR_VIEW(alias_tensor_ptr));
				// Will use that to determine whether insert reference or not.
				const int is_multiview = CCV_IS_TENSOR_MULTIVIEW(alias_tensor_ptr);
				while (CCV_IS_TENSOR_MULTIVIEW(alias_tensor_ptr))
				{
					const ccv_nnc_tensor_multiview_t* const mv = (const ccv_nnc_tensor_multiview_t*)alias_tensor_ptr;
					alias_tensor_ptr = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(mv)[0]);
				}
				const ccv_nnc_tensor_t alias_tensor = *alias_tensor_ptr;
				// If there is no ofs, and inc is the same as dim, we take a shortcut and just init as normal tensor.
				int pos;
				if (memcmp(ccv_nnc_no_ofs, tensor_symbol_info[block_ref].ofs, sizeof(ccv_nnc_no_ofs)) == 0 &&
					memcmp(tensor_symbol_info[block_ref].inc, tensor_symbol_info[block_ref].info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
				{
					pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
					ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
					*tensor = ccv_nnc_tensor(alias_tensor.data.u8, tensor_symbol_info[block_ref].info, 0);
				} else {
					pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_view_t));
					ccv_nnc_tensor_view_t* const tensor_view = (ccv_nnc_tensor_view_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, pos);
					// Otherwise initialize a tensor view
					*tensor_view = ccv_nnc_tensor_view(&alias_tensor, tensor_symbol_info[block_ref].info.dim, tensor_symbol_info[block_ref].ofs, tensor_symbol_info[block_ref].inc);
					tensor_view->alias_ref = (uintptr_t)alias_pos;
				}
				tensor_arena->vt_tensors[block_ref] = (ccv_nnc_tensor_t*)(intptr_t)pos;
				if (is_multiview)
				{
					ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, alias_pos);
					ccv_nnc_tensor_synchronize_to_multiview(mv, (ccv_nnc_tensor_t*)(intptr_t)pos);
				}
			}
		}
	// Replacing the tensor placeholder within sub arena's multi-view to the input tensor.
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (tensor_arena->sub_arenas[i])
		{
			const int exec_idx = graph_prep->sub_preps[i]->exec_idx - 1;
			const ccv_nnc_graph_exec_symbol_info_t* const node = graph_prep->exec_symbol_info + exec_idx;
			for (j = 0; j < node->input_size; j++)
			{
				const int idx = node->inputs[j];
				const int s_idx = (tensor_symbol_info[idx].s_ref && tensor_symbol_info[idx].s_ref->rnum > i) ? *(int*)ccv_array_get(tensor_symbol_info[idx].s_ref, i) - 1 : -1;
				if (s_idx < 0)
					continue;
				ccv_nnc_tensor_t* sub_tensor = tensor_arena->sub_arenas[i]->vt_tensors[s_idx];
				// Only do the replacement if it is a multi-view tensor.
				// sub_tensor can be unassigned if it is a tape variable. It will get fixed up later from its peer.
				if (sub_tensor && CCV_IS_TENSOR_MULTIVIEW(sub_tensor) && !TENSOR_EXPECT_UNASSIGNED(tensor_blocks[idx]))
				{
					// It cannot be binded tensor.
					assert(CCV_NNC_IS_METADATA_POS(tensor_arena->vt_tensors[idx]));
					const int vt_pos = (int)(intptr_t)tensor_arena->vt_tensors[idx];
					const int is_sub_arena_out_tensor = (sub_arena_out_tensors && sub_arena_out_tensors[idx]);
					ccv_nnc_tensor_t* const vt_tensor = is_sub_arena_out_tensor ? sub_arena_out_tensors[idx] : _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, vt_pos);
					// If this tensor is also an multiview, we need to first generate a new tensor, and then generate a reference
					// to this tensor.
					if (CCV_IS_TENSOR_MULTIVIEW(vt_tensor))
					{
						const int ref_pos = _ccv_nnc_tensor_metadata_pos_new(tensor_arena->tensor_metadata, sizeof(ccv_nnc_tensor_t));
						ccv_nnc_tensor_t* const ref_tensor = _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, ref_pos);
						ccv_nnc_tensor_multiview_t* const multiview = (ccv_nnc_tensor_multiview_t*)(is_sub_arena_out_tensor ? vt_tensor : _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, vt_pos));
						ref_tensor->alias_ref = is_sub_arena_out_tensor ? (uintptr_t)vt_tensor : (uintptr_t)vt_pos;
						ccv_nnc_tensor_synchronize_to_multiview(multiview, (ccv_nnc_tensor_t*)(intptr_t)ref_pos);
						ccv_nnc_tensor_t* tv = (ccv_nnc_tensor_t*)(CCV_NNC_IS_METADATA_POS(CCV_NNC_MULTIVIEW_DATA(multiview)[0]) ? _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA(multiview)[0]) : CCV_NNC_MULTIVIEW_DATA(multiview)[0]);
						while (CCV_IS_TENSOR_MULTIVIEW(tv))
							tv = (ccv_nnc_tensor_t*)(CCV_NNC_IS_METADATA_POS(CCV_NNC_MULTIVIEW_DATA((ccv_nnc_tensor_multiview_t*)tv)[0]) ? _ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, (int)(intptr_t)CCV_NNC_MULTIVIEW_DATA((ccv_nnc_tensor_multiview_t*)tv)[0]) : CCV_NNC_MULTIVIEW_DATA((ccv_nnc_tensor_multiview_t*)tv)[0]);
						*ref_tensor = ccv_nnc_tensor(tv->data.ptr, tv->info, 0);
						_ccv_nnc_tensor_multiview_full_pos((ccv_nnc_tensor_multiview_t*)sub_tensor, (ccv_nnc_tensor_t*)(intptr_t)ref_pos);
					} else
						_ccv_nnc_tensor_multiview_full_pos((ccv_nnc_tensor_multiview_t*)sub_tensor, is_sub_arena_out_tensor ? vt_tensor : (ccv_nnc_tensor_t*)(intptr_t)vt_pos);
				}
			}
		}
	// After alias created, for case..of statement, we now revert back to flat tensor rather than multi-view.
	// No worries though, this new tensor is subscribed for the phi multi-view. More over, we have logic
	// when initialize case..of node, which will take the phi multi-view again.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_blocks[i].bypass_ref && tensor_arena->vt_tensors[i])
		{
			assert(CCV_NNC_IS_METADATA_POS(tensor_arena->vt_tensors[i]));
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, (int)(intptr_t)tensor_arena->vt_tensors[i]);
			assert(mv->anchor == CCV_NNC_MULTIVIEW_PHI);
			tensor_arena->vt_tensors[i] = (ccv_nnc_tensor_t*)(intptr_t)_ccv_nnc_tensor_flat_if_multiview(tensor_arena->tensor_metadata, (int)(intptr_t)tensor_arena->vt_tensors[i]);
		}
	// rewire the rest. I can rewire multiple times because I can identify whether this is wired or not.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_arena->vt_tensors[i])
			tensor_arena->vt_tensors[i] = _ccv_nnc_tensor_metadata_rewire(tensor_arena->tensor_metadata, tensor_arena->vt_tensors[i]);
	// Associate multiview tensors from sub arena to the parent.
	if (sub_arena_out_tensors)
	{
		for (i = 0; i < alloc_prep->block_size; i++)
			if (alloc_prep->blocks[i].block_ref < tensor_symbol_info_size)
			{
				const int block_ref = alloc_prep->blocks[i].block_ref;
				if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[block_ref]))
					continue;
				int sub_arena_ref = block_ref;
				if (TENSOR_EXPECT_ALIAS(tensor_blocks[block_ref]))
				{
					// Assigning out the tensor aliases.
					assert(tensor_symbol_info[block_ref].alias_ref);
					const int alias_ref = tensor_symbol_info[block_ref].alias_ref - 1;
					// It referenced to is not an alias.
					assert(tensor_arena->vt_tensors[alias_ref]);
					sub_arena_ref = alias_ref;
					if (!sub_arena_out_tensors[sub_arena_ref])
						continue;
				}
				if (!sub_arena_out_tensors[sub_arena_ref])
					continue;
				ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)(CCV_IS_TENSOR_MULTIVIEW(sub_arena_out_tensors[sub_arena_ref]) ? sub_arena_out_tensors[sub_arena_ref] : (ccv_nnc_tensor_t*)sub_arena_out_tensors[sub_arena_ref]->alias_ref);
				assert(CCV_IS_TENSOR_MULTIVIEW(mv));
				// This is only possible if the vt_tensors is a phi node.
				if (tensor_arena->vt_tensors[block_ref]->alias_ref)
				{
					// For phi node, the sub_arena_out_tensors are only relevant to its selected output. Therefore, setting that to be the receiver of the broadcast.
					ccv_nnc_tensor_multiview_t* const phi = (ccv_nnc_tensor_multiview_t*)(tensor_arena->vt_tensors[block_ref]->alias_ref);
					assert(phi->anchor == CCV_NNC_MULTIVIEW_PHI);
					assert(!CCV_IS_TENSOR_MULTIVIEW(CCV_NNC_MULTIVIEW_DATA(phi)[1]));
					CCV_NNC_MULTIVIEW_DATA(phi)[1]->alias_ref = (uintptr_t)mv;
					ccv_nnc_tensor_synchronize_to_multiview(mv, CCV_NNC_MULTIVIEW_DATA(phi)[1]);
				} else {
					tensor_arena->vt_tensors[block_ref]->alias_ref = (uintptr_t)mv;
					ccv_nnc_tensor_synchronize_to_multiview(mv, tensor_arena->vt_tensors[block_ref]);
				}
			}
	}
	// Go over all the tensors that has assign_ref. If the tensor it is assigned from is:
	// 1). From sub_arena_out_tensors, it could be possible that it now pointing to an area this arena doesn't know.
	// 2). From phi multi-view, for this case, it is in fact that this arena won't know which memory I am going to use prior.
	// Therefore, for above two scenarios, the tensor has assign_ref, even it is a multiview tensor, need to subscribe
	// to the output of assign_ref tensor.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_arena->vt_tensors[i] && tensor_symbol_info[i].assign_ref)
		{
			const int assign_ref = tensor_symbol_info[i].assign_ref - 1;
			ccv_nnc_tensor_t* assign_tensor;
			if (sub_arena_out_tensors && sub_arena_out_tensors[assign_ref])
				assign_tensor = CCV_IS_TENSOR_MULTIVIEW(sub_arena_out_tensors[assign_ref]) ? sub_arena_out_tensors[assign_ref] : (ccv_nnc_tensor_t*)sub_arena_out_tensors[assign_ref]->alias_ref;
			else
				assign_tensor = tensor_arena->vt_tensors[assign_ref];
			ccv_nnc_graph_add_carry_over(graph_prep->graph, assign_tensor, tensor_arena->vt_tensors[i]);
		}
	if (sub_arena_out_tensors)
		ccfree(sub_arena_out_tensors);
	// Rewire sub arena's tensor references.
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (tensor_arena->sub_arenas[i])
		{
			const int exec_idx = graph_prep->sub_preps[i]->exec_idx - 1;
			const ccv_nnc_graph_exec_symbol_info_t* const node = graph_prep->exec_symbol_info + exec_idx;
			for (j = 0; j < node->input_size; j++)
			{
				const int idx = node->inputs[j];
				const int s_idx = (tensor_symbol_info[idx].s_ref && tensor_symbol_info[idx].s_ref->rnum > i) ? *(int*)ccv_array_get(tensor_symbol_info[idx].s_ref, i) - 1 : -1;
				if (s_idx < 0)
					continue;
				ccv_nnc_tensor_t* sub_tensor = tensor_arena->sub_arenas[i]->vt_tensors[s_idx];
				// Only do the replacement if it is a multi-view tensor.
				// sub_tensor can be unassigned if it is a tape variable. It will get fixed up later from its peer.
				if (sub_tensor && CCV_IS_TENSOR_MULTIVIEW(sub_tensor))
				{
					// This is binded tensor, bind it now.
					if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[idx]))
						_ccv_nnc_tensor_multiview_full_pos((ccv_nnc_tensor_multiview_t*)sub_tensor, tensor_arena->vt_tensors[idx]);
					else
						_ccv_nnc_tensor_multiview_full_pos_rewire(tensor_arena->tensor_metadata, (ccv_nnc_tensor_multiview_t*)sub_tensor);
				}
			}
		}
	return tensor_arena;
}

static ccv_nnc_tensor_t* _ccv_nnc_tensor_arena_find_peer_ref(const ccv_nnc_tensor_arena_t* const tensor_arena, const ccv_nnc_symbolic_graph_t* const graph, const int peer_ref)
{
	assert(graph);
	if ((intptr_t)graph == tensor_arena->graph_ref)
	{
		assert(peer_ref >= 0 && peer_ref < tensor_arena->vt_tensor_size);
		return tensor_arena->vt_tensors[peer_ref];
	}
	int i;
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (tensor_arena->sub_arenas[i])
		{
			ccv_nnc_tensor_t* const tensor = _ccv_nnc_tensor_arena_find_peer_ref(tensor_arena->sub_arenas[i], graph, peer_ref);
			if (tensor)
				return tensor;
		}
	return 0;
}

static void _ccv_nnc_tensor_mark_as_tape_var(ccv_nnc_tensor_t* const tensor)
{
	if (!CCV_IS_TENSOR_MULTIVIEW(tensor))
		tensor->type |= CCV_TAPE_ALLOC;
	else {
		ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)tensor;
		mv->type |= CCV_TAPE_ALLOC;
		int i;
		for (i = 0; i < mv->repeat + mv->kind; i++)
			_ccv_nnc_tensor_mark_as_tape_var(CCV_NNC_MULTIVIEW_DATA(mv)[i]);
	}
}

static void _ccv_nnc_tensor_arena_fixup_peer_ref_and_tape_var(const ccv_nnc_tensor_arena_t* const root_arena, const ccv_nnc_symbolic_graph_prep_t* const graph_prep, ccv_nnc_tensor_arena_t* const tensor_arena)
{
	assert(tensor_arena->graph_ref == (intptr_t)graph_prep->symbolic_graph);
	int i;
	for (i = 0; i < graph_prep->tensor_symbol_info_size; i++)
	{
		if (graph_prep->tensor_symbol_info[i].peer_ref)
		{
			tensor_arena->vt_tensors[i] = _ccv_nnc_tensor_arena_find_peer_ref(root_arena, graph_prep->symbolic_graph->peer, graph_prep->tensor_symbol_info[i].peer_ref - 1);
			// No need to continue check this if it is from its peer.
			continue;
		}
		if ((graph_prep->tensor_symbol_info[i].flags & CCV_NNC_TENSOR_SYMBOL_TAPE_VAR) && tensor_arena->vt_tensors[i])
		{
			// If it is a normal tensor, and the buffer it relies on is read only, no need to mark as tape var.
			if (!CCV_IS_TENSOR_MULTIVIEW(tensor_arena->vt_tensors[i]))
			{
				const int vt_ref = graph_prep->alloc_prep->vt_blocks[i];
				if (vt_ref >= 0 &&
					TENSOR_READ_WRITE(graph_prep->alloc_prep->buffers[graph_prep->alloc_prep->blocks[vt_ref].buffer_ref]) == READ_ONLY)
					continue;
			}
			_ccv_nnc_tensor_mark_as_tape_var(tensor_arena->vt_tensors[i]);
		}
	}
	for (i = 0; i < graph_prep->sub_prep_size; i++)
		if (graph_prep->sub_preps[i])
			_ccv_nnc_tensor_arena_fixup_peer_ref_and_tape_var(root_arena, graph_prep->sub_preps[i], tensor_arena->sub_arenas[i]);
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
		ccv_nnc_tensor_t* tensor = tensor_arena->vt_tensors[symbol.d];
		if (tensor && CCV_IS_TENSOR_MULTIVIEW(tensor))
		{
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
			while (CCV_IS_TENSOR_MULTIVIEW(mv))
				mv = (ccv_nnc_tensor_multiview_t*)(mv->it ? mv->it : CCV_NNC_MULTIVIEW_DATA(mv)[0]);
			return (ccv_nnc_tensor_t*)mv;
		}
		return tensor;
	}
	int i;
	for (i = 0; i < tensor_arena->sub_arena_size; i++)
		if (tensor_arena->sub_arenas[i])
		{
			ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_from_symbol(tensor_arena->sub_arenas[i], symbol);
			if (tensor)
				return tensor;
		}
	return 0;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_from_symbol(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena, const ccv_nnc_graph_exec_symbol_t symbol)
{
	if ((intptr_t)symbol.graph == graph_exec_arena->graph_ref)
	{
		assert(symbol.d >= 0 && symbol.d < graph_exec_arena->graph_exec_size);
		return graph_exec_arena->graph_execs[symbol.d];
	}
	int i;
	for (i = 0; i < graph_exec_arena->sub_arena_size; i++)
		if (graph_exec_arena->sub_arenas[i])
		{
			ccv_nnc_graph_exec_t exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena->sub_arenas[i], symbol);
			if (!CCV_NO_GRAPH_EXEC(exec))
				return exec;
		}
	return (ccv_nnc_graph_exec_t){}; // 0.
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_source(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	return graph_exec_arena->source;
}

ccv_nnc_graph_exec_t ccv_nnc_graph_exec_destination(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	return graph_exec_arena->destination;
}

// Check whether the head is the beginning of this block.
static int _ccv_nnc_tensor_block_check_head(const ccv_nnc_tensor_block_t* const tensor_block, const int head_node)
{
	assert(tensor_block->head);
	return (tensor_block->head->rnum == 1 && *(int*)ccv_array_get(tensor_block->head, 0) == head_node);
}

// Check whether the tail is the end of this block.
static int _ccv_nnc_tensor_block_check_tail(const ccv_nnc_tensor_block_t* const tensor_block, const int tail_node)
{
	assert(tensor_block->tail);
	return (tensor_block->tail->rnum == 1 && *(int*)ccv_array_get(tensor_block->tail, 0) == tail_node);
}

// Make two tensor blocks one. Return 1 if that happened.
static int _ccv_nnc_tensor_blocks_try_fold(ccv_nnc_tensor_block_t* const tensor_blocks, const int p_ref_0, const int p_ref_1)
{
	// Now we are sure p_ref_0 points to the input, p_ref_1 points to the output.
	if (!TENSOR_IS_UNFOLDABLE_AS_INPUT(tensor_blocks[p_ref_0]) &&
		(!TENSOR_IS_UNFOLDABLE_AS_OUTPUT(tensor_blocks[p_ref_1]) || tensor_blocks[p_ref_1].unfoldable_except_ref == p_ref_0 + 1) &&
		tensor_blocks[p_ref_0].tail->rnum == 1 &&
		tensor_blocks[p_ref_1].head->rnum == 1 &&
		tensor_blocks[p_ref_0].type == tensor_blocks[p_ref_1].type && // Must be the same type.
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
		if (TENSOR_IS_UNFOLDABLE_AS_INPUT(tensor_blocks[p_ref_1]))
			TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[p_ref_0]);
		// Don't need to check UNFOLDABLE_AS_OUTPUT for p_ref_1 because if it is so, we cannot fold right now.
		TENSOR_EXPECT_SET_UNASSIGNED(tensor_blocks[p_ref_1]);
		tensor_blocks[p_ref_1].ref = p_ref_0 + 1;
		if (!tensor_blocks[p_ref_0].r_refs)
			tensor_blocks[p_ref_0].r_refs = ccv_array_new(sizeof(int), 0, 0);
		ccv_array_add_unique_int(tensor_blocks[p_ref_0].r_refs, p_ref_1 + 1);
		tensor_blocks[p_ref_1].size = 0;
		tensor_blocks[p_ref_1].head = 0;
		tensor_blocks[p_ref_1].tail = 0;
		return 1;
	}
	return 0;
}

static void _ccv_nnc_exec_dep_and_tensor_blocks_prep(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_info_t* const p_node_info, const ccv_nnc_graph_visit_t* const visit, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const int unroll_count, const int* const dup_tensor_block_ref, const int* const dup_tensor_from_ref, const int* const dup_exec_from_ref, ccv_nnc_graph_exec_flag_t* const exec_flags, ccv_sparse_matrix_t** r_exec_dep, ccv_nnc_tensor_block_t** r_tensor_blocks)
{
	int i, j, k;
	// Generate exec dependencies (or, in other words, partial ordering of executions).
	ccv_sparse_matrix_t* exec_dep = ccv_sparse_matrix_new(symbolic_graph->exec_symbol_info->rnum, symbolic_graph->exec_symbol_info->rnum, CCV_32S | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	int* buf = (int*)ccmalloc(sizeof(int) * symbolic_graph->exec_symbol_info->rnum * 2);
	int buf_size;
	if (p_node_info)
		{ assert(output_size == 0); }
#define for_block(x, val) \
	do { \
		if (((int32_t*)val)[0] > 0) \
		{ \
			buf[buf_size * 2] = x; \
			buf[buf_size * 2 + 1] = ((int32_t*)val)[0] + 1; \
			++buf_size; \
		} \
	} while (0)
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx, _, term) {
		buf_size = 0; /* save all its parent deps to this buffer */
		ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx);
		if (vector)
			CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
		if (!node->outgoings)
			continue;
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			const int32_t one = 1;
			ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, outgoing, idx);
			/* If not found, set, if the current node is the destination node, no need 
			 * set itself as parent of subsequent nodes because its terminal nature. */
			if (!term && (!cell.i32 || cell.i32[0] == 0))
				ccv_set_sparse_matrix_cell(exec_dep, outgoing, idx, &one);
			for (j = 0; j < buf_size; j++) /* set with all idx's dependencies as well */
			{
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, outgoing, buf[j * 2]);
				/* If not found, set */
				if (!cell.i32 || cell.i32[0] == 0)
					ccv_set_sparse_matrix_cell(exec_dep, outgoing, buf[j * 2], &buf[j * 2 + 1]);
				else {
					/* Otherwise, set to the longest one */
					int32_t dep = ccv_max(cell.i32[0], buf[j * 2 + 1]);
					ccv_set_sparse_matrix_cell(exec_dep, outgoing, buf[j * 2], &dep);
				}
			}
		}
	} ccv_nnc_graph_visit_endfor
#undef for_block
	ccfree(buf);
	// This struct is allocated earlier to collect information about the tensor's expected start / end execs.
	const int tensor_block_size = symbolic_graph->tensor_symbol_info->rnum;
	ccv_nnc_tensor_block_t* tensor_blocks = (ccv_nnc_tensor_block_t*)cccalloc(tensor_block_size, sizeof(ccv_nnc_tensor_block_t));
	// The reason is that I need to make everyone of them to be unassigned unless it is used somewhere. It
	// happens that I have to loop through all relevant node to find out if one is used or not.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		tensor_blocks[i].flags = UNASSIGNED, tensor_blocks[i].type = tensor_symbol_info[i].info.type, tensor_blocks[i].bypass_ref = tensor_symbol_info[i].bypass_ref;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		for (i = 0; i < node->input_size; i++)
			if (node->inputs[i] >= 0)
				tensor_blocks[node->inputs[i]].flags = 0;
		for (i = 0; i < node->output_size; i++)
			if (node->outputs[i] >= 0)
				tensor_blocks[node->outputs[i]].flags = 0;
	} ccv_nnc_graph_visit_endfor
	if (p_node_info)
	{
		assert(p_tensor_symbol_info);
		// Mark it as used if it is used in either input or output.
		for (i = 0; i < p_node_info->input_size; i++)
			if (p_node_info->inputs[i] >= 0)
			{
				const int d = p_node_info->inputs[i];
				if (p_tensor_symbol_info[d].s_ref && p_tensor_symbol_info[d].s_ref->rnum >= symbolic_graph->p_idx)
				{
					const int dd = *(int*)ccv_array_get(p_tensor_symbol_info[d].s_ref, symbolic_graph->p_idx - 1) - 1;
					if (dd >= 0) // If this exists in this sub-graph, great.
						tensor_blocks[dd].flags = 0;
				}
			}
		for (i = 0; i < p_node_info->output_size; i++)
			if (p_node_info->outputs[i] >= 0)
			{
				const int d = p_node_info->outputs[i];
				if (p_tensor_symbol_info[d].s_ref && p_tensor_symbol_info[d].s_ref->rnum >= symbolic_graph->p_idx)
				{
					const int dd = *(int*)ccv_array_get(p_tensor_symbol_info[d].s_ref, symbolic_graph->p_idx - 1) - 1;
					if (dd >= 0) // If this exists in this sub-graph, great.
						tensor_blocks[dd].flags = 0;
				}
			}
	}
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		if (!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]))
		{
			// If this tensor is used in assign_ref, set it to be un-foldable. (It will be used as parameter,
			// therefore, itself life-cycle almost certainly won't concatenate properly with the tensor to
			// fold to).
			if (tensor_symbol_info[i].assign_ref)
			{
				// TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[i]);
				// It can be folded as input (it is fine to be overwritten), but it cannot as output (when folded as input,
				// it kept its own representation, which is not the case for output).
				TENSOR_SET_UNFOLDABLE_AS_OUTPUT(tensor_blocks[i]);
				const int assign_ref = tensor_symbol_info[i].assign_ref - 1;
				// But for where it comes from, it cannot be folded as input, because it cannot be overwritten any time.
				TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[assign_ref]);
				// It also cannot be folded as output (except i), because we need to keep its own representation.
				TENSOR_SET_UNFOLDABLE_AS_OUTPUT(tensor_blocks[assign_ref]);
				assert(tensor_blocks[assign_ref].unfoldable_except_ref == 0);
				tensor_blocks[assign_ref].unfoldable_except_ref = i + 1;
				for (j = 0; j < unroll_count; j++)
				{
					TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[dup_tensor_block_ref[i * unroll_count + j]]);
					TENSOR_SET_UNFOLDABLE_AS_OUTPUT(tensor_blocks[dup_tensor_block_ref[i * unroll_count + j]]);
				}
				if (tensor_blocks[assign_ref].bypass_ref)
				{
					// If it contains a bypass_ref, that means we can fold into both the bypass and except_ref, making it untenable.
					tensor_blocks[assign_ref].unfoldable_except_ref = 0;
					const int bypass_ref = tensor_blocks[assign_ref].bypass_ref - 1;
					TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[bypass_ref]);
					TENSOR_SET_UNFOLDABLE_AS_OUTPUT(tensor_blocks[bypass_ref]);
					// On the other hand, it can be folded into the except_ref for the bypass_ref.
					tensor_blocks[bypass_ref].unfoldable_except_ref = i + 1;
					if (dup_tensor_from_ref)
					{
						const int bypass_from_ref = dup_tensor_from_ref[bypass_ref];
						if (bypass_from_ref >= 0)
						{
							TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[bypass_from_ref]);
							TENSOR_SET_UNFOLDABLE_AS_OUTPUT(tensor_blocks[bypass_from_ref]);
							assert(dup_tensor_block_ref[bypass_from_ref * unroll_count + unroll_count - 1] == bypass_ref);
							for (j = 0; j < unroll_count - 1; j++)
							{
								// Mark every incarnation as unfold-able.
								TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[dup_tensor_block_ref[bypass_from_ref * unroll_count + j]]);
								TENSOR_SET_UNFOLDABLE_AS_OUTPUT(tensor_blocks[dup_tensor_block_ref[bypass_from_ref * unroll_count + j]]);
							}
						}
					}
				}
			}
		}
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		// If it has a peer reference, we don't need to allocate this tensor at all,
		// set it to be unassigned.
		if (tensor_symbol_info[i].peer_ref)
			TENSOR_EXPECT_SET_UNASSIGNED(tensor_blocks[i]);
		// If it is a tape variable, set it to be un-foldable as too (otherwise we cannot use tape properly).
		else if (tensor_symbol_info[i].flags & CCV_NNC_TENSOR_SYMBOL_TAPE_VAR) {
			TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[i]);
			TENSOR_SET_UNFOLDABLE_AS_OUTPUT(tensor_blocks[i]);
			// For this case, there is no exception.
			tensor_blocks[i].unfoldable_except_ref = 0;
		} else if (tensor_symbol_info[i].p_ref) {
			const int p_ref_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(tensor_symbol_info[i].p_ref - 1, p_node_info);
			// If I am a case of graph, and this tensor is the input from the parent graph, you cannot fold it as input.
			if (p_node_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
				// TODO: This check can be lifted if we can fold in the parent graph.
				if (-1 == p_ref_is_in_or_out)
					TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[i]);
			if (1 == p_ref_is_in_or_out) // If p_ref is out, it cannot be fold as input.
				TENSOR_SET_UNFOLDABLE_AS_INPUT(tensor_blocks[i]);
		}
	}
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		// Check no tensor info is auto now.
		assert(!ccv_nnc_is_tensor_auto(tensor_symbol_info[i].info));
		if (tensor_symbol_info[i].alias_ref)
		{
			const int ref = tensor_symbol_info[i].alias_ref - 1;
			// If the referenced one is unassigned, mark this as assigned only if current one is assigned.
			if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[ref]) && !TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]))
				tensor_blocks[ref].flags = 0;
			// An alias cannot ref to another alias.
			assert(!tensor_symbol_info[ref].alias_ref);
			tensor_blocks[i].flags = ALIAS;
			tensor_blocks[i].ref = ref + 1; // Assign the ref.
			if (!tensor_blocks[ref].r_refs)
				tensor_blocks[ref].r_refs = ccv_array_new(sizeof(int), 0, 0);
			ccv_array_add_unique_int(tensor_blocks[ref].r_refs, i + 1);
		}
	}
	// Scan again and if the ref is not assigned, mark the alias not assigned.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		if (TENSOR_EXPECT_ALIAS(tensor_blocks[i]))
		{
			const int ref = tensor_blocks[i].ref - 1;
			if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[ref]))
			{
				// Mark this as unassigned.
				tensor_blocks[i].flags = UNASSIGNED;
				tensor_blocks[i].ref = 0;
			}
		}
	// Ignore tensors that are already binded, no matter if it is used or not.
	for (i = 0; i < tensor_bind_size; i++)
	{
		const ccv_nnc_tensor_symbol_t resolved_symbol = ccv_nnc_tensor_symbol_resolve(symbolic_graph, tensor_binds[i].symbol);
		// If there is a tensor binded, then it is unassigned.
		if (resolved_symbol.d >= 0)
		{
			// Doesn't work if this is a loop carrying variable.
			assert(!tensor_symbol_info[resolved_symbol.d].assign_ref);
			tensor_blocks[resolved_symbol.d].flags = UNASSIGNED;
			tensor_blocks[resolved_symbol.d].ref = 0; // No need to have ref as well.
		}
	}
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		// Check no tensor info is auto now.
		assert(!ccv_nnc_is_tensor_auto(tensor_symbol_info[i].info));
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
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		for (i = 0; i < node->input_size; i++)
		{
			int d = node->inputs[i];
			if (d < 0)
				continue;
			if (TENSOR_EXPECT_ALIAS(tensor_blocks[d]))
				d = tensor_symbol_info[d].alias_ref - 1;
			tensor_blocks[d].flags |= READ_ONLY;
			if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[d]))
				continue;
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[d]));
			/* If this is first encounter, its head starts (this tensor is init'ed outside of the graph
			 * from the very beginning of the graph life-cycle and ends here. */
			if (tensor_blocks[d].head->rnum == 0 && !(tensor_symbol_info[d].flags & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS))
			{
				for (j = 0; j < source_size; j++)
				{
					// If the source is connecting to current node, add (otherwise we will create tensor blocks that used in other streams, which is unneccessary).
					const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, idx, sources[j].d);
					if (cell.i32 && cell.i32[0] > 0)
						_ccv_nnc_tensor_block_add_exec(exec_dep, sources[j].d, tensor_blocks[d]);
				}
				/* If this is a read-only (based on SSA, if first encountered as read), and this is
				 * sub-graph (TODO: this condition can be lifted for case..of that is never in a while
				 * loop, however, in that case, you need to prevent read-only gets reused for the
				 * output tensor, which is not obvious how to implement correctly), and it is not
				 * assign_ref from anywhere (not a parameterized loop). We cannot reuse this region
				 * of memory anyway (because on second loop, we want to read the same value out).
				 * Mark it to the end of the graph. */
				if (p_node_info && !tensor_symbol_info[d].assign_ref)
					for (j = 0; j < destination_size; j++)
					{
						// If the destination is connecting to current node, add (otherwise we will create tensor blocks that used in other streams, which is unneccessary).
						const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, destinations[j].d, idx);
						if (cell.i32 && cell.i32[0] > 0)
							_ccv_nnc_tensor_block_add_exec(exec_dep, destinations[j].d, tensor_blocks[d]);
					}
			}
			_ccv_nnc_tensor_block_add_exec(exec_dep, idx, tensor_blocks[d]);
		}
		for (i = 0; i < node->output_size; i++)
		{
			int d = node->outputs[i];
			if (d < 0)
				continue;
			if (TENSOR_EXPECT_ALIAS(tensor_blocks[d]))
				d = tensor_symbol_info[d].alias_ref - 1;
			tensor_blocks[d].flags |= WRITE_ONLY;
			if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[d]))
				continue;
			assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[d]));
			_ccv_nnc_tensor_block_add_exec(exec_dep, idx, tensor_blocks[d]);
		}
	} ccv_nnc_graph_visit_endfor
	// For any assign_ref, its life-time kept until the end and wrap over.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		// If this tensor is not unassigned (or alias) and it is assigned from somewhere else,
		// that "somewhere else" need to keep its life-time til the end.
		if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]) &&
			p_node_info && tensor_symbol_info[i].assign_ref)
		{
			const int assign_ref = tensor_symbol_info[i].assign_ref - 1;
			for (j = 0; j < destination_size; j++)
			{
				// This logic is to be more conservative about which destination we add to.
				// As of now, if we add everything, it is fine most likely. However, it may
				// cause issues in the future to do so naively. Thus, instead, we only add
				// the destination to it iff either the tensor is not used at all, or, the
				// destination is on the same stream as of the tensor block some way.
				int flag = !tensor_blocks[assign_ref].tail;
				for (k = 0; !flag && k < tensor_blocks[assign_ref].tail->rnum; k++)
				{
					const int idx = *(int*)ccv_array_get(tensor_blocks[assign_ref].tail, k);
					const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, destinations[j].d, idx);
					flag = (cell.i32 && cell.i32[0] > 0);
				}
				if (flag) // If there is no tail at all, add it. Otherwise, only add it if the destination is on the same stream with this tensor block somehow.
					_ccv_nnc_tensor_block_add_exec(exec_dep, destinations[j].d, tensor_blocks[assign_ref]);
			}
		}
	for (i = 0; i < output_size; i++)
	{
		assert(outputs[i].graph == symbolic_graph);
		int d = outputs[i].d;
		if (d < 0)
			continue;
		if (TENSOR_EXPECT_ALIAS(tensor_blocks[d]))
			d = tensor_symbol_info[d].alias_ref - 1;
		if (TENSOR_EXPECT_UNASSIGNED(tensor_blocks[d]))
			continue;
		assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[d]));
		for (j = 0; j < destination_size; j++)
		{
			int flag = !tensor_blocks[d].tail;
			for (k = 0; !flag && k < tensor_blocks[d].tail->rnum; k++)
			{
				const int idx = *(int*)ccv_array_get(tensor_blocks[d].tail, k);
				const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, destinations[j].d, idx);
				flag = (cell.i32 && cell.i32[0] > 0);
			}
			if (flag) // If there is no tail at all, add it. Otherwise, only add it if the destination is on the same stream with this tensor block somehow.
				_ccv_nnc_tensor_block_add_exec(exec_dep, destinations[j].d, tensor_blocks[d]);
		}
	}
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		/* Maximum tensor reuse by collapse tensors allows in-place operations (and it matches the start, end tensor). */
		int x, y;
		for (x = 0; x < node->input_size; x++)
			for (y = 0; y < node->output_size; y++)
				/* Some operations enforces some tensors to be the same for inputs / outputs. */
				if (ccv_nnc_cmd_enforce_inplace(node->cmd, x, y))
				{
					// If both unassigned, it is fine.
					if (node->inputs[x] < 0 && node->outputs[y] < 0)
						continue;
					int ref = node->inputs[x];
					assert(ref >= 0);
					while (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) && tensor_blocks[ref].ref)
						ref = tensor_blocks[ref].ref - 1;
					const int node_output_y = node->outputs[y];
					assert(node_output_y >= 0);
					// If both are not computable, it is fine, we don't need to enforce.
					if (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) &&
						!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[node_output_y]))
						continue;
					// Otherwise, enforce and error out if failed.
					if (!_ccv_nnc_tensor_blocks_try_fold(tensor_blocks, ref, node_output_y))
						{ assert(0 && "cannot enforce inplace for the two tensors"); }
				}
		for (x = 0; x < node->input_size; x++)
		{
			/* If the input is not assigned, it can be referenced, find the referenced one */
			int ref = node->inputs[x];
			if (ref < 0)
				continue;
			while (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) && tensor_blocks[ref].ref)
				ref = tensor_blocks[ref].ref - 1;
			assert(tensor_blocks[ref].ref == 0);
			const ccv_nnc_tensor_symbol_info_t x_symbol = tensor_symbol_info[ref];
			if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[ref]) &&
				tensor_blocks[ref].tail->rnum == 1)
				for (y = 0; y < node->output_size; y++)
					/* Only proceed if the input symbol is different from the output symbol, */
					/* and the input symbol meets the output symbol exactly at the same spot. */
					if (ccv_nnc_cmd_allow_inplace(node->cmd, x, y) &&
						node->outputs[y] >= 0 &&
						ref != node->outputs[y] &&
						TENSOR_EXPECT_COMPUTABLE(tensor_blocks[node->outputs[y]]))
					{
						const int node_output_y = node->outputs[y];
						const ccv_nnc_tensor_symbol_info_t y_symbol = tensor_symbol_info[node_output_y];
						/* If dimension matches perfectly, then we can assign y_symbol to x. */
						if (memcmp(x_symbol.info.dim, y_symbol.info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
							_ccv_nnc_tensor_blocks_try_fold(tensor_blocks, ref, node_output_y);
					}
		}
	} ccv_nnc_graph_visit_endfor
	// Specifically handle the bypass. This need to be done after the first pass.
	// I need to extend the bypass life-time to the same as the one I am going with.
	// It is important we visit these nodes and assign bypass_ref to its dependents in topological order.
	ccv_nnc_tensor_block_t empty_block = {};
	empty_block.head = ccv_array_new(sizeof(int), 0, 0);
	empty_block.tail = ccv_array_new(sizeof(int), 0, 0);
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		if (node->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			int can_bypass = 1;
			for (i = 0; can_bypass && i < node->output_size; i++)
			{
				int d = node->outputs[i];
				if (d < 0)
					continue;
				if (!tensor_blocks[d].bypass_ref)
					continue;
				while (tensor_blocks[d].ref)
					d = tensor_blocks[d].ref - 1;
				int bypass_ref = tensor_blocks[node->outputs[i]].bypass_ref - 1;
				while (tensor_blocks[bypass_ref].ref)
					bypass_ref = tensor_blocks[bypass_ref].ref - 1;
				// If this doesn't participate in the while loop, we don't need to check the while loop constraint.
				if (!tensor_symbol_info[bypass_ref].assign_ref && !tensor_symbol_info[bypass_ref].r_assign_ref)
					continue;
				ccv_array_clear(empty_block.head);
				for (j = 0; tensor_blocks[bypass_ref].head && j < tensor_blocks[bypass_ref].head->rnum; j++)
					ccv_array_push(empty_block.head, ccv_array_get(tensor_blocks[bypass_ref].head, j));
				ccv_array_clear(empty_block.tail);
				for (j = 0; tensor_blocks[bypass_ref].tail && j < tensor_blocks[bypass_ref].tail->rnum; j++)
					ccv_array_push(empty_block.tail, ccv_array_get(tensor_blocks[bypass_ref].tail, j));
				for (j = 0; tensor_blocks[d].head && j < tensor_blocks[d].head->rnum; j++)
					_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[d].head, j), empty_block);
				for (j = 0; tensor_blocks[d].tail && j < tensor_blocks[d].tail->rnum; j++)
					_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[d].tail, j), empty_block);
				// It can only be unfoldable due to while constraint. Check whether this satisfies the while loop constraint.
				assert(!(tensor_symbol_info[bypass_ref].assign_ref && tensor_symbol_info[bypass_ref].r_assign_ref));
				int b_ref = (tensor_symbol_info[bypass_ref].assign_ref) ? tensor_symbol_info[bypass_ref].assign_ref - 1 : tensor_symbol_info[bypass_ref].r_assign_ref - 1;
				while (tensor_blocks[b_ref].ref)
					b_ref = tensor_blocks[b_ref].ref - 1;
				int a_hop_b = _ccv_nnc_tensor_block_head_after_tail(exec_dep, empty_block, tensor_blocks[b_ref]);
				int b_hop_a = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[b_ref], empty_block);
				// These two can be assigned to the same region of memory without issue (because their life-time doesn't interfere)
				// even after we extend the life-time of bypass_ref. Then we are in a good shape.
				can_bypass = can_bypass && (a_hop_b || b_hop_a);
			}
			if (can_bypass)
			{
				for (i = 0; i < node->output_size; i++)
				{
					int d = node->outputs[i];
					if (d < 0)
						continue;
					if (!tensor_blocks[d].bypass_ref)
						continue;
					while (tensor_blocks[d].ref)
						d = tensor_blocks[d].ref - 1;
					int bypass_ref = tensor_blocks[node->outputs[i]].bypass_ref - 1;
					while (tensor_blocks[bypass_ref].ref)
						bypass_ref = tensor_blocks[bypass_ref].ref - 1;
					// The bypass_ref can extend its life-time.
					for (j = 0; tensor_blocks[d].head && j < tensor_blocks[d].head->rnum; j++)
						_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[d].head, j), tensor_blocks[bypass_ref]);
					for (j = 0; tensor_blocks[d].tail && j < tensor_blocks[d].tail->rnum; j++)
						_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[d].tail, j), tensor_blocks[bypass_ref]);
				}
			} else {
				for (i = 0; i < node->output_size; i++)
					tensor_blocks[node->outputs[i]].bypass_ref = 0;
				const int exec_idx = (dup_exec_from_ref) ? dup_exec_from_ref[idx] : idx;
				// Mark this exec as no bypass IO (thus, I need to insert explicit data transfer.
				exec_flags[exec_idx].flags |= CCV_NNC_GRAPH_EXEC_ATTR_CASE_OF_NO_BYPASS_IO;
			}
		}
	} ccv_nnc_graph_visit_endfor
	ccv_array_free(empty_block.head);
	ccv_array_free(empty_block.tail);
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

static ccv_nnc_tensor_symbol_t _ccv_nnc_dup_tensor_symbol(ccv_nnc_symbolic_graph_t* const dup_graph, const int unroll_count, int* const dup_tensor_block_ref, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const int input)
{
	if (dup_tensor_block_ref[input * unroll_count] < 0) // No tensor ref, create one.
	{
		if (tensor_symbol_info[input].alias_ref)
		{
			const int alias_ref = tensor_symbol_info[input].alias_ref - 1;
			assert(tensor_symbol_info[alias_ref].alias_ref == 0);
			ccv_nnc_tensor_symbol_t tensor_symbol = {};
			if (dup_tensor_block_ref[alias_ref * unroll_count] < 0)
			{
				tensor_symbol = ccv_nnc_tensor_symbol_new(dup_graph, tensor_symbol_info[alias_ref].info, 0);
				if (tensor_symbol_info[alias_ref].peer_ref)
					ccv_nnc_tensor_symbol_set_peer(dup_graph, tensor_symbol, (ccv_nnc_tensor_symbol_t){
						.d = tensor_symbol_info[alias_ref].peer_ref - 1,
						.graph = dup_graph->peer
					});
				ccv_nnc_tensor_symbol_set_flags(dup_graph, tensor_symbol, tensor_symbol_info[alias_ref].flags);
				dup_tensor_block_ref[alias_ref * unroll_count] = tensor_symbol.d;
			} else {
				tensor_symbol.d = dup_tensor_block_ref[alias_ref * unroll_count];
				tensor_symbol.graph = dup_graph;
			}
			ccv_nnc_tensor_symbol_t alias_symbol = ccv_nnc_tensor_symbol_alias_new(dup_graph, tensor_symbol, tensor_symbol_info[input].ofs, tensor_symbol_info[input].inc, tensor_symbol_info[input].info, 0);
			if (tensor_symbol_info[input].peer_ref)
				ccv_nnc_tensor_symbol_set_peer(dup_graph, alias_symbol, (ccv_nnc_tensor_symbol_t){
					.d = tensor_symbol_info[input].peer_ref - 1,
					.graph = dup_graph->peer
				});
			ccv_nnc_tensor_symbol_set_flags(dup_graph, alias_symbol, tensor_symbol_info[input].flags);
			dup_tensor_block_ref[input * unroll_count] = alias_symbol.d;
		} else {
			ccv_nnc_tensor_symbol_t tensor_symbol = ccv_nnc_tensor_symbol_new(dup_graph, tensor_symbol_info[input].info, 0);
			if (tensor_symbol_info[input].peer_ref)
				ccv_nnc_tensor_symbol_set_peer(dup_graph, tensor_symbol, (ccv_nnc_tensor_symbol_t){
					.d = tensor_symbol_info[input].peer_ref - 1,
					.graph = dup_graph->peer
				});
			ccv_nnc_tensor_symbol_set_flags(dup_graph, tensor_symbol, tensor_symbol_info[input].flags);
			dup_tensor_block_ref[input * unroll_count] = tensor_symbol.d;
		}
		if (tensor_symbol_info[input].bypass_ref)
		{
			const int dup_bypass_ref = dup_tensor_block_ref[(tensor_symbol_info[input].bypass_ref - 1) * unroll_count];
			assert(dup_bypass_ref >= 0);
			ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(dup_graph->tensor_symbol_info, dup_tensor_block_ref[input * unroll_count]);
			symbol_info->bypass_ref = dup_bypass_ref + 1;
		}
	}
	return (ccv_nnc_tensor_symbol_t) {
		.d = dup_tensor_block_ref[input * unroll_count],
		.graph = dup_graph,
	};
}

static ccv_nnc_graph_exec_symbol_t _ccv_nnc_dup_graph_exec_symbol(ccv_nnc_symbolic_graph_t* const dup_graph, const int unroll_count, int* const dup_exec_ref, int* const dup_tensor_block_ref, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const ccv_nnc_graph_exec_symbol_info_t* const node, const int idx, ccv_nnc_tensor_symbol_t* const max_inputs, ccv_nnc_tensor_symbol_t* const max_outputs)
{
	int i;
	if (dup_exec_ref[idx * unroll_count] < 0)
	{
		// Input has to come before output, because output could has a bypass reference to the input.
		for (i = 0; i < node->input_size; i++)
			max_inputs[i] = (node->inputs[i] >= 0) ? _ccv_nnc_dup_tensor_symbol(dup_graph, unroll_count, dup_tensor_block_ref, tensor_symbol_info, node->inputs[i]) : (ccv_nnc_tensor_symbol_t){ .d = node->inputs[i], .graph = dup_graph };
		for (i = 0; i < node->output_size; i++)
			max_outputs[i] = (node->outputs[i] >= 0) ? _ccv_nnc_dup_tensor_symbol(dup_graph, unroll_count, dup_tensor_block_ref, tensor_symbol_info, node->outputs[i]) : (ccv_nnc_tensor_symbol_t){ .d = node->outputs[i], .graph = dup_graph };
		ccv_nnc_graph_exec_symbol_t exec_symbol = ccv_nnc_graph_exec_symbol_new(dup_graph, node->cmd, max_inputs, node->input_size, max_outputs, node->output_size, 0);
		dup_exec_ref[idx * unroll_count] = exec_symbol.d;
	}
	return (ccv_nnc_graph_exec_symbol_t) {
		.d = dup_exec_ref[idx * unroll_count],
		.graph = dup_graph,
	};
}

static void _ccv_nnc_tensor_blocks_free(ccv_nnc_tensor_block_t* const tensor_blocks, const int tensor_block_size)
{
	int i;
	for (i = 0; i < tensor_block_size; i++)
	{
		if (tensor_blocks[i].head)
			ccv_array_free(tensor_blocks[i].head);
		if (tensor_blocks[i].tail)
			ccv_array_free(tensor_blocks[i].tail);
		if (tensor_blocks[i].r_refs)
			ccv_array_free(tensor_blocks[i].r_refs);
		if (tensor_blocks[i].dup_p_refs)
			ccv_array_free(tensor_blocks[i].dup_p_refs);
	}
	ccfree(tensor_blocks);
}

// Find tensors that cannot be solved by co-allocating to the same location.
static int _ccv_nnc_exec_dep_and_tensor_blocks_unroll_count(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const ccv_sparse_matrix_t* const exec_dep, ccv_nnc_tensor_block_t* const tensor_blocks)
{
	int i, j, unroll_count = 0;
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
				// Can it be folded
				// These two can be assigned to the same region of memory without issue (because their life-time doesn't interfere).
				if (a_hop_b || b_hop_a)
				{
					tensor_blocks[a_ref].companion_ref = b_ref + 1;
					tensor_blocks[b_ref].companion_ref = a_ref + 1;
					continue;
				}
				int c_ref = tensor_symbol_info[b_ref].assign_ref - 1;
				for (j = 0; c_ref >= 0; j++)
				{
					while (tensor_blocks[c_ref].ref)
						c_ref = tensor_blocks[c_ref].ref - 1;
					c_ref = tensor_symbol_info[c_ref].assign_ref - 1;
				}
				unroll_count = ccv_max(unroll_count, j + 1);
			}
		}
	// Reset companion_ref if need to unroll.
	if (unroll_count)
		for (j = 0; j < symbolic_graph->tensor_symbol_info->rnum; j++)
			tensor_blocks[j].companion_ref = 0;
	return unroll_count;
}

static void _ccv_nnc_exec_dep_and_tensor_blocks_unroll_n(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_visit_t* const visit, const int unroll_count, const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const ccv_sparse_matrix_t* const exec_dep, const ccv_nnc_tensor_block_t* const tensor_blocks, ccv_nnc_symbolic_graph_t* const dup_graph, int* const r_dup_tensor_block_ref, int* const r_dup_exec_ref)
{
	int i, j, n;
	// The inout exec nodes, these are the nodes we are going to extend.
	uint8_t* inout = (uint8_t*)cccalloc(symbolic_graph->exec_symbol_info->rnum, sizeof(uint8_t));
	int max_input_size = 0;
	int max_output_size = 0;
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
	{
		max_input_size = ccv_max(exec_symbol_info[i].input_size, max_input_size);
		max_output_size = ccv_max(exec_symbol_info[i].output_size, max_output_size);
	}
	ccv_nnc_tensor_symbol_t max_inputs[ccv_max(1, max_input_size)];
	ccv_nnc_tensor_symbol_t max_outputs[ccv_max(1, max_output_size)];
	// Doing graph expansion
	// It goes without saying, we must have more than one tensors / execs (otherwise I cannot use 0 as no exec ref).
	assert(dup_graph->exec_symbol_info->rnum > 0);
	assert(dup_graph->tensor_symbol_info->rnum > 0);
#define INCOMING_NODE (1)
#define OUTGOING_NODE (2)
	// Unroll the graph n times.
	for (n = 0; n < unroll_count; n++)
	{
		int* const dup_exec_ref = r_dup_exec_ref + n;
		const int* const prev_dup_tensor_block_ref = n > 0 ? r_dup_tensor_block_ref + (n - 1) : 0;
		int* const dup_tensor_block_ref = r_dup_tensor_block_ref + n;
		for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
			dup_exec_ref[i * unroll_count] = -1;
		for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		{
			// If there is a assign_ref, that means I don't need to dup the tensor.
			if (tensor_symbol_info[i].assign_ref)
			{
				const int assign_ref = tensor_symbol_info[i].assign_ref - 1;
				dup_tensor_block_ref[i * unroll_count] = prev_dup_tensor_block_ref ? prev_dup_tensor_block_ref[assign_ref * unroll_count] : assign_ref;
			} else if (TENSOR_EXPECT_COMPUTABLE(tensor_blocks[i]) && TENSOR_READ_WRITE(tensor_blocks[i]) == READ_ONLY)
			// If this is a read-only tensor block, no need to duplicate because the value never changes
			// (note we handled assign_ref first), therefore, no need to generate duplicate.
				dup_tensor_block_ref[i * unroll_count] = i;
			else
				dup_tensor_block_ref[i * unroll_count] = -1;
		}
		// Go through the original graph, make copies of the node if it is inout.
		ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
			ccv_nnc_graph_exec_symbol_t exec_symbol = _ccv_nnc_dup_graph_exec_symbol(dup_graph, unroll_count, dup_exec_ref, dup_tensor_block_ref, tensor_symbol_info, node, idx, max_inputs, max_outputs);
			inout[idx] |= INCOMING_NODE; /* Mark this node as incoming. */
			if (!node->outgoings)
				continue;
			for (i = 0; i < node->outgoings->rnum; i++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, i);
				inout[outgoing_idx] |= OUTGOING_NODE; /* Mark this node as outgoing. */
				ccv_nnc_graph_exec_symbol_t outgoing_symbol = _ccv_nnc_dup_graph_exec_symbol(dup_graph, unroll_count, dup_exec_ref, dup_tensor_block_ref, tensor_symbol_info, exec_symbol_info + outgoing_idx, outgoing_idx, max_inputs, max_outputs);
				ccv_nnc_graph_exec_symbol_concat(dup_graph, exec_symbol, outgoing_symbol);
			}
		} ccv_nnc_graph_visit_endfor
		// Check the visitor are all marked as either incoming or outgoing.
		const ccv_nnc_graph_exec_symbol_t* const dup_destinations = ccv_nnc_symbolic_graph_destinations(dup_graph);
		const int dup_destination_size = ccv_nnc_symbolic_graph_destination_size(dup_graph);
		for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
		{
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info[i].flags))
				continue;
			assert((inout[i] & INCOMING_NODE) || (inout[i] & OUTGOING_NODE));
			// If this is pure incoming nodes, then I need to concat this one with all original destination node
			if (inout[i] == INCOMING_NODE)
				for (j = 0; j < dup_destination_size; j++)
				{
					ccv_nnc_graph_exec_symbol_concat(dup_graph, (ccv_nnc_graph_exec_symbol_t) {
						.d = dup_destinations[j].d,
						.graph = dup_graph,
					}, (ccv_nnc_graph_exec_symbol_t) {
						.d = dup_exec_ref[i * unroll_count],
						.graph = dup_graph,
					});
				}
		}
		if (dup_graph->destinations)
			ccv_array_clear(dup_graph->destinations);
		for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
		{
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info[i].flags))
				continue;
			const int d = dup_exec_ref[i * unroll_count];
			ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(dup_graph->exec_symbol_info, d);
			// If this has no outgoing node, add to the destination.
			if (!exec_symbol_info->outgoings || exec_symbol_info->outgoings->rnum == 0)
				ccv_nnc_symbolic_graph_add_destination(dup_graph, (ccv_nnc_graph_exec_symbol_t) {
					.graph = dup_graph,
					.d = d,
				});
		}
	}
#undef INCOMING_NODE
#undef OUTGOING_NODE
	ccfree(inout);
}

static void _ccv_nnc_fixup_assign_ref_after_unroll(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const int unroll_count, const ccv_nnc_tensor_block_t* const tensor_blocks, const int* const dup_tensor_block_ref, ccv_nnc_tensor_symbol_info_t* const dup_tensor_symbol_info)
{
	int i;
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++) // symbolic graph is the old graph and tensor blocks is the old tensor blocks.
		// Now can assign them (The dup) as companion.
		// Get to the last one, which we will wrap over.
		if (dup_tensor_symbol_info[i].assign_ref)
		{
			dup_tensor_symbol_info[dup_tensor_symbol_info[i].assign_ref - 1].r_assign_ref = 0;
			dup_tensor_symbol_info[i].assign_ref = dup_tensor_block_ref[(dup_tensor_symbol_info[i].assign_ref - 1) * unroll_count + unroll_count - 1] + 1;
			assert(dup_tensor_symbol_info[i].assign_ref);
			dup_tensor_symbol_info[dup_tensor_symbol_info[i].assign_ref - 1].r_assign_ref = i + 1;
		}
}

// If the tensor blocks are the outputs of this graph, its life-time should be extended to the end of this graph.
// However, it is not that simple if the graph is unrolled. For unrolled graph, it needs to reach the end of
// the "original" graph and all its duplicated ends (for their duplicated tensor blocks).
static void _ccv_nnc_fixup_tensor_blocks_for_outputs(ccv_sparse_matrix_t* const exec_dep, ccv_nnc_tensor_block_t* const tensor_blocks, const ccv_nnc_graph_exec_symbol_info_t* const  p_node_info, const int unroll_count, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const int p_idx, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const int* const dup_exec_ref, const int* const dup_tensor_block_ref)
{
	int i, j, k;
	for (i = 0; i < p_node_info->output_size; i++)
	{
		const int d = p_node_info->outputs[i];
		const int s_ref = *(int*)ccv_array_get(p_tensor_symbol_info[d].s_ref, p_idx) - 1;
		if (!TENSOR_EXPECT_COMPUTABLE(tensor_blocks[s_ref]))
			continue;
		for (k = 0; k < destination_size; k++)
			_ccv_nnc_tensor_block_add_exec(exec_dep, destinations[k].d, tensor_blocks[s_ref]);
		// Add the duplicated destinations to the tensor_block_ref.
		for (j = 0; j < unroll_count; j++)
			for (k = 0; k < destination_size; k++)
			{
				const int dup_exec_idx = dup_exec_ref[destinations[k].d * unroll_count + j];
				const int dup_tensor_block_idx = dup_tensor_block_ref[s_ref * unroll_count + j];
				if (dup_exec_idx >= 0 && dup_tensor_block_idx >= 0)
					_ccv_nnc_tensor_block_add_exec(exec_dep, dup_exec_idx, tensor_blocks[dup_tensor_block_idx]);
			}
	}
}

static void _ccv_nnc_redo_exec_dep_and_tensor_blocks_when_unroll(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_info_t* const p_node_info, const ccv_nnc_graph_visit_t* const visit, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_nnc_graph_exec_flag_t* const exec_flags, ccv_sparse_matrix_t** r_exec_dep, ccv_nnc_tensor_block_t** r_tensor_blocks, int* r_tensor_block_size, ccv_nnc_symbolic_graph_t** r_dup_graph, int* r_unroll_count, int** r_dup_exec_ref, int** r_dup_tensor_block_ref)
{
	int i, j;
	ccv_sparse_matrix_t* exec_dep = *r_exec_dep;
	ccv_nnc_tensor_block_t* tensor_blocks = *r_tensor_blocks;
	// blocks that cannot be simply solved with either in-place operation tensor block folding or using the same memory region.
	// Unfortunately, I cannot do this analysis to the block folding done for sub-graphs, because we do sub-graph placement later.
	// No need to change anything, we are good.
	const int unroll_count = _ccv_nnc_exec_dep_and_tensor_blocks_unroll_count(symbolic_graph, tensor_symbol_info, exec_dep, tensor_blocks);
	if (!unroll_count)
		return;
	// Have conditions that cannot be satisfied with simple solution (allocate to the same memory region).
	// Doing graph expansion, first duplicate the old graph, but replace all sub graphs with noop.
	ccv_nnc_symbolic_graph_t* dup_graph = ccv_nnc_symbolic_graph_dup(symbolic_graph, _ccv_nnc_subst_sub_graph_with_noop);
	int* dup_exec_ref = (int*)ccmalloc(sizeof(int) * symbolic_graph->exec_symbol_info->rnum * unroll_count);
	int* dup_tensor_block_ref = (int*)ccmalloc(sizeof(int) * symbolic_graph->tensor_symbol_info->rnum * unroll_count);
	_ccv_nnc_exec_dep_and_tensor_blocks_unroll_n(symbolic_graph, visit, unroll_count, exec_symbol_info, tensor_symbol_info, exec_dep, tensor_blocks, dup_graph, dup_tensor_block_ref, dup_exec_ref);
	ccv_nnc_tensor_symbol_info_t* const dup_tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * dup_graph->tensor_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* const dup_exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * dup_graph->exec_symbol_info->rnum);
	ccv_nnc_graph_visit_t* dup_visit = ccv_nnc_graph_visit_new(dup_graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(dup_graph->exec_symbol_info, 0), dup_graph->exec_symbol_info->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->sources, 0), dup_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->destinations, 0), dup_graph->destinations->rnum, 0);
	ccv_nnc_symbolic_graph_symbol_infer(dup_graph, dup_visit, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->sources, 0), dup_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->destinations, 0), dup_graph->destinations->rnum, p_tensor_symbol_info, p_tensor_symbol_info_size, dup_tensor_symbol_info, dup_exec_symbol_info);
	_ccv_nnc_fixup_assign_ref_after_unroll(symbolic_graph, unroll_count, tensor_blocks, dup_tensor_block_ref, dup_tensor_symbol_info);
	// Free out the old exec_dep
	ccv_matrix_free(exec_dep);
	// and the tensor blocks, prepare for the new.
	_ccv_nnc_tensor_blocks_free(tensor_blocks, symbolic_graph->tensor_symbol_info->rnum);
	// A reverse map to find where the original tensor comes from.
	int* dup_tensor_from_ref = (int*)ccmalloc(sizeof(int) * dup_graph->tensor_symbol_info->rnum);
	for (i = 0; i < dup_graph->tensor_symbol_info->rnum; i++)
		dup_tensor_from_ref[i] = -1;
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		for (j = 0; j < unroll_count; j++)
			if (dup_tensor_block_ref[i * unroll_count + j] >= 0)
				dup_tensor_from_ref[dup_tensor_block_ref[i * unroll_count + j]] = i;
	int* dup_exec_from_ref = (int*)ccmalloc(sizeof(int) * dup_graph->exec_symbol_info->rnum);
	for (i = 0; i < dup_graph->exec_symbol_info->rnum; i++)
		dup_exec_from_ref[i] = -1;
	for (i = 0; i < symbolic_graph->exec_symbol_info->rnum; i++)
	{
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_symbol_info[i].flags))
			continue;
		dup_exec_from_ref[i] = i; // Reference back.
		for (j = 0; j < unroll_count; j++)
			if (dup_exec_ref[i * unroll_count + j] >= 0)
				dup_exec_from_ref[dup_exec_ref[i * unroll_count + j]] = i;
	}
	// Reset all attr.
	memset(exec_flags, 0, sizeof(ccv_nnc_graph_exec_flag_t) * symbolic_graph->exec_symbol_info->rnum);
	_ccv_nnc_exec_dep_and_tensor_blocks_prep(dup_graph, p_node_info, dup_visit, tensor_binds, tensor_bind_size, 0, 0, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->sources, 0), dup_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_graph->destinations, 0), dup_graph->destinations->rnum, p_tensor_symbol_info, p_tensor_symbol_info_size, dup_exec_symbol_info, dup_tensor_symbol_info, unroll_count, dup_tensor_block_ref, dup_tensor_from_ref, dup_exec_from_ref, exec_flags, &exec_dep, &tensor_blocks);
	ccv_nnc_graph_visit_free(dup_visit);
	ccfree(dup_exec_symbol_info);
	ccfree(dup_exec_from_ref);
	ccfree(dup_tensor_from_ref);
	// Assign out dup_p_ref, which will be used to extended the anonymous block life-time.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		// Loop over all possible duplications to assign dup_p_ref properly.
		for (j = 0; j < unroll_count; j++)
		{
			const int dup_idx = dup_tensor_block_ref[j + i * unroll_count];
			if (dup_idx >= 0 && (tensor_blocks[i].p_refs[0] || tensor_blocks[i].p_refs[1]))
			{
				const int p_ref_0 = tensor_blocks[i].p_refs[0] - 1;
				const int p_ref_0_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_0, p_node_info);
				if (p_ref_0_is_in_or_out == 1) // If it is out tensor, mark dup_p_ref for this.
				{
					if (!tensor_blocks[dup_idx].dup_p_refs)
						tensor_blocks[dup_idx].dup_p_refs = ccv_array_new(sizeof(int), 1, 0);
					ccv_array_add_unique_int(tensor_blocks[dup_idx].dup_p_refs, p_ref_0);
				}
				if (p_ref_0_is_in_or_out == 1 || tensor_blocks[i].p_refs[1] == 0)
					continue;
				const int p_ref_1 = tensor_blocks[i].p_refs[1] - 1;
				const int p_ref_1_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_1, p_node_info);
				if (p_ref_1_is_in_or_out == 1)
				{
					if (!tensor_blocks[dup_idx].dup_p_refs)
						tensor_blocks[dup_idx].dup_p_refs = ccv_array_new(sizeof(int), 1, 0);
					ccv_array_add_unique_int(tensor_blocks[dup_idx].dup_p_refs, p_ref_1);
				}
			}
		}
	// companion_ref
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
		// Now can assign them (The dup) as companion.
		if (!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[i]) && dup_tensor_symbol_info[i].assign_ref)
		{
			// Get to the last one, which we will wrap over.
			const int assign_ref = dup_tensor_symbol_info[i].assign_ref - 1;
			if (assign_ref >= 0)
			{
				int b_ref = assign_ref;
				while (tensor_blocks[b_ref].ref)
					b_ref = tensor_blocks[b_ref].ref - 1;
				int a_hop_b = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[i], tensor_blocks[b_ref]);
				int b_hop_a = _ccv_nnc_tensor_block_head_after_tail(exec_dep, tensor_blocks[b_ref], tensor_blocks[i]);
				// It cannot be that both i can hop to j can j can hop to i.
				// And it can be hop from one to another now after duplication.
				assert(a_hop_b > 0 || b_hop_a > 0);
				tensor_blocks[i].companion_ref = b_ref + 1;
				tensor_blocks[b_ref].companion_ref = i + 1;
			}
		}
	ccfree(dup_tensor_symbol_info);
	// Extend the dup tensor block ref, prepare for future extensions.
	dup_tensor_block_ref = (int*)ccrealloc(dup_tensor_block_ref, sizeof(int) * dup_graph->tensor_symbol_info->rnum * unroll_count);
	for (i = symbolic_graph->tensor_symbol_info->rnum * unroll_count; i < dup_graph->tensor_symbol_info->rnum * unroll_count; i++)
		dup_tensor_block_ref[i] = -1;
	// Assign out changed properties.
	*r_exec_dep = exec_dep;
	*r_tensor_blocks = tensor_blocks;
	*r_tensor_block_size = dup_graph->tensor_symbol_info->rnum;
	*r_dup_graph = dup_graph;
	*r_unroll_count = unroll_count;
	*r_dup_exec_ref = dup_exec_ref;
	*r_dup_tensor_block_ref = dup_tensor_block_ref;
}

static int _ccv_nnc_anonymous_tensor_block_from_free_list(const ccv_nnc_tensor_block_t* const tensor_blocks, const int tensor_block_size, const ccv_array_t* const anonymous_block_free_list, const int anonymous_block_free_list_cap, const int type, const uint64_t size, const ccv_sparse_matrix_t* const exec_dep, const ccv_array_t* const dup_p_refs)
{
	if (!anonymous_block_free_list || !anonymous_block_free_list_cap)
		return tensor_block_size;
	int i;
	const int no_dup_p_refs = (!dup_p_refs || !dup_p_refs->rnum);
	int found_idx = tensor_block_size;
	for (i = 0; i < anonymous_block_free_list_cap; i++)
	{
		const int idx = *(int*)ccv_array_get(anonymous_block_free_list, i);
		assert(idx < tensor_block_size);
		// If the type doesn't match, ignore.
		if (tensor_blocks[idx].type != type)
			continue;
		// Heuristic about how to select the best tensor block to move forward.
		// If the size is larger, and no dup_p_refs, found, I cannot do better than this, just return directly.
		if (tensor_blocks[idx].size >= size)
		{
			if (no_dup_p_refs)
				return idx;
			// Otherwise, only if the current tensor block's dup_p_refs is after (or at) the dup_p_refs,
			// then we cannot do better than this, if that is the case, just return.
			if (tensor_blocks[idx].dup_p_refs && tensor_blocks[idx].dup_p_refs->rnum &&
				_ccv_nnc_tensor_block_a_after_b_inclusively(exec_dep, tensor_blocks[idx].dup_p_refs, dup_p_refs))
				return idx;
		}
		int64_t found_idx_size_diff;
		int64_t idx_size_diff;
		if (found_idx == tensor_block_size || // If no found_idx yet, set the current one to be the found one, and continue.
			// Now, compare whether this one or the found_idx one is better.
			// At this point, there is no point of comparing the dup_p_refs, we only care about which one
			// is closer to the size we request. Only on a tie, dup_p_refs or not is important again.
			(found_idx_size_diff = llabs((int64_t)tensor_blocks[found_idx].size - (int64_t)size)) < (idx_size_diff = llabs((int64_t)tensor_blocks[idx].size - (int64_t)size)))
		{
			found_idx = idx;
			continue;
		}
		// No need to update if found_idx is better than idx.
		if (found_idx_size_diff > idx_size_diff)
			continue;
		// We bias towards the bigger one in case of similar.
		if (found_idx_size_diff == idx_size_diff && tensor_blocks[idx].size > tensor_blocks[found_idx].size)
		{
			found_idx = idx;
			continue;
		}
		assert(tensor_blocks[idx].size == tensor_blocks[found_idx].size);
		// On a tie, check which one has tighter life-cycle.
		if (tensor_blocks[idx].size >= size) // If this block size covers the size we request, we prefer longer life-cycle ones.
		{
			// Check whether the current tensor blocks life-cycle is longer than the previous one.
			if (tensor_blocks[idx].dup_p_refs && tensor_blocks[idx].dup_p_refs->rnum > 0 &&
				(!tensor_blocks[found_idx].dup_p_refs || !tensor_blocks[found_idx].dup_p_refs->rnum ||
				 _ccv_nnc_tensor_block_a_after_b_inclusively(exec_dep, tensor_blocks[idx].dup_p_refs, tensor_blocks[found_idx].dup_p_refs)))
				found_idx = idx;
			continue;
		}
		// Now both our size is smaller than requested size, in this case, we need to increase the tensor block size.
		// We prefer to choose the one that has life-cycle closer to the expected ones.
		if (no_dup_p_refs)
		{
			// Whoever is shorter wins.
			if (tensor_blocks[found_idx].dup_p_refs && tensor_blocks[found_idx].dup_p_refs->rnum > 0 &&
				(!tensor_blocks[idx].dup_p_refs || !tensor_blocks[idx].dup_p_refs->rnum ||
				 _ccv_nnc_tensor_block_a_after_b_inclusively(exec_dep, tensor_blocks[found_idx].dup_p_refs, tensor_blocks[idx].dup_p_refs)))
				found_idx = idx;
			continue;
		}
		if (!tensor_blocks[idx].dup_p_refs || !tensor_blocks[idx].dup_p_refs->rnum)
			continue;
		if (!tensor_blocks[found_idx].dup_p_refs || !tensor_blocks[found_idx].dup_p_refs->rnum)
		{
			found_idx = idx;
			continue;
		}
		// If both covers the request dup_p_refs, we prefer the shorter one, otherwise we prefer the longer one.
		const int idx_after_request = _ccv_nnc_tensor_block_a_after_b_inclusively(exec_dep, tensor_blocks[idx].dup_p_refs, dup_p_refs);
		const int found_idx_after_request = _ccv_nnc_tensor_block_a_after_b_inclusively(exec_dep, tensor_blocks[found_idx].dup_p_refs, dup_p_refs);
		if (idx_after_request && found_idx_after_request)
		{
			if (_ccv_nnc_tensor_block_a_after_b_inclusively(exec_dep, tensor_blocks[found_idx].dup_p_refs, tensor_blocks[idx].dup_p_refs))
				found_idx = idx;
			continue;
		} else {
			// We entered this branch must be either idx_after_request is false or found_idx_after_request is false or both.
			// If found_idx_after_request is not false, we are currently doing fine, no need to proceed.
			// Otherwise, if idx_after_request is true, it is preferred. If both are false, then prefer the longer one.
			if (!found_idx_after_request && (idx_after_request ||
				_ccv_nnc_tensor_block_a_after_b_inclusively(exec_dep, tensor_blocks[idx].dup_p_refs, tensor_blocks[found_idx].dup_p_refs)))
				found_idx = idx;
			continue;
		}
	}
	return found_idx;
}

static ccv_array_t* _ccv_nnc_dup_breakpoints_with_p_node_inputs(ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_info_t* const p_node_info)
{
	if (!(p_node_info && (p_node_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)))
		return 0;
	int i, j, k;
	int input_size = 0;
	for (i = 0; i < p_node_info->p_while.input_size; i++)
		if (p_node_info->p_while.inputs[i] >= 0)
			++input_size;
	// If doesn't have tensor inputs (thus, only special inputs), just return.
	if (!input_size)
		return 0;
	ccv_nnc_tensor_symbol_t inputs[input_size];
	input_size = 0;
	for (i = 0; i < p_node_info->p_while.input_size; i++)
		if (p_node_info->p_while.inputs[i] >= 0)
			inputs[input_size++] = (ccv_nnc_tensor_symbol_t){
				.d = p_node_info->p_while.inputs[i],
				.graph = symbolic_graph,
			};
	assert(symbolic_graph->breakpoint_size > 0);
	ccv_array_t* dup_breakpoints = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), symbolic_graph->breakpoint_size, 0);
	const int exec_symbol_info_size = symbolic_graph->exec_symbol_info->rnum;
	for (i = 0; i < symbolic_graph->breakpoint_size; i++)
	{
		// Make a noop copy of the breakpoint, but with some tensor inputs.
		ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), inputs, input_size, 0, 0, 0);
		ccv_array_push(dup_breakpoints, &noop);
		// Connect this noop to the outgoing nodes of breakpoints.
		const ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(symbolic_graph->exec_symbol_info, symbolic_graph->breakpoints[i].d);
		if (symbol_info->outgoings)
			for (j = 0; j < symbol_info->outgoings->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(symbol_info->outgoings, j);
				ccv_nnc_graph_exec_symbol_concat(symbolic_graph, noop, (ccv_nnc_graph_exec_symbol_t){
					.d = d,
					.graph = symbolic_graph,
				});
			}
	}
	for (i = 0; i < exec_symbol_info_size; i++)
	{
		const ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(symbolic_graph->exec_symbol_info, i);
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(symbol_info->flags))
			continue;
		if (symbol_info->outgoings)
		{
			const int outgoing_size = symbol_info->outgoings->rnum;
			for (j = 0; j < outgoing_size; j++)
			{
				const int d = *(int*)ccv_array_get(symbol_info->outgoings, j);
				for (k = 0; k < symbolic_graph->breakpoint_size; k++)
					if (d == symbolic_graph->breakpoints[k].d)
					{
						ccv_nnc_graph_exec_symbol_t noop = *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_breakpoints, k);
						ccv_nnc_graph_exec_symbol_concat(symbolic_graph, (ccv_nnc_graph_exec_symbol_t){
							.d = i,
							.graph = symbolic_graph,
						}, noop);
						// Found, connected, exit.
						break;
					}
			}
		}
	}
	// Add the dup_breakpoints to source if neccessary.
	assert(symbolic_graph->sources);
	const int source_size = symbolic_graph->sources->rnum;
	for (i = 0; i < source_size; i++)
	{
		const int d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(symbolic_graph->sources, i))->d;
		for (j = 0; j < symbolic_graph->breakpoint_size; j++)
			if (d == symbolic_graph->breakpoints[j].d)
			{
				ccv_nnc_graph_exec_symbol_t noop = *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_breakpoints, j);
				ccv_nnc_symbolic_graph_add_source(symbolic_graph, noop);
				// Found, made, exit.
				break;
			}
	}
	// Add the dup_breakpoints to destination if neccessary.
	assert(symbolic_graph->destinations);
	const int destination_size = symbolic_graph->destinations->rnum;
	for (i = 0; i < destination_size; i++)
	{
		const int d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(symbolic_graph->destinations, i))->d;
		for (j = 0; j < symbolic_graph->breakpoint_size; j++)
			if (d == symbolic_graph->breakpoints[j].d)
			{
				ccv_nnc_graph_exec_symbol_t noop = *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dup_breakpoints, j);
				ccv_nnc_symbolic_graph_add_destination(symbolic_graph, noop);
				// Found, made, exit.
				break;
			}
	}
	return dup_breakpoints;
}

// Plan out how we allocate tensor (should I do optimizations on graph here or not at all?).
static ccv_nnc_symbolic_graph_prep_t* _ccv_nnc_symbolic_graph_prep_new(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, const ccv_nnc_graph_exec_symbol_info_t* const p_exec_symbol_info, const int p_exec_symbol_info_size)
{
	assert(source_size > 0);
	assert(destination_size > 0);
	// First, fill all the "auto" holes.
	// This is the symbol table that with "auto" info filled up.
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * symbolic_graph->tensor_symbol_info->rnum);
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * symbolic_graph->exec_symbol_info->rnum);
	ccv_nnc_graph_exec_flag_t* exec_flags = (ccv_nnc_graph_exec_flag_t*)cccalloc(symbolic_graph->exec_symbol_info->rnum, sizeof(ccv_nnc_graph_exec_flag_t));
	ccv_nnc_graph_visit_t* visit = ccv_nnc_graph_visit_new(symbolic_graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(symbolic_graph->exec_symbol_info, 0), symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
	ccv_nnc_symbolic_graph_symbol_infer(symbolic_graph, visit, sources, source_size, destinations, destination_size, p_tensor_symbol_info, p_tensor_symbol_info_size, tensor_symbol_info, exec_symbol_info);
	int i, j, k, p, q;
	const ccv_nnc_graph_exec_symbol_info_t* const  p_node_info = p_exec_symbol_info ? p_exec_symbol_info + (symbolic_graph->exec_idx - 1) : 0;
	ccv_sparse_matrix_t* exec_dep;
	ccv_nnc_tensor_block_t* tensor_blocks;
	_ccv_nnc_exec_dep_and_tensor_blocks_prep(symbolic_graph, p_node_info, visit, tensor_binds, tensor_bind_size, outputs, output_size, sources, source_size, destinations, destination_size, p_tensor_symbol_info, p_tensor_symbol_info_size, exec_symbol_info, tensor_symbol_info, 0, 0, 0, 0, exec_flags, &exec_dep, &tensor_blocks);
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
	int unroll_count = 0;
	// In true recursive fashion, I need to call all the sub graphs and do the pre compilation for them one by one.
	ccv_nnc_symbolic_graph_prep_t* prep = (ccv_nnc_symbolic_graph_prep_t*)ccmalloc(sizeof(ccv_nnc_symbolic_graph_prep_t));
	prep->graph = ccv_nnc_graph_new(); // Just allocate the graph right now.
	prep->flags = 0;
	// Cannot handle dup a node that is a graph as well.
	if (p_exec_symbol_info)
	{
		prep->flags = p_node_info->flags;
		if (p_node_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
		{
			_ccv_nnc_redo_exec_dep_and_tensor_blocks_when_unroll(symbolic_graph, p_node_info, visit, tensor_binds, tensor_bind_size, sources, source_size, destinations, destination_size, p_tensor_symbol_info, p_tensor_symbol_info_size, exec_symbol_info, tensor_symbol_info, exec_flags, &exec_dep, &tensor_blocks, &tensor_block_size, &dup_graph, &unroll_count, &dup_exec_ref, &dup_tensor_block_ref);
			_ccv_nnc_fixup_tensor_blocks_for_outputs(exec_dep, tensor_blocks, p_node_info, unroll_count, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(symbolic_graph->destinations, 0), symbolic_graph->destinations->rnum, symbolic_graph->p_idx - 1, p_tensor_symbol_info, p_tensor_symbol_info_size, tensor_symbol_info, dup_exec_ref, dup_tensor_block_ref);
		} else if (p_node_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF) {
			// TODO: We want to try our best to fit as much of its corresponding inputs / outputs into companion_ref group.
		}
	}
	ccv_nnc_symbolic_graph_prep_t** sub_preps = symbolic_graph->sub_graphs && symbolic_graph->sub_graphs->rnum ? (ccv_nnc_symbolic_graph_prep_t**)cccalloc(symbolic_graph->sub_graphs->rnum, sizeof(ccv_nnc_symbolic_graph_prep_t*)) : 0;
	ccv_array_t* anonymous_block_free_list = 0;
	const int tensor_fold_size = (tensor_block_size + 31) >> 5;
	// Record whether this tensor is folded in this round.
	uint32_t* const tensor_fold = (uint32_t*)ccmalloc(sizeof(uint32_t) * tensor_fold_size);
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		for (p = 0; p < node->graph_ref_size; p++)
		{
			ccv_nnc_symbolic_graph_t* const sub_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, CCV_NNC_GRAPH_REF(node)[p] - 1);
			ccv_array_t* const dup_breakpoints = _ccv_nnc_dup_breakpoints_with_p_node_inputs(sub_graph, node);
			ccv_nnc_symbolic_graph_prep_t* const sub_prep = _ccv_nnc_symbolic_graph_prep_new(sub_graph, tensor_binds, tensor_bind_size, 0, 0, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum, tensor_symbol_info, symbolic_graph->tensor_symbol_info->rnum, exec_symbol_info, symbolic_graph->exec_symbol_info->rnum);
			sub_prep->dup_breakpoints = dup_breakpoints;
			sub_prep->p = prep;
			sub_preps[CCV_NNC_GRAPH_REF(node)[p] - 1] = sub_prep;
			const ccv_nnc_tensor_alloc_prep_t* const s_alloc_prep = sub_prep->alloc_prep;
			const ccv_nnc_tensor_block_t* const s_tensor_blocks = sub_prep->tensor_blocks;
			for (i = 0; i < s_alloc_prep->block_size; i++)
			{
				const int block_ref = s_alloc_prep->blocks[i].block_ref;
				const int buffer_ref = s_alloc_prep->blocks[i].buffer_ref;
				if (block_ref < sub_prep->tensor_symbol_info_size)
				{
					// If this block has a bypass, and its bypass has a different p_refs, then it doesn't matter.
					// I cannot assign p_refs to its parent buffer, and that buffer has to be anonymous.
					if (s_tensor_blocks[block_ref].bypass_ref)
					{
						int bypass_ref = s_tensor_blocks[block_ref].bypass_ref - 1;
						while (s_tensor_blocks[bypass_ref].ref)
							bypass_ref = s_tensor_blocks[bypass_ref].ref - 1;
						if (s_tensor_blocks[block_ref].p_refs[0] != s_tensor_blocks[bypass_ref].p_refs[0] ||
							s_tensor_blocks[block_ref].p_refs[1] != s_tensor_blocks[bypass_ref].p_refs[1])
							continue;
					}
					if (s_tensor_blocks[block_ref].p_refs[0])
					{
						/* If it is already properly assigned, next. */
						if (s_alloc_prep->buffers[buffer_ref].p_refs[0] != s_tensor_blocks[block_ref].p_refs[0] &&
							s_alloc_prep->buffers[buffer_ref].p_refs[1] != s_tensor_blocks[block_ref].p_refs[0])
						{
							if (!s_alloc_prep->buffers[buffer_ref].p_refs[0])
								s_alloc_prep->buffers[buffer_ref].p_refs[0] = s_tensor_blocks[block_ref].p_refs[0];
							else {
								assert(!s_alloc_prep->buffers[buffer_ref].p_refs[1]);
								s_alloc_prep->buffers[buffer_ref].p_refs[1] = s_tensor_blocks[block_ref].p_refs[0];
							}
						}
						/* When entering this branch, s_alloc_prep->buffers[buffer_ref].p_refs[0] cannot be 0. */
						if (s_tensor_blocks[block_ref].p_refs[1] &&
							s_alloc_prep->buffers[buffer_ref].p_refs[0] != s_tensor_blocks[block_ref].p_refs[1] &&
							s_alloc_prep->buffers[buffer_ref].p_refs[1] != s_tensor_blocks[block_ref].p_refs[1])
						{
							assert(s_alloc_prep->buffers[buffer_ref].p_refs[0]);
							assert(!s_alloc_prep->buffers[buffer_ref].p_refs[1]);
							s_alloc_prep->buffers[buffer_ref].p_refs[1] = s_tensor_blocks[block_ref].p_refs[1];
						}
					}
				} else if (s_tensor_blocks[block_ref].dup_p_refs) {
					/* In this case, only relevant bit is dup_p_ref. dup_p_ref extends the life-time of anonymous block
					 * which by default only has life-cycle shared with this sub-graph node. The reason to extend is that
					 * these anonymous blocks that has dup_p_ref may contain data that will be used as output (thus, dup_p_ref
					 * always points to an output tensor of this sub-graph node) therefore, the memory region must extend
					 * its life-time to the end of the output tensor. */
					if (!s_alloc_prep->buffers[buffer_ref].dup_p_refs)
						s_alloc_prep->buffers[buffer_ref].dup_p_refs = ccv_array_new(sizeof(int), s_tensor_blocks[block_ref].dup_p_refs->rnum, 0);
					for (j = 0; j < s_tensor_blocks[block_ref].dup_p_refs->rnum; j++)
						ccv_array_add_unique_int(s_alloc_prep->buffers[buffer_ref].dup_p_refs, *(int*)ccv_array_get(s_tensor_blocks[block_ref].dup_p_refs, j));
				}
			}
		}
		const int init_tensor_block_size = tensor_block_size;
		int rw_anonymous_buffer_size_cap = 0;
		int ro_anonymous_buffer_size_cap = 0;
		if (anonymous_block_free_list)
			ccv_array_clear(anonymous_block_free_list);
		memset(tensor_fold, 0, sizeof(uint32_t) * tensor_fold_size);
		for (p = 0; p < node->graph_ref_size; p++)
		{
			ccv_nnc_symbolic_graph_prep_t* const sub_prep = sub_preps[CCV_NNC_GRAPH_REF(node)[p] - 1];
			const ccv_nnc_tensor_alloc_prep_t* const s_alloc_prep = sub_prep->alloc_prep;
			int rw_anonymous_buffer_size = 0;
			int ro_anonymous_buffer_size = 0;
			for (i = 0; i < s_alloc_prep->buffer_size; i++)
				if (s_alloc_prep->buffers[i].p_refs[0])
				{
					/* Reduce 2 p_refs, if it is, to 1 p_ref (by doing block folding). */
					int p_ref_0 = s_alloc_prep->buffers[i].p_refs[0] - 1;
					/* Need to go through refs. Since we reuse the tensor block for this input, it now has to have allocate at least this much space. */
					int p_ref_0_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_0, node);
					assert(p_ref_0_is_in_or_out != 0);
					int unref_p_ref_0 = p_ref_0;
					while (tensor_blocks[unref_p_ref_0].ref)
						unref_p_ref_0 = tensor_blocks[unref_p_ref_0].ref - 1;
					/* This parent tensor block cannot be unassigned because it is either input / output of this sub-graph node. */
					assert(!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[unref_p_ref_0]));
					if (s_alloc_prep->buffers[i].p_refs[1])
					{
						int p_ref_1 = s_alloc_prep->buffers[i].p_refs[1] - 1;
						const int p_ref_1_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_1, node);
						assert(p_ref_1_is_in_or_out != 0);
						int unref_p_ref_1 = p_ref_1;
						while (tensor_blocks[unref_p_ref_1].ref)
							unref_p_ref_1 = tensor_blocks[unref_p_ref_1].ref - 1;
						/* See above comment for the similar p_ref_0 check. */
						assert(!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[unref_p_ref_1]));
						assert(p_ref_0_is_in_or_out != p_ref_1_is_in_or_out);
						int p_ref_t;
						if (p_ref_0_is_in_or_out < p_ref_1_is_in_or_out) /* if p_ref_0 is input and p_ref_1 is output, switch. */
						{
							CCV_SWAP(p_ref_0, p_ref_1, p_ref_t);
							CCV_SWAP(unref_p_ref_0, unref_p_ref_1, p_ref_t);
						}
						p_ref_0_is_in_or_out = 1; /* Now p_ref_0 surely is the output tensor. */
						/* If the dimension matches, can fold. */
						if (memcmp(tensor_symbol_info[unref_p_ref_1].info.dim, tensor_symbol_info[unref_p_ref_0].info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
						{
							const int folded = _ccv_nnc_tensor_blocks_try_fold(tensor_blocks, unref_p_ref_1, unref_p_ref_0);
							if (folded)
							{
								p_ref_0 = p_ref_1;
								unref_p_ref_0 = unref_p_ref_1; // p_ref_0 now folded into p_ref_1, therefore, pointing to p_ref_1 now.
								tensor_fold[unref_p_ref_0 >> 5] |= (1u << (unref_p_ref_0 & 0x1f));
								for (j = 0; j < unroll_count; j++) /* Fold its duplicates as well. */
								{
									const int folded = _ccv_nnc_tensor_blocks_try_fold(tensor_blocks, dup_tensor_block_ref[unref_p_ref_1 * unroll_count + j], dup_tensor_block_ref[unref_p_ref_0 * unroll_count + j]);
									assert(folded && "the subsequent duplicates can be folded too.");
								}
							}
						}
					}
					/* Only proceed if it is folded here (thus, the input / output tensor can be connected, reuse is not a problem
					 * Or if the p_ref_0 is the output, it is the first started from this node (thus, I have full control over
					 * its life-cycle). Or if the p_ref_0 is the input, it is ended in this node (thus, I can take over i
					 * life-cycle freely within this sub-graph (otherwise, if it is used anywhere, I cannot change the content
					 * within its memory region)). Unless this buffer is used as read-only, and we don't have any output
					 * associated with it, then we are good. */
					if ((tensor_fold[unref_p_ref_0 >> 5] & (1u << (unref_p_ref_0 & 0x1f))) ||
						(p_ref_0_is_in_or_out == 1 && _ccv_nnc_tensor_block_check_head(tensor_blocks + unref_p_ref_0, idx)) ||
						(p_ref_0_is_in_or_out == -1 && _ccv_nnc_tensor_block_check_tail(tensor_blocks + unref_p_ref_0, idx)) ||
						TENSOR_READ_WRITE(s_alloc_prep->buffers[i]) == READ_ONLY)
					{
						if (TENSOR_READ_WRITE(s_alloc_prep->buffers[i]) == READ_ONLY)
							{ assert(s_alloc_prep->buffers[i].p_refs[1] == 0); }
						/* p_ref_0 is either the only one, or the output tensor, we always prefer the output tensor (there
						 * is a long argument why that is the case, the digest is, it is much easier to control your output
						 * than your input). */
						s_alloc_prep->buffers[i].p_refs[0] = p_ref_0 + 1;
						s_alloc_prep->buffers[i].p_refs[1] = 0;
						/* This parent tensor block cannot be unassigned because it is either input / output of this sub-graph node. */
						assert(!TENSOR_EXPECT_UNASSIGNED(tensor_blocks[unref_p_ref_0]));
						tensor_blocks[unref_p_ref_0].size = ccv_max(s_alloc_prep->buffers[i].size, tensor_blocks[unref_p_ref_0].size);
						for (j = 0; j < unroll_count; j++) /* Change the size of its duplicates as well. */
							tensor_blocks[dup_tensor_block_ref[p_ref_0 * unroll_count + j]].size =
								tensor_blocks[dup_tensor_block_ref[unref_p_ref_0 * unroll_count + j]].size =
									tensor_blocks[unref_p_ref_0].size;
					} else {
						s_alloc_prep->buffers[i].p_refs[0] = s_alloc_prep->buffers[i].p_refs[1] = 0;
						if (TENSOR_READ_WRITE(s_alloc_prep->buffers[i]) == READ_ONLY)
							++ro_anonymous_buffer_size;
						else
							++rw_anonymous_buffer_size;
					}
				} else {
					if (TENSOR_READ_WRITE(s_alloc_prep->buffers[i]) == READ_ONLY)
						++ro_anonymous_buffer_size;
					else
						++rw_anonymous_buffer_size;
				}
			if (ro_anonymous_buffer_size || rw_anonymous_buffer_size)
			{
				const int anonymous_block_free_list_cap = anonymous_block_free_list ? anonymous_block_free_list->rnum : 0;
				// All read-write buffer (potentially) can be reused between each case..of branch.
				rw_anonymous_buffer_size_cap += rw_anonymous_buffer_size;
				// Read-only buffer cannot be reused between each case..of branch.
				ro_anonymous_buffer_size_cap += ro_anonymous_buffer_size;
				/* Anonymous block, allocate additional tensor blocks for this. */
				/* This is either because this is an internal tensor (don't have p_ref) */
				/* or it is an anonymous block itself within the sub graphs of this while graph. */
				tensor_blocks = (ccv_nnc_tensor_block_t*)ccrealloc(tensor_blocks, sizeof(ccv_nnc_tensor_block_t) * (init_tensor_block_size + (unroll_count + 1) * rw_anonymous_buffer_size_cap + ro_anonymous_buffer_size_cap));
				memset(tensor_blocks + tensor_block_size, 0, sizeof(ccv_nnc_tensor_block_t) * (init_tensor_block_size + (unroll_count + 1) * rw_anonymous_buffer_size_cap + ro_anonymous_buffer_size_cap - tensor_block_size));
				if (dup_tensor_block_ref)
					dup_tensor_block_ref = (int*)ccrealloc(dup_tensor_block_ref, sizeof(int) * unroll_count * (init_tensor_block_size + rw_anonymous_buffer_size_cap + ro_anonymous_buffer_size_cap));
				for (i = 0; i < s_alloc_prep->buffer_size; i++)
					if (!s_alloc_prep->buffers[i].p_refs[0])
					{
						if (TENSOR_READ_WRITE(s_alloc_prep->buffers[i]) == READ_ONLY) /* If it is read-only, add all sources (destinations) to it. */
						{
							TENSOR_SET_ANONYMOUS(tensor_blocks[tensor_block_size]);
							TENSOR_SET_READ_WRITE(tensor_blocks[tensor_block_size], TENSOR_READ_WRITE(s_alloc_prep->buffers[i]));
							tensor_blocks[tensor_block_size].type = s_alloc_prep->buffers[i].type;
							tensor_blocks[tensor_block_size].size = s_alloc_prep->buffers[i].size;
							s_alloc_prep->buffers[i].p_refs[0] = tensor_block_size + 1;
							tensor_blocks[tensor_block_size].head = ccv_array_new(sizeof(int), 1, 0);
							ccv_array_push(tensor_blocks[tensor_block_size].head, &idx);
							ccv_array_t* const dup_p_refs = s_alloc_prep->buffers[i].dup_p_refs;
							if (dup_p_refs && dup_p_refs->rnum > 0)
							{
								for (j = 0; j < dup_p_refs->rnum; j++)
								{
									const int dup_p_ref = *(int*)ccv_array_get(dup_p_refs, j);
									assert(dup_p_ref >= 0);
									assert(dup_p_ref < symbolic_graph->tensor_symbol_info->rnum);
									assert(tensor_blocks[dup_p_ref].tail);
									// If it points to a p_ref upwards, check whether this is an output, if it is an output, add it to
									// this block's dup_p_refs. It propagates back all the way to upper layer's buffer object.
									if (tensor_symbol_info[dup_p_ref].p_ref)
									{
										const int p_ref_0 = tensor_symbol_info[dup_p_ref].p_ref - 1;
										assert(p_node_info);
										const int p_ref_0_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_0, p_node_info);
										if (p_ref_0_is_in_or_out == 1) // If it is out tensor, mark dup_p_ref for this.
										{
											if (!tensor_blocks[tensor_block_size].dup_p_refs)
												tensor_blocks[tensor_block_size].dup_p_refs = ccv_array_new(sizeof(int), 1, 0);
											ccv_array_add_unique_int(tensor_blocks[tensor_block_size].dup_p_refs, p_ref_0);
										}
									}
									if (!tensor_blocks[tensor_block_size].tail)
										tensor_blocks[tensor_block_size].tail = ccv_array_new(sizeof(int), tensor_blocks[dup_p_ref].tail->rnum, 0);
									for (k = 0; k < tensor_blocks[dup_p_ref].tail->rnum; k++)
										_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[dup_p_ref].tail, k), tensor_blocks[tensor_block_size]);
								}
							} else {
								tensor_blocks[tensor_block_size].tail = ccv_array_new(sizeof(int), 1, 0);
								ccv_array_push(tensor_blocks[tensor_block_size].tail, &idx);
							}
							for (j = 0; j < source_size; j++)
								_ccv_nnc_tensor_block_add_exec(exec_dep, sources[j].d, tensor_blocks[tensor_block_size]);
							/* If this is a read-only (based on SSA, if first encountered as read), and this is
							 * sub-graph. Mark it to the end of the graph. */
							if (p_exec_symbol_info)
								for (j = 0; j < destination_size; j++)
									_ccv_nnc_tensor_block_add_exec(exec_dep, destinations[j].d, tensor_blocks[tensor_block_size]);
							/* If it is read-only, it is self-reflecting. */
							for (k = 0; k < unroll_count; k++)
							{
								for (j = 0; j < destination_size; j++)
									if (dup_exec_ref[destinations[j].d * unroll_count + k] >= 0)
									_ccv_nnc_tensor_block_add_exec(exec_dep, dup_exec_ref[destinations[j].d * unroll_count + k], tensor_blocks[tensor_block_size]);
								/* No need to extend life-time, because this is a sub-graph and we already extended read-only to the end of destination. */
								assert(symbolic_graph->p);
								dup_tensor_block_ref[tensor_block_size * unroll_count + k] = tensor_block_size;
							}
							++tensor_block_size;
						} else {
							ccv_array_t* const dup_p_refs = s_alloc_prep->buffers[i].dup_p_refs;
							const int tensor_block_idx = _ccv_nnc_anonymous_tensor_block_from_free_list(tensor_blocks, tensor_block_size, anonymous_block_free_list, anonymous_block_free_list_cap, s_alloc_prep->buffers[i].type, s_alloc_prep->buffers[i].size, exec_dep, dup_p_refs);
							const int new_anonymous_tensor_block = (tensor_block_idx == tensor_block_size);
							// Find suitable tensor block from the free list.
							TENSOR_SET_ANONYMOUS(tensor_blocks[tensor_block_idx]);
							TENSOR_SET_READ_WRITE(tensor_blocks[tensor_block_idx], TENSOR_READ_WRITE(s_alloc_prep->buffers[i]));
							s_alloc_prep->buffers[i].p_refs[0] = tensor_block_idx + 1;
							if (new_anonymous_tensor_block)
							{
								tensor_blocks[tensor_block_idx].type = s_alloc_prep->buffers[i].type;
								tensor_blocks[tensor_block_idx].size = s_alloc_prep->buffers[i].size;
								tensor_blocks[tensor_block_idx].head = ccv_array_new(sizeof(int), 1, 0);
								ccv_array_push(tensor_blocks[tensor_block_idx].head, &idx);
							} else
								tensor_blocks[tensor_block_idx].size = ccv_max(tensor_blocks[tensor_block_idx].size, s_alloc_prep->buffers[i].size);
							if (dup_p_refs && dup_p_refs->rnum > 0)
							{
								for (j = 0; j < dup_p_refs->rnum; j++)
								{
									const int dup_p_ref = *(int*)ccv_array_get(dup_p_refs, j);
									assert(dup_p_ref >= 0);
									assert(dup_p_ref < symbolic_graph->tensor_symbol_info->rnum);
									// If it points to a p_ref upwards, check whether this is an output, if it is an output, add it to
									// this block's dup_p_refs. It propagates back all the way to upper layer's buffer object.
									if (tensor_symbol_info[dup_p_ref].p_ref)
									{
										const int p_ref_0 = tensor_symbol_info[dup_p_ref].p_ref - 1;
										assert(p_node_info);
										const int p_ref_0_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_0, p_node_info);
										if (p_ref_0_is_in_or_out == 1) // If it is out tensor, mark dup_p_ref for this.
										{
											if (!tensor_blocks[tensor_block_idx].dup_p_refs)
												tensor_blocks[tensor_block_idx].dup_p_refs = ccv_array_new(sizeof(int), 1, 0);
											ccv_array_add_unique_int(tensor_blocks[tensor_block_idx].dup_p_refs, p_ref_0);
										}
									}
									assert(tensor_blocks[dup_p_ref].tail);
									if (!tensor_blocks[tensor_block_idx].tail)
										tensor_blocks[tensor_block_idx].tail = ccv_array_new(sizeof(int), tensor_blocks[dup_p_ref].tail->rnum, 0);
									for (k = 0; k < tensor_blocks[dup_p_ref].tail->rnum; k++)
										_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[dup_p_ref].tail, k), tensor_blocks[tensor_block_idx]);
										// We have to add it to the warp around companion_ref as well.
										// TODO: Although we know this wasted space (any space in between current one and its companion_ref will still
										// be occupied and unlikely to be reused), but we cannot really do too much about it because the companion_ref's
										// definition is too free-form and if we enforce stronger gaurantee on this (such as it must wrap around), this
										// gaurantee may be broken down in the line.
										if (tensor_blocks[dup_p_ref].companion_ref)
										{
											const int companion_ref = tensor_blocks[dup_p_ref].companion_ref - 1;
											for (q = 0; tensor_blocks[companion_ref].head && q < tensor_blocks[companion_ref].head->rnum; q++)
												_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[companion_ref].head, q), tensor_blocks[tensor_block_idx]);
											for (q = 0; tensor_blocks[companion_ref].tail && q < tensor_blocks[companion_ref].tail->rnum; q++)
												_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[companion_ref].tail, q), tensor_blocks[tensor_block_idx]);
										}
								}
							} else if (new_anonymous_tensor_block) {
								tensor_blocks[tensor_block_idx].tail = ccv_array_new(sizeof(int), 1, 0);
								ccv_array_push(tensor_blocks[tensor_block_idx].tail, &idx);
							}
							const int prev_tensor_block_idx = tensor_block_idx;
							if (new_anonymous_tensor_block)
							{
								if (!anonymous_block_free_list)
									anonymous_block_free_list = ccv_array_new(sizeof(int), 0, 0);
								ccv_array_push(anonymous_block_free_list, &tensor_block_size);
								++tensor_block_size;
							}
							for (k = 0; k < unroll_count; k++)
							{
								const int tensor_block_idx = new_anonymous_tensor_block ?
									(dup_tensor_block_ref[prev_tensor_block_idx * unroll_count + k] = tensor_block_size) :
									dup_tensor_block_ref[prev_tensor_block_idx * unroll_count + k];
								TENSOR_SET_ANONYMOUS(tensor_blocks[tensor_block_idx]);
								TENSOR_SET_READ_WRITE(tensor_blocks[tensor_block_idx], TENSOR_READ_WRITE(s_alloc_prep->buffers[i]));
								if (new_anonymous_tensor_block)
								{
									tensor_blocks[tensor_block_idx].type = s_alloc_prep->buffers[i].type;
									tensor_blocks[tensor_block_idx].size = s_alloc_prep->buffers[i].size;
									tensor_blocks[tensor_block_idx].head = ccv_array_new(sizeof(int), 1, 0);
									/* Attach to duplicated exec for this tensor block. */
									ccv_array_push(tensor_blocks[tensor_block_idx].head, &dup_exec_ref[idx * unroll_count + k]);
								} else {
									tensor_blocks[tensor_block_idx].size = ccv_max(tensor_blocks[tensor_block_idx].size, s_alloc_prep->buffers[i].size);
									_ccv_nnc_tensor_block_add_exec(exec_dep, dup_exec_ref[idx * unroll_count + k], tensor_blocks[tensor_block_idx]);

								}
								if (dup_p_refs && dup_p_refs->rnum > 0)
								{
									/* Not nil, not self-reflecting. */
									for (j = 0; j < dup_p_refs->rnum; j++)
									{
										const int dup_p_ref = *(int*)ccv_array_get(dup_p_refs, j);
										assert(dup_p_ref >= 0);
										assert(dup_p_ref < symbolic_graph->tensor_symbol_info->rnum);
										// If it points to a p_ref upwards, check whether this is an output, if it is an output, add it to
										// this block's dup_p_refs. It propagates back all the way to upper layer's buffer object.
										if (tensor_symbol_info[dup_p_ref].p_ref)
										{
											const int p_ref_0 = tensor_symbol_info[dup_p_ref].p_ref - 1;
											assert(p_node_info);
											const int p_ref_0_is_in_or_out = _ccv_nnc_is_symbolic_graph_exec_input_or_output(p_ref_0, p_node_info);
											if (p_ref_0_is_in_or_out == 1) // If it is out tensor, mark dup_p_ref for this.
											{
												if (!tensor_blocks[tensor_block_idx].dup_p_refs)
													tensor_blocks[tensor_block_idx].dup_p_refs = ccv_array_new(sizeof(int), 1, 0);
												ccv_array_add_unique_int(tensor_blocks[tensor_block_idx].dup_p_refs, p_ref_0);
											}
										}
										assert(dup_tensor_block_ref[dup_p_ref * unroll_count + k] >= 0 && dup_tensor_block_ref[dup_p_ref * unroll_count + k] != dup_p_ref);
										const int dup_dup_p_ref = dup_tensor_block_ref[dup_p_ref * unroll_count + k];
										assert(tensor_blocks[dup_dup_p_ref].tail);
										if (!tensor_blocks[tensor_block_idx].tail)
											tensor_blocks[tensor_block_idx].tail = ccv_array_new(sizeof(int), tensor_blocks[dup_dup_p_ref].tail->rnum, 0);
										for (q = 0; q < tensor_blocks[dup_dup_p_ref].tail->rnum; q++)
											_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[dup_dup_p_ref].tail, q), tensor_blocks[tensor_block_idx]);
										// We have to add it to the warp around companion_ref as well.
										if (tensor_blocks[dup_dup_p_ref].companion_ref)
										{
											const int companion_ref = tensor_blocks[dup_dup_p_ref].companion_ref - 1;
											for (q = 0; tensor_blocks[companion_ref].head && q < tensor_blocks[companion_ref].head->rnum; q++)
												_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[companion_ref].head, q), tensor_blocks[tensor_block_idx]);
											for (q = 0; tensor_blocks[companion_ref].tail && q < tensor_blocks[companion_ref].tail->rnum; q++)
												_ccv_nnc_tensor_block_add_exec(exec_dep, *(int*)ccv_array_get(tensor_blocks[companion_ref].tail, q), tensor_blocks[tensor_block_idx]);
										}
									}
								} else if (new_anonymous_tensor_block) {
									tensor_blocks[tensor_block_idx].tail = ccv_array_new(sizeof(int), 1, 0);
									ccv_array_push(tensor_blocks[tensor_block_idx].tail, &dup_exec_ref[idx * unroll_count + k]);
								}
								if (new_anonymous_tensor_block)
									++tensor_block_size;
							}
						}
					}
			}
		}
	} ccv_nnc_graph_visit_endfor
	if (anonymous_block_free_list)
		ccv_array_free(anonymous_block_free_list);
	ccfree(tensor_fold);
	// It is time to guess what's the best tensor placement and create the opaque tensor arena. The alloc_dep will return
	// the allocation dependencies, thus, which tensor is reused to the existing tensor.
	ccv_nnc_tensor_alloc_prep_t* alloc_prep = _ccv_nnc_tensor_alloc_prep_new(exec_dep, tensor_blocks, tensor_block_size);
	ccv_matrix_free(exec_dep);
	prep->while_count_tensor = 0;
	prep->dup_breakpoints = 0;
	prep->p = 0;
	prep->symbolic_graph = symbolic_graph;
	prep->p_idx = symbolic_graph->p_idx;
	prep->exec_idx = symbolic_graph->exec_idx;
	prep->sub_prep_size = symbolic_graph->sub_graphs ? symbolic_graph->sub_graphs->rnum : 0;
	prep->sub_preps = sub_preps;
	prep->exec_symbol_info_size = symbolic_graph->exec_symbol_info->rnum;
	prep->exec_symbol_info = exec_symbol_info;
	prep->tensor_symbol_info_size = symbolic_graph->tensor_symbol_info->rnum;
	prep->tensor_symbol_info = tensor_symbol_info;
	prep->unroll_count = unroll_count;
	prep->dup_tensor_block_ref = dup_tensor_block_ref;
	prep->tensor_block_size = tensor_block_size;
	prep->tensor_blocks = tensor_blocks;
	prep->exec_flags = exec_flags;
	prep->visit = visit;
	prep->alloc_prep = alloc_prep;
	if (dup_graph)
		ccv_nnc_symbolic_graph_free(dup_graph);
	if (dup_exec_ref)
		ccfree(dup_exec_ref);
	return prep;
}

static void _ccv_nnc_symbolic_graph_prep_free(ccv_nnc_symbolic_graph_prep_t* const prep)
{
	int i;
	_ccv_nnc_tensor_blocks_free(prep->tensor_blocks, prep->tensor_block_size);
	ccfree(prep->exec_flags);
	for (i = 0; i < prep->sub_prep_size; i++)
		if (prep->sub_preps[i])
			_ccv_nnc_symbolic_graph_prep_free(prep->sub_preps[i]);
	if (prep->sub_preps)
		ccfree(prep->sub_preps);
	ccfree(prep->tensor_symbol_info);
	ccfree(prep->exec_symbol_info);
	if (prep->dup_tensor_block_ref)
		ccfree(prep->dup_tensor_block_ref);
	_ccv_nnc_tensor_alloc_prep_free(prep->alloc_prep);
	ccv_nnc_graph_visit_free(prep->visit);
	ccfree(prep);
}

static void _ccv_nnc_symbolic_graph_prep_while_count_tensor(ccv_nnc_symbolic_graph_prep_t* const graph_prep)
{
	int i, j;
	ccv_nnc_graph_visit_for(graph_prep->visit, graph_prep->exec_symbol_info, node, idx) {
		if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
		{
			const int graph_ref = CCV_NNC_GRAPH_REF(node)[0] - 1;
			assert(graph_ref >= 0);
			ccv_nnc_symbolic_graph_prep_t* const sub_prep = graph_prep->sub_preps[graph_ref];
			for (i = 0; i < node->p_while.input_size; i++)
				if (CCV_NNC_IS_WHILE_COUNT_TENSOR_SYMBOL(node->p_while.inputs[i]))
				{
					ccv_nnc_symbolic_graph_prep_t* prep = sub_prep;
					const int d = CCV_NNC_DECODE_WHILE_COUNT_SYMBOL(node->p_while.inputs[i]);
					for (j = 0; j < d; j++)
						prep = prep->p;
					prep->while_count_tensor = 1;
				}
		}
		for (i = 0; i < node->graph_ref_size; i++)
		{
			const int graph_ref = CCV_NNC_GRAPH_REF(node)[i] - 1;
			if (graph_ref >= 0)
				_ccv_nnc_symbolic_graph_prep_while_count_tensor(graph_prep->sub_preps[graph_ref]);
		}
	} ccv_nnc_graph_visit_endfor
}

static ccv_nnc_tensor_t* _ccv_nnc_tensor_from_graph_prep(const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const int symbol)
{
	if (symbol >= 0)
		return graph_prep->tensor_arena->vt_tensors[symbol];
	if (symbol == CCV_NNC_NO_TENSOR_SYMBOL)
		return 0;
	assert(CCV_NNC_IS_WHILE_COUNT_TENSOR_SYMBOL(symbol));
	const ccv_nnc_symbolic_graph_prep_t* prep = graph_prep;
	int i;
	const int d = CCV_NNC_DECODE_WHILE_COUNT_SYMBOL(symbol);
	for (i = 0; i < d; i++)
		prep = prep->p;
	assert(prep->while_count_tensor);
	return (ccv_nnc_tensor_t*)_ccv_nnc_tensor_metadata_get(prep->tensor_arena->tensor_metadata, (0 << 1) + 1);
}

static void _ccv_nnc_graph_exec_arena_sequential(ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	int i;
	int* const exec_cvt = (int*)ccmalloc(sizeof(int) * graph->exec_info->rnum);
	ccv_nnc_graph_sequential(graph, exec_cvt, graph->exec_info->rnum);
	graph_exec_arena->source.d = exec_cvt[graph_exec_arena->source.d];
	graph_exec_arena->destination.d = exec_cvt[graph_exec_arena->destination.d];
	ccv_nnc_graph_exec_t* const graph_execs = graph_exec_arena->graph_execs;
	for (i = 0; i < graph_exec_arena->graph_exec_size; i++)
		if (graph_execs[i].graph == graph)
			graph_execs[i].d = exec_cvt[graph_execs[i].d];
	ccfree(exec_cvt);
}

static ccv_nnc_graph_exec_arena_t* _ccv_nnc_graph_exec_arena_new(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const ccv_nnc_tensor_arena_t* const tensor_arena)
{
	int i, j, k;
	ccv_nnc_graph_t* const graph = graph_prep->graph;
	const int exec_symbol_info_size = symbolic_graph->exec_symbol_info->rnum;
	ccv_nnc_graph_exec_arena_t* const graph_exec_arena = (ccv_nnc_graph_exec_arena_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_arena_t) + sizeof(ccv_nnc_graph_exec_arena_t*) * graph_prep->sub_prep_size + sizeof(ccv_nnc_graph_exec_t) * (exec_symbol_info_size - 1));
	graph_exec_arena->graph_ref = (intptr_t)symbolic_graph;
	graph_exec_arena->graph_exec_size = exec_symbol_info_size;
	graph_exec_arena->sub_arena_size = graph_prep->sub_prep_size;
	graph_exec_arena->sub_arenas = (ccv_nnc_graph_exec_arena_t**)(graph_exec_arena->graph_execs + exec_symbol_info_size);
	memset(graph_exec_arena->sub_arenas, 0, sizeof(ccv_nnc_graph_exec_arena_t*) * graph_exec_arena->sub_arena_size);
	ccv_nnc_graph_exec_t* const graph_execs = graph_exec_arena->graph_execs;
	int max_input_size = 0, max_output_size = 0, max_breakpoint_size = 0;
	for (i = 0; i < exec_symbol_info_size; i++)
	{
		max_input_size = ccv_max(max_input_size, graph_prep->exec_symbol_info[i].input_size);
		max_output_size = ccv_max(max_input_size, graph_prep->exec_symbol_info[i].output_size);
		if (graph_prep->exec_symbol_info[i].flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
			max_input_size = ccv_max(max_input_size, graph_prep->exec_symbol_info[i].p_while.input_size);
		graph_execs[i].d = CCV_NNC_NO_TENSOR_SYMBOL;
		graph_execs[i].graph = 0;
	}
	for (i = 0; i < graph_prep->sub_prep_size; i++)
		max_breakpoint_size = ccv_max(max_breakpoint_size, (*(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, i))->breakpoint_size);
	ccv_nnc_tensor_t* max_inputs[ccv_max(1, max_input_size)];
	ccv_nnc_tensor_t* max_outputs[ccv_max(1, max_output_size)];
	ccv_nnc_graph_exec_t max_breakpoints[ccv_max(1, max_breakpoint_size)];
	const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = graph_prep->exec_symbol_info;
	const ccv_nnc_graph_exec_flag_t* const exec_flags = graph_prep->exec_flags;
	// Create node, this is in topological order.
	ccv_nnc_graph_visit_for(graph_prep->visit, exec_symbol_info, node, idx) {
		if (CCV_NO_GRAPH_EXEC(graph_execs[idx]))
		{
			for (i = 0; i < node->input_size; i++)
				max_inputs[i] = _ccv_nnc_tensor_from_graph_prep(graph_prep, node->inputs[i]);
			for (i = 0; i < node->output_size; i++)
				max_outputs[i] = node->outputs[i] >= 0 ? tensor_arena->vt_tensors[node->outputs[i]] : 0;
			if (node->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
			{
				const int graph_ref = CCV_NNC_GRAPH_REF(node)[0] - 1;
				assert(graph_ref >= 0);
				ccv_nnc_symbolic_graph_prep_t* const sub_prep = graph_prep->sub_preps[graph_ref];
				ccv_nnc_graph_t* const sub_graph = sub_prep->graph;
				graph_execs[idx] = ccv_nnc_graph_while(graph, node->cmd.cmd, sub_graph);
				const ccv_nnc_symbolic_graph_t* const sub_symbolic_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, graph_ref);
				ccv_nnc_graph_exec_arena_t* const sub_arena = graph_exec_arena->sub_arenas[graph_ref] = _ccv_nnc_graph_exec_arena_new(sub_symbolic_graph, ccv_nnc_symbolic_graph_sources(sub_symbolic_graph), ccv_nnc_symbolic_graph_source_size(sub_symbolic_graph), ccv_nnc_symbolic_graph_destinations(sub_symbolic_graph), ccv_nnc_symbolic_graph_destination_size(sub_symbolic_graph), graph_prep->sub_preps[graph_ref], tensor_arena->sub_arenas[graph_ref]);
				ccv_nnc_graph_exec_set_io(graph, graph_execs[idx], max_inputs, node->input_size, max_outputs, node->output_size);
				for (i = 0; i < node->p_while.input_size; i++)
					max_inputs[i] = _ccv_nnc_tensor_from_graph_prep(sub_prep, node->p_while.inputs[i]);
				for (i = 0; i < sub_symbolic_graph->breakpoint_size; i++)
					max_breakpoints[i] = ccv_nnc_graph_exec_from_symbol(sub_arena, sub_symbolic_graph->breakpoints[i]);
				ccv_nnc_graph_set_while_expr(sub_graph, node->p_while.expr, node->p_while.data, max_inputs, node->p_while.input_size, max_breakpoints, sub_symbolic_graph->breakpoint_size);
				_ccv_nnc_graph_exec_arena_sequential(sub_graph, sub_arena);
			} else if (node->flags & CCV_NNC_GRAPH_EXEC_CASE_OF) {
				for (i = 0; i < node->output_size; i++)
					if (max_outputs[i] && max_outputs[i]->alias_ref)
						max_outputs[i] = (ccv_nnc_tensor_t*)max_outputs[i]->alias_ref;
				graph_execs[idx] = ccv_nnc_graph_case_of_new(graph, node->cmd.cmd, max_inputs, node->input_size, max_outputs, node->output_size, node->case_of.argument.offset, node->case_of.argument.size);
				const int offset = (exec_flags[idx].flags & CCV_NNC_GRAPH_EXEC_ATTR_CASE_OF_NO_BYPASS_IO) ? 1 : 0;
				ccv_nnc_graph_set_case_of_expr(graph, graph_execs[idx], node->case_of.expr, node->case_of.data, offset);
				if (exec_flags[idx].flags & CCV_NNC_GRAPH_EXEC_ATTR_CASE_OF_NO_BYPASS_IO)
				{
					// Add another graph for data transfer.
					ccv_nnc_graph_t* sub_graph = ccv_nnc_graph_new();
					for (i = 0; i < node->output_size; i++)
						max_outputs[i] = node->outputs[i] >= 0 ? tensor_arena->vt_tensors[node->outputs[i]] : 0;
					ccv_nnc_graph_exec_t io = ccv_nnc_graph_exec_new(sub_graph, ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, max_inputs, ccv_min(node->input_size, node->output_size), max_outputs, ccv_min(node->input_size, node->output_size));
					ccv_nnc_graph_set_sources(sub_graph, &io, 1);
					ccv_nnc_graph_set_destinations(sub_graph, &io, 1);
					ccv_nnc_graph_set_case_of(graph, graph_execs[idx], sub_graph, 0);
					int exec_cvt;
					ccv_nnc_graph_sequential(sub_graph, &exec_cvt, 1);
				}
				for (i = 0; i < node->graph_ref_size; i++)
				{
					const int graph_ref = CCV_NNC_GRAPH_REF(node)[i] - 1;
					if (graph_ref < 0)
						continue;
					ccv_nnc_graph_t* const sub_graph = graph_prep->sub_preps[graph_ref]->graph;
					const ccv_nnc_symbolic_graph_t* const sub_symbolic_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(symbolic_graph->sub_graphs, graph_ref);
					ccv_nnc_graph_exec_arena_t* const sub_arena = graph_exec_arena->sub_arenas[graph_ref] = _ccv_nnc_graph_exec_arena_new(sub_symbolic_graph, ccv_nnc_symbolic_graph_sources(sub_symbolic_graph), ccv_nnc_symbolic_graph_source_size(sub_symbolic_graph), ccv_nnc_symbolic_graph_destinations(sub_symbolic_graph), ccv_nnc_symbolic_graph_destination_size(sub_symbolic_graph), graph_prep->sub_preps[graph_ref], tensor_arena->sub_arenas[graph_ref]);
					ccv_nnc_graph_set_case_of(graph, graph_execs[idx], sub_graph, i + offset);
					_ccv_nnc_graph_exec_arena_sequential(sub_graph, sub_arena);
				}
			} else {
				graph_execs[idx] = ccv_nnc_graph_exec_new(graph, node->cmd, node->hint, max_inputs, node->input_size, max_outputs, node->output_size);
			}
			ccv_nnc_graph_exec_set_io_flags(graph, graph_execs[idx], 0, 0, 0, 0);
		}
	} ccv_nnc_graph_visit_endfor
	// Then connect them.
	ccv_nnc_graph_visit_for(graph_prep->visit, exec_symbol_info, node, idx) {
		if (node->outgoings)
			for (i = 0; i < node->outgoings->rnum; i++)
			{
				const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
				if (graph_execs[outgoing].graph)
					ccv_nnc_graph_exec_concat(graph, graph_execs[idx], graph_execs[outgoing]);
			}
	} ccv_nnc_graph_visit_endfor
	int source_exec_created = 0;
	const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = graph_prep->tensor_symbol_info;
	const ccv_nnc_tensor_block_t* const tensor_blocks = graph_prep->tensor_blocks;
	ccv_array_t* const* const alloc_dep = graph_prep->alloc_prep->alloc_dep;
	// After the graph is materialized, we need to handle the case that some of these tensors require to be initialized to zero before use.
	for (i = 0; i < symbolic_graph->tensor_symbol_info->rnum; i++)
	{
		if (tensor_symbol_info[i].flags & CCV_NNC_TENSOR_SYMBOL_INIT_ZEROS)
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
				if (outgoing >= exec_symbol_info_size)
					continue;
				assert(outgoing >= 0);
				assert(graph_execs[outgoing].graph);
				ccv_nnc_graph_exec_concat(graph, set_exec, graph_execs[outgoing]);
			}
			int flags = 0;
			if (alloc_dep[ref])
				for (j = 0; j < alloc_dep[ref]->rnum; j++)
				{
					const int d = *(int*)ccv_array_get(alloc_dep[ref], j);
					// This is from alloc_dep, it should be computable.
					assert(TENSOR_EXPECT_COMPUTABLE(tensor_blocks[d]));
					if (tensor_blocks[d].tail)
						for (k = 0; k < tensor_blocks[d].tail->rnum; k++)
						{
							const int incoming = *(int*)ccv_array_get(tensor_blocks[d].tail, k);
							if (incoming >= exec_symbol_info_size)
								continue;
							assert(incoming >= 0);
							assert(graph_execs[incoming].graph);
							ccv_nnc_graph_exec_concat(graph, graph_execs[incoming], set_exec);
							flags = 1;
						}
				}
			// If cannot find a start node for this exec, we need to append it to the no-op of the start.
			if (!flags)
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
	// Now go through the list of tensors to see whether we need to do explicit broadcast for these tensor multi-views
	// (we need that if it is not associated as inputs / outputs of any execs, this is possible if all execs associate
	// with its alias).
	assert(tensor_arena->vt_tensor_size == graph_prep->tensor_symbol_info_size);
	for (i = 0; i < tensor_arena->vt_tensor_size; i++)
	{
		ccv_nnc_tensor_t* mv = tensor_arena->vt_tensors[i];
		// If it is multiview tensor, inspect all its head to see whether we already associated with the node.
		if (mv && CCV_IS_TENSOR_MULTIVIEW(mv))
		{
			const ccv_array_t* const head = tensor_blocks[i].head;
			if (head && head->rnum > 0)
				for (j = 0; j < head->rnum; j++)
				{
					const int idx = *(int*)ccv_array_get(head, j);
					if (idx >= exec_symbol_info_size)
						continue;
					assert(idx >= 0);
					const int d = graph_execs[idx].d;
					ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, d);
					int flag = 0;
					for (k = 0; k < exec_info->tensor_wrap_size && !flag; k++)
						flag = (exec_info->tensor_wraps[k] && exec_info->tensor_wraps[k]->tensors[0] == mv);
					// If none is in the flag, it need to be included in the cast.
					if (!flag)
						ccv_nnc_graph_exec_add_update(graph, graph_execs[idx], mv);
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
	ccv_nnc_graph_set_sources(graph, &graph_exec_arena->source, 1);
	ccv_nnc_graph_set_destinations(graph, &graph_exec_arena->destination, 1);
	return graph_exec_arena;
}

static ccv_nnc_graph_t* _ccv_nnc_graph_find_peer(const ccv_nnc_symbolic_graph_prep_t* const graph_prep, const ccv_nnc_symbolic_graph_t* const peer)
{
	if (graph_prep->symbolic_graph == peer)
		return graph_prep->graph;
	int i;
	for (i = 0; i < graph_prep->sub_prep_size; i++)
		if (graph_prep->sub_preps[i])
		{
			ccv_nnc_graph_t* const graph = _ccv_nnc_graph_find_peer(graph_prep->sub_preps[i], peer);
			if (graph)
				return graph;
		}
	return 0;
}

static void _ccv_nnc_graph_fixup_peer(const ccv_nnc_symbolic_graph_prep_t* const root_prep, ccv_nnc_symbolic_graph_prep_t* const graph_prep)
{
	int i;
	for (i = 0; i < graph_prep->sub_prep_size; i++)
		if (graph_prep->sub_preps[i])
		{
			if (graph_prep->sub_preps[i]->symbolic_graph->peer)
				graph_prep->sub_preps[i]->graph->peer = _ccv_nnc_graph_find_peer(root_prep, graph_prep->sub_preps[i]->symbolic_graph->peer);
		}
}

static void _ccv_nnc_graph_exec_arena_fixup_peer_ref(const ccv_nnc_graph_exec_arena_t* const root_arena, const ccv_nnc_symbolic_graph_prep_t* const graph_prep, ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	assert(graph_exec_arena->graph_ref == (intptr_t)graph_prep->symbolic_graph);
	int i;
	for (i = 0; i < graph_prep->exec_symbol_info_size; i++)
	{
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(graph_prep->exec_symbol_info[i].flags))
			continue;
		if (graph_exec_arena->graph_execs[i].graph && graph_prep->exec_symbol_info[i].peer_ref)
		{
			ccv_nnc_graph_exec_t peer_exec = ccv_nnc_graph_exec_from_symbol(root_arena, (ccv_nnc_graph_exec_symbol_t){
				.d = graph_prep->exec_symbol_info[i].peer_ref - 1,
				.graph = graph_prep->symbolic_graph->peer ? graph_prep->symbolic_graph->peer : graph_prep->symbolic_graph,
			});
			if (peer_exec.d >= 0)
				ccv_nnc_graph_exec_set_peer(graph_prep->graph, graph_exec_arena->graph_execs[i], peer_exec);
		}
	}
	for (i = 0; i < graph_prep->sub_prep_size; i++)
		if (graph_prep->sub_preps[i])
			_ccv_nnc_graph_exec_arena_fixup_peer_ref(root_arena, graph_prep->sub_preps[i], graph_exec_arena->sub_arenas[i]);
}

static void _ccv_nnc_symbolic_graph_prep_dup_breakpoints_free(ccv_nnc_symbolic_graph_prep_t* const graph_prep)
{
	int i;
	if (graph_prep->dup_breakpoints)
	{
		// Strip the const modifier only possible because it is a sub-graph.
		ccv_nnc_symbolic_graph_t* const symbolic_graph = (ccv_nnc_symbolic_graph_t*)graph_prep->symbolic_graph;
		for (i = 0; i < graph_prep->dup_breakpoints->rnum; i++)
			ccv_nnc_graph_exec_symbol_free(symbolic_graph, *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(graph_prep->dup_breakpoints, i));
		ccv_array_free(graph_prep->dup_breakpoints);
		graph_prep->dup_breakpoints = 0;
		graph_prep->exec_symbol_info_size = symbolic_graph->exec_symbol_info->rnum;
		// Afterwards, we have to regenerate the exec_symbol_info, fill in the information (through symbol_infer).
		memcpy(graph_prep->exec_symbol_info, ccv_array_get(symbolic_graph->exec_symbol_info, 0), sizeof(ccv_nnc_graph_exec_symbol_info_t) * graph_prep->exec_symbol_info_size);
		// Since exec_symbol_info changed, create a new visit object.
		assert(symbolic_graph->sources);
		assert(symbolic_graph->destinations);
		ccv_nnc_graph_exec_symbol_t* const sources = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(symbolic_graph->sources, 0);
		const int source_size = symbolic_graph->sources->rnum;
		ccv_nnc_graph_exec_symbol_t* const destinations = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(symbolic_graph->destinations, 0);
		const int destination_size = symbolic_graph->destinations->rnum;
		ccv_nnc_graph_visit_t* visit = ccv_nnc_graph_visit_new(symbolic_graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(symbolic_graph->exec_symbol_info, 0), symbolic_graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
		ccv_nnc_graph_visit_free(graph_prep->visit);
		graph_prep->visit = visit;
		assert(graph_prep->p);
		ccv_nnc_symbolic_graph_symbol_infer(symbolic_graph, visit, sources, source_size, destinations, destination_size, graph_prep->p->tensor_symbol_info, graph_prep->p->tensor_symbol_info_size, graph_prep->tensor_symbol_info, graph_prep->exec_symbol_info);
	}
	ccv_nnc_graph_visit_for(graph_prep->visit, graph_prep->exec_symbol_info, node, idx) {
		for (i = 0; i < node->graph_ref_size; i++)
		{
			const int graph_ref = CCV_NNC_GRAPH_REF(node)[i] - 1;
			if (graph_ref >= 0)
				_ccv_nnc_symbolic_graph_prep_dup_breakpoints_free(graph_prep->sub_preps[graph_ref]);
		}
	} ccv_nnc_graph_visit_endfor
}

void ccv_nnc_symbolic_graph_compile(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, ccv_nnc_graph_t** const graph_ref, ccv_nnc_tensor_arena_t** const tensor_arena_ref, ccv_nnc_graph_exec_arena_t** const graph_exec_arena_ref)
{
	assert(graph_ref);
	assert(tensor_arena_ref);
	assert(graph_exec_arena_ref);
	int i;
	// Cannot bind the multi-view.
	for (i = 0; i < tensor_bind_size; i++)
	{
		assert(tensor_binds[i].tensor);
		assert(!CCV_IS_TENSOR_MULTIVIEW(tensor_binds[i].tensor));
	}
	ccv_nnc_symbolic_graph_prep_t* graph_prep = _ccv_nnc_symbolic_graph_prep_new(symbolic_graph, tensor_binds, tensor_bind_size, outputs, output_size, sources, source_size, destinations, destination_size, 0, 0, 0, 0);
	_ccv_nnc_symbolic_graph_prep_while_count_tensor(graph_prep);
	ccv_nnc_tensor_arena_t* tensor_arena = _ccv_nnc_tensor_arena_new(graph_prep, 0, tensor_binds, tensor_bind_size);
	_ccv_nnc_tensor_arena_fixup_peer_ref_and_tape_var(tensor_arena, graph_prep, tensor_arena);
	*tensor_arena_ref = tensor_arena;
	// The above handled tensor allocation, now we need to materialize the graph from symbolic to real.
	_ccv_nnc_graph_fixup_peer(graph_prep, graph_prep);
	// Now tensor allocation is done, if there are any dup_breakpoints, I need to clean it up.
	_ccv_nnc_symbolic_graph_prep_dup_breakpoints_free(graph_prep);
	*graph_ref = graph_prep->graph;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = _ccv_nnc_graph_exec_arena_new(symbolic_graph, sources, source_size, destinations, destination_size, graph_prep, tensor_arena);
	_ccv_nnc_graph_exec_arena_sequential(graph_prep->graph, graph_exec_arena);
	_ccv_nnc_graph_exec_arena_fixup_peer_ref(graph_exec_arena, graph_prep, graph_exec_arena);
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
	for (i = 0; i < tensor_arena->m_tensor_idx->rnum; i++)
	{
		ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)_ccv_nnc_tensor_metadata_get(tensor_arena->tensor_metadata, *(int*)ccv_array_get(tensor_arena->m_tensor_idx, i));;
		assert(mv && CCV_IS_TENSOR_MULTIVIEW(mv));
		ccv_nnc_tensor_multiview_free(*mv);
	}
	ccv_array_free(tensor_arena->tensor_metadata);
	ccv_array_free(tensor_arena->m_tensor_idx);
	ccfree(tensor_arena);
}

int ccv_nnc_tensor_bind_symbol(const ccv_nnc_tensor_arena_t* const tensor_arena, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_t* const tensor)
{
	assert(tensor_arena->graph_ref == (intptr_t)symbol.graph);
	assert(symbol.d < tensor_arena->vt_tensor_size);
	tensor_arena->vt_tensors[symbol.d]->data.ptr = tensor->data.ptr;
	return 0;
}

void ccv_nnc_tensor_arena_free(ccv_nnc_tensor_arena_t* const tensor_arena)
{
	int i;
	for (i = 0; i < tensor_arena->buffer_size; i++)
	{
		const int memory_type = CCV_TENSOR_GET_MEMORY(tensor_arena->buffers[i].type);
#ifdef HAVE_CUDA
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(tensor_arena->buffers[i].type);
		if (memory_type == CCV_TENSOR_GPU_MEMORY)
			cufree(device_id, tensor_arena->buffers[i].ptr);
		else {
			assert(memory_type == CCV_TENSOR_CPU_MEMORY);
			ccfree(tensor_arena->buffers[i].ptr);
		}
#else
		assert(memory_type == CCV_TENSOR_CPU_MEMORY);
		ccfree(tensor_arena->buffers[i].ptr);
#endif
	}
	_ccv_nnc_tensor_arena_free(tensor_arena);
}

void ccv_nnc_graph_exec_arena_free(ccv_nnc_graph_exec_arena_t* const graph_exec_arena)
{
	int i;
	for (i = 0; i < graph_exec_arena->sub_arena_size; i++)
		if (graph_exec_arena->sub_arenas[i])
			ccv_nnc_graph_exec_arena_free(graph_exec_arena->sub_arenas[i]);
	ccfree(graph_exec_arena);
}
