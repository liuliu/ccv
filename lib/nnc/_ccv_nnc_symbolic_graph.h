/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_symbolic_graph_internal_h
#define GUARD_ccv_nnc_symbolic_graph_internal_h

#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"

typedef struct {
	// Start for while loop handling
	int assign_ref; // Reference to the tensor that the value will be copied from (for parameter passing). Starts at 1.
	int r_assign_ref; // It is a reverse of the assign_ref. Starts at 1.
	int bypass_ref; // Some exec may not generate output for this tensor. In that case, use the content from tensor bypass (typical case for case..of). Starts at 1.
	int r_bypass_ref; // It is a reverse of the bypass_ref. Starts at 1.
	int p_ref; // Reference to the tensor number in its parent graph. Starts at 1.
	// End of while loop handling.
	int alias_ref; // Reference to the tensor. Starts at 1.
	int peer_ref; // Reference to its peer. Starts at 1.
	int flags;
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_array_t* s_ref; // Reference to the tensor number in its sub graphs, Starts at 1.
	char* name;
	ccv_nnc_tensor_param_t info;
} ccv_nnc_tensor_symbol_info_t;

enum {
	CCV_NNC_GRAPH_EXEC_CASE_OF_NO_BYPASS = 0x01, // If this flag is set, this case..of won't have any bypass.
};

typedef struct {
	int input_size;
	int output_size;
	int graph_ref_size;
	int flags; // Mark this node as dead.
	int peer_ref; // Reference to its peer. Starts at 1.
	int* inputs;
	int* outputs;
	ccv_array_t* outgoings; // Outgoing nodes
	char* name;
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
	// Below are only relevant to sub-graph nodes (case_of, while).
	int _inline_graph_ref[2]; // Reference to the sub-graph. Starts at 1.
	int* _heap_graph_ref;
	union {
		struct {
			ccv_nnc_graph_case_of_f expr;
			const void* data;
			int flags;
			struct {
				int offset;
				int size;
			} argument; // range for which data as inputs from input section.
		} case_of;
		struct {
			ccv_nnc_graph_while_f expr;
			const void* data;
			int* inputs;
			int input_size;
		} p_while;
	};
} ccv_nnc_graph_exec_symbol_info_t;

struct ccv_nnc_symbolic_graph_s {
	ccv_array_t* tensor_symbol_info; // A lit of info for tensor symbols.
	ccv_array_t* exec_symbol_info; // A list of info for exec symbols.
	// I think that I can be more explicit about which are sources and which are destinations.
	ccv_array_t* sources;
	ccv_array_t* destinations;
	// Some extra information piggy-back on symbolic graph struct.
	// Start for while loop handling
	ccv_array_t* sub_graphs; // A list of its sub-graphs (for while loop).
	struct ccv_nnc_symbolic_graph_s* peer; // The peer graph (only useful for backward prop graph).
	struct ccv_nnc_symbolic_graph_s* p; // The parent graph (if current one is a sub-graph).
	int p_idx; // Reference to the index in its parent graph's sub-graph array, Starts at 1.
	int exec_idx; // Reference to the index in its parent graph's exec (the graph exec), Starts at 1.
	// Why some of these I choose to be flat int* array, some of these I choose to be ccv_array_t?
	// for flat int* array, these are not going to be modified until next time call ccv_nnc_symbolic_graph_backward
	// for ccv_array_t, we can continue to modify what's inside.
	int breakpoint_size;
	ccv_nnc_graph_exec_symbol_t* breakpoints;
	// End of while loop handling.
	struct {
		int tensor;
		int exec;
	} reuse; // The reuse slot for tensor or graph exec symbols.
	// Start for backward (automatic differentiation) handling
	struct {
		int tensor_symbol_size;
		int* tensor_symbol_idx;
		int exec_symbol_size;
		int* exec_symbol_idx;
	} backward;
	// End of backward (automatic differentiation) handling.
	// For parallel, get duplicated tensors.
	struct {
		int count;
		int tensor_symbol_size;
		int* tensor_symbol_idx;
		int exec_symbol_size;
		int* exec_symbol_idx;
	} data_parallel;
	// Hooks
	struct {
		struct {
			ccv_nnc_tensor_symbol_new_hook_f func;
			void* context;
		} tensor_symbol_new;
		struct {
			ccv_nnc_tensor_symbol_alias_new_hook_f func;
			void* context;
		} tensor_symbol_alias_new;
		struct {
			ccv_nnc_graph_exec_symbol_new_hook_f func;
			void* context;
		} graph_exec_symbol_new;
	} hooks;
};

struct ccv_nnc_tensor_arena_s {
	intptr_t graph_ref; // A value contains the pointer name of the graph.
	int sub_arena_size;
	struct ccv_nnc_tensor_arena_s** sub_arenas; // Corresponding to sub graphs.
	// This is a table of tensor references to real allocated tensors.
	int vt_tensor_size;
	ccv_nnc_tensor_t** vt_tensors;
	// This is the allocated non-continuous buffers.
	int buffer_size;
	struct {
		int type; // The type from tensor blocks.
		int pin_mem; // Whether this memory is pinned.
		uint64_t size;
		uint8_t* ptr;
	}* buffers;
	// Real allocated tensor header metadata (this is a mixed pool of ccv_tensor_t, ccv_tensor_view_t,
	// ccv_tensor_multiview_t, thus, it is aligned to a 16-byte boundary).
	ccv_array_t* tensor_metadata;
	ccv_array_t* m_tensor_idx; // The index into multi-view tensors in tensor_metadata.
};

struct ccv_nnc_graph_exec_arena_s {
	intptr_t graph_ref; // A value contains the pointer name of the graph.
	int sub_arena_size;
	struct ccv_nnc_graph_exec_arena_s** sub_arenas; // Corresponding to sub graphs.
	ccv_nnc_graph_exec_t source;
	ccv_nnc_graph_exec_t destination;
	int graph_exec_size;
	ccv_nnc_graph_exec_t graph_execs[1];
};

#define CCV_NNC_ENCODE_WHILE_COUNT_SYMBOL(d) ((int)((~(uint32_t)d) << 4 | 0xe))
#define CCV_NNC_DECODE_WHILE_COUNT_SYMBOL(symbol) ((~(uint32_t)(symbol)) >> 4)

inline static void ccv_array_replace_unique_int(ccv_array_t* ints, const int idx, const int outgoing)
{
	int i;
	int flag = 0;
	for (i = 0; i < ints->rnum;)
	{
		if (*(int*)ccv_array_get(ints, i) == idx)
		{
			if (flag)
			{
				if (i < ints->rnum - 1)
					*(int*)ccv_array_get(ints, i) = *(int*)ccv_array_get(ints, ints->rnum - 1);
				--ints->rnum;
				continue;
			}
			*(int*)ccv_array_get(ints, i) = outgoing;
			flag = 1;
		} else if (*(int*)ccv_array_get(ints, i) == outgoing) {
			// Remove this from the list.
			if (flag)
			{
				if (i < ints->rnum - 1)
					*(int*)ccv_array_get(ints, i) = *(int*)ccv_array_get(ints, ints->rnum - 1);
				--ints->rnum;
				continue;
			}
			flag = 1;
		}
		++i;
	}
	if (!flag)
		ccv_array_push(ints, &outgoing);
}

void ccv_nnc_symbolic_graph_symbol_infer(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_visit_t* const visit, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info);

void ccv_nnc_symbolic_graph_add_source(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t source);
void ccv_nnc_symbolic_graph_add_destination(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t destination);
int ccv_nnc_over_tensor_symbol_aliases(const ccv_nnc_tensor_symbol_info_t* const tensor_a, const ccv_nnc_tensor_symbol_info_t* const tensor_b);
int ccv_nnc_tensor_symbol_map_raw(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol);

#endif

