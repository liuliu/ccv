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

typedef struct {
	// Start for while loop handling
	int assign_ref; // Reference to the tensor that the value will be copied from (for parameter passing). Starts at 1.
	int p_ref; // Reference to the tensor number in its parent graph. Starts at 1.
	ccv_array_t* s_ref; // Reference to the tensor number in its sub graphs, Starts at 1.
	// End of while loop handling.
	int alias_ref; // Reference to the tensor. Starts at 1.
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
	ccv_array_t* outgoings; // Outgoing nodes
	int graph_ref; // Reference to the sub-graph. Starts at 1.
	int dead; // Mark this node as dead.
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
	char* name;
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
	struct ccv_nnc_symbolic_graph_s* p; // The parent graph (if current one is a sub-graph).
	int p_idx; // Reference to the index in its parent graph's sub-graph array, Starts at 1.
	int exec_idx; // Reference to the index in its parent graph's exec (the graph exec), Starts at 1.
	// Why some of these I choose to be flat int* array, some of these I choose to be ccv_array_t?
	// for flat int* array, these are not going to be modified until next time call ccv_nnc_symbolic_graph_backward
	// for ccv_array_t, we can continue to modify what's inside.
	int cond_eval_size;
	ccv_nnc_graph_exec_symbol_t* cond_evals;
	ccv_nnc_graph_while_f while_expr;
	const void* while_data;
	// Map between parent / sub-graph's tensor symbols.
	// End of while loop handling.
	// Start for backward (automatic differentiation) handling
	int forward_symbol_size;
	int* backward_tensor_symbols;
	int backward_symbol_size;
	int* backward_exec_symbols;
	// End of backward (automatic differentiation) handling.
};

struct ccv_nnc_tensor_arena_s {
	int memory_type;
	int device_id;
	intptr_t graph_ref; // A value contains the pointer name of the graph.
	int sub_arena_size;
	struct ccv_nnc_tensor_arena_s** sub_arenas; // Corresponding to sub graphs.
	// This is a table of tensor references to real allocated tensors.
	int vt_tensor_size;
	ccv_nnc_tensor_t** vt_tensors;
	// This is the allocated non-continuous buffers.
	int buffer_size;
	struct {
		uint64_t size;
		uint8_t* ptr;
	}* buffers;
	// Real allocated tensor header metadata (this is a mixed pool of ccv_tensor_t, ccv_tensor_view_t,
	// ccv_tensor_multiview_t, thus, it is aligned to a 16-byte boundary).
	ccv_array_t* tensor_metadata;
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

inline static void ccv_array_replace_int(ccv_array_t* ints, const int idx, const int outgoing)
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

void ccv_nnc_symbolic_graph_symbol_infer(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_info_t* const p_tensor_symbol_info, const int p_tensor_symbol_info_size, ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info);

void ccv_nnc_symbolic_graph_add_source(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t source);
void ccv_nnc_symbolic_graph_add_destination(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t destination);

#endif

