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
	ccv_array_t* while_graphs; // A list of its sub-graphs (for while loop).
	struct ccv_nnc_symbolic_graph_s* p; // The parent graph (if current one is a sub-graph).
	// Why some of these I choose to be flat int* array, some of these I choose to be ccv_array_t?
	// for flat int* array, these are not going to be modified until next time call ccv_nnc_symbolic_graph_backward
	// for ccv_array_t, we can continue to modify what's inside.
	ccv_array_t* conditionals;
	ccv_nnc_graph_while_f while_func;
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

void ccv_nnc_symbolic_graph_symbol_organize(const ccv_nnc_symbolic_graph_t* const symbolic_graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info);

#endif

