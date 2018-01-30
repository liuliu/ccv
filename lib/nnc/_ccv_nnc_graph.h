/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_graph_internal_h
#define GUARD_ccv_nnc_graph_internal_h

#include "ccv_nnc.h"

typedef struct {
	int broadcast_required;
	int count;
	int index;
	ccv_nnc_tensor_t* tensors[1];
} ccv_nnc_graph_tensor_nest_t;

typedef struct {
	int input_size;
	int output_size;
	int flags;
	int peer_ref; // Reference to its peer. Starts at 1.
	int graph_ref_size;
	ccv_nnc_tensor_t** inputs;
	int* input_flags;
	ccv_nnc_tensor_t** outputs;
	int* output_flags;
	ccv_array_t* outgoings; // outgoing nodes
	intptr_t alias_ref; // Link to some reference data.
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
	// These correlates to tensors that need to be unwrapped, but not in either inputs / outputs (thus, only relevant if this graph exec symbol points to a sub-graph.)
	int broadcast_size;
	ccv_nnc_tensor_t** broadcasts;
	int tensor_nest_size; // This should be input_size + output_size + rest that need to be broadcast.
	ccv_nnc_graph_tensor_nest_t** tensor_nests;
	// Below are only relevant to sub-graph nodes (case_of, while).
	int _inline_graph_ref[2]; // Reference to the sub-graph. Starts at 1.
	int* _heap_graph_ref;
	union {
		struct {
			ccv_nnc_graph_case_of_f expr;
			const void* data;
			int offset;
		} case_of;
		struct {
			ccv_nnc_graph_while_f expr;
			const void* data;
		} p_while;
	};
} ccv_nnc_graph_exec_info_t;

struct ccv_nnc_graph_s {
	int p_idx; // Reference to the index in its parent graph's sub-graph array, Starts at 1.
	int exec_idx; // Reference to the index in its parent graph's exec (the graph exec), Starts at 1.
	ccv_array_t* exec_info; // deferred exec info
	// I think that I can be more explicit about which are sources and which are destinations.
	ccv_array_t* sources;
	ccv_array_t* destinations;
	// Extra information, this logs all the exec that need to be unwrapped (including all sub-graphs).
	ccv_array_t* nest_execs; // It contains a ccv_nnc_graph_exec_t struct. This points to execs that has nested tensors.
	// Some extra information piggy-back on graph struct.
	struct ccv_nnc_graph_s* p; // The parent graph (if current one is a sub-graph).
	struct ccv_nnc_graph_s* peer; // The peer graph (only useful for backward prop graph).
	ccv_array_t* sub_graphs; // A list of its sub-graphs (for while loop).
	// Why some of these I choose to be flat * array, some of these I choose to be ccv_array_t?
	// for flat * array, these are not going to be modified until next time call ccv_nnc_symbolic_graph_backward
	// for ccv_array_t, we can continue to modify what's inside.
	uint64_t while_count;
	int breakpoint_size;
	ccv_nnc_graph_exec_t* breakpoints;
	// End of while loop handling.
};

#endif
