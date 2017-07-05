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
	int input_size;
	int output_size;
	ccv_nnc_tensor_t** inputs;
	ccv_nnc_tensor_t** outputs;
	ccv_array_t* outgoings; // outgoing nodes
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
	int graph_ref; // Reference to the sub-graph. Starts at 1.
	int wraps; // How many wraps for the inputs / outputs (thus, the inputs and outputs must contain multi-view tensor).
	int wrap_ptr; // At which level of the wrap we are currently at. Starts at 0.
} ccv_nnc_graph_exec_info_t;

struct ccv_nnc_graph_s {
	ccv_array_t* exec_info; // deferred exec info
	// I think that I can be more explicit about which are sources and which are destinations.
	ccv_array_t* sources;
	ccv_array_t* destinations;
	// Extra information, this logs all the exec that need to be unwrapped (including all sub-graphs).
	ccv_array_t* wraps; // It contains a ccv_nnc_graph_exec_t struct. This points to execs that has wrapped nodes.
	// Some extra information piggy-back on graph struct.
	struct ccv_nnc_graph_s* p; // The parent graph (if current one is a sub-graph).
	int exec_idx; // Reference to the index in its parent graph's exec (the graph exec), Starts at 1.
	ccv_array_t* sub_graphs; // A list of its sub-graphs (for while loop).
	// Why some of these I choose to be flat * array, some of these I choose to be ccv_array_t?
	// for flat * array, these are not going to be modified until next time call ccv_nnc_symbolic_graph_backward
	// for ccv_array_t, we can continue to modify what's inside.
	int cond_eval_size;
	ccv_nnc_graph_exec_t* cond_evals;
	ccv_nnc_graph_while_f while_expr;
	const void* while_data;
	// End of while loop handling.
};

typedef struct {
	off_t offset; // If the tensor points to a tensor view, tensor->data.u8 - offset is the origin of the tensor.
	ccv_nnc_tensor_t* tensor;
} ccv_nnc_tensor_reference_t;

#endif
