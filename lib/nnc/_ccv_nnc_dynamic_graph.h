/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_dynamic_graph_internal_h
#define GUARD_ccv_nnc_dynamic_graph_internal_h

#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"


struct ccv_nnc_tensor_variable_s {
	int index;
	int alias_ref;
	ccv_nnc_tensor_param_t info;
	ccv_nnc_tensor_symbol_t symbol;
	ccv_nnc_tensor_view_t* tensor_view;
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
};

enum {
	CCV_NNC_TENSOR_NO_VARIABLE = -1,
	CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED = -2,
};

typedef struct { // Extra information kept per tensor symbol along with symbolic graph.
	int index; // The index back into the tensor variable. -1 meant no associated tensor vairable.
	ccv_array_t* sources; // array of graph_exec_symbol, use this tensor symbol as output.
	ccv_array_t* destinations; // array of graph_exec_symbol, use this tensor symbol as input.
	ccv_nnc_tensor_view_t* tensor_view; // Transfer ownership of the tensor view to here.
} ccv_nnc_tensor_variable_graph_bind_t;

struct ccv_nnc_dynamic_graph_s {
	int reuse_var; // -1 if no var can be reused. Otherwise first locate the reuse var without increase array size.
	ccv_array_t* vars; // Array keeps track of all allocated tensor variable.
	ccv_array_t* binds; // Array keeps track of extra information for a tensor symbol.
	ccv_nnc_symbolic_graph_t* tape; // Symbolic graph to keep track of computation.
	ccv_array_t* ws; // array of integers as workspace
};

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_exchange_new(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable);

#endif
