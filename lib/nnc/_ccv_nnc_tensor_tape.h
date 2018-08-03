/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_tensor_tape_internal_h
#define GUARD_ccv_nnc_tensor_tape_internal_h

#include "ccv_nnc.h"

typedef struct {
	int type;
	ccv_numeric_data_t data;
} ccv_nnc_tape_tensor_data_t;

// These are data structure that directly take pointers.
typedef struct {
	int dim_count;
	int* dim;
	ccv_nnc_tape_tensor_data_t* data;
} ccv_nnc_tape_tensor_data_array_t;

typedef struct {
	int dim_count;
	int* dim;
	uint64_t* data;
} ccv_nnc_tape_exec_data_array_t;

// Tape's structure mimics the graph, but it only uses that to index into specific tensor.
struct ccv_nnc_tensor_tape_s {
	ccv_array_t* tensor_data; // struct of ccv_nnc_tape_tensor_data_array_t
	ccv_array_t* exec_data; // struct of ccv_nnc_tape_exec_data_array_t
};

#endif
