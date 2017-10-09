/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#include "ccv_nnc.h"

// These are data structure that directly take pointers.
typedef struct {
	uint64_t while_max_count;
	ccv_numeric_data_t* data;
} ccv_nnc_tape_tensor_data_t;

typedef struct ccv_nnc_tape_graph_data_s {
	uint64_t while_max_count; // Default to 1.
	ccv_nnc_tape_tensor_data_t* tensor_data; // ??? I need a hash table for this.
	int sub_graph_data_size;
	struct ccv_nnc_tape_graph_data_s* sub_graph_data;
} ccv_nnc_tape_graph_data_t;

// Tape's structure mimics the graph, but it only uses that to index into specific tensor.
struct ccv_nnc_tensor_tape_s {
	ccv_nnc_tape_graph_data_t* graph_data;
};
