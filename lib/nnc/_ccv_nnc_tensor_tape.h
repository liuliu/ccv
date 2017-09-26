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
	int input_size;
	int output_size;
	ccv_numeric_data_t* inputs;
	ccv_numeric_data_t* outputs;
} ccv_nnc_tape_graph_exec_data_t;

typedef struct ccv_nnc_tape_graph_data_s {
	int graph_exec_data_size;
	ccv_nnc_tape_graph_exec_data_t* graph_exec_data;
	int sub_graph_data_size;
	struct ccv_nnc_tape_graph_data_s* sub_graph_data;
} ccv_nnc_tape_graph_data_t;

// Tape's structure mimics the graph, but it only uses that to index into specific tensor.
struct ccv_nnc_tensor_tape_s {
};

CCV_WARN_UNUSED(ccv_nnc_tensor_tape_t*) ccv_nnc_tensor_tape_new(void);
void ccv_nnc_tensor_tape_free(ccv_nnc_tensor_tape_t* const tape);
