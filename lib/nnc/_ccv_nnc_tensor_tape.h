/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#include "ccv_nnc.h"

typedef struct {
	int input_size;
	int output_size;
	ccv_nnc_tensor_t** inputs;
	ccv_nnc_tensor_t** outputs;
} ccv_nnc_tape_graph_exec_inst_t;

typedef struct ccv_nnc_tape_graph_inst_s {
	int graph_exec_inst_size;
	ccv_nnc_tape_graph_exec_inst_t* graph_exec_insts;
	int sub_graph_inst_size;
	struct ccv_nnc_tape_graph_inst_s* sub_graph_insts;
} ccv_nnc_tape_graph_inst_t;

// These are data structure that directly take pointers.
typedef struct {
	int input_size;
	int output_size;
	ccv_numeric_data_t* inputs;
	ccv_numeric_data_t* outputs;
} ccv_nnc_tape_graph_exec_data_t;

typedef struct ccv_nnc_tape_graph_data_s {
	uint64_t while_max_count; // Default to 1.
	int graph_exec_data_size; // This maps to the same exec of the underlying graph.
	ccv_nnc_tape_graph_exec_data_t* graph_exec_data;
	int sub_graph_data_size;
	struct ccv_nnc_tape_graph_data_s* sub_graph_data;
} ccv_nnc_tape_graph_data_t;

// Tape's structure mimics the graph, but it only uses that to index into specific tensor.
struct ccv_nnc_tensor_tape_s {
	ccv_nnc_tape_graph_inst_t* graph_inst;
	ccv_nnc_tape_graph_data_t* graph_data;
};
