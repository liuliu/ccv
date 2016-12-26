#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "ccv_nnc_symbolic_graph_internal.h"

/**
 * Level-4 API
 */

int ccv_nnc_symbolic_graph_while(const ccv_nnc_graph_exec_symbol_t* sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* conditions, const int condition_size, const ccv_nnc_graph_exec_symbol_t* destinations, const int destination_size, ccv_nnc_graph_while_f while_func, const ccv_nnc_tensor_symbol_t* inputs, const int input_size, const ccv_nnc_tensor_symbol_t* outputs, const int output_size, const ccv_nnc_tensor_symbol_map_t* symbol_map, const int symbol_map_size)
{
	return 0;
}

int ccv_nnc_graph_while_run(const ccv_nnc_graph_t* graph, const ccv_nnc_tensor_tape_t* tensor_tape, const int flags, const ccv_nnc_graph_exec_t* sources, const int source_size, const ccv_nnc_graph_exec_t* destinations, const int destination_size)
{
	return 0;
}
