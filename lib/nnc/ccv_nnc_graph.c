#include "ccv_nnc.h"

ccv_nnc_net_graph_t* ccv_nnc_net_graph_new(const ccv_nnc_net_node_t node, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	return 0;
}

ccv_nnc_net_graph_t* ccv_nnc_net_graph_concat(ccv_nnc_net_graph_t* const* inputs, const int input_size, const ccv_nnc_net_graph_t* output)
{
	return 0;
}

void ccv_nnc_net_graph_run(const ccv_nnc_net_graph_t* graph)
{
}

void ccv_nnc_net_graph_free(ccv_nnc_net_graph_t* graph)
{
}
