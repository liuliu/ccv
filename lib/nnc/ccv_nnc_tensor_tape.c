/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#include "_ccv_nnc_tensor_tape.h"
#include "_ccv_nnc_graph.h"

static void _ccv_nnc_tape_graph_data_new(ccv_nnc_tape_graph_data_t* const graph_data, const ccv_nnc_graph_t* const graph)
{
	graph_data->while_max_count = 1;
	graph_data->sub_graph_data_size = graph->sub_graphs ? graph->sub_graphs->rnum : 0;
	graph_data->sub_graph_data = graph_data->sub_graph_data_size ? ccmalloc(sizeof(ccv_nnc_tape_graph_data_t) * graph_data->sub_graph_data_size) : 0;
	int i;
	for (i = 0; i < graph_data->sub_graph_data_size; i++)
		_ccv_nnc_tape_graph_data_new(graph_data->sub_graph_data + i, *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, i));
}

static void _ccv_nnc_tape_graph_data_free(ccv_nnc_tape_graph_data_t* const graph_data)
{
	int i;
	for (i = 0; i < graph_data->sub_graph_data_size; i++)
		_ccv_nnc_tape_graph_data_free(graph_data->sub_graph_data + i);
	ccfree(graph_data->sub_graph_data);
}

ccv_nnc_tensor_tape_t* ccv_nnc_tensor_tape_new(const ccv_nnc_graph_t* const graph)
{
	// Its parent should be nil (we make tape from the root graph).
	assert(graph->p == 0);
	ccv_nnc_tensor_tape_t* tape = (ccv_nnc_tensor_tape_t*)ccmalloc(sizeof(ccv_nnc_tensor_tape_t) + sizeof(ccv_nnc_tape_graph_data_t));
	tape->graph_data = (ccv_nnc_tape_graph_data_t*)(tape + 1);
	_ccv_nnc_tape_graph_data_new(tape->graph_data, graph);
	return tape;
}

void ccv_nnc_tensor_tape_io(ccv_nnc_tensor_tape_t* const tape, const ccv_nnc_graph_t* const graph, const int exec_index, ccv_nnc_tensor_t** const inputs, ccv_nnc_tensor_t** const outputs)
{
	// Go to the root graph, record which was taken along the way.
	const ccv_nnc_graph_t* curr_graph = graph;
	int d;
	for (d = 0; curr_graph; d++)
		curr_graph = curr_graph->p;
	curr_graph = graph;
	int trace[d];
	for (d = 0; curr_graph; d++)
	{
		const int p_idx = curr_graph->p_idx - 1;
		trace[d] = p_idx;
		curr_graph = curr_graph->p;
	}
	const ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec_index);
	int i, tape_io = 0;
	for (i = 0; i < exec_info->input_size && !tape_io; i++)
		if (CCV_GET_TAPE_ALLOC(inputs[i]->type))
			tape_io = 1;
	for (i = 0; i < exec_info->output_size && !tape_io; i++)
		if (CCV_GET_TAPE_ALLOC(outputs[i]->type))
			tape_io = 1;
	// If doesn't need to update with tape io, just pointing to the inputs and outputs directly.
	if (!tape_io)
		return;
	// Now, go through the inputs / outputs and update.
	for (i = 0; i < exec_info->input_size; i++)
		if (CCV_GET_TAPE_ALLOC(inputs[i]->type))
		{
		}
	for (i = 0; i < exec_info->output_size; i++)
		if (CCV_GET_TAPE_ALLOC(outputs[i]->type))
		{
		}
}

void ccv_nnc_tensor_tape_free(ccv_nnc_tensor_tape_t* const tape)
{
	_ccv_nnc_tape_graph_data_free(tape->graph_data);
	ccfree(tape);
}
