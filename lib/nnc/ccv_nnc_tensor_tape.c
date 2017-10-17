/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#include "_ccv_nnc_tensor_tape.h"
#include "_ccv_nnc_graph.h"

ccv_nnc_tensor_tape_t* ccv_nnc_tensor_tape_new(const ccv_nnc_graph_t* const graph)
{
	// Its parent should be nil (we make tape from the root graph).
	assert(graph->p == 0);
	ccv_nnc_tensor_tape_t* tape = (ccv_nnc_tensor_tape_t*)ccmalloc(sizeof(ccv_nnc_tensor_tape_t));
	tape->tensor_data = ccv_array_new(sizeof(ccv_nnc_tape_tensor_data_t), 0, 0);
	return tape;
}

static ccv_nnc_tensor_t* _ccv_nnc_tensor_from_tensor_multiview(const ccv_nnc_graph_t* const* const graphs, const int graph_size, ccv_nnc_tensor_multiview_t* const mv)
{
	int i;
	ccv_nnc_tensor_t* tensor = (ccv_nnc_tensor_t*)mv;
	for (i = 0; i < graph_size; i++)
	{
		const int count = (int)graphs[i]->while_count;
		while (CCV_IS_TENSOR_MULTIVIEW(tensor) &&
			   ((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graphs[i])
		{
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
			const int off = mv->kind;
			const int mod = mv->repeat;
			// If reached the root.
			if (mv->tv)
				tensor = mv->tv;
			else
				tensor = (ccv_nnc_tensor_t*)mv->data[count >= off ? ((count - off) % mod) + off : count].ptr; // Unwrap.
		}
	}
	return tensor;
}

/*
// Simple allocator from ccv_array_t.
static int _ccv_nnc_tensor_metadata_pos_new(ccv_array_t* const tensor_metadata, const size_t size)
{
	int pos = tensor_metadata->rnum + 1; // Appending 1 so it starts from non-zero.
	int rsize = (size + 15) / 16;
	ccv_array_resize(tensor_metadata, pos + rsize - 1);
	return (pos << 1) + 1;
}

static ccv_numeric_data_t _ccv_nnc_tape_tensor_data_get(const ccv_array_t* const tensor_data, const int pos)
{
	assert((pos >> 1) <= tensor_data->rnum);
	return (ccv_nnc_)ccv_array_get(tensor_data, (pos >> 1) - 1);
}

#define CCV_NNC_IS_TAPE_TENSOR_DATA_POS(ptr) ((uintptr_t)(ptr) & 1)
*/

void ccv_nnc_tensor_tape_io(ccv_nnc_tensor_tape_t* const tape, const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i, tape_io = 0;
	for (i = 0; i < input_size && !tape_io; i++)
		if (CCV_GET_TAPE_ALLOC(inputs[i]->type))
			tape_io = 1;
	for (i = 0; i < output_size && !tape_io; i++)
		if (CCV_GET_TAPE_ALLOC(outputs[i]->type))
			tape_io = 1;
	// If doesn't need to update with tape io, just pointing to the inputs and outputs directly.
	if (!tape_io)
		return;
	// Go to the root graph, record which was taken along the way.
	// In this way, we can then unwrap multi-view tensors.
	const ccv_nnc_graph_t* curr_graph = graph;
	int d;
	for (d = 0; curr_graph; d++)
		curr_graph = curr_graph->p;
	curr_graph = graph;
	const int graph_size = d;
	const ccv_nnc_graph_t* graphs[graph_size];
	for (d = graph_size - 1; curr_graph; d--, curr_graph = curr_graph->p)
		graphs[d] = curr_graph;
	// Now, go through the inputs / outputs and update.
	for (i = 0; i < input_size; i++)
		if (CCV_GET_TAPE_ALLOC(inputs[i]->type))
		{
		}
	for (i = 0; i < output_size; i++)
		if (CCV_GET_TAPE_ALLOC(outputs[i]->type))
		{
		}
}

void ccv_nnc_tensor_tape_free(ccv_nnc_tensor_tape_t* const tape)
{
	ccv_array_free(tape->tensor_data);
	ccfree(tape);
}
