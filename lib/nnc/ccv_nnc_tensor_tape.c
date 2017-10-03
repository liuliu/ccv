/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#include "_ccv_nnc_tensor_tape.h"
#include "_ccv_nnc_graph.h"

static void _ccv_nnc_tape_graph_inst_new(ccv_nnc_tape_graph_inst_t* const graph_inst, const ccv_nnc_graph_t* const graph)
{
	graph_inst->graph_exec_inst_size = graph->exec_info->rnum;
	graph_inst->graph_exec_insts = cccalloc(graph->exec_info->rnum, sizeof(ccv_nnc_tape_graph_exec_inst_t));
	graph_inst->sub_graph_inst_size = graph->sub_graphs ? graph->sub_graphs->rnum : 0;
	graph_inst->sub_graph_insts = graph_inst->sub_graph_inst_size ? ccmalloc(sizeof(ccv_nnc_tape_graph_inst_t) * graph_inst->sub_graph_inst_size) : 0;
	int i;
	for (i = 0; i < graph_inst->sub_graph_inst_size; i++)
		_ccv_nnc_tape_graph_inst_new(graph_inst->sub_graph_insts + i, *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, i));
}

static void _ccv_nnc_tape_graph_inst_free(ccv_nnc_tape_graph_inst_t* const graph_inst)
{
	if (graph_inst->graph_exec_insts)
		ccfree(graph_inst->graph_exec_insts);
	int i;
	for (i = 0; i < graph_inst->sub_graph_inst_size; i++)
		_ccv_nnc_tape_graph_inst_free(graph_inst->sub_graph_insts + i);
	ccfree(graph_inst->sub_graph_insts);
}

static void _ccv_nnc_tape_graph_data_new(ccv_nnc_tape_graph_data_t* const graph_data, const ccv_nnc_graph_t* const graph)
{
	graph_data->while_max_count = 1;
	graph_data->graph_exec_data_size = graph->exec_info->rnum;
	graph_data->graph_exec_data = cccalloc(graph->exec_info->rnum, sizeof(ccv_nnc_tape_graph_exec_data_t));
	graph_data->sub_graph_data_size = graph->sub_graphs ? graph->sub_graphs->rnum : 0;
	graph_data->sub_graph_data = graph_data->sub_graph_data_size ? ccmalloc(sizeof(ccv_nnc_tape_graph_data_t) * graph_data->sub_graph_data_size) : 0;
	int i;
	for (i = 0; i < graph_data->sub_graph_data_size; i++)
		_ccv_nnc_tape_graph_data_new(graph_data->sub_graph_data + i, *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, i));
}

static void _ccv_nnc_tape_graph_data_free(ccv_nnc_tape_graph_data_t* const graph_data)
{
	if (graph_data->graph_exec_data)
		ccfree(graph_data->graph_exec_data);
	int i;
	for (i = 0; i < graph_data->sub_graph_data_size; i++)
		_ccv_nnc_tape_graph_data_free(graph_data->sub_graph_data + i);
	ccfree(graph_data->sub_graph_data);
}

ccv_nnc_tensor_tape_t* ccv_nnc_tensor_tape_new(const ccv_nnc_graph_t* const graph)
{
	// Its parent should be nil (we make tape from the root graph).
	assert(graph->p == 0);
	ccv_nnc_tensor_tape_t* tape = (ccv_nnc_tensor_tape_t*)ccmalloc(sizeof(ccv_nnc_tensor_tape_t) + sizeof(ccv_nnc_tape_graph_inst_t) + sizeof(ccv_nnc_tape_graph_data_t));
	tape->graph_inst = (ccv_nnc_tape_graph_inst_t*)(tape + 1);
	tape->graph_data = (ccv_nnc_tape_graph_data_t*)(tape->graph_inst + 1);
	_ccv_nnc_tape_graph_inst_new(tape->graph_inst, graph);
	_ccv_nnc_tape_graph_data_new(tape->graph_data, graph);
	return tape;
}

void ccv_nnc_tensor_tape_io(ccv_nnc_tensor_tape_t* const tape, const ccv_nnc_graph_t* const graph, const int exec_index, const ccv_nnc_tensor_t** const inputs, const ccv_nnc_tensor_t** outputs, ccv_nnc_tensor_t*** const input_ref, ccv_nnc_tensor_t*** const output_ref)
{
}

void ccv_nnc_tensor_tape_free(ccv_nnc_tensor_tape_t* const tape)
{
	_ccv_nnc_tape_graph_inst_free(tape->graph_inst);
	_ccv_nnc_tape_graph_data_free(tape->graph_data);
	ccfree(tape);
}
