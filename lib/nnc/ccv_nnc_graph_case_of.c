#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"

ccv_nnc_graph_exec_t ccv_nnc_graph_case_of_new(ccv_nnc_graph_t* const graph, const uint32_t cmd, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const int argument_offset, const int argument_size)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	assert(argument_offset >= 0);
	assert(argument_offset + argument_size <= input_size);
	ccv_nnc_graph_exec_t exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, inputs, input_size, outputs, output_size);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	exec_info->flags |= CCV_NNC_GRAPH_EXEC_CASE_OF;
	exec_info->case_of.argument.offset = argument_offset;
	exec_info->case_of.argument.size = argument_size;
	int i, j;
	for (i = 0; i < output_size; i++)
		if (outputs[i] && ((ccv_nnc_tensor_multiview_t*)outputs[i])->anchor == CCV_NNC_MULTIVIEW_PHI)
			for (j = 0; j < ((ccv_nnc_tensor_multiview_t*)outputs[i])->kind + ((ccv_nnc_tensor_multiview_t*)outputs[i])->repeat; j++)
			{
				ccv_nnc_tensor_t* const mv = (ccv_nnc_tensor_t*)CCV_NNC_MULTIVIEW_DATA((ccv_nnc_tensor_multiview_t*)outputs[i])[j]->alias_ref;
				if (mv && CCV_IS_TENSOR_MULTIVIEW(mv))
					ccv_nnc_graph_exec_add_update(graph, exec, mv);
			}
	return exec;
}

void ccv_nnc_graph_set_case_of_expr(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_graph_case_of_f case_of, const void* case_of_data, const int offset)
{
	assert(exec.graph == graph);
	assert(exec.d >= 0);
	assert(exec.d < graph->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	assert(exec_info->flags == CCV_NNC_GRAPH_EXEC_CASE_OF);
	exec_info->case_of.data = case_of_data;
	exec_info->case_of.expr = case_of;
	exec_info->case_of.offset = offset;
}

void ccv_nnc_graph_set_case_of(ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, ccv_nnc_graph_t* const case_graph, const int case_of)
{
	assert(exec.graph == graph);
	assert(exec.d >= 0);
	assert(exec.d < graph->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	assert(exec_info->flags == CCV_NNC_GRAPH_EXEC_CASE_OF);
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_graph_t*), 1, 0);
	int i;
	if (case_graph->exec_wraps)
	{
		// Copy wraps from sub graph to parent graph.
		if (!graph->exec_wraps)
			graph->exec_wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), case_graph->exec_wraps->rnum, 0);
		for (i = 0; i < case_graph->exec_wraps->rnum; i++)
			ccv_array_push(graph->exec_wraps, ccv_array_get(case_graph->exec_wraps, i));
	}
	ccv_array_push(graph->sub_graphs, &case_graph);
	case_graph->p = graph;
	case_graph->p_idx = graph->sub_graphs->rnum;
	case_graph->exec_idx = exec.d + 1;
	// If case_of is larger than the inline graph_ref, we need to allocate.
	if (case_of >= sizeof(exec_info->_inline_graph_ref) / sizeof(exec_info->_inline_graph_ref[0]))
	{
		if (!exec_info->_heap_graph_ref)
		{
			exec_info->_heap_graph_ref = cccalloc(case_of + 1, sizeof(int));
			// Copy from inline data.
			memcpy(exec_info->_heap_graph_ref, exec_info->_inline_graph_ref, sizeof(exec_info->_inline_graph_ref));
			exec_info->graph_ref_size = case_of + 1;
		} else if (exec_info->graph_ref_size <= case_of) {
			exec_info->_heap_graph_ref = ccrealloc(exec_info->_heap_graph_ref, sizeof(int) * (case_of + 1));
			// Reset the newly allocated ones to 0.
			memset(exec_info->_heap_graph_ref + exec_info->graph_ref_size, 0, sizeof(int) * (case_of + 1 - exec_info->graph_ref_size));
			exec_info->graph_ref_size = case_of + 1;
		}
	} else
		exec_info->graph_ref_size = ccv_max(exec_info->graph_ref_size, case_of + 1);
	// Set the branch with the graph.
	CCV_NNC_GRAPH_REF(exec_info)[case_of] = graph->sub_graphs->rnum;
}
