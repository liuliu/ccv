#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_graph.h"

void ccv_nnc_tensor_multiview(ccv_nnc_tensor_t* data[], const uint8_t kind, const uint16_t repeat, const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	assert(kind == CCV_NNC_MULTIVIEW_K0N || kind == CCV_NNC_MULTIVIEW_K1N);
	assert(repeat > 0);
	tensor_multiview->type = CCV_TENSOR_MULTIVIEW;
	tensor_multiview->kind = kind;
	tensor_multiview->repeat = repeat;
	tensor_multiview->anchor = (intptr_t)graph;
	tensor_multiview->it = 0;
	tensor_multiview->p = 0;
	tensor_multiview->offset = 0;
	tensor_multiview->sp = 0;
	tensor_multiview->_heap_data = (repeat + kind <= sizeof(tensor_multiview->_inline_data) / sizeof(tensor_multiview->_inline_data[0])) ? 0 : ccmalloc(sizeof(ccv_nnc_tensor_t*) * (repeat + kind));
	int i;
	// Currently, only CCV_NNC_MULTIVIEW_K12 uses 3 tensors.
	for (i = 0; i < repeat + kind; i++)
	{
		CCV_NNC_MULTIVIEW_DATA(tensor_multiview)[i] = data[i];
		ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)data[i];
		if (data[i] != CCV_NNC_TENSOR_PLACEHOLDER && CCV_IS_TENSOR_MULTIVIEW(mv))
			mv->p = tensor_multiview;
	}
}

void ccv_nnc_tensor_multiview_free(const ccv_nnc_tensor_multiview_t tensor_multiview)
{
	if (tensor_multiview.sp)
		ccv_array_free(tensor_multiview.sp);
	if (tensor_multiview._heap_data)
		ccfree(tensor_multiview._heap_data);
}

void ccv_nnc_tensor_synchronize_to_multiview(ccv_nnc_tensor_multiview_t* const tensor_multiview, ccv_nnc_tensor_t* const tensor)
{
	if (!tensor_multiview->sp)
		tensor_multiview->sp = ccv_array_new(sizeof(ccv_nnc_tensor_t*), 0, 0);
	ccv_array_push(tensor_multiview->sp, &tensor);
}

void ccv_nnc_tensor_multiview_synchronize(ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	assert(tensor_multiview->it && !CCV_IS_TENSOR_MULTIVIEW(tensor_multiview->it));
	// Update the pointer on tv only if it is not a single tensor pointer.
	unsigned char* const data = tensor_multiview->it->data.u8 - tensor_multiview->offset;
	const ccv_nnc_tensor_multiview_t* c = tensor_multiview;
	int i;
	do {
		if (c->sp)
			for (i = 0; i < c->sp->rnum; i++)
			{
				ccv_nnc_tensor_t* const tensor = *(ccv_nnc_tensor_t**)ccv_array_get(c->sp, i);
				if (CCV_IS_TENSOR_VIEW(tensor))
				{
					ccv_nnc_tensor_view_t* const tensor_view = (ccv_nnc_tensor_view_t*)tensor;
					tensor_view->data.u8 = data + tensor_view->off;
				} else
					tensor->data.u8 = data;
			}
		c = c->p;
	} while (c);
}

ccv_nnc_graph_exec_t ccv_nnc_graph_while(ccv_nnc_graph_t* const graph, const uint32_t cmd, ccv_nnc_graph_t* const while_graph)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	ccv_nnc_graph_exec_t while_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_info_t* while_exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, while_exec.d);
	while_exec_info->flags |= CCV_NNC_GRAPH_EXEC_P_WHILE;
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_graph_t*), 1, 0);
	int i;
	if (while_graph->exec_wraps)
	{
		// Copy wraps from sub graph to parent graph.
		if (!graph->exec_wraps)
			graph->exec_wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), while_graph->exec_wraps->rnum, 0);
		for (i = 0; i < while_graph->exec_wraps->rnum; i++)
			ccv_array_push(graph->exec_wraps, ccv_array_get(while_graph->exec_wraps, i));
	}
	ccv_array_push(graph->sub_graphs, &while_graph);
	while_graph->p = graph;
	while_graph->p_idx = graph->sub_graphs->rnum;
	while_graph->exec_idx = while_exec.d + 1;
	while_exec_info->graph_ref_size = 1;
	CCV_NNC_GRAPH_REF(while_exec_info)[0] = graph->sub_graphs->rnum;
	return while_exec;
}

ccv_nnc_graph_t* ccv_nnc_graph_from_graph_exec(const ccv_nnc_graph_t* const graph, ccv_nnc_graph_exec_t exec)
{
	assert(exec.graph == graph);
	assert(exec.d < graph->exec_info->rnum);
	assert(graph->sub_graphs);
	ccv_nnc_graph_exec_info_t* exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, exec.d);
	assert(CCV_NNC_GRAPH_REF(exec_info)[0]);
	const int graph_ref = CCV_NNC_GRAPH_REF(exec_info)[0] - 1;
	assert(graph_ref < graph->sub_graphs->rnum);
	ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, graph_ref);
	return sub_graph;
}

ccv_nnc_tensor_t ccv_nnc_tensor_for_while_count(const ccv_nnc_graph_t* const while_graph)
{
	return ccv_nnc_tensor(&while_graph->while_count, ONE_CPU_TENSOR(1), 0);
}

void ccv_nnc_graph_set_while_expr(ccv_nnc_graph_t* const while_graph, const ccv_nnc_graph_while_f while_expr, const void* const while_data, ccv_nnc_tensor_t* const* const inputs, const int input_size, const ccv_nnc_graph_exec_t* const breakpoints, const int breakpoint_size)
{
	assert(while_graph->p);
	const int exec_idx = while_graph->exec_idx - 1;
	assert(exec_idx >= 0 && exec_idx < while_graph->p->exec_info->rnum);
	ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(while_graph->p->exec_info, exec_idx);
	exec_info->p_while.expr = while_expr;
	exec_info->p_while.data = while_data;
	if (input_size > 0)
	{
		exec_info->p_while.input_size = input_size;
		exec_info->p_while.inputs = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * input_size);
		memcpy(exec_info->p_while.inputs, inputs, sizeof(ccv_nnc_tensor_t*) * input_size);
	}
	assert(breakpoint_size > 0);
	while_graph->breakpoint_size = breakpoint_size;
	while_graph->breakpoints = (ccv_nnc_graph_exec_t*)((while_graph->breakpoints) ? ccrealloc(while_graph->breakpoints, sizeof(ccv_nnc_graph_exec_t) * breakpoint_size) : ccmalloc(sizeof(ccv_nnc_graph_exec_t) * breakpoint_size));
	memcpy(while_graph->breakpoints, breakpoints, sizeof(ccv_nnc_graph_exec_t) * breakpoint_size);
}
