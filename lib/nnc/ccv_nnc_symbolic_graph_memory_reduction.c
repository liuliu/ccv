#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

// MARK - Level-3.5 API

static void _ccv_nnc_remove_unused_from_marked(const uint32_t* const tensor_used, const int size, uint32_t* const tensor_marked)
{
	int i;
	for (i = 0; i < size; i++)
		tensor_marked[i] &= tensor_used[i];
}

static ccv_sparse_matrix_t* _ccv_nnc_exec_dep_new(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_visit_t* const visit)
{
	ccv_sparse_matrix_t* exec_dep = ccv_sparse_matrix_new(graph->exec_symbol_info->rnum, graph->exec_symbol_info->rnum, CCV_32S | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	int* buf = (int*)ccmalloc(sizeof(int) * graph->exec_symbol_info->rnum * 2);
	int buf_size;
#define for_block(x, val) \
	do { \
		if (((int32_t*)val)[0] > 0) \
		{ \
			buf[buf_size * 2] = x; \
			buf[buf_size * 2 + 1] = ((int32_t*)val)[0] + 1; \
			++buf_size; \
		} \
	} while (0)
	const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
	int i, j;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx, term) {
		buf_size = 0; /* save all its parent deps to this buffer */
		ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx);
		if (vector)
			CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
		if (!node->outgoings)
			continue;
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			const int32_t one = 1;
			ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, outgoing, idx);
			/* If not found, set, if the current node is the destination node, no need 
			 * set itself as parent of subsequent nodes because its terminal nature. */
			if (!cell.i32 || cell.i32[0] == 0)
				ccv_set_sparse_matrix_cell(exec_dep, outgoing, idx, &one);
			if (buf_size > 0)
			{
				ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, outgoing);
				assert(vector);
				for (j = 0; j < buf_size; j++) /* set with all idx's dependencies as well */
				{
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell_from_vector(exec_dep, vector, buf[j * 2]);
					/* If not found, set */
					if (!cell.i32 || cell.i32[0] == 0)
						ccv_set_sparse_matrix_cell_from_vector(exec_dep, vector, buf[j * 2], &buf[j * 2 + 1]);
					else {
						/* Otherwise, set to the longest one */
						int32_t dep = ccv_max(cell.i32[0], buf[j * 2 + 1]);
						ccv_set_sparse_matrix_cell_from_vector(exec_dep, vector, buf[j * 2], &dep);
					}
				}
			}
		}
	} ccv_nnc_graph_visit_endfor
#undef for_block
	ccfree(buf);
	return exec_dep;
}

typedef struct {
	int okay;
	int original;
	ccv_nnc_tensor_param_t info;
	ccv_array_t* old_conversion_nodes;
	struct {
		ccv_array_t* sources;
		ccv_array_t* nodes;
	} reconversion;
} ccv_nnc_conversion_info_t;

typedef struct {
	ccv_array_t* outgoings;
} ccv_nnc_graph_exec_symbol_reverse_t;

void ccv_nnc_symbolic_graph_memory_reduction(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	// Note all these exec_symbol_info and tensor_symbol_info cannot be accessed once I start to mutate the graph. Therefore, I will do the
	// mutation at the last step, to carefully step away from that possibility.
	ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
	ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, 0);
	ccv_nnc_graph_visit_t* const visit = ccv_nnc_graph_visit_new(graph, exec_symbol_info, graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
	ccv_nnc_symbolic_graph_symbol_infer(graph, visit, sources, source_size, destinations, destination_size, 0, 0, tensor_symbol_info, exec_symbol_info);
	const int tensor_symbol_info_size = graph->tensor_symbol_info->rnum;
	const int exec_symbol_info_size = graph->exec_symbol_info->rnum;
	uint32_t* const tensor_marked = (uint32_t*)cccalloc(((tensor_symbol_info_size + 31) >> 5) * 2, sizeof(uint32_t));
	uint32_t* const tensor_used = tensor_marked + ((tensor_symbol_info_size + 31) >> 5);
	ccv_nnc_graph_exec_symbol_reverse_t* const reversed_nodes = cccalloc(exec_symbol_info_size, sizeof(ccv_nnc_graph_exec_symbol_reverse_t));
	int i, j, k;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		if (node->flags & CCV_NNC_GRAPH_EXEC_DEAD)
			continue;
		if (node->outgoings)
			for (i = 0; i < node->outgoings->rnum; i++)
			{
				const int d = *(int*)ccv_array_get(node->outgoings, i);
				if (!reversed_nodes[d].outgoings)
					reversed_nodes[d].outgoings = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_add_unique_int(reversed_nodes[d].outgoings, idx);
			}
		if (node->cmd.cmd == CCV_NNC_DATATYPE_CONVERSION_FORWARD && node->output_size >= 1 && node->outputs[0] >= 0)
		{
			const int d = node->outputs[0];
			// If this tensor is alias, or assigned (while loop), or bypassed (case..of), skip.
			if (tensor_symbol_info[d].alias_ref || tensor_symbol_info[d].assign_ref || tensor_symbol_info[d].bypass_ref ||
					tensor_symbol_info[d].r_assign_ref || tensor_symbol_info[d].r_bypass_ref)
				continue;
			tensor_marked[d >> 5] |= (1u << (d & 0x1f));
		} else if (ccv_nnc_cmd_is_backward(node->cmd))
			for (i = 0; i < node->input_size; i++)
			{
				const int d = node->inputs[i];
				if (d >= 0)
					tensor_used[d >> 5] |= (1u << (d & 0x1f));
			}
	} ccv_nnc_graph_visit_endfor
	// If a tensor is marked but never used in backward pass, no need to reduce it.
	_ccv_nnc_remove_unused_from_marked(tensor_used, (tensor_symbol_info_size + 31) >> 5, tensor_marked);
	// If this tensor is pointed to by an alias, we don't want to reconversion.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_symbol_info[i].alias_ref)
		{
			const int d = tensor_symbol_info[i].alias_ref - 1;
			// unmark.
			if ((tensor_marked[d >> 5] & (1u << (d & 0x1f))))
				tensor_marked[d >> 5] &= ~(1u << (d & 0x1f));
		}
	// Now tensor_marked only contains the tensors that we think beneficial to reconvert. Find the best place to insert conversion.
	ccv_nnc_conversion_info_t* const conversion_info = cccalloc(tensor_symbol_info_size, sizeof(ccv_nnc_conversion_info_t));
	ccv_sparse_matrix_t* const exec_dep = _ccv_nnc_exec_dep_new(graph, visit);
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		if (node->cmd.cmd == CCV_NNC_DATATYPE_CONVERSION_FORWARD && node->output_size >= 1 && node->outputs[0] >= 0)
		{
			const int d = node->outputs[0];
			if (d >= 0 && (tensor_marked[d >> 5] & (1u << (d & 0x1f))))
			{
				conversion_info[d].original = node->inputs[0];
				if (!conversion_info[d].old_conversion_nodes)
					conversion_info[d].old_conversion_nodes = ccv_array_new(sizeof(int), 0, 0);
				ccv_array_add_unique_int(conversion_info[d].old_conversion_nodes, idx);
			}
		} else if (ccv_nnc_cmd_is_backward(node->cmd))
			for (i = 0; i < node->input_size; i++)
			{
				const int d = node->inputs[i];
				if (d >= 0 && (tensor_marked[d >> 5] & (1u << (d & 0x1f))))
				{
					if (!conversion_info[d].reconversion.nodes)
						conversion_info[d].reconversion.nodes = ccv_array_new(sizeof(int), 0, 0);
					ccv_array_add_unique_int(conversion_info[d].reconversion.nodes, idx);
				}
			}
	} ccv_nnc_graph_visit_endfor
	for (i = 0; i < tensor_symbol_info_size; i++)
	{
		if (!conversion_info[i].reconversion.nodes)
			continue;
		// Check to see if it is beneficial for reconversion (i.e. the output is larger than the input).
		const int original_datatype = tensor_symbol_info[conversion_info[i].original].info.datatype;
		const int converted_datatype = tensor_symbol_info[i].info.datatype;
		if (CCV_GET_DATA_TYPE_SIZE(original_datatype) >= CCV_GET_DATA_TYPE_SIZE(converted_datatype))
			continue;
		// If we have more than one destination, need to find the common ancestor.
		ccv_array_t* const nodes = conversion_info[i].reconversion.nodes;
		ccv_array_t* const old_conversion_nodes = conversion_info[i].old_conversion_nodes;
		assert(nodes->rnum > 0);
		assert(old_conversion_nodes && old_conversion_nodes->rnum > 0);
		int flag = 0;
		for (j = 0; j < nodes->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(nodes, j);
			ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, d);
			assert(vector);
			for (k = 0; k < old_conversion_nodes->rnum; k++)
			{
				const int dd = *(int*)ccv_array_get(old_conversion_nodes, k);
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell_from_vector(exec_dep, vector, dd);
				if (cell.i32 && cell.i32[0] <= 3) // If the old conversion node to existing node has only hop distance of 3, no need to insert new conversion nodes. This is an empirical value.
					flag = 1;
			}
			if (flag)
				break;
		}
		// If there is no need to reconvert. Abort the whole thing.
		if (flag)
			continue;
		ccv_array_t* const reconversion_sources = ccv_array_new(sizeof(int), 0, 0);
		for (j = 0; j < nodes->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(nodes, j);
			ccv_array_t* const outgoings = reversed_nodes[d].outgoings;
			if (!outgoings)
				continue;
			int x, y;
			for (x = 0; x < outgoings->rnum; x++)
			{
				const int dd = *(int*)ccv_array_get(outgoings, x);
				int flag = 0;
				for (y = 0; !flag && y < nodes->rnum; y++)
				{
					if (j == y)
						continue;
					const int ddd = *(int*)ccv_array_get(nodes, y);
					// If the outgoing is one of the nodes, we cannot add it as source.
					if (ddd == dd)
					{
						flag = 1;
						continue;
					}
					// Check dependencies, if there is a dependency from y node to dd, dd cannot be source.
					const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, dd, ddd);
					if (cell.i32 && cell.i32[0] > 0)
						flag = 1;
				}
				if (!flag)
					ccv_array_add_unique_int(reconversion_sources, dd);
			}
		}
		// If there is no sources. Abort the whole thing.
		if (reconversion_sources->rnum == 0)
		{
			ccv_array_free(reconversion_sources);
			continue;
		}
		// Mark it as ready to be compressed.
		conversion_info[i].reconversion.sources = reconversion_sources;
		conversion_info[i].info = tensor_symbol_info[i].info;
		conversion_info[i].okay = 1;
	}
	// Do the graph mutation now based on the conversion info.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (conversion_info[i].okay)
		{
			const ccv_nnc_tensor_symbol_t reconverted = ccv_nnc_tensor_symbol_new(graph, conversion_info[i].info, 0);
			const ccv_nnc_tensor_symbol_t original = (ccv_nnc_tensor_symbol_t){
				.graph = graph,
				.d = conversion_info[i].original
			};
			const ccv_nnc_graph_exec_symbol_t reconversion_node = ccv_nnc_graph_exec_symbol_new(graph, CMD_DATATYPE_CONVERSION_FORWARD(), TENSOR_SYMBOL_LIST(original), TENSOR_SYMBOL_LIST(reconverted), 0);
			ccv_array_t* const nodes = conversion_info[i].reconversion.nodes;
			assert(nodes && nodes->rnum > 0);
			ccv_array_t* const sources = conversion_info[i].reconversion.sources;
			assert(sources && sources->rnum > 0);
			for (j = 0; j < sources->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(sources, j);
				ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
					.graph = graph,
					.d = d,
				}, reconversion_node);
			}
			for (j = 0; j < nodes->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(nodes, j);
				ccv_nnc_graph_exec_symbol_concat(graph, reconversion_node, (ccv_nnc_graph_exec_symbol_t){
					.graph = graph,
					.d = d
				});
				ccv_nnc_graph_exec_symbol_info_t* const destination_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
				for (k = 0; k < destination_info->input_size; k++)
					if (destination_info->inputs[k] == i)
						destination_info->inputs[k] = reconverted.d;
			}
		}
	ccv_nnc_graph_visit_free(visit);
	ccv_matrix_free(exec_dep);
	ccfree(tensor_marked);
	for (i = 0; i < tensor_symbol_info_size; i++)
	{
		if (conversion_info[i].old_conversion_nodes)
			ccv_array_free(conversion_info[i].old_conversion_nodes);
		if (conversion_info[i].reconversion.nodes)
			ccv_array_free(conversion_info[i].reconversion.nodes);
		if (conversion_info[i].reconversion.sources)
			ccv_array_free(conversion_info[i].reconversion.sources);
	}
	for (i = 0; i < exec_symbol_info_size; i++)
		if (reversed_nodes[i].outgoings)
			ccv_array_free(reversed_nodes[i].outgoings);
	ccfree(reversed_nodes);
	ccfree(conversion_info);
}
