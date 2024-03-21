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
			int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			const int32_t one = 1;
			ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, outgoing, idx);
			/* If not found, set, if the current node is the destination node, no need 
			 * set itself as parent of subsequent nodes because its terminal nature. */
			if (!cell.i32 || cell.i32[0] == 0)
				ccv_set_sparse_matrix_cell(exec_dep, outgoing, idx, &one);
			if (buf_size > 0)
			{
				ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, outgoing);
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
	int should_compress;
	ccv_nnc_tensor_param_t info;
	struct {
		int source;
		int destination;
	} compress;
	struct {
		int source;
		int destination;
		ccv_array_t* nodes;
	} decompress;
} ccv_nnc_compress_info_t;

void ccv_nnc_symbolic_graph_memory_compression(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
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
	int i, j, k;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		// If this node is a convolution backward node, check the original input tensor symbol, and see if it is generated
		// by this graph. If it is, we can track it back and compress it immediately after its generation.
		// I am not certain what's better (whether to "overlap" the compression with the last bits of usage on forward pass)
		// or just compress immediately after. I certainly need to decompress just before this convolution backward node.
		if (node->cmd.cmd == CCV_NNC_CONVOLUTION_BACKWARD && node->input_size > 1 && node->inputs[1] >= 0)
		{
			const int d = node->inputs[1];
			// If this tensor is alias, or assigned (while loop), or bypassed (case..of), skip.
			if (tensor_symbol_info[d].alias_ref || tensor_symbol_info[d].assign_ref || tensor_symbol_info[d].bypass_ref ||
					tensor_symbol_info[d].r_assign_ref || tensor_symbol_info[d].r_bypass_ref)
				continue;
			tensor_marked[d >> 5] |= (1u << (d & 0x1f));
		}
		if (node->cmd.cmd == CCV_NNC_CONVOLUTION_FORWARD && node->output_size >= 1 && node->outputs[0] >= 0)
		{
			const int d = node->outputs[0];
			// If this tensor is alias, or assigned (while loop), or bypassed (case..of), skip.
			if (tensor_symbol_info[d].alias_ref || tensor_symbol_info[d].assign_ref || tensor_symbol_info[d].bypass_ref ||
					tensor_symbol_info[d].r_assign_ref || tensor_symbol_info[d].r_bypass_ref)
				continue;
			tensor_marked[d >> 5] |= (1u << (d & 0x1f));
		}
		if (ccv_nnc_cmd_is_backward(node->cmd))
			for (i = 0; i < node->input_size; i++)
			{
				const int d = node->inputs[i];
				if (d >= 0)
					tensor_used[d >> 5] |= (1u << (d & 0x1f));
			}
	} ccv_nnc_graph_visit_endfor
	// If a tensor is marked but never used in backward pass, no need to compress it.
	_ccv_nnc_remove_unused_from_marked(tensor_used, (tensor_symbol_info_size + 31) >> 5, tensor_marked);
	// If a tensor is not generated on the forward pass, no need to compress it.
	memset(tensor_used, 0, sizeof(uint32_t) * ((tensor_symbol_info_size + 31) >> 5));
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		if (ccv_nnc_cmd_is_forward(node->cmd))
			for (i = 0; i < node->output_size; i++)
			{
				const int d = node->outputs[i];
				if (d >= 0)
					tensor_used[d >> 5] |= (1u << (d & 0x1f));
			}
	} ccv_nnc_graph_visit_endfor
	_ccv_nnc_remove_unused_from_marked(tensor_used, (tensor_symbol_info_size + 31) >> 5, tensor_marked);
	// If this tensor is pointed to by an alias, we don't want to compress as well.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_symbol_info[i].alias_ref)
		{
			const int d = tensor_symbol_info[i].alias_ref - 1;
			// unmark.
			if ((tensor_marked[d >> 5] & (1u << (d & 0x1f))))
				tensor_marked[d >> 5] &= ~(1u << (d & 0x1f));
		}
	// Now tensor_marked only contains the tensors that we think beneficial to compress. Find the best place to insert compression / decompression.
	ccv_nnc_compress_info_t* const compress_info = cccalloc(tensor_symbol_info_size, sizeof(ccv_nnc_compress_info_t));
	ccv_sparse_matrix_t* const exec_dep = _ccv_nnc_exec_dep_new(graph, visit);
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		if (ccv_nnc_cmd_is_forward(node->cmd))
			for (i = 0; i < node->output_size; i++)
			{
				const int d = node->outputs[i];
				if (d >= 0 && (tensor_marked[d >> 5] & (1u << (d & 0x1f))))
					compress_info[d].compress.source = idx;
			}
		else if (ccv_nnc_cmd_is_backward(node->cmd))
			for (i = 0; i < node->input_size; i++)
			{
				const int d = node->inputs[i];
				if (d >= 0 && (tensor_marked[d >> 5] & (1u << (d & 0x1f))))
				{
					if (!compress_info[d].decompress.nodes)
						compress_info[d].decompress.nodes = ccv_array_new(sizeof(int), 0, 0);
					ccv_array_push(compress_info[d].decompress.nodes, &idx);
				}
			}
	} ccv_nnc_graph_visit_endfor
	ccv_array_t* const commons = ccv_array_new(sizeof(int), 0, 0);
	for (i = 0; i < tensor_symbol_info_size; i++)
	{
		if (!compress_info[i].decompress.nodes)
			continue;
		// If we have more than one destination, need to find the common ancestor.
		ccv_array_t* decompress_nodes = compress_info[i].decompress.nodes;
		if (decompress_nodes && decompress_nodes->rnum > 1)
		{
			ccv_array_clear(commons);
			ccv_array_t* const nodes = compress_info[i].decompress.nodes;
			const int d = *(int*)ccv_array_get(nodes, 0);
			ccv_array_push(commons, &d);
#define for_block(x, val) \
			do { \
				const int dd = ((int32_t*)val)[0]; \
				if (dd > 0) \
					ccv_array_push(commons, &x); \
			} while (0)
			ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, d);
			if (vector)
				CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
#undef for_block
			for (j = 0; j < commons->rnum;)
			{
				const int d = *(int*)ccv_array_get(commons, j);
				int flag = 0;
				for (k = 1; k < nodes->rnum && !flag; k++)
				{
					const int dd = *(int*)ccv_array_get(nodes, k);
					if (dd == d) // If it is the same as the commons, keep.
						continue;
					const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, dd, d);
					// If I cannot reach from this destination to the ancestor. This is not an ancestor.
					// Remove it from the list.
					if (!cell.i32 || cell.i32[0] == 0)
						flag = 1;
				}
				if (flag)
				{
					if (j < commons->rnum - 1)
						*(int*)ccv_array_get(commons, j) = *(int*)ccv_array_get(commons, commons->rnum - 1);
					--commons->rnum;
					continue;
				}
				++j;
			}
			// If there is no common ancestor. We cannot do this. Abort the whole thing.
			if (commons->rnum == 0)
				continue;
			decompress_nodes = commons;
		}
		// Find source / destination for compress nodes.
		const int compress_source = compress_info[i].compress.source;
		ccv_array_t* const outgoings = exec_symbol_info[compress_source].outgoings;
		if (!outgoings || outgoings->rnum == 0)
			continue;
		int hop = exec_symbol_info_size;
		for (j = 0; j < outgoings->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(outgoings, j);
			const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, d, compress_source);
			if (cell.i32[0] < hop)
			{
				compress_info[i].compress.destination = d;
				hop = cell.i32[0];
			}
		}
		if (hop == exec_symbol_info_size)
			continue;
		// Find source / destination for decompress nodes.
		// First, find the node that everyone can reach it.
		int decompress_destination = -1;
		for (j = 0; j < decompress_nodes->rnum; j++)
		{
			int flag = 0;
			const int dj = *(int*)ccv_array_get(decompress_nodes, j);
			for (k = 0; !flag && k < decompress_nodes->rnum; k++)
				if (j != k)
				{
					const int dk = *(int*)ccv_array_get(decompress_nodes, k);
					const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, dj, dk);
					if (!cell.i32 || cell.i32[0] == 0)
						flag = 1;
				}
			if (!flag)
				decompress_destination = (decompress_destination == -1) ? dj : -2;
		}
		// Cannot continue, either we cannot find the node that is child node of everyone, or
		// it has more than one of these.
		if (decompress_destination < 0)
			continue;
		compress_info[i].decompress.destination = decompress_destination;
		hop = exec_symbol_info_size;
#define for_block(x, val) \
		do { \
			const int dd = ((int32_t*)val)[0]; \
			if (dd > 0 && dd < hop) \
			{ \
				compress_info[i].decompress.source = x; \
				hop = dd; \
			} \
		} while (0)
		ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, decompress_destination);
		if (vector)
			CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
		// Final check, the destination of compression should be smaller than the source of decompression.
		const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep, compress_info[i].decompress.source, compress_info[i].compress.destination);
		if (compress_info[i].decompress.source != compress_info[i].compress.destination && (!cell.i32 || cell.i32[0] == 0))
			continue;
		// Mark it as ready to be compressed.
		compress_info[i].info = tensor_symbol_info[i].info;
		compress_info[i].should_compress = 1;
	}
	// Do the graph mutation now based on the compression info.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (compress_info[i].should_compress)
		{
			ccv_nnc_tensor_param_t compressed_params;
			ccv_nnc_hint_tensor_auto(CMD_COMPRESSION_LSSC_FORWARD(), &compress_info[i].info, 1, ccv_nnc_no_hint, &compressed_params, 1);
			const ccv_nnc_tensor_symbol_t original = (ccv_nnc_tensor_symbol_t){
				.graph = graph,
				.d = i
			};
			const ccv_nnc_tensor_symbol_t compressed = ccv_nnc_tensor_symbol_new(graph, compressed_params, 0);
			const ccv_nnc_graph_exec_symbol_t compress_node = ccv_nnc_graph_exec_symbol_new(graph, CMD_COMPRESSION_LSSC_FORWARD(), TENSOR_SYMBOL_LIST(original), TENSOR_SYMBOL_LIST(compressed), 0);
			ccv_nnc_graph_exec_symbol_disjoin(graph, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].compress.source,
			}, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].compress.destination
			});
			ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].compress.source,
			}, compress_node);
			ccv_nnc_graph_exec_symbol_concat(graph, compress_node, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].compress.destination
			});
			const ccv_nnc_tensor_symbol_t decompressed = ccv_nnc_tensor_symbol_new(graph, compress_info[i].info, 0);
			const ccv_nnc_graph_exec_symbol_t decompress_node = ccv_nnc_graph_exec_symbol_new(graph, CMD_COMPRESSION_LSSC_BACKWARD(), TENSOR_SYMBOL_LIST(compressed), TENSOR_SYMBOL_LIST(decompressed), 0);
			ccv_nnc_graph_exec_symbol_disjoin(graph, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].decompress.source,
			}, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].decompress.destination
			});
			ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].decompress.source,
			}, decompress_node);
			ccv_nnc_graph_exec_symbol_concat(graph, decompress_node, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = compress_info[i].decompress.destination
			});
			for (j = 0; j < compress_info[i].decompress.nodes->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(compress_info[i].decompress.nodes, j);
				ccv_nnc_graph_exec_symbol_info_t* const destination_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
				for (k = 0; k < destination_info->input_size; k++)
					if (destination_info->inputs[k] == i)
						destination_info->inputs[k] = decompressed.d;
			}
		}
	ccv_nnc_graph_visit_free(visit);
	ccv_array_free(commons);
	ccv_matrix_free(exec_dep);
	ccfree(tensor_marked);
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (compress_info[i].decompress.nodes)
			ccv_array_free(compress_info[i].decompress.nodes);
	ccfree(compress_info);
}
