#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

#pragma mark - Level-3.5 API

enum {
	CCV_NNC_PARALLEL_BROADCAST = 0x1,
	CCV_NNC_PARALLEL_ALLREDUCER = 0x2,
	CCV_NNC_PARALLEL_REDUCER = 0x3,
};

static int _ccv_nnc_exec_inputs_contain(const ccv_nnc_graph_exec_symbol_info_t* const node, const int d)
{
	int i;
	for (i = 0; i < node->input_size; i++)
		if (node->inputs[i] == d)
			return 1;
	return 0;
}

void ccv_nnc_symbolic_graph_data_parallel(ccv_nnc_symbolic_graph_t* const graph, const int parallel, const ccv_nnc_tensor_symbol_t* const broadcasts, const int broadcast_size, const ccv_nnc_tensor_symbol_t* const allreducers, const int allreducer_size, ccv_nnc_tensor_symbol_t* const allreducer_outs, const ccv_nnc_tensor_symbol_t* const reducers, const int reducer_size, ccv_nnc_tensor_symbol_t* const reducer_outs, const int reduce_op_type, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	assert(reduce_op_type == CCV_NNC_PARALLEL_REDUCE_OP_SUM);
	const int parallel_count = (parallel == 0) ? ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) : parallel;
	if (parallel_count == 1)
		return;
	assert(parallel_count > 1);
	ccv_nnc_graph_visit_t* const visit = ccv_nnc_graph_visit_new(graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
	int i, j, k;
	// Tensor symbol has to be on device 0 or any.
	ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), node, idx) {
		for (i = 0; i < node->input_size; i++)
			if (node->inputs[i] >= 0)
			{
				ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, node->inputs[i]);
				if (CCV_TENSOR_GET_MEMORY(tensor_symbol->info.type) == CCV_TENSOR_GPU_MEMORY &&
					CCV_TENSOR_GET_DEVICE(tensor_symbol->info.type) != CCV_COMPUTE_DEVICE_ANY)
					{ assert(CCV_TENSOR_GET_DEVICE(tensor_symbol->info.type) == CCV_COMPUTE_DEVICE_000); }
			}
		for (i = 0; i < node->output_size; i++)
			if (node->outputs[i] >= 0)
			{
				ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, node->outputs[i]);
				if (CCV_TENSOR_GET_MEMORY(tensor_symbol->info.type) == CCV_TENSOR_GPU_MEMORY &&
					CCV_TENSOR_GET_DEVICE(tensor_symbol->info.type) != CCV_COMPUTE_DEVICE_ANY)
					{ assert(CCV_TENSOR_GET_DEVICE(tensor_symbol->info.type) == CCV_COMPUTE_DEVICE_000); }
			}
	} ccv_nnc_graph_visit_endfor
	// Run infer in the graph to get all tensors shaped.
	ccv_nnc_symbolic_graph_symbol_infer(graph, visit, sources, source_size, destinations, destination_size, 0, 0, 0, (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, 0), (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0));
	// Set ANY device to default device. Make a list of execution nodes / tensors to be duplicated.
	ccv_array_t* const dup_tensors = ccv_array_new(sizeof(int), 0, 0);
	ccv_array_t* const dup_execs = ccv_array_new(sizeof(int), 0, 0);
	ccv_array_t* const broadcast_reduce_execs = ccv_array_new(sizeof(int), 0, 0);
	int* const allreduce_inputs = allreducer_size > 0 ? (int*)ccmalloc(sizeof(int) * allreducer_size) : 0;
	for (i = 0; i < allreducer_size; i++)
		allreduce_inputs[i] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_symbol_params(graph, allreducers[i]), 0).d;
	const int tensor_symbol_size = graph->tensor_symbol_info->rnum;
	const int graph_exec_symbol_size = graph->exec_symbol_info->rnum;
	int* const tensor_flags = (int*)cccalloc(tensor_symbol_size + graph_exec_symbol_size, sizeof(int));
	int* const exec_flags = tensor_flags + tensor_symbol_size;
	for (i = 0; i < broadcast_size; i++)
	{
		// Doesn't support alias for these.
		tensor_flags[broadcasts[i].d] = CCV_NNC_PARALLEL_BROADCAST;
		assert(graph == broadcasts[i].graph);
		assert(!((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, broadcasts[i].d))->alias_ref);
	}
	int* const allreduce_producers = allreducer_size > 0 ? (int*)cccalloc(tensor_symbol_size, sizeof(int)) : 0;
	for (i = 0; i < allreducer_size; i++)
	{
		// Doesn't support alias for these.
		tensor_flags[allreducers[i].d] = CCV_NNC_PARALLEL_ALLREDUCER;
		assert(graph == allreducers[i].graph);
		assert(!((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, allreducers[i].d))->alias_ref);
	}
	for (i = 0; i < reducer_size; i++)
	{
		// Doesn't support alias for these.
		tensor_flags[reducers[i].d] = CCV_NNC_PARALLEL_REDUCER;
		assert(graph == reducers[i].graph);
		assert(!((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, reducers[i].d))->alias_ref);
	}
	// No overlap between broadcasts, allreducers, reducers.
	for (i = 0; i < broadcast_size; i++)
		for (j = 0; j < reducer_size; j++)
			{ assert(broadcasts[i].d != reducers[j].d); }
	for (i = 0; i < broadcast_size; i++)
		for (j = 0; j < allreducer_size; j++)
			{ assert(broadcasts[i].d != allreducers[j].d); }
	for (i = 0; i < allreducer_size; i++)
		for (j = 0; j < reducer_size; j++)
			{ assert(allreducers[i].d != reducers[j].d); }
	ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), node, idx) {
		int parallelizable_data = 0;
		int reduce_inputs = 0;
		int broadcast_outputs = 0;
		for (i = 0; i < node->input_size; i++)
			if (node->inputs[i] >= 0)
			{
				ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, node->inputs[i]);
				if (CCV_TENSOR_GET_MEMORY(tensor_symbol->info.type) == CCV_TENSOR_GPU_MEMORY)
				{
					if (CCV_TENSOR_GET_DEVICE(tensor_symbol->info.type) == CCV_COMPUTE_DEVICE_ANY)
						CCV_TENSOR_SET_DEVICE_ID(tensor_symbol->info.type, 0);
					// Don't support alias for broadcast / allreducer / reducer.
					assert(!tensor_symbol->alias_ref || tensor_flags[tensor_symbol->alias_ref - 1] != CCV_NNC_PARALLEL_BROADCAST);
					assert(!tensor_symbol->alias_ref || tensor_flags[tensor_symbol->alias_ref - 1] != CCV_NNC_PARALLEL_ALLREDUCER);
					assert(!tensor_symbol->alias_ref || tensor_flags[tensor_symbol->alias_ref - 1] != CCV_NNC_PARALLEL_REDUCER);
					const int d = node->inputs[i];
					if (tensor_flags[d] == CCV_NNC_PARALLEL_REDUCER)
						reduce_inputs = 1;
					parallelizable_data = 1;
				}
			}
		for (i = 0; i < node->output_size; i++)
			if (node->outputs[i] >= 0)
			{
				ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, node->outputs[i]);
				if (CCV_TENSOR_GET_MEMORY(tensor_symbol->info.type) == CCV_TENSOR_GPU_MEMORY)
				{
					if (CCV_TENSOR_GET_DEVICE(tensor_symbol->info.type) == CCV_COMPUTE_DEVICE_ANY)
						CCV_TENSOR_SET_DEVICE_ID(tensor_symbol->info.type, 0);
					// Don't support alias for broadcast / allreducer / reducer.
					assert(!tensor_symbol->alias_ref || tensor_flags[tensor_symbol->alias_ref - 1] != CCV_NNC_PARALLEL_BROADCAST);
					assert(!tensor_symbol->alias_ref || tensor_flags[tensor_symbol->alias_ref - 1] != CCV_NNC_PARALLEL_ALLREDUCER);
					assert(!tensor_symbol->alias_ref || tensor_flags[tensor_symbol->alias_ref - 1] != CCV_NNC_PARALLEL_REDUCER);
					const int d = node->outputs[i];
					if (tensor_flags[d] == CCV_NNC_PARALLEL_BROADCAST)
						broadcast_outputs = 1;
					else if (tensor_flags[d] == CCV_NNC_PARALLEL_ALLREDUCER)
						allreduce_producers[d] = idx + 1;
					parallelizable_data = 1;
				}
			}
		assert(!(broadcast_outputs && reduce_inputs)); // This node cannot be both broadcast and reducer.
		if (broadcast_outputs ^ reduce_inputs)
		{
			if (broadcast_outputs)
				exec_flags[idx] = CCV_NNC_PARALLEL_BROADCAST;
			else if (reduce_inputs)
				exec_flags[idx] = CCV_NNC_PARALLEL_REDUCER;
			ccv_array_push(broadcast_reduce_execs, &idx);
		} else if (parallelizable_data && !broadcast_outputs && !reduce_inputs) {
			// If this node contains GPU data that need to be parallelized, and this node itself is not a broadcast node or a reducer node..
			ccv_array_push(dup_execs, &idx);
			for (i = 0; i < node->input_size; i++)
				if (node->inputs[i] >= 0)
				{
					ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, node->inputs[i]);
					if (CCV_TENSOR_GET_MEMORY(tensor_symbol->info.type) == CCV_TENSOR_GPU_MEMORY)
					{
						// Add the symbol alias to first.
						if (tensor_symbol->alias_ref)
							ccv_array_add_unique_int(dup_tensors, tensor_symbol->alias_ref - 1);
						ccv_array_add_unique_int(dup_tensors, node->inputs[i]);
					}
				}
			for (i = 0; i < node->output_size; i++)
				if (node->outputs[i] >= 0)
				{
					ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, node->outputs[i]);
					if (CCV_TENSOR_GET_MEMORY(tensor_symbol->info.type) == CCV_TENSOR_GPU_MEMORY)
					{
						if (tensor_symbol->alias_ref)
							ccv_array_add_unique_int(dup_tensors, tensor_symbol->alias_ref - 1);
						ccv_array_add_unique_int(dup_tensors, node->outputs[i]);
					}
				}
		}
	} ccv_nnc_graph_visit_endfor
	// Now, actually create these tensors.
	if (!graph->data_parallel.tensor_symbol_idx)
		graph->data_parallel.tensor_symbol_idx = (int*)ccmalloc(sizeof(int) * (parallel_count - 1) * tensor_symbol_size);
	else if (graph->data_parallel.tensor_symbol_size * (graph->data_parallel.count - 1) != tensor_symbol_size * (parallel_count - 1))
		// This may shrink too, but that is OK.
		graph->data_parallel.tensor_symbol_idx = (int*)ccrealloc(graph->data_parallel.tensor_symbol_idx, sizeof(int) * (parallel_count - 1) * tensor_symbol_size);
	graph->data_parallel.tensor_symbol_size = tensor_symbol_size;
	graph->data_parallel.count = parallel_count;
	int* const dup_tensor_idx = graph->data_parallel.tensor_symbol_idx;
	// dup_tensor_idx is the array starts with 0 here.
	for (i = 0; i < (parallel_count - 1) * tensor_symbol_size; i++)
		dup_tensor_idx[i] = -1;
	// Make the duplicated tensors (on different devices).
	for (i = 0; i < dup_tensors->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(dup_tensors, i);
		ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
		ccv_nnc_tensor_param_t info = tensor_symbol->info;
		const int flags = tensor_symbol->flags;
		if (tensor_symbol->alias_ref)
		{
			const int alias_ref = tensor_symbol->alias_ref - 1;
			for (j = 0; j < parallel_count - 1; j++)
			{
				const int dup_d = dup_tensor_idx[alias_ref * (parallel_count - 1) + j];
				CCV_TENSOR_SET_DEVICE_ID(info.type, j + 1); // Set the device id.
				assert(dup_d >= 0);
				// Get tensor symbol again, it may be invalid after added new symbol (we use it for ofs and inc).
				ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, d);
				const ccv_nnc_tensor_symbol_t new_symbol = ccv_nnc_tensor_symbol_alias_new(graph, (ccv_nnc_tensor_symbol_t){
					.d = dup_d,
					.graph = graph,
				}, tensor_symbol->ofs, tensor_symbol->inc, info, 0);
				ccv_nnc_tensor_symbol_set_flags(graph, new_symbol, flags);
				dup_tensor_idx[d * (parallel_count - 1) + j] = new_symbol.d;
			}
		} else {
			for (j = 0; j < parallel_count - 1; j++)
			{
				CCV_TENSOR_SET_DEVICE_ID(info.type, j + 1); // Set the device id.
				const ccv_nnc_tensor_symbol_t new_symbol = ccv_nnc_tensor_symbol_new(graph, info, 0);
				ccv_nnc_tensor_symbol_set_flags(graph, new_symbol, flags);
				dup_tensor_idx[d * (parallel_count - 1) + j] = new_symbol.d;
			}
		}
	}
	ccv_array_free(dup_tensors);
	// Now, create execs.
	if (!graph->data_parallel.exec_symbol_idx)
		graph->data_parallel.exec_symbol_idx = (int*)ccmalloc(sizeof(int) * (parallel_count - 1) * graph_exec_symbol_size);
	else if (graph->data_parallel.exec_symbol_size * (graph->data_parallel.count - 1) != graph_exec_symbol_size * (parallel_count - 1))
		// This may shrink too, but that is OK.
		graph->data_parallel.exec_symbol_idx = (int*)ccrealloc(graph->data_parallel.exec_symbol_idx, sizeof(int) * (parallel_count - 1) * graph_exec_symbol_size);
	graph->data_parallel.exec_symbol_size = graph_exec_symbol_size;
	int* const dup_exec_idx = graph->data_parallel.exec_symbol_idx;
	// dup_exec_idx is the array starts with 0 here.
	for (i = 0; i < (parallel_count - 1) * graph_exec_symbol_size; i++)
		dup_exec_idx[i] = -1;
	int max_io_size = 1 + parallel_count;
	// Now make the duplicated execs nodes (on different devices).
	for (i = 0; i < dup_execs->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(dup_execs, i);
		ccv_nnc_graph_exec_symbol_info_t* const exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
		max_io_size = ccv_max(max_io_size, exec_symbol->input_size + exec_symbol->output_size);
	}
	max_io_size = ccv_max(max_io_size, parallel_count * 2); // tensors from all parallel_count, the output is to all parallel_count (thus, allreduce).
	ccv_nnc_tensor_symbol_t max_io[max_io_size];
	for (i = 0; i < dup_execs->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(dup_execs, i);
		for (j = 0; j < parallel_count - 1; j++)
		{
			ccv_nnc_graph_exec_symbol_info_t* const exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
			const ccv_nnc_cmd_t cmd = exec_symbol->cmd;
			const ccv_nnc_hint_t hint = exec_symbol->hint;
			const int input_size = exec_symbol->input_size;
			const int output_size = exec_symbol->output_size;
			ccv_nnc_tensor_symbol_t* const inputs = max_io;
			for (k = 0; k < input_size; k++)
			{
				const int idx = exec_symbol->inputs[k];
				if (idx >= 0)
					inputs[k].d = dup_tensor_idx[idx * (parallel_count - 1) + j] >= 0 ? dup_tensor_idx[idx * (parallel_count - 1) + j] : idx;
				else
					inputs[k].d = idx;
				inputs[k].graph = idx != CCV_NNC_NO_TENSOR_SYMBOL ? graph : 0;
			}
			ccv_nnc_tensor_symbol_t* const outputs = max_io + input_size;
			for (k = 0; k < output_size; k++)
			{
				const int idx = exec_symbol->outputs[k];
				if (idx >= 0)
					outputs[k].d = dup_tensor_idx[idx * (parallel_count - 1) + j] >= 0 ? dup_tensor_idx[idx * (parallel_count - 1) + j] : idx;
				else
					outputs[k].d = idx;
				outputs[k].graph = idx != CCV_NNC_NO_TENSOR_SYMBOL ? graph : 0;
			}
			const ccv_nnc_graph_exec_symbol_t new_symbol = ccv_nnc_graph_exec_symbol_new(graph, cmd, inputs, input_size, outputs, output_size, 0);
			ccv_nnc_graph_exec_symbol_set_hint(graph, new_symbol, hint);
			dup_exec_idx[d * (parallel_count - 1) + j] = new_symbol.d;
		}
	}
	// Create new tensors for broadcast / reduce.
	int* const broadcast_reduce_tensor_idx = (int*)cccalloc(tensor_symbol_size, sizeof(int));
	for (i = 0; i < broadcast_size + reducer_size; i++)
	{
		const int idx = i >= broadcast_size ? reducers[i - broadcast_size].d : broadcasts[i].d;
		ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, idx);
		ccv_nnc_tensor_param_t info = tensor_symbol->info;
		const int flags = tensor_symbol->flags;
		// No alias handling.
		assert(!tensor_symbol->alias_ref);
		const ccv_nnc_tensor_symbol_t new_symbol = ccv_nnc_tensor_symbol_new(graph, info, 0);
		ccv_nnc_tensor_symbol_set_flags(graph, new_symbol, flags);
		broadcast_reduce_tensor_idx[idx] = new_symbol.d + 1;
	}
	int* const broadcast_exec_idx = (int*)cccalloc(tensor_symbol_size, sizeof(int));
	int* const reduce_exec_idx = (int*)cccalloc(tensor_symbol_size, sizeof(int));
	// Create node for broadcast (thus, transfer data to different parallel_count) and reducer (transfer data back to a device, and sum).
	for (i = 0; i < broadcast_reduce_execs->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(broadcast_reduce_execs, i);
		// For broadcast, we create data transfers as our dup node, and create connections to these data transfers.
		if (exec_flags[d] == CCV_NNC_PARALLEL_BROADCAST)
		{
			ccv_nnc_graph_exec_symbol_info_t* const exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
			ccv_nnc_tensor_symbol_t* const inputs = max_io;
			ccv_nnc_tensor_symbol_t* const outputs = max_io + 1;
			const ccv_nnc_graph_exec_symbol_t source = {
				.d = d,
				.graph = graph,
			};
			for (j = 0; j < exec_symbol->output_size; j++)
			{
				const int idx = exec_symbol->outputs[j];
				if (idx >= 0 && tensor_flags[idx] == CCV_NNC_PARALLEL_BROADCAST)
				{
					inputs[0] = (ccv_nnc_tensor_symbol_t){
						.d = idx,
						.graph = graph,
					};
					// Reset the tensor flags, it is broadcasted now.
					tensor_flags[idx] = 0;
					outputs[0] = (ccv_nnc_tensor_symbol_t){
						.d = broadcast_reduce_tensor_idx[idx] - 1,
						.graph = graph,
					};
					assert(broadcast_reduce_tensor_idx[idx] > 0);
					for (k = 0; k < parallel_count - 1; k++)
						outputs[k + 1] = (ccv_nnc_tensor_symbol_t){
							.d = dup_tensor_idx[idx * (parallel_count - 1) + k],
							.graph = graph,
						};
					const ccv_nnc_graph_exec_symbol_t bcast = ccv_nnc_graph_exec_symbol_new(graph, CMD_COMM_BROADCAST_FORWARD(), inputs, 1, outputs, parallel_count, 0);
					ccv_nnc_graph_exec_symbol_concat(graph, source, bcast);
					assert(!broadcast_exec_idx[idx]);
					broadcast_exec_idx[idx] = bcast.d + 1;
				}
			}
		} else if (exec_flags[d] == CCV_NNC_PARALLEL_REDUCER) {
			// Gather is a bit more sophisticated, we need to use the new tensor to hold the summed value.
			// This is what we have right now, I will use NCCL later.
			ccv_nnc_tensor_symbol_t* const inputs = max_io;
			ccv_nnc_tensor_symbol_t* const outputs = max_io + parallel_count;
			ccv_nnc_graph_exec_symbol_info_t* exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
			for (j = 0; j < exec_symbol->input_size; j++)
			{
				const int idx = exec_symbol->inputs[j];
				if (idx >= 0 && tensor_flags[idx] == CCV_NNC_PARALLEL_REDUCER && !reduce_exec_idx[idx])
				{
					inputs[0] = (ccv_nnc_tensor_symbol_t){
						.d = idx,
						.graph = graph,
					};
					for (k = 0; k < parallel_count - 1; k++)
						inputs[k + 1] = (ccv_nnc_tensor_symbol_t){
							.d = dup_tensor_idx[idx * (parallel_count - 1) + k],
							.graph = graph,
						};
					outputs[0] = (ccv_nnc_tensor_symbol_t){
						.d = broadcast_reduce_tensor_idx[idx] - 1,
						.graph = graph,
					};
					// Create new symbol for all other tensors to facilitate copy (this is not useful for NCCL, but useful for REF implementation).
					ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, idx);
					ccv_nnc_tensor_param_t info = tensor_symbol->info;
					const int flags = tensor_symbol->flags;
					// No alias handling.
					assert(!tensor_symbol->alias_ref);
					for (k = 1; k < parallel_count; k++)
					{
						const ccv_nnc_tensor_symbol_t new_symbol = ccv_nnc_tensor_symbol_new(graph, info, 0);
						ccv_nnc_tensor_symbol_set_flags(graph, new_symbol, flags);
						outputs[k] = new_symbol;
					}
					const ccv_nnc_graph_exec_symbol_t reduce = ccv_nnc_graph_exec_symbol_new(graph, CMD_COMM_REDUCE_FORWARD(), inputs, parallel_count, outputs, parallel_count, 0);
					// Refresh the pointer to keep it up to date.
					exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
					ccv_nnc_graph_exec_symbol_concat(graph, reduce, (ccv_nnc_graph_exec_symbol_t){
						.d = d,
						.graph = graph,
					});
					reduce_exec_idx[idx] = reduce.d + 1;
				}
			}
			// Update the inputs pointing to the summed value.
			for (j = 0; j < exec_symbol->input_size; j++)
			{
				const int idx = exec_symbol->inputs[j];
				if (idx >= 0 && tensor_flags[idx] == CCV_NNC_PARALLEL_REDUCER)
					exec_symbol->inputs[j] = broadcast_reduce_tensor_idx[idx] - 1;
			}
		}
	}
	ccv_array_free(broadcast_reduce_execs);
	// If this tensor is not broadcasted yet, that means there is no exec to generate this tensor. We just generate headless copy.
	for (i = 0; i < dup_execs->rnum; i++)
	{
		const int idx = *(int*)ccv_array_get(dup_execs, i);
		ccv_nnc_graph_exec_symbol_info_t* const node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, idx);
		if (exec_flags[idx] == CCV_NNC_PARALLEL_REDUCER)
			continue;
		// We try to make copy command as compact as possible by having one copy for multiple tensors if they used together.
		ccv_nnc_tensor_symbol_t* const inputs = max_io;
		ccv_nnc_tensor_symbol_t* const outputs = max_io + 1;
		for (j = 0; j < node->input_size; j++)
		{
			const int idx = node->inputs[j];
			// Now, figure out whether we need to create copy command.
			if (idx >= 0 && idx < tensor_symbol_size && tensor_flags[idx] == CCV_NNC_PARALLEL_BROADCAST)
			{
				inputs[0] = (ccv_nnc_tensor_symbol_t){
					.d = idx,
					.graph = graph,
				};
				// Reset the tensor flags, it is broadcasted now.
				tensor_flags[idx] = 0;
				outputs[0] = (ccv_nnc_tensor_symbol_t){
					.d = broadcast_reduce_tensor_idx[idx] - 1,
					.graph = graph,
				};
				assert(broadcast_reduce_tensor_idx[idx] > 0);
				for (k = 0; k < parallel_count - 1; k++)
					outputs[k + 1] = (ccv_nnc_tensor_symbol_t){
						.d = dup_tensor_idx[idx * (parallel_count - 1) + k],
						.graph = graph,
					};
				const ccv_nnc_graph_exec_symbol_t bcast = ccv_nnc_graph_exec_symbol_new(graph, CMD_COMM_BROADCAST_FORWARD(), inputs, 1, outputs, parallel_count, 0);
				broadcast_exec_idx[idx] = bcast.d + 1;
			}
		}
	}
	// Write reducer_outs last, because it may be the same pointer as reducers.
	if (reducer_outs)
		for (i = 0; i < reducer_size; i++)
		{
			reducer_outs[i].d = broadcast_reduce_tensor_idx[i + broadcast_size] - 1;
			reducer_outs[i].graph = graph;
		}
	ccfree(broadcast_reduce_tensor_idx);
	ccv_array_free(dup_execs);
	// Now everything is dup'ed, connect them all.
	ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), node, idx) {
		for (i = 0; i < node->input_size; i++)
		{
			const int input = node->inputs[i];
			// If it is broadcast worthy.
			if (input >= 0 && input < tensor_symbol_size && broadcast_exec_idx[input])
				ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
					.d = broadcast_exec_idx[input] - 1,
					.graph = graph,
				}, (ccv_nnc_graph_exec_symbol_t){
					.d = idx,
					.graph = graph,
				});
		}
		// Check whether this node has outgoing to the reducer node, if so, replace that to the sum node.
		if (node->outgoings && node->outgoings->rnum)
			for (i = 0; i < node->outgoings->rnum; i++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, i);
				if (outgoing_idx >= graph_exec_symbol_size)
					continue;
				if (exec_flags[outgoing_idx] == CCV_NNC_PARALLEL_REDUCER)
					for (j = 0; j < node->output_size; j++)
					{
						const int output_idx = node->outputs[j];
						if (output_idx >= 0 && tensor_flags[output_idx] == CCV_NNC_PARALLEL_REDUCER)
						{
							assert(reduce_exec_idx[output_idx]);
							ccv_array_replace_unique_int(node->outgoings, outgoing_idx, reduce_exec_idx[output_idx] - 1);
						}
					}
			}
		for (i = 0; i < parallel_count - 1; i++)
		{
			const int d = dup_exec_idx[idx * (parallel_count - 1) + i];
			if (d < 0)
				continue;
			const ccv_nnc_graph_exec_symbol_t source = {
				.d = d,
				.graph = graph,
			};
			// If it is broadcast worthy.
			for (j = 0; j < node->input_size; j++)
			{
				const int input = node->inputs[j];
				if (input >= 0 && input < tensor_symbol_size && broadcast_exec_idx[input])
					ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
						.d = broadcast_exec_idx[input] - 1,
						.graph = graph,
					}, source);
			}
			// If it is reduce worthy.
			for (j = 0; j < node->output_size; j++)
			{
				const int output = node->outputs[j];
				if (output >= 0 && output < tensor_symbol_size && reduce_exec_idx[output])
					ccv_nnc_graph_exec_symbol_concat(graph, source, (ccv_nnc_graph_exec_symbol_t){
						.d = reduce_exec_idx[output] - 1,
						.graph = graph,
					});
			}
			if (node->outgoings && node->outgoings->rnum)
				for (j = 0; j < node->outgoings->rnum; j++)
				{
					const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, j);
					if (outgoing_idx > graph_exec_symbol_size)
						continue;
					const int outgoing_d = dup_exec_idx[outgoing_idx * (parallel_count - 1) + i];
					if (outgoing_d < 0)
						continue;
					ccv_nnc_graph_exec_symbol_concat(graph, source, (ccv_nnc_graph_exec_symbol_t){
						.d = outgoing_d,
						.graph = graph,
					});
				}
		}
	} ccv_nnc_graph_visit_endfor
	ccfree(broadcast_exec_idx);
	ccfree(reduce_exec_idx);
	ccfree(tensor_flags);
	ccv_nnc_graph_visit_free(visit);
	// Allreduce is easier to do, we do that the last. It consists of two steps:
	// 1. Generate allreduce node for each symbol;
	// 2. Disconnect them from source and connect them through all reduce nodes.
	for (i = 0; i < allreducer_size; i++)
	{
		ccv_nnc_tensor_symbol_t* const outputs = max_io + parallel_count;
		outputs[0] = allreducers[i];
		// Copy over allreducers output symbols (as the old symbol).
		for (j = 0; j < parallel_count - 1; j++)
		{
			const int d = allreducers[i].d;
			outputs[j + 1].graph = graph;
			assert(dup_tensor_idx[d * (parallel_count - 1) + j] >= 0);
			outputs[j + 1].d = dup_tensor_idx[d * (parallel_count - 1) + j];
		}
		ccv_nnc_tensor_symbol_t* const inputs = max_io;
		inputs[0].graph = graph;
		inputs[0].d = allreduce_inputs[i];
		// Create identical new tensor symbols
		for (j = 0; j < parallel_count - 1; j++)
		{
			if (dup_tensor_idx[allreduce_inputs[i] * (parallel_count - 1) + j] < 0)
				dup_tensor_idx[allreduce_inputs[i] * (parallel_count - 1) + j] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_symbol_params(graph, outputs[j + 1]), 0).d;
			inputs[j + 1].graph = graph;
			inputs[j + 1].d = dup_tensor_idx[allreduce_inputs[i] * (parallel_count - 1) + j];
		}
		// Create allreduce node.
		const ccv_nnc_graph_exec_symbol_t allreduce = ccv_nnc_graph_exec_symbol_new(graph, CMD_COMM_ALLREDUCE_FORWARD(), inputs, parallel_count, outputs, parallel_count, 0);
		const int exec_idx = allreduce_producers[allreducers[i].d] - 1;
		assert(exec_idx >= 0);
		ccv_nnc_graph_exec_symbol_info_t* const node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, exec_idx);
		for (j = 0; j < node->output_size; j++)
			if (node->outputs[j] == outputs[0].d)
				node->outputs[j] = inputs[0].d;
		ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
			.graph = graph,
			.d = exec_idx,
		}, allreduce);
		// Remove connections from current node directly to its following nodes (these should follow allreduce node now).
		for (j = 0; j < node->outgoings->rnum;)
		{
			const int d = *(int*)ccv_array_get(node->outgoings, j);
			if (d == allreduce.d)
			{
				++j;
				continue;
			}
			// Get the destination nodes, and check whether they have inputs matches our outputs.
			ccv_nnc_graph_exec_symbol_info_t* const outgoing_node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
			if (_ccv_nnc_exec_inputs_contain(outgoing_node, allreducers[i].d))
			{
				ccv_nnc_graph_exec_symbol_concat(graph, allreduce, (ccv_nnc_graph_exec_symbol_t){
					.graph = graph,
					.d = d,
				});
				// Remove the connection.
				if (j < node->outgoings->rnum - 1)
					*(int*)ccv_array_get(node->outgoings, j) = *(int*)ccv_array_get(node->outgoings, node->outgoings->rnum - 1);
				--node->outgoings->rnum;
			} else
				++j;
		}
		for (j = 0; j < parallel_count - 1; j++)
		{
			const int new_exec_idx = dup_exec_idx[exec_idx * (parallel_count - 1) + j];
			ccv_nnc_graph_exec_symbol_info_t* const node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, new_exec_idx);
			for (k = 0; k < node->output_size; k++)
				if (node->outputs[k] == outputs[j + 1].d)
					node->outputs[k] = inputs[j + 1].d;
			ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
				.graph = graph,
				.d = new_exec_idx,
			}, allreduce);
			for (k = 0; k < node->outgoings->rnum;)
			{
				const int d = *(int*)ccv_array_get(node->outgoings, k);
				if (d == allreduce.d)
				{
					++k;
					continue;
				}
				// Get the destination nodes, and check whether they have inputs matches our outputs.
				ccv_nnc_graph_exec_symbol_info_t* const outgoing_node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
				if (_ccv_nnc_exec_inputs_contain(outgoing_node, outputs[j + 1].d))
				{
					ccv_nnc_graph_exec_symbol_concat(graph, allreduce, (ccv_nnc_graph_exec_symbol_t){
						.graph = graph,
						.d = d,
					});
					// Remove the connection.
					if (k < node->outgoings->rnum - 1)
						*(int*)ccv_array_get(node->outgoings, k) = *(int*)ccv_array_get(node->outgoings, node->outgoings->rnum - 1);
					--node->outgoings->rnum;
				} else
					++k;
			}
		}
	}
	ccfree(allreduce_producers);
	// Write allreducer_outs last, because it may be the same pointer as allreducers.
	if (allreducer_outs)
		for (i = 0; i < allreducer_size; i++)
		{
			allreducer_outs[i].d = allreduce_inputs[i];
			allreducer_outs[i].graph = graph;
		}
	ccfree(allreduce_inputs);
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_copy(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol, const int device_id)
{
	if (!graph->data_parallel.tensor_symbol_idx)
		return NO_TENSOR_SYMBOL;
	assert(graph->data_parallel.tensor_symbol_idx);
	assert(symbol.d >= 0);
	assert(symbol.d < graph->data_parallel.tensor_symbol_size);
	assert(symbol.graph == graph);
	if (device_id == 0)
		return symbol;
	const int parallel_count = graph->data_parallel.count;
	if (graph->data_parallel.tensor_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1] < 0)
		return NO_TENSOR_SYMBOL;
	ccv_nnc_tensor_symbol_t tensor = {
		.d = graph->data_parallel.tensor_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1],
		.graph = graph,
	};
	return tensor;
}

void ccv_nnc_tensor_symbol_set_copy(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol, const int device_id, const ccv_nnc_tensor_symbol_t copy)
{
	assert(graph->data_parallel.tensor_symbol_idx);
	assert(symbol.d >= 0);
	assert(symbol.d < graph->tensor_symbol_info->rnum);
	assert(symbol.graph == graph);
	const int parallel_count = graph->data_parallel.count;
	if (copy.d == CCV_NNC_NO_TENSOR_SYMBOL)
	{
		assert(symbol.d < graph->data_parallel.tensor_symbol_size);
		graph->data_parallel.tensor_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1] = -1;
		return;
	}
	assert(copy.d >= 0);
	assert(copy.d < graph->tensor_symbol_info->rnum);
	assert(copy.graph == graph);
	assert(parallel_count > 1);
	if (symbol.d >= graph->data_parallel.tensor_symbol_size)
	{
		graph->data_parallel.tensor_symbol_idx = ccrealloc(graph->data_parallel.tensor_symbol_idx, sizeof(int) * (parallel_count - 1) * (symbol.d + 1));
		int i;
		for (i = graph->data_parallel.tensor_symbol_size * (parallel_count - 1); i < (symbol.d + 1) * (parallel_count - 1); i++)
			graph->data_parallel.tensor_symbol_idx[i] = -1;
		graph->data_parallel.tensor_symbol_size = symbol.d + 1;
	}
	graph->data_parallel.tensor_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1] = copy.d;
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol_copy(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol, const int device_id)
{
	if (!graph->data_parallel.exec_symbol_idx)
		return NO_GRAPH_EXEC_SYMBOL;
	assert(graph->data_parallel.exec_symbol_idx);
	assert(symbol.d >= 0);
	assert(symbol.d < graph->data_parallel.exec_symbol_size);
	assert(symbol.graph == graph);
	if (device_id == 0)
		return symbol;
	const int parallel_count = graph->data_parallel.count;
	if (graph->data_parallel.exec_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1] < 0)
		return NO_GRAPH_EXEC_SYMBOL;
	ccv_nnc_graph_exec_symbol_t graph_exec = {
		.d = graph->data_parallel.exec_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1],
		.graph = graph,
	};
	return graph_exec;
}

void ccv_nnc_graph_exec_symbol_set_copy(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol, const int device_id, const ccv_nnc_graph_exec_symbol_t copy)
{
	assert(graph->data_parallel.exec_symbol_idx);
	assert(symbol.d >= 0);
	assert(symbol.d < graph->exec_symbol_info->rnum);
	assert(symbol.graph == graph);
	const int parallel_count = graph->data_parallel.count;
	if (copy.d == CCV_NNC_NO_GRAPH_EXEC_SYMBOL)
	{
		assert(symbol.d < graph->data_parallel.exec_symbol_size);
		graph->data_parallel.exec_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1] = -1;
		return;
	}
	assert(copy.d >= 0);
	assert(copy.d < graph->exec_symbol_info->rnum);
	assert(copy.graph == graph);
	assert(parallel_count > 1);
	if (symbol.d >= graph->data_parallel.exec_symbol_size)
	{
		graph->data_parallel.exec_symbol_idx = ccrealloc(graph->data_parallel.exec_symbol_idx, sizeof(int) * (parallel_count - 1) * (symbol.d + 1));
		int i;
		for (i = graph->data_parallel.exec_symbol_size * (parallel_count - 1); i < (symbol.d + 1) * (parallel_count - 1); i++)
			graph->data_parallel.exec_symbol_idx[i] = -1;
		graph->data_parallel.exec_symbol_size = symbol.d + 1;
	}
	graph->data_parallel.exec_symbol_idx[symbol.d * (parallel_count - 1) + device_id - 1] = copy.d;
}
