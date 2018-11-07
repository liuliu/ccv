#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

#pragma mark - Level-3.5 API

enum {
	CCV_NNC_PARALLEL_SCATTER = 0x1,
	CCV_NNC_PARALLEL_GATHER = 0x2,
};

void ccv_nnc_symbolic_graph_data_parallel(ccv_nnc_symbolic_graph_t* const graph, int parallel, const ccv_nnc_tensor_symbol_t* const scatters, const int scatter_size, const ccv_nnc_tensor_symbol_t* const gathers, const int gather_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	if (parallel == 1)
		return;
	if (parallel == 0)
		parallel = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
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
	ccv_nnc_symbolic_graph_symbol_infer(graph, visit, sources, source_size, destinations, destination_size, 0, 0, (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, 0), (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0));
	// Set ANY device to default device. Make a list of execution nodes / tensors to be duplicated.
	ccv_array_t* const dup_tensors = ccv_array_new(sizeof(int), 0, 0);
	ccv_array_t* const dup_execs = ccv_array_new(sizeof(int), 0, 0);
	ccv_array_t* const scatter_gather_execs = ccv_array_new(sizeof(int), 0, 0);
	const int tensor_symbol_size = graph->tensor_symbol_info->rnum;
	const int graph_exec_symbol_size = graph->exec_symbol_info->rnum;
	int* const tensor_flags = cccalloc(tensor_symbol_size + graph_exec_symbol_size, sizeof(int));
	int* const exec_flags = tensor_flags + tensor_symbol_size;
	for (i = 0; i < scatter_size; i++)
		tensor_flags[scatters[i].d] = CCV_NNC_PARALLEL_SCATTER;
	for (i = 0; i < gather_size; i++)
		tensor_flags[gathers[i].d] = CCV_NNC_PARALLEL_GATHER;
	ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), node, idx) {
		int parallelizable_data = 0;
		int gather_inputs = 0;
		int scatter_outputs = 0;
		for (i = 0; i < node->input_size; i++)
			if (node->inputs[i] >= 0)
			{
				ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, node->inputs[i]);
				if (CCV_TENSOR_GET_MEMORY(tensor_symbol->info.type) == CCV_TENSOR_GPU_MEMORY)
				{
					if (CCV_TENSOR_GET_DEVICE(tensor_symbol->info.type) == CCV_COMPUTE_DEVICE_ANY)
						CCV_TENSOR_SET_DEVICE_ID(tensor_symbol->info.type, 0);
					if (tensor_flags[node->inputs[i]] == CCV_NNC_PARALLEL_GATHER)
						gather_inputs = 1;
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
					if (tensor_flags[node->outputs[i]] == CCV_NNC_PARALLEL_SCATTER)
						scatter_outputs = 1;
					parallelizable_data = 1;
				}
			}
		assert(!(scatter_outputs && gather_inputs)); // This node cannot be both scatter and gather.
		if (scatter_outputs ^ gather_inputs)
		{
			if (scatter_outputs)
				exec_flags[idx] = CCV_NNC_PARALLEL_SCATTER;
			else if (gather_inputs)
				exec_flags[idx] = CCV_NNC_PARALLEL_GATHER;
			ccv_array_push(scatter_gather_execs, &idx);
		} else if (parallelizable_data && !scatter_outputs && !gather_inputs) {
			// If this node contains GPU data that need to be parallelized, and this node itself is not a scatter node or a gather node..
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
	const int parallel_count = parallel;
	assert(parallel_count > 1);
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
	int max_io_size = 2;
	for (i = 0; i < dup_execs->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(dup_execs, i);
		ccv_nnc_graph_exec_symbol_info_t* const exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
		max_io_size = ccv_max(max_io_size, exec_symbol->input_size + exec_symbol->output_size);
		max_io_size = ccv_max(max_io_size, 2 * exec_symbol->input_size); // For scatter copy.
	}
	for (i = 0; i < scatter_gather_execs->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(scatter_gather_execs, i);
		ccv_nnc_graph_exec_symbol_info_t* const exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
		max_io_size = ccv_max(max_io_size, 2 * exec_symbol->output_size); // For scatter node.
		max_io_size = ccv_max(max_io_size, 2 * exec_symbol->input_size); // For gather node.
	}
	max_io_size = ccv_max(max_io_size, parallel_count + 1); // tensors from all parallel_count, the output is the summed one.
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
	// Create new tensors for gathering.
	int* const gather_tensor_idx = cccalloc(parallel_count * tensor_symbol_size, sizeof(int));
	for (i = 0; i < gather_size; i++)
	{
		const int idx = gathers[i].d;
		ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, idx);
		ccv_nnc_tensor_param_t info = tensor_symbol->info;
		const int flags = tensor_symbol->flags;
		if (tensor_symbol->alias_ref)
		{
			const int alias_ref = tensor_symbol->alias_ref - 1;
			for (j = 0; j < parallel_count; j++)
			{
				int d = gather_tensor_idx[alias_ref * parallel_count + j] - 1;
				if (d < 0)
				{
					// Create the tensor refers to.
					ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, alias_ref);
					const ccv_nnc_tensor_symbol_t new_symbol = ccv_nnc_tensor_symbol_new(graph, tensor_symbol->info, 0);
					ccv_nnc_tensor_symbol_set_flags(graph, new_symbol, flags);
					gather_tensor_idx[alias_ref] = new_symbol.d + 1;
					d = new_symbol.d;
				}
				// Get the tensor symbol again, the original pointer may be invalid after new symbol created.
				ccv_nnc_tensor_symbol_info_t* const tensor_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, idx);
				const ccv_nnc_tensor_symbol_t new_symbol = ccv_nnc_tensor_symbol_alias_new(graph, (ccv_nnc_tensor_symbol_t){
					.d = d,
					.graph = graph,
				}, tensor_symbol->ofs, tensor_symbol->inc, info, 0);
				ccv_nnc_tensor_symbol_set_flags(graph, new_symbol, flags);
				gather_tensor_idx[idx * parallel_count + j] = new_symbol.d + 1;
			}
		} else {
			for (j = 0; j < parallel_count; j++)
			{
				const ccv_nnc_tensor_symbol_t new_symbol = ccv_nnc_tensor_symbol_new(graph, info, 0);
				ccv_nnc_tensor_symbol_set_flags(graph, new_symbol, flags);
				gather_tensor_idx[idx * parallel_count + j] = new_symbol.d + 1;
			}
		}
	}
	const int sum_size = max_io_size / 2; // This is the size of the inputs.
	ccv_nnc_graph_exec_symbol_t sum_symbols[sum_size];
	int gather_inputs[sum_size];
	int* const gather_exec_idx = cccalloc(tensor_symbol_size, sizeof(int));
	// Create node for scatter (thus, transfer data to different parallel_count) and gather (transfer data back to a device, and sum).
	for (i = 0; i < scatter_gather_execs->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(scatter_gather_execs, i);
		// For scatter, we create data transfers as our dup node, and create connections to these data transfers.
		if (exec_flags[d] == CCV_NNC_PARALLEL_SCATTER)
		{
			ccv_nnc_graph_exec_symbol_info_t* const exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
			ccv_nnc_tensor_symbol_t* const inputs = max_io;
			int input_size = 0;
			for (j = 0; j < exec_symbol->output_size; j++)
			{
				const int idx = exec_symbol->outputs[j];
				if (idx >= 0 && tensor_flags[idx] == CCV_NNC_PARALLEL_SCATTER)
				{
					inputs[input_size++] = (ccv_nnc_tensor_symbol_t){
						.d = idx,
						.graph = graph,
					};
					// Reset the tensor flags, it is scattered now.
					tensor_flags[idx] = 0;
				}
			}
			ccv_nnc_tensor_symbol_t* const outputs = max_io + input_size;
			const ccv_nnc_graph_exec_symbol_t source = {
				.d = d,
				.graph = graph,
			};
			for (j = 0; j < parallel_count - 1; j++)
			{
				for (k = 0; k < input_size; k++)
				{
					const int idx = dup_tensor_idx[inputs[k].d * (parallel_count - 1) + j];
					assert(idx >= 0);
					outputs[k].d = idx;
					outputs[k].graph = graph;
				}
				const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_new(graph, CMD_DATA_TRANSFER_FORWARD(), inputs, input_size, outputs, input_size, 0);
				dup_exec_idx[d * (parallel_count - 1) + j] = copy.d;
				ccv_nnc_graph_exec_symbol_concat(graph, source, copy);
			}
		} else if (exec_flags[d] == CCV_NNC_PARALLEL_GATHER) {
			// Gather is a bit more sophisticated, we need to use the new tensor to hold the summed value.
			// This is what we have right now, I will use NCCL later.
			ccv_nnc_tensor_symbol_t* const inputs = max_io;
			int input_size = 0;
			ccv_nnc_graph_exec_symbol_info_t* exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
			for (j = 0; j < exec_symbol->input_size; j++)
			{
				const int idx = exec_symbol->inputs[j];
				if (idx >= 0 && tensor_flags[idx] == CCV_NNC_PARALLEL_GATHER)
				{
					if (gather_exec_idx[idx])
						ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
							.d = gather_exec_idx[idx] - 1,
							.graph = graph,
						}, (ccv_nnc_graph_exec_symbol_t){
							.d = d,
							.graph = graph,
						});
					else {
						// Create a new sum symbol.
						inputs[0].d = idx;
						inputs[0].graph = graph;
						for (k = 1; k < parallel_count; k++)
							inputs[k] = (ccv_nnc_tensor_symbol_t){
								.d = gather_tensor_idx[idx * parallel_count + k] - 1,
								.graph = graph,
							};
						inputs[parallel_count].d = gather_tensor_idx[idx * parallel_count] - 1;
						inputs[parallel_count].graph = graph;
						gather_inputs[input_size] = idx;
						sum_symbols[input_size] = ccv_nnc_graph_exec_symbol_new(graph, CMD_EWSUM_FORWARD(), inputs, parallel_count, inputs + parallel_count, 1, 0);
						// Refresh the pointer to keep it up to date.
						exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
						ccv_nnc_graph_exec_symbol_concat(graph, sum_symbols[input_size], (ccv_nnc_graph_exec_symbol_t){
							.d = d,
							.graph = graph,
						});
						gather_exec_idx[idx] = sum_symbols[input_size].d + 1;
						++input_size;
					}
				}
			}
			if (input_size > 0)
			{
				ccv_nnc_tensor_symbol_t* const outputs = max_io + input_size;
				for (j = 0; j < parallel_count - 1; j++)
				{
					for (k = 0; k < input_size; k++)
					{
						const int idx = gather_inputs[k];
						inputs[k].d = dup_tensor_idx[idx * (parallel_count - 1) + j];
						assert(inputs[k].d >= 0);
						inputs[k].graph = graph,
						outputs[k].d = gather_tensor_idx[idx * parallel_count + j + 1] - 1;
					}
					const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_new(graph, CMD_DATA_TRANSFER_FORWARD(), inputs, input_size, outputs, input_size, 0);
					// Refresh the pointer to keep it up to date.
					exec_symbol = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
					dup_exec_idx[d * (parallel_count - 1) + j] = copy.d;
					for (k = 0; k < input_size; k++)
						ccv_nnc_graph_exec_symbol_concat(graph, copy, sum_symbols[k]);
				}
			}
			// Update the inputs pointing to the summed value.
			for (j = 0; j < exec_symbol->input_size; j++)
			{
				const int idx = exec_symbol->inputs[j];
				if (idx >= 0 && tensor_flags[idx] == CCV_NNC_PARALLEL_GATHER)
					exec_symbol->inputs[j] = gather_tensor_idx[idx * parallel_count] - 1;
			}
		}
	}
	ccv_array_free(scatter_gather_execs);
	ccfree(gather_tensor_idx);
	// If this tensor is not scattered, that means there is no exec to generate this tensor. We just generate headless copy.
	int* const scatter_exec_idx = cccalloc((parallel_count - 1) * tensor_symbol_size, sizeof(int));
	for (i = 0; i < dup_execs->rnum; i++)
	{
		const int idx = *(int*)ccv_array_get(dup_execs, i);
		ccv_nnc_graph_exec_symbol_info_t* const node = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, idx);
		if (exec_flags[idx] == CCV_NNC_PARALLEL_GATHER)
			continue;
		// We try to make copy command as compact as possible by having one copy for multiple tensors if they used together.
		ccv_nnc_tensor_symbol_t* const inputs = max_io;
		int input_size = 0;
		for (j = 0; j < node->input_size; j++)
		{
			const int input = node->inputs[j];
			// Now, figure out whether we need to create copy command.
			if (input >= 0 && input < tensor_symbol_size && tensor_flags[input] == CCV_NNC_PARALLEL_SCATTER)
			{
				inputs[input_size++].d = input;
				tensor_flags[input] = 0; // Rest to 0.
			}
		}
		if (input_size > 0)
		{
			ccv_nnc_tensor_symbol_t* const outputs = max_io + input_size;
			for (j = 0; j < parallel_count - 1; j++)
			{
				for (k = 0; k < input_size; k++)
				{
					assert(dup_tensor_idx[inputs[k].d * (parallel_count - 1) + j] >= 0);
					inputs[k].graph = graph;
					outputs[k].d = dup_tensor_idx[inputs[k].d * (parallel_count - 1) + j];
					outputs[k].graph = graph;
				}
				const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_new(graph, CMD_DATA_TRANSFER_FORWARD(), inputs, input_size, outputs, input_size, 0);
				for (k = 0; k < input_size; k++)
					scatter_exec_idx[inputs[k].d * (parallel_count - 1) + j] = copy.d + 1;
			}
		}
	}
	ccv_array_free(dup_execs);
	// Now everything is dup'ed, connect them all.
	ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), node, idx) {
		for (i = 0; i < parallel_count - 1; i++)
		{
			const int d = dup_exec_idx[idx * (parallel_count - 1) + i];
			if (d < 0)
				continue;
			const ccv_nnc_graph_exec_symbol_t source = {
				.d = d,
				.graph = graph,
			};
			// If it is gather node, no need to scatter for this.
			if (exec_flags[idx] != CCV_NNC_PARALLEL_GATHER)
				for (j = 0; j < node->input_size; j++)
				{
					const int input = node->inputs[j];
					if (input >= 0 && input < tensor_symbol_size && scatter_exec_idx[input * (parallel_count - 1) + i])
						ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
							.d = scatter_exec_idx[input * (parallel_count - 1) + i] - 1,
							.graph = graph,
						}, source);
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
	ccfree(scatter_exec_idx);
	// Check whether this node has outgoing to the gather node, if so, replace that to the sum node.
	ccv_nnc_graph_visit_for(visit, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), node, idx) {
		if (node->outgoings && node->outgoings->rnum)
			for (i = 0; i < node->outgoings->rnum; i++)
			{
				const int outgoing_idx = *(int*)ccv_array_get(node->outgoings, i);
				if (outgoing_idx >= graph_exec_symbol_size)
					continue;
				if (exec_flags[outgoing_idx] == CCV_NNC_PARALLEL_GATHER)
					for (j = 0; j < node->output_size; j++)
					{
						const int output_idx = node->outputs[j];
						if (output_idx >= 0 && tensor_flags[output_idx] == CCV_NNC_PARALLEL_GATHER)
						{
							assert(gather_exec_idx[output_idx]);
							ccv_array_replace_unique_int(node->outgoings, outgoing_idx, gather_exec_idx[output_idx] - 1);
						}
					}
			}
	} ccv_nnc_graph_visit_endfor
	ccfree(gather_exec_idx);
	ccfree(tensor_flags);
	ccv_nnc_graph_visit_free(visit);
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_copy(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol, const int device_id)
{
	assert(graph->data_parallel.tensor_symbol_idx);
	assert(symbol.d >= 0);
	assert(symbol.d < graph->data_parallel.tensor_symbol_size);
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

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol_copy(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t symbol, const int device_id)
{
	assert(graph->data_parallel.exec_symbol_idx);
	assert(symbol.d >= 0);
	assert(symbol.d < graph->data_parallel.exec_symbol_size);
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
