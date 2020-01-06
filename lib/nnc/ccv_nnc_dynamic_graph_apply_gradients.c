#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"

#pragma mark - Level-4.5 API

void ccv_nnc_dynamic_graph_apply_gradients(ccv_nnc_dynamic_graph_t* const dynamic_graph, const ccv_nnc_cmd_t minimizer, const ccv_nnc_tensor_variable_t* const gradients, const int gradient_size, ccv_nnc_tensor_variable_t* const parameters, const int parameter_size, ccv_nnc_tensor_variable_t* const saved_aux, const int parallel, ccv_nnc_stream_context_t* const stream_context)
{
	assert(gradient_size == parameter_size);
	assert(!dynamic_graph->no_grad);
	// Call apply gradients to stateful execs first.
	if (dynamic_graph->stateful_execs)
	{
		khiter_t k;
		for (k = kh_begin(dynamic_graph->stateful_execs); k != kh_end(dynamic_graph->stateful_execs); ++k)
		{
			if (!kh_exist(dynamic_graph->stateful_execs, k))
				continue;
			const int d = kh_key(dynamic_graph->stateful_execs, k);
			const ccv_nnc_cmd_t cmd = ccv_nnc_graph_exec_symbol_cmd(dynamic_graph->tape, (ccv_nnc_graph_exec_symbol_t){
				.graph = dynamic_graph->tape,
				.d = d
			});
			const ccv_nnc_stateful_cmd_vtab_t* const isa = (ccv_nnc_stateful_cmd_vtab_t*)cmd.isa;
			if (isa->apply_gradients)
				isa->apply_gradients(cmd, minimizer, stream_context);
		}
	}
	if (parameter_size == 0)
	{
		ccv_nnc_stream_context_wait(stream_context);
		return;
	}
	const int aux_size = ccv_nnc_minimizer_saved_aux_size(minimizer);
	const int saved_aux_size = parameter_size * aux_size;
	int i, j;
	ccv_nnc_tensor_symbol_t update_inputs[aux_size + 2];
	ccv_nnc_tensor_symbol_t update_outputs[aux_size + 1];
	int freeable_size = 0;
	ccv_nnc_graph_exec_symbol_t minimizes[parameter_size];
	ccv_nnc_tensor_variable_t freeables[parameter_size + saved_aux_size];
	ccv_array_t* const symbol_stack = ccv_array_new(sizeof(ccv_nnc_tape_symbol_t), 1, 0);
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_tensor_symbol, symbol_stack);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_tensor_symbol_alias, symbol_stack);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_graph_exec_symbol, symbol_stack);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), parameter_size * 3 + saved_aux_size * 2, 0);
	const int parallel_count = ccv_max(parallel, 1);
	const int per_parameter_size = parameter_size / parallel_count;
	assert((parameter_size % parallel_count) == 0);
	ccv_nnc_tensor_symbol_t* const allreduce_inputs = parallel_count > 1 ? (ccv_nnc_tensor_symbol_t*)alloca(sizeof(ccv_nnc_tensor_symbol_t) * parallel_count * 2 + sizeof(ccv_nnc_graph_exec_symbol_t) * per_parameter_size) : 0;
	ccv_nnc_tensor_symbol_t* const allreduce_outputs = allreduce_inputs ? allreduce_inputs + parallel_count : 0;
	ccv_nnc_graph_exec_symbol_t* const allreduces = allreduce_outputs ? (ccv_nnc_graph_exec_symbol_t*)(allreduce_outputs + parallel_count) : 0;
	if (parallel_count > 1) // Doing allreduce first.
	{
		for (i = 0; i < per_parameter_size; i++)
		{
			for (j = 0; j < parallel_count; j++)
			{
				const int idx = i + j * per_parameter_size;
				assert(parameters[idx]->symbol.d >= 0);
				const ccv_nnc_tensor_param_t info = parameters[idx]->info;
				const ccv_nnc_tensor_symbol_t gradient = allreduce_inputs[j] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
				const ccv_nnc_tensor_bind_t bind = {
					.symbol = gradient,
					.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, gradients[idx], stream_context)
				};
				ccv_array_push(tensor_binds, &bind);
				allreduce_outputs[j] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
			}
			allreduces[i] = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, CMD_COMM_ALLREDUCE_FORWARD(), allreduce_inputs, parallel_count, allreduce_outputs, parallel_count, 0);
			for (j = 0; j < parallel_count; j++)
			{
				const int idx = i + j * per_parameter_size;
				assert(parameters[idx]->symbol.d >= 0);
				const ccv_nnc_tensor_param_t info = parameters[idx]->info;
				update_inputs[0] = allreduce_outputs[j];
				update_inputs[1] = parameters[idx]->symbol;
				ccv_nnc_tensor_bind_t bind = {
					.symbol = parameters[idx]->symbol,
					.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, parameters[idx], stream_context)
				};
				ccv_array_push(tensor_binds, &bind);
				freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, parameters[idx]);
				bind.symbol = update_outputs[0] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
				bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, parameters[idx], stream_context);
				ccv_array_push(tensor_binds, &bind);
				int k;
				for (k = 0; k < aux_size; k++)
				{
					bind.symbol = update_inputs[2 + k] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
					bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, saved_aux[idx * aux_size + k], stream_context);
					ccv_array_push(tensor_binds, &bind);
					freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, saved_aux[idx * aux_size + k]);
					bind.symbol = update_outputs[1 + k] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
					bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, saved_aux[idx * aux_size + k], stream_context);
					ccv_array_push(tensor_binds, &bind);
				}
				minimizes[idx] = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, minimizer, update_inputs, aux_size + 2, update_outputs, aux_size + 1, 0);
				ccv_nnc_graph_exec_symbol_concat(dynamic_graph->tape, allreduces[i], minimizes[idx]);
			}
		}
	} else {
		for (i = 0; i < per_parameter_size; i++)
		{
			assert(parameters[i]->symbol.d >= 0);
			const ccv_nnc_tensor_param_t info = parameters[i]->info;
			const ccv_nnc_tensor_symbol_t gradient = update_inputs[0] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
			ccv_nnc_tensor_bind_t bind = {
				.symbol = gradient,
				.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, gradients[i], stream_context)
			};
			ccv_array_push(tensor_binds, &bind);
			update_inputs[1] = parameters[i]->symbol;
			bind.symbol = parameters[i]->symbol;
			bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, parameters[i], stream_context);
			ccv_array_push(tensor_binds, &bind);
			freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, parameters[i]);
			bind.symbol = update_outputs[0] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
			bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, parameters[i], stream_context);
			ccv_array_push(tensor_binds, &bind);
			for (j = 0; j < aux_size; j++)
			{
				bind.symbol = update_inputs[2 + j] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
				bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, saved_aux[i * aux_size + j], stream_context);
				ccv_array_push(tensor_binds, &bind);
				freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, saved_aux[i * aux_size + j]);
				bind.symbol = update_outputs[1 + j] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, info, 0);
				bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, saved_aux[i * aux_size + j], stream_context);
				ccv_array_push(tensor_binds, &bind);
			}
			minimizes[i] = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, minimizer, update_inputs, aux_size + 2, update_outputs, aux_size + 1, 0);
		}
	}
	ccv_nnc_dy_xpu_alloc_t xpu_alloc = {
		.graph = dynamic_graph,
		.stream = (intptr_t)stream_context
	};
	ccv_nnc_symbolic_graph_compile_param_t compile_params = {
		.allocator = {
			.isa = &ccv_nnc_dy_allocator_isa,
			.context = {
				.alloc = &xpu_alloc,
				.free = dynamic_graph,
			}
		}
	};
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(dynamic_graph->tape, compile_params,
		(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
		0, 0,
		parallel_count > 1 ? allreduces : minimizes, parallel_count > 1 ? per_parameter_size : parameter_size,
		minimizes, parameter_size,
		&graph, &tensor_arena, &exec_arena);
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, 0, 0);
	ccv_array_free(tensor_binds);
	// Remove newly added symbols to restore the graph.
	for (i = 0; i < symbol_stack->rnum; i++)
	{
		const ccv_nnc_tape_symbol_t* const symbol = (ccv_nnc_tape_symbol_t*)ccv_array_get(symbol_stack, i);
		if (symbol->type == CCV_NNC_SYMBOL_TENSOR || symbol->type == CCV_NNC_SYMBOL_TENSOR_ALIAS)
			ccv_nnc_tensor_symbol_free(dynamic_graph->tape, (ccv_nnc_tensor_symbol_t){
				.d = symbol->d,
				.graph = dynamic_graph->tape
			});
		else if (symbol->type == CCV_NNC_SYMBOL_GRAPH_EXEC)
			ccv_nnc_graph_exec_symbol_free(dynamic_graph->tape, (ccv_nnc_graph_exec_symbol_t){
				.d = symbol->d,
				.graph = dynamic_graph->tape
			});
	}
	ccv_array_free(symbol_stack);
	if (stream_context)
	{
		ccv_nnc_graph_set_default_static_schedule(graph, ccv_nnc_stream_context_type(stream_context));
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, stream_context);
		ccv_nnc_stream_context_wait(stream_context);
	} else if (parallel_count > 1) { // We need to schedule it, now to figure out what stream type we are at.
		int flag = 0;
		for (i = 0; !flag && i < parameter_size; i++)
			flag = (CCV_TENSOR_GET_MEMORY(parameters[i]->info.type) == CCV_TENSOR_GPU_MEMORY);
		const int stream_type = flag ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
		ccv_nnc_graph_set_default_static_schedule(graph, stream_type);
		ccv_nnc_stream_context_t* const default_stream = ccv_nnc_graph_default_stream(graph);
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, default_stream);
		ccv_nnc_stream_context_wait(default_stream);
	} else
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(exec_arena);
	// Now, able to free some of the reused outputs. This need to be the last step otherwise some of the exec symbols
	// above may be freed by this operation.
	for (i = 0; i < freeable_size; i++)
		ccv_nnc_tensor_variable_free(dynamic_graph, freeables[i]);
}
