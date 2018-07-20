#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"

/**
 * Level-4.5 API
 */

void ccv_nnc_dynamic_graph_minimize(ccv_nnc_dynamic_graph_t* const dynamic_graph, const ccv_nnc_cmd_t minimizer, const ccv_nnc_tensor_variable_t* const losses, const int loss_size, const ccv_nnc_tensor_variable_t* const dlosses_optional, ccv_nnc_tensor_variable_t* const parameters, const int parameter_size, ccv_nnc_tensor_variable_t* const saved_aux)
{
	assert(parameter_size > 0);
	assert(loss_size > 0);
	int d, i, j, k;
	int losses_source_size = 0;
	// Both f_variable and tensor_variable should be, at least, executed. Otherwise we cannot differentiate.
	for (i = 0; i < loss_size; i++)
	{
		assert(losses[i]->symbol.d >= 0);
		const ccv_nnc_tensor_variable_graph_bind_t* const loss_symbol_extra = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, losses[i]->symbol.d);
		assert(loss_symbol_extra->sources && loss_symbol_extra->sources->rnum > 0);
		losses_source_size += loss_symbol_extra->sources->rnum;
	}
	for (i = 0; i < parameter_size; i++)
	{
		assert(parameters[i]->symbol.d >= 0);
		assert(((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, parameters[i]->symbol.d))->destinations &&
			((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, parameters[i]->symbol.d))->destinations->rnum > 0);
	}
	const int exec_symbol_info_size = ccv_nnc_graph_exec_symbol_count(dynamic_graph->tape);
	ccv_array_t* const sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
	if (!dynamic_graph->ws)
		dynamic_graph->ws = ccv_array_new(sizeof(int), exec_symbol_info_size * 2 + ((exec_symbol_info_size + 31) >> 5), 0);
	ccv_array_t* const ws = dynamic_graph->ws;
	ccv_array_resize(ws, exec_symbol_info_size * 2 + ((exec_symbol_info_size + 31) >> 5));
	// set visited to all 0.
	memset((uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2), 0, sizeof(uint32_t) * ((exec_symbol_info_size + 31) >> 5));
	for (i = 0; i < parameter_size; i++)
	{
		ccv_array_t* const destinations = ((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, parameters[i]->symbol.d))->destinations;
		for (j = 0; j < destinations->rnum; j++)
			ccv_nnc_insert_if_prior_to_any(dynamic_graph->tape,
				*(int*)ccv_array_get(destinations, j),
				sources, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
	}
	ccv_array_t* const destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), losses_source_size, 0);
	for (i = 0; i < loss_size; i++)
	{
		const ccv_nnc_tensor_variable_graph_bind_t* const loss_symbol_extra = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, losses[i]->symbol.d);
		for (j = 0; j < loss_symbol_extra->sources->rnum; j++)
		{
			const int symbol_d = *(int*)ccv_array_get(loss_symbol_extra->sources, j);
			int flag = 0;
			for (k = 0; !flag && k < destinations->rnum; k++)
				flag = (symbol_d == ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, k))->d);
			if (!flag)
			{
				const ccv_nnc_graph_exec_symbol_t symbol = {
					.d = symbol_d,
					.graph = dynamic_graph->tape
				};
				ccv_array_push(destinations, &symbol);
			}
		}
	}
	// Go over sources, because destinations will get removed all the time, thus, the index is not accurate.
	if (destinations->rnum > 1)
		for (i = 0 ; i < destinations->rnum; i++)
		{
			memset((uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2), 0, sizeof(uint32_t) * ((exec_symbol_info_size + 31) >> 5));
			ccv_nnc_remove_if_prior_to_any(dynamic_graph->tape,
				((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, i))->d,
				destinations, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
		}
	ccv_nnc_tensor_symbol_t loss_symbols[loss_size];
	for (i = 0; i < loss_size; i++)
		loss_symbols[i] = losses[i]->symbol;
	ccv_nnc_tensor_symbol_t parameter_symbols[parameter_size];
	for (i = 0; i < parameter_size; i++)
		parameter_symbols[i] = parameters[i]->symbol;
	ccv_array_t* const symbol_stack = ccv_array_new(sizeof(ccv_nnc_tape_symbol_t), 1, 0);
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_tensor_symbol, symbol_stack);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_tensor_symbol_alias, symbol_stack);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_graph_exec_symbol, symbol_stack);
	ccv_nnc_tensor_symbol_t updated_parameter_symbols[parameter_size];
	const int saved_aux_size = parameter_size * ccv_nnc_minimizer_saved_aux_size(minimizer);
	ccv_nnc_tensor_symbol_map_t saved_aux_symbols[saved_aux_size];
	ccv_nnc_graph_exec_symbol_t update_exec_symbols[parameter_size];
	ccv_nnc_symbolic_graph_minimize(dynamic_graph->tape, minimizer,
		loss_symbols, loss_size, parameter_symbols, parameter_size,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum,
		updated_parameter_symbols, saved_aux_symbols, update_exec_symbols);
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, 0, 0);
	// Bind generated tensors.
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), dynamic_graph->vars->rnum + 2, 0);
	for (i = 0; i < dynamic_graph->vars->rnum; i++)
	{
		ccv_nnc_tensor_variable_t var = *(ccv_nnc_tensor_variable_t*)ccv_array_get(dynamic_graph->vars, i);
		if (var && var->tensor_view && var->symbol.d >= 0)
		{
			ccv_nnc_tensor_bind_t bind = {
				.symbol = var->symbol,
				.tensor = (ccv_nnc_tensor_t*)var->tensor_view
			};
			ccv_array_push(tensor_binds, &bind);
		}
	}
	for (i = 0; i < dynamic_graph->binds->rnum; i++)
	{
		ccv_nnc_tensor_variable_graph_bind_t* const bind = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, i);
		if (bind->index == CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED && bind->tensor_view)
		{
			ccv_nnc_tensor_bind_t b = {
				.symbol = {
					.d = i,
					.graph = dynamic_graph->tape,
				},
				.tensor = (ccv_nnc_tensor_t*)bind->tensor_view
			};
			ccv_array_push(tensor_binds, &b);
		}
	}
	// Compiled graph comes from the dloss.
	ccv_array_clear(sources);
	ccv_nnc_tensor_symbol_t dloss_symbols[loss_size];
	for (i = 0; i < loss_size; i++)
	{
		dloss_symbols[i] = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, losses[i]->symbol);
		assert(dloss_symbols[i].d >= 0);
	}
	for (d = 0; d < destinations->rnum; d++)
	{
		const ccv_nnc_graph_exec_symbol_t* const destination = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, d);
		const int* outgoings; int outgoing_size;
		ccv_nnc_graph_exec_symbol_to(dynamic_graph->tape, *destination, &outgoings, &outgoing_size);
		for (i = 0; i < outgoing_size; i++)
		{
			const int exec_idx = outgoings[i];
			const int* inputs; int input_size;
			ccv_nnc_graph_exec_symbol_io(dynamic_graph->tape, (ccv_nnc_graph_exec_symbol_t){
				.d = exec_idx,
				.graph = dynamic_graph->tape
			}, &inputs, &input_size, 0, 0);
			for (j = 0; j < input_size; j++)
			{
				const int input = inputs[j];
				const int alias_ref = input >= 0 ? ccv_nnc_tensor_symbol_alias_to(dynamic_graph->tape, (ccv_nnc_tensor_symbol_t){
					.d = input,
					.graph = dynamic_graph->tape
				}).d : CCV_NNC_NO_TENSOR_SYMBOL; // This could be CCV_NNC_NO_TENSOR_SYMBOL, which is negative.
				// alias_ref is either exists, or -1.
				int flag = 0;
				for (k = 0; !flag && k < loss_size; k++)
					flag = (dloss_symbols[k].d == input || dloss_symbols[k].d == alias_ref);
				if (flag)
				{
					flag = 0;
					for (k = 0; !flag && k < sources->rnum; k++)
						flag = (exec_idx == ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, k))->d);
					if (!flag)
					{
						const ccv_nnc_graph_exec_symbol_t source = {
							.d = exec_idx,
							.graph = dynamic_graph->tape
						};
						ccv_array_push(sources, &source);
					}
					break;
				}
			}
		}
	}
	ccv_array_free(destinations);
	int freeable_size = 0;
	ccv_nnc_tensor_variable_t freeables[parameter_size + saved_aux_size];
	// Bind dt tensor.
	for (i = 0; i < parameter_size; i++)
	{
		const ccv_nnc_tensor_symbol_t symbol = updated_parameter_symbols[i];
		if (parameters[i]->symbol.d >= 0)
			freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, parameters[i]);
		ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_from_variable(dynamic_graph, parameters[i]);
		const ccv_nnc_tensor_bind_t dt_bind = {
			.symbol = symbol,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &dt_bind);
	}
	for (i = 0; i < saved_aux_size; i++)
	{
		const ccv_nnc_tensor_symbol_map_t symbol_map = saved_aux_symbols[i];
		if (saved_aux[i]->symbol.d >= 0)
			freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, saved_aux[i]);
		ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_from_variable(dynamic_graph, saved_aux[i]);
		ccv_nnc_tensor_bind_t aux_bind = {
			.symbol = symbol_map.source,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &aux_bind);
		aux_bind.symbol = symbol_map.destination;
		ccv_array_push(tensor_binds, &aux_bind);
	}
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* exec_arena = 0;
	if (dlosses_optional)
	{
		// If provided df variable, no need to set to all ones.
		for (i = 0; i < loss_size; i++)
		{
			const ccv_nnc_tensor_bind_t df_bind = {
				.symbol = dloss_symbols[i],
				.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, dlosses_optional[i])
			};
			ccv_array_push(tensor_binds, &df_bind);
		}
		ccv_nnc_symbolic_graph_compile(dynamic_graph->tape,
			(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
			0, 0,
			(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
			update_exec_symbols, parameter_size,
			&graph, &tensor_arena, &exec_arena);
		ccv_array_free(sources);
	} else {
		ccv_nnc_graph_exec_symbol_t set_ones = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, CMD_SET_FORWARD(1), 0, 0, dloss_symbols, loss_size, 0);
		for (i = 0; i < sources->rnum; i++)
			ccv_nnc_graph_exec_symbol_concat(dynamic_graph->tape, set_ones, *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, i));
		ccv_array_free(sources);
		ccv_nnc_symbolic_graph_compile(dynamic_graph->tape,
			(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
			0, 0,
			&set_ones, 1,
			update_exec_symbols, parameter_size,
			&graph, &tensor_arena, &exec_arena);
		ccv_nnc_graph_exec_symbol_free(dynamic_graph->tape, set_ones);
	}
	ccv_array_free(tensor_binds);
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(exec_arena);
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
	// Now, able to free some of the reused outputs. This need to be the last step otherwise some of the exec symbols
	// above may be freed by this operation.
	for (i = 0; i < freeable_size; i++)
		ccv_nnc_tensor_variable_free(dynamic_graph, freeables[i]);
	ccv_array_free(symbol_stack);
}
