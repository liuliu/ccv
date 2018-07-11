#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"

/**
 * Level-4.5 API
 */

static void _ccv_nnc_insert_if_prior_to_any(const ccv_nnc_symbolic_graph_t* const graph, const int d, ccv_array_t* const sources, uint32_t* const visited, int* const buf0, int* const buf1)
{
	if (visited[(d >> 5)] & (1u << (d & 31)))
		return;
	visited[(d >> 5)] |= (1u << (d & 31));
	buf0[0] = d;
	int* buf[2] = {
		buf0, buf1
	};
	int buf_size[2] = {
		1, 0
	};
	int p = 0, q = 1;
	int i, j, k;
	int flag = 0;
	while (buf_size[p] > 0)
	{
		buf_size[q] = 0;
		for (i = 0; i < buf_size[p]; i++)
		{
			const int* outgoings; int outgoing_size;
			ccv_nnc_graph_exec_symbol_to(graph, (ccv_nnc_graph_exec_symbol_t){
				.d = buf[p][i],
				.graph = graph
			}, &outgoings, &outgoing_size);
			for (j = 0; j < outgoing_size; j++)
			{
				const int outgoing_idx = outgoings[j];
				for (k = 0; k < sources->rnum; k++)
				{
					ccv_nnc_graph_exec_symbol_t* const source_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, k);
					// If this outgoing idx is one of the source, replace it with d, or delete it.
					if (source_symbol->d == outgoing_idx)
					{
						if (!flag)
						{
							source_symbol->d = d;
							flag = 1;
						} else {
							// Delete this from the list.
							if (k < sources->rnum - 1)
								source_symbol->d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, sources->rnum - 1))->d;
							--sources->rnum;
						}
						break;
					}
				}
				if (visited[(outgoing_idx >> 5)] & (1u << (outgoing_idx & 31)))
					continue;
				visited[(outgoing_idx >> 5)] |= (1u << (outgoing_idx & 31));
				buf[q][buf_size[q]] = outgoing_idx;
				++buf_size[q];
			}
		}
		CCV_SWAP(p, q, i);
	}
	// If this node is not visited, and we cannot find anything in the sources to replace, this is a new top node.
	if (!flag)
	{
		const ccv_nnc_graph_exec_symbol_t source_symbol = {
			.d = d,
			.graph = graph
		};
		ccv_array_push(sources, &source_symbol);
	}
}

static void _ccv_nnc_remove_if_prior_to_any(const ccv_nnc_symbolic_graph_t* const graph, const int d, ccv_array_t* const destinations, uint32_t* const visited, int* const buf0, int* const buf1)
{
	int i, j, k;
	// If it is already visited, this is the later one, we are good.
	if (visited[(d >> 5)] & (1u << (d & 31)))
		return;
	visited[(d >> 5)] |= (1u << (d & 31));
	buf0[0] = d;
	int* buf[2] = {
		buf0, buf1
	};
	int buf_size[2] = {
		1, 0
	};
	int p = 0, q = 1;
	int flag = 0;
	while (!flag && buf_size[p] > 0)
	{
		buf_size[q] = 0;
		for (i = 0; !flag && i < buf_size[p]; i++)
		{
			const int* outgoings; int outgoing_size;
			ccv_nnc_graph_exec_symbol_to(graph, (ccv_nnc_graph_exec_symbol_t){
				.d = buf[p][i],
				.graph = graph
			}, &outgoings, &outgoing_size);
			for (j = 0; j < outgoing_size; j++)
			{
				const int outgoing_idx = outgoings[j];
				// If this node happens to be visited, do nothing.
				if (visited[(outgoing_idx >> 5)] & (1u << (outgoing_idx & 31)))
					continue;
				for (k = 0; !flag && k < destinations->rnum; k++)
				{
					ccv_nnc_graph_exec_symbol_t* const destination_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, k);
					// If this outgoing idx is one of the destination, delete current node.
					flag = (destination_symbol->d == outgoing_idx);
				}
				visited[(outgoing_idx >> 5)] |= (1u << (outgoing_idx & 31));
				buf[q][buf_size[q]] = outgoing_idx;
				++buf_size[q];
			}
		}
		CCV_SWAP(p, q, i);
	}
	if (flag)
		for (i = 0; i < destinations->rnum; i++)
		{
			ccv_nnc_graph_exec_symbol_t* const destination_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, i);
			// If this outgoing idx is one of the destination, delete current node.
			if (destination_symbol->d == d)
			{
				// Delete this from the list.
				if (i < destinations->rnum - 1)
					destination_symbol->d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, destinations->rnum - 1))->d;
				--destinations->rnum;
				break;
			}
		}
}

enum {
	CCV_NNC_SYMBOL_TENSOR,
	CCV_NNC_SYMBOL_TENSOR_ALIAS,
	CCV_NNC_SYMBOL_GRAPH_EXEC,
};

typedef struct {
	int type;
	int d;
} ccv_nnc_tape_symbol_t;

static void _ccv_nnc_dynamic_graph_push_backward_tensor_symbol(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_array_t* const stack = (ccv_array_t*)context;
	ccv_nnc_tape_symbol_t tape_symbol = {
		.d = symbol.d,
		.type = CCV_NNC_SYMBOL_TENSOR,
	};
	ccv_array_push(stack, &tape_symbol);
}

static void _ccv_nnc_dynamic_graph_push_backward_tensor_symbol_alias(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_symbol_t from_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_array_t* const stack = (ccv_array_t*)context;
	ccv_nnc_tape_symbol_t tape_symbol = {
		.d = symbol.d,
		.type = CCV_NNC_SYMBOL_TENSOR_ALIAS,
	};
	ccv_array_push(stack, &tape_symbol);
}

static void _ccv_nnc_dynamic_graph_push_backward_graph_exec_symbol(void* context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
{
	ccv_array_t* const stack = (ccv_array_t*)context;
	ccv_nnc_tape_symbol_t tape_symbol = {
		.d = symbol.d,
		.type = CCV_NNC_SYMBOL_GRAPH_EXEC,
	};
	ccv_array_push(stack, &tape_symbol);
}

void ccv_nnc_dynamic_graph_backward(ccv_nnc_dynamic_graph_t* const dynamic_graph, const ccv_nnc_tensor_variable_t f_variable, const ccv_nnc_tensor_variable_t df_optional, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size)
{
	int d, i, j, k;
	assert(input_size == output_size);
	assert(input_size > 0);
	// Both f_variable and tensor_variable should be, at least, executed. Otherwise we cannot differentiate.
	assert(f_variable->symbol.d >= 0);
	const ccv_nnc_tensor_variable_graph_bind_t* const f_symbol_extra = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, f_variable->symbol.d);
	assert(f_symbol_extra->sources && f_symbol_extra->sources->rnum > 0);
	for (i = 0; i < input_size; i++)
	{
		assert(inputs[i]->symbol.d >= 0);
		assert(((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations &&
			((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations->rnum > 0);
	}
	// Fill in the symbol info for outputs.
	for (i = 0; i < output_size; i++)
		if (ccv_nnc_is_tensor_auto(outputs[i]->info))
			outputs[i]->info = inputs[i]->info;
	const int exec_symbol_info_size = ccv_nnc_graph_exec_symbol_count(dynamic_graph->tape);
	ccv_array_t* const sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
	if (!dynamic_graph->ws)
		dynamic_graph->ws = ccv_array_new(sizeof(int), exec_symbol_info_size * 2 + ((exec_symbol_info_size + 31) >> 5), 0);
	ccv_array_t* const ws = dynamic_graph->ws;
	ccv_array_resize(ws, exec_symbol_info_size * 2 + ((exec_symbol_info_size + 31) >> 5));
	// set visited to all 0.
	memset((uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2), 0, sizeof(uint32_t) * ((exec_symbol_info_size + 31) >> 5));
	for (i = 0; i < input_size; i++)
	{
		ccv_array_t* const destinations = ((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations;
		for (j = 0; j < destinations->rnum; j++)
			_ccv_nnc_insert_if_prior_to_any(dynamic_graph->tape,
				*(int*)ccv_array_get(destinations, j),
				sources, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
	}
	ccv_array_t* const destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), f_symbol_extra->sources->rnum, 0);
	ccv_array_resize(destinations, f_symbol_extra->sources->rnum);
	for (i = 0; i < destinations->rnum; i++)
	{
		ccv_nnc_graph_exec_symbol_t* const symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, i);
		symbol->d = *(int*)ccv_array_get(f_symbol_extra->sources, i);
		symbol->graph = dynamic_graph->tape;
	}
	// Go over sources, because destinations will get removed all the time, thus, the index is not accurate.
	if (destinations->rnum > 1)
		for (i = 0 ; i < f_symbol_extra->sources->rnum; i++)
		{
			memset((uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2), 0, sizeof(uint32_t) * ((exec_symbol_info_size + 31) >> 5));
			_ccv_nnc_remove_if_prior_to_any(dynamic_graph->tape,
				*(int*)ccv_array_get(f_symbol_extra->sources, i),
				destinations, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
		}
	ccv_nnc_tensor_symbol_t input_symbols[input_size];
	for (i = 0; i < input_size; i++)
		input_symbols[i] = inputs[i]->symbol;
	ccv_array_t* const symbol_stack = ccv_array_new(sizeof(ccv_nnc_tape_symbol_t), 1, 0);
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, _ccv_nnc_dynamic_graph_push_backward_tensor_symbol, symbol_stack);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, _ccv_nnc_dynamic_graph_push_backward_tensor_symbol_alias, symbol_stack);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, _ccv_nnc_dynamic_graph_push_backward_graph_exec_symbol, symbol_stack);
	ccv_nnc_symbolic_graph_backward(dynamic_graph->tape,
		&f_variable->symbol, 1, input_symbols, input_size,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum);
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, 0, 0);
	const ccv_nnc_tensor_symbol_t df = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, f_variable->symbol);
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
	// Compiled graph comes from the df.
	ccv_array_clear(sources);
	assert(df.d >= 0);
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
				if (df.d == input || df.d == alias_ref)
				{
					int flag = 0;
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
	int freeable_size = 0;
	ccv_nnc_tensor_variable_t freeables[output_size];
	// Bind dt tensor.
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t symbol = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, input_symbols[i]);
		if (outputs[i]->symbol.d >= 0)
			freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, outputs[i]);
		ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_from_variable(dynamic_graph, outputs[i]);
		const ccv_nnc_tensor_bind_t dt_bind = {
			.symbol = symbol,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &dt_bind);
	}
	ccv_array_clear(destinations);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t symbol = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, input_symbols[i]);
		const ccv_nnc_graph_exec_symbol_t destination = ccv_nnc_graph_exec_symbol_for_backward(dynamic_graph->tape, symbol);
		ccv_array_push(destinations, &destination);
	}
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* exec_arena = 0;
	if (df_optional)
	{
		// If provided df variable, no need to set to all ones.
		const ccv_nnc_tensor_bind_t df_bind = {
			.symbol = df,
			.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, df_optional)
		};
		ccv_array_push(tensor_binds, &df_bind);
		ccv_nnc_symbolic_graph_compile(dynamic_graph->tape,
			(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
			0, 0,
			(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
			(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum,
			&graph, &tensor_arena, &exec_arena);
		ccv_array_free(sources);
	} else {
		ccv_nnc_graph_exec_symbol_t set_ones = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, CMD_SET_FORWARD(1), 0, 0, &df, 1, 0);
		for (i = 0; i < sources->rnum; i++)
			ccv_nnc_graph_exec_symbol_concat(dynamic_graph->tape, set_ones, *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, i));
		ccv_array_free(sources);
		ccv_nnc_symbolic_graph_compile(dynamic_graph->tape,
			(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
			0, 0,
			&set_ones, 1,
			(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum,
			&graph, &tensor_arena, &exec_arena);
		ccv_nnc_graph_exec_symbol_free(dynamic_graph->tape, set_ones);
	}
	ccv_array_free(destinations);
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
