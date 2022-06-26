#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_dynamic_graph.h"

// MARK - Level-4.5 API

static void* _ccv_nnc_dynamic_compile_alloc(const int type, const int pinned_mem, const size_t size, void* const arg)
{
	assert(type & CCV_TENSOR_GPU_MEMORY);
	ccv_nnc_dy_xpu_alloc_t* const xpu_alloc  = (ccv_nnc_dy_xpu_alloc_t*)arg;
	const int device = CCV_TENSOR_GET_DEVICE_ID(type);
	return ccv_nnc_xpu_alloc(xpu_alloc->xpu_alloc, device, xpu_alloc->stream, size);
}

static void _ccv_nnc_dynamic_compile_free(void* const ptr, void* const arg)
{
	ccv_nnc_xpu_alloc_t* const xpu_alloc = (ccv_nnc_xpu_alloc_t*)arg;
	ccv_nnc_xpu_free(xpu_alloc, ptr);
}

const ccv_nnc_symbolic_graph_compile_allocator_vtab_t ccv_nnc_dy_allocator_isa = {
	.alloc = _ccv_nnc_dynamic_compile_alloc,
	.free = _ccv_nnc_dynamic_compile_free
};

void ccv_nnc_dynamic_graph_backward(ccv_nnc_dynamic_graph_t* const dynamic_graph, const ccv_nnc_tensor_variable_t* const f_variables, const int f_variable_size, const ccv_nnc_tensor_variable_t* const df_optionals, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int d, i, j, k;
	assert(input_size == output_size);
	assert(input_size > 0);
	assert(output_size > 0);
	assert(f_variable_size > 0);
	int f_source_size = 0;
	// Both f_variable and tensor_variable should be, at least, executed. Otherwise we cannot differentiate.
	for (i = 0; i < f_variable_size; i++)
	{
		assert(f_variables[i]->symbol.d >= 0);
		const ccv_nnc_tensor_variable_graph_bind_t* const f_symbol_extra = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, f_variables[i]->symbol.d);
		assert(f_symbol_extra->sources && f_symbol_extra->sources->rnum > 0);
		f_source_size += f_symbol_extra->sources->rnum;
	}
	assert(!dynamic_graph->no_grad);
	for (i = 0; i < input_size; i++)
	{
		assert(inputs[i]->type != CCV_NNC_TENSOR_CONSTANT);
		assert(inputs[i]->symbol.d >= 0);
		assert(((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations &&
			((ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, inputs[i]->symbol.d))->destinations->rnum > 0);
	}
	// Fill in the symbol info for outputs.
	for (i = 0; i < output_size; i++)
		if (outputs[i] && ccv_nnc_is_tensor_auto(outputs[i]->info))
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
			ccv_nnc_insert_if_prior_to_any(dynamic_graph->tape,
				*(int*)ccv_array_get(destinations, j),
				sources, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
	}
	ccv_array_t* const destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), f_source_size, 0);
	for (i = 0; i < f_variable_size; i++)
	{
		const ccv_nnc_tensor_variable_graph_bind_t* const loss_symbol_extra = (ccv_nnc_tensor_variable_graph_bind_t*)ccv_array_get(dynamic_graph->binds, f_variables[i]->symbol.d);
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
		for (i = 0; i < destinations->rnum; i++)
		{
			memset((uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2), 0, sizeof(uint32_t) * ((exec_symbol_info_size + 31) >> 5));
			ccv_nnc_remove_if_prior_to_any(dynamic_graph->tape,
				((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, i))->d,
				destinations, (uint32_t*)ccv_array_get(ws, exec_symbol_info_size * 2),
				(int*)ccv_array_get(ws, 0), (int*)ccv_array_get(ws, exec_symbol_info_size));
		}
	ccv_nnc_tensor_symbol_t f_symbols[f_variable_size];
	for (i = 0; i < f_variable_size; i++)
		f_symbols[i] = f_variables[i]->symbol;
	ccv_nnc_tensor_symbol_t input_symbols[input_size];
	for (i = 0; i < input_size; i++)
		input_symbols[i] = inputs[i]->symbol;
	ccv_array_t* const symbol_stack = ccv_array_new(sizeof(ccv_nnc_tape_symbol_t), 1, 0);
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_tensor_symbol, symbol_stack);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_tensor_symbol_alias, symbol_stack);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, ccv_nnc_dynamic_graph_push_backward_graph_exec_symbol, symbol_stack);
	ccv_nnc_symbolic_graph_backward(dynamic_graph->tape,
		f_symbols, f_variable_size, input_symbols, input_size,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
		(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum);
	// Bind generated tensors.
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), dynamic_graph->vars->rnum + 2, 0);
	for (i = 0; i < dynamic_graph->vars->rnum; i++)
	{
		ccv_nnc_tensor_variable_t var = *(ccv_nnc_tensor_variable_t*)ccv_array_get(dynamic_graph->vars, i);
		if (var && var->tensor_view && var->symbol.d >= 0)
		{
			ccv_nnc_tensor_bind_t bind = {
				.symbol = var->symbol,
				.tensor = (ccv_nnc_tensor_t*)CCV_NNC_TENSOR_VIEW(var->tensor_view)
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
				.tensor = (ccv_nnc_tensor_t*)CCV_NNC_TENSOR_VIEW(bind->tensor_view)
			};
			ccv_array_push(tensor_binds, &b);
		}
	}
	// Compiled graph comes from the df.
	ccv_array_clear(sources);
	ccv_nnc_tensor_symbol_t df_symbols[f_variable_size];
	for (i = 0; i < f_variable_size; i++)
	{
		df_symbols[i] = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, f_variables[i]->symbol);
		assert(f_symbols[i].d >= 0);
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
				for (k = 0; !flag && k < f_variable_size; k++)
					flag = (df_symbols[k].d == input || df_symbols[k].d == alias_ref);
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
	int freeable_size = 0;
	ccv_nnc_tensor_variable_t freeables[output_size];
	ccv_array_clear(destinations);
	int max_input_size = 1;
	int max_output_size = 1;
	for (i = 0; i < sources->rnum; i++)
	{
		const ccv_nnc_graph_exec_symbol_t source = *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, i);
		int input_size; int output_size;
		ccv_nnc_graph_exec_symbol_io(dynamic_graph->tape, source, 0, &input_size, 0, &output_size);
		max_input_size = ccv_max(input_size, max_input_size);
		max_output_size = ccv_max(output_size, max_output_size);
	}
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t symbol = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, input_symbols[i]);
		ccv_nnc_graph_exec_symbol_t destination = ccv_nnc_graph_exec_symbol_for_backward(dynamic_graph->tape, symbol);
		int input_size; int output_size;
		ccv_nnc_graph_exec_symbol_io(dynamic_graph->tape, destination, 0, &input_size, 0, &output_size);
		max_input_size = ccv_max(input_size, max_input_size);
		max_output_size = ccv_max(output_size, max_output_size);
	}
	const int max_input_bitmask_size = ((max_input_size + 63) >> 6);
	const int max_output_bitmask_size =  ((max_output_size + 63) >> 6);
	ccv_nnc_tensor_symbol_t temp_input_symbols[max_input_size];
	ccv_nnc_tensor_symbol_t temp_output_symbols[max_output_size];
	uint64_t temp_input_bitmasks[max_input_bitmask_size];
	uint64_t temp_output_bitmasks[max_output_bitmask_size];
	// Bind dt tensor.
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t symbol = ccv_nnc_tensor_symbol_for_backward(dynamic_graph->tape, input_symbols[i]);
		ccv_nnc_graph_exec_symbol_t destination = ccv_nnc_graph_exec_symbol_for_backward(dynamic_graph->tape, symbol);
		if (outputs[i])
		{
			if (ccv_nnc_tensor_variable_contains_value(outputs[i]))
			{
				// If the output tensors already exist, we need to accumulate the result.
				// However, if this tensor is set from outside, we don't accumulate on that
				// (these maybe people just want to collect the result in explicit way).
				// On the other hand, if these external tensor views has a symbol associated
				// with them, they are not made to collect results. They are probably bind in
				// previous computations.
				// The above logic is convoluted, but it should make intuitive sense in many
				// cases.
				ccv_nnc_tensor_symbol_t inputs[2];
				inputs[0] = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, outputs[i]->info, 0);
				inputs[1] = symbol;
				const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(dynamic_graph->tape, outputs[i]->info, 0);
				ccv_nnc_tensor_bind_t dt_bind = {
					.symbol = inputs[0],
					.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, outputs[i], stream_context)
				};
				ccv_array_push(tensor_binds, &dt_bind);
				ccv_nnc_graph_exec_symbol_t accum = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, CMD_EWSUM_FORWARD(), inputs, 2, &output, 1, 0);
				ccv_nnc_graph_exec_symbol_concat(dynamic_graph->tape, destination, accum);
				destination = accum; // The accumulation unit becomes the new destination.
				freeables[freeable_size++] = ccv_nnc_tensor_variable_exchange_new(dynamic_graph, outputs[i]);
				dt_bind.symbol = output;
				dt_bind.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, outputs[i], stream_context);
				ccv_array_push(tensor_binds, &dt_bind);
			} else {
				assert(outputs[i]->symbol.d < 0);
				// Otherwise, we can directly bind to the backward output.
				ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_from_variable(dynamic_graph, outputs[i], stream_context);
				const ccv_nnc_tensor_bind_t dt_bind = {
					.symbol = symbol,
					.tensor = tensor
				};
				ccv_array_push(tensor_binds, &dt_bind);
			}
		} else {
			// Remove this symbol if it is possible, since we don't have any use of it.
			// This won't cover cases where we need to merge them together (hence, the cmd will be sum), so it is the best guess.
			const int* inputs; int input_size;
			const int* outputs; int output_size;
			ccv_nnc_graph_exec_symbol_io(dynamic_graph->tape, destination, &inputs, &input_size, &outputs, &output_size);
			ccv_nnc_tensor_symbol_t* input_symbols = temp_input_symbols;
			ccv_nnc_tensor_symbol_t* output_symbols = temp_output_symbols;
			uint64_t* input_bitmasks = temp_input_bitmasks;
			uint64_t* output_bitmasks = temp_output_bitmasks;
			memset(input_bitmasks, 0, sizeof(uint64_t) * ccv_max(1, ((input_size + 63) >> 6)));
			memset(output_bitmasks, 0, sizeof(uint64_t) * ccv_max(1, ((output_size + 63) >> 6)));
			const ccv_nnc_cmd_t cmd = ccv_nnc_graph_exec_symbol_cmd(dynamic_graph->tape, destination);
			// Now, check to see if we can remove this symbol from this source.
			for (k = 0; k < input_size; k++)
				if (inputs[k] >= 0)
					input_bitmasks[k >> 6] |= ((uint64_t)1 << (k & 63));
			int flag = 0;
			for (k = 0; k < output_size; k++)
				if (outputs[k] >= 0 && outputs[k] != symbol.d)
				{
					output_bitmasks[k >> 6] |= ((uint64_t)1 << (k & 63));
					flag = 1;
				}
			// If we can omit this output (or there is no output at all).
			if (!flag || ccv_nnc_cmd_bitmask(cmd, input_size, output_size, input_bitmasks, (input_size + 63) >> 6, output_bitmasks, (output_size + 63) >> 6))
			{
				// Set the new outputs by omitting the one.
				for (k = 0; k < input_size; k++)
					input_symbols[k] = (ccv_nnc_tensor_symbol_t){
						.d = inputs[k],
						.graph = inputs[k] != CCV_NNC_NO_TENSOR_SYMBOL ? dynamic_graph->tape : 0,
					};
				for (k = 0; k < output_size; k++)
					if (outputs[k] != symbol.d)
						output_symbols[k] = (ccv_nnc_tensor_symbol_t){
							.d = outputs[k],
							.graph = outputs[k] != CCV_NNC_NO_TENSOR_SYMBOL ? dynamic_graph->tape : 0,
						};
					else
						output_symbols[k] = (ccv_nnc_tensor_symbol_t){
							.d = CCV_NNC_NO_TENSOR_SYMBOL,
							.graph = 0,
						};
				ccv_nnc_graph_exec_symbol_set_io(dynamic_graph->tape, destination, input_symbols, input_size, output_symbols, output_size);
				// If there is no output, and this is not custom (custom may have side effect,
				// whereas the normal ops are side-effect free), set this symbol to be a noop.
				// TODO: This could be other cases regarding CCV_NNC_GRAPH_BACKWARD.
				if (!flag &&
					cmd.cmd != CCV_NNC_CUSTOM_FORWARD &&
					cmd.cmd != CCV_NNC_CUSTOM_BACKWARD)
					ccv_nnc_graph_exec_symbol_set(dynamic_graph->tape, destination, ccv_nnc_cmd(CCV_NNC_NOOP, 0, ccv_nnc_cmd_auto, 0));
			}
		}
		ccv_array_push(destinations, &destination);
	}
	// Remove the hook only at this point.
	ccv_nnc_tensor_symbol_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_tensor_symbol_alias_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_graph_exec_symbol_new_hook(dynamic_graph->tape, 0, 0);
	ccv_nnc_dy_xpu_alloc_t xpu_alloc = {
		.xpu_alloc = &dynamic_graph->xpu_alloc,
		.stream = stream_context
	};
	ccv_nnc_symbolic_graph_compile_param_t compile_params = {
		.allocator = {
			.isa = &ccv_nnc_dy_allocator_isa,
			.context = {
				.alloc = &xpu_alloc,
				.free = &dynamic_graph->xpu_alloc,
			}
		}
	};
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* exec_arena = 0;
	// TODO: Should apply simplification right after the backward pass generated.
	// Remove these if it is not needed by the cmd, for example, if absence assumed to be 1.
	for (i = 0; i < f_variable_size; i++)
	{
		if (df_optionals && df_optionals[i])
		{
			const ccv_nnc_tensor_bind_t df_bind = {
				.symbol = df_symbols[i],
				.tensor = ccv_nnc_tensor_from_variable(dynamic_graph, df_optionals[i], stream_context)
			};
			ccv_array_push(tensor_binds, &df_bind);
			continue;
		}
		if (!df_symbols[i].graph) // Skip.
			continue;
		int no_set = 0; // If we cannot find the df_symbols in all sources, we cannot predict whether it is used or not.
		for (j = 0; j < sources->rnum; j++)
		{
			const ccv_nnc_graph_exec_symbol_t source = *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, j);
			const int* inputs; int input_size;
			const int* outputs; int output_size;
			ccv_nnc_graph_exec_symbol_io(dynamic_graph->tape, source, &inputs, &input_size, &outputs, &output_size);
			const ccv_nnc_cmd_t cmd = ccv_nnc_graph_exec_symbol_cmd(dynamic_graph->tape, source);
			int flag = 0;
			for (k = 0; !flag && k < input_size; k++)
			{
				const int alias_ref = inputs[k] >= 0 ? ccv_nnc_tensor_symbol_alias_to(dynamic_graph->tape, (ccv_nnc_tensor_symbol_t){
					.d = inputs[k],
					.graph = dynamic_graph->tape
				}).d : CCV_NNC_NO_TENSOR_SYMBOL; // This could be CCV_NNC_NO_TENSOR_SYMBOL, which is negative.
				flag = (df_symbols[i].d == inputs[k] || df_symbols[i].d == alias_ref);
			}
			if (flag)
			{
				no_set = 1;
				// Now, check to see if we can remove this symbol from this source.
				memset(temp_input_bitmasks, 0, sizeof(uint64_t) * ccv_max(1, ((input_size + 63) >> 6)));
				memset(temp_output_bitmasks, 0, sizeof(uint64_t) * ccv_max(1, ((output_size + 63) >> 6)));
				for (k = 0; k < input_size; k++)
					if (inputs[k] >= 0)
					{
						const int alias_ref = inputs[k] >= 0 ? ccv_nnc_tensor_symbol_alias_to(dynamic_graph->tape, (ccv_nnc_tensor_symbol_t){
							.d = inputs[k],
							.graph = dynamic_graph->tape
						}).d : CCV_NNC_NO_TENSOR_SYMBOL; // This could be CCV_NNC_NO_TENSOR_SYMBOL, which is negative.
						if (df_symbols[i].d != inputs[k] && df_symbols[i].d != alias_ref)
							temp_input_bitmasks[k >> 6] |= ((uint64_t)1 << (k & 63));
					}
				for (k = 0; k < output_size; k++)
					if (outputs[k] >= 0)
						temp_output_bitmasks[k >> 6] |= ((uint64_t)1 << (k & 63));
				if (!ccv_nnc_cmd_bitmask(cmd, input_size, output_size, temp_input_bitmasks, (input_size + 63) >> 6, temp_output_bitmasks, (output_size + 63) >> 6))
					no_set = 0;
			}
		}
		if (no_set) // Remove this flag from all sources and continue.
		{
			for (j = 0; j < sources->rnum; j++)
			{
				const ccv_nnc_graph_exec_symbol_t source = *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, j);
				const int* inputs; int input_size;
				const int* outputs; int output_size;
				ccv_nnc_graph_exec_symbol_io(dynamic_graph->tape, source, &inputs, &input_size, &outputs, &output_size);
				int flag = 0;
				for (k = 0; !flag && k < input_size; k++)
				{
					const int alias_ref = inputs[k] >= 0 ? ccv_nnc_tensor_symbol_alias_to(dynamic_graph->tape, (ccv_nnc_tensor_symbol_t){
						.d = inputs[k],
						.graph = dynamic_graph->tape
					}).d : CCV_NNC_NO_TENSOR_SYMBOL; // This could be CCV_NNC_NO_TENSOR_SYMBOL, which is negative.
					flag = (df_symbols[i].d == inputs[k] || df_symbols[i].d == alias_ref);
				}
				if (flag)
				{
					for (k = 0; k < input_size; k++)
						if (inputs[k] >= 0)
						{
							const int alias_ref = inputs[k] >= 0 ? ccv_nnc_tensor_symbol_alias_to(dynamic_graph->tape, (ccv_nnc_tensor_symbol_t){
								.d = inputs[k],
								.graph = dynamic_graph->tape
							}).d : CCV_NNC_NO_TENSOR_SYMBOL; // This could be CCV_NNC_NO_TENSOR_SYMBOL, which is negative.
							const int no_symbol = df_symbols[i].d == inputs[k] || df_symbols[i].d == alias_ref;
							temp_input_symbols[k] = (ccv_nnc_tensor_symbol_t){
								.d = no_symbol ? CCV_NNC_NO_TENSOR_SYMBOL : inputs[k],
								.graph = no_symbol ? 0 : dynamic_graph->tape,
							};
						} else {
							temp_input_symbols[k] = (ccv_nnc_tensor_symbol_t){
								.d = inputs[k],
								.graph = inputs[k] != CCV_NNC_NO_TENSOR_SYMBOL ? dynamic_graph->tape : 0,
							};
						}
					for (k = 0; k < output_size; k++)
						temp_output_symbols[k] = (ccv_nnc_tensor_symbol_t){
							.d = outputs[k],
							.graph = outputs[k] != CCV_NNC_NO_TENSOR_SYMBOL ? dynamic_graph->tape : 0,
						};
					ccv_nnc_graph_exec_symbol_set_io(dynamic_graph->tape, source, temp_input_symbols, input_size, temp_output_symbols, output_size);
				}
			}
			df_symbols[i].graph = 0;
		}
	}
	// Aggregate them into one set command.
	ccv_nnc_tensor_symbol_t df_symbols_0[f_variable_size];
	ccv_nnc_graph_exec_symbol_t set_ones[f_variable_size];
	int set_one_size = 0;
	for (i = 0; i < f_variable_size;)
		if ((df_optionals && df_optionals[i]) || !df_symbols[i].graph) // Skip.
			++i;
		else {
			df_symbols_0[0] = df_symbols[i];
			k = 1;
			int idx = f_variable_size;
			const ccv_nnc_tensor_param_t params_0 = ccv_nnc_tensor_symbol_params(dynamic_graph->tape, df_symbols_0[0]);
			for (j = i + 1; j < f_variable_size; j++)
				if (df_symbols[j].graph)
				{
					const ccv_nnc_tensor_param_t params_j = ccv_nnc_tensor_symbol_params(dynamic_graph->tape, df_symbols[j]);
					if (params_j.type != params_0.type)
					{
						if (idx == f_variable_size)
							idx = j;
					} else {
						df_symbols_0[k++] = df_symbols[j];
						assert(df_symbols[j].graph == dynamic_graph->tape);
						df_symbols[j].graph = 0;
					}
				}
			i = idx;
			set_ones[set_one_size] = ccv_nnc_graph_exec_symbol_new(dynamic_graph->tape, CMD_SET_FORWARD(1), 0, 0, df_symbols_0, k, 0);
			for (j = 0; j < sources->rnum; j++)
				ccv_nnc_graph_exec_symbol_concat(dynamic_graph->tape, set_ones[set_one_size], *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, j));
			++set_one_size;
		}
	// Reset it back.
	for (i = 0; i < f_variable_size; i++)
		df_symbols[i].graph = dynamic_graph->tape;
	if (set_one_size > 0)
	{
		ccv_nnc_symbolic_graph_compile(dynamic_graph->tape, compile_params,
			(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
			0, 0,
			set_ones, set_one_size,
			(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum,
			&graph, &tensor_arena, &exec_arena);
	} else {
		// Otherwise we don't have a single set ones, in this case, we still compile from source.
		ccv_nnc_symbolic_graph_compile(dynamic_graph->tape, compile_params,
			(ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum,
			0, 0,
			(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, 0), sources->rnum,
			(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, 0), destinations->rnum,
			&graph, &tensor_arena, &exec_arena);
	}
	ccv_array_free(sources);
	for (i = 0; i < set_one_size; i++)
		ccv_nnc_graph_exec_symbol_free(dynamic_graph->tape, set_ones[i]);
	ccv_array_free(destinations);
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
	// Go through inputs and outputs to find out stream type and parallel counts.
	int multi_device = 0;
	for (i = 1; !multi_device && i < input_size; i++)
		multi_device = (CCV_TENSOR_GET_DEVICE(inputs[i - 1]->info.type) != CCV_TENSOR_GET_DEVICE(inputs[i]->info.type));
	if (stream_context)
	{
		ccv_nnc_graph_set_default_static_schedule(graph, ccv_nnc_stream_context_type(stream_context));
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, stream_context);
		ccv_nnc_tensor_arena_buffer_free(tensor_arena);
		ccv_nnc_compilation_artifact_t* const artifact = ccv_nnc_compilation_artifact_new(graph, tensor_arena, exec_arena);
		ccv_nnc_stream_context_add_callback(stream_context, (ccv_nnc_callback_f)ccv_nnc_compilation_artifact_free, artifact);
	} else {
		if (multi_device)
		{
			int flag = 0;
			for (i = 0; !flag && i < input_size; i++)
				flag = (CCV_TENSOR_GET_MEMORY(inputs[i]->info.type) == CCV_TENSOR_GPU_MEMORY);
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
	}
	// Now, able to free some of the reused outputs. This need to be the last step otherwise some of the exec symbols
	// above may be freed by this operation.
	for (i = 0; i < freeable_size; i++)
		ccv_nnc_tensor_variable_free(dynamic_graph, freeables[i]);
}
