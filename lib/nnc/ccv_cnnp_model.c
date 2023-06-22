#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"

// MARK - Level-5 API

ccv_cnnp_model_io_t ccv_cnnp_model_apply(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t* const inputs, const int input_size)
{
	if (!model->io)
		model->io = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
	ccv_cnnp_model_io_t model_io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s) + sizeof(ccv_nnc_tensor_symbol_t) * model->output_size);
	model_io->param_ref = 0;
	model_io->param_sel = 0;
	model_io->visit = 0;
	model_io->model = model;
	model_io->dependencies = 0;
	model_io->dependents = 0;
	model_io->outgoings = 0;
	model_io->outputs = (ccv_nnc_tensor_symbol_t*)(model_io + 1);
	ccv_array_push(model->io, &model_io);
	if (input_size > 0)
	{
		model_io->incomings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), input_size, 0);
		ccv_array_resize(model_io->incomings, input_size);
		int i;
		memcpy(ccv_array_get(model_io->incomings, 0), inputs, sizeof(ccv_cnnp_model_io_t) * input_size);
		for (i = 0; i < input_size; i++)
		{
			if (!inputs[i]->outgoings)
				inputs[i]->outgoings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
			ccv_array_push(inputs[i]->outgoings, &model_io);
		}
	} else {
		model_io->incomings = 0;
	}
	return model_io;
}

void ccv_cnnp_model_add_dependencies(ccv_cnnp_model_io_t model_io, const ccv_cnnp_model_io_t* const dependencies, const int dependency_size)
{
	assert(dependency_size > 0);
	if (!model_io->dependencies)
		model_io->dependencies = ccv_array_new(sizeof(ccv_cnnp_model_io_t), dependency_size, 0);
	int i, j;
	for (i = 0; i < dependency_size; i++)
	{
		int flag = 0;
		// Check if it is already exist or not.
		for (j = 0; !flag && j < model_io->dependencies->rnum; j++)
			if (*(ccv_cnnp_model_io_t*)ccv_array_get(model_io->dependencies, j) == dependencies[i])
				flag = 1;
		if (flag)
			continue;
		ccv_array_push(model_io->dependencies, dependencies + i);
		++dependencies[i]->dependents;
	}
}

int ccv_cnnp_model_output_size(const ccv_cnnp_model_t* const model)
{
	return model->output_size;
}

ccv_cnnp_model_io_t ccv_cnnp_model_parameters(ccv_cnnp_model_t* const model, const int selector, const int index)
{
	if (!model->io)
		model->io = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
	ccv_cnnp_model_io_t model_io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
	model_io->param_ref = index >= 0 ? index + 1 : ALL_PARAMETERS;
	model_io->param_sel = selector >= 0 ? selector + 1 : ALL_PARAMETERS;
	model_io->visit = 0;
	model_io->model = model;
	model_io->outputs = 0;
	model_io->dependencies = 0;
	model_io->dependents = 0;
	model_io->incomings = 0;
	model_io->outgoings = 0;
	ccv_array_push(model->io, &model_io);
	return model_io;
}

void ccv_cnnp_model_notify_hook(ccv_cnnp_model_t* const model, ccv_cnnp_model_notify_f func, void* const context)
{
	model->notify_hook.func = func;
	model->notify_hook.context = context;
}

void ccv_cnnp_model_notify(const ccv_cnnp_model_t* const model, const int tag, void* const payload)
{
	if (model->notify_hook.func)
		model->notify_hook.func(model, tag, payload, model->notify_hook.context);
	if (model->isa->notify)
		model->isa->notify(model, tag, payload);
}

static int _ccv_nnc_array_dedup_graph_exec_symbols(ccv_nnc_graph_exec_symbol_t* const graph_exec_symbols, int graph_exec_symbol_size)
{
	int i, j;
	for (i = 0; i < graph_exec_symbol_size; i++)
	{
		ccv_nnc_graph_exec_symbol_t* const graph_exec_symbol = graph_exec_symbols + i;
		// Check whether this tensor symbol has any duplicate.
		for (j = i + 1; j < graph_exec_symbol_size;)
		{
			ccv_nnc_graph_exec_symbol_t* const other_symbol = graph_exec_symbols + j;
			// If there is a same tensor symbol, remove it.
			if (other_symbol->d == graph_exec_symbol->d && other_symbol->graph == graph_exec_symbol->graph)
			{
				if (j + 1 < graph_exec_symbol_size)
					*other_symbol = graph_exec_symbols[graph_exec_symbol_size - 1];
				--graph_exec_symbol_size;
				continue;
			}
			++j;
		}
	}
	return graph_exec_symbol_size;
}

typedef struct {
	ccv_cnnp_model_sequence_t* sequence;
	char prefix;
	ccv_array_t* symbols;
	ccv_array_t* ids;
	ccv_array_t* trainables;
} ccv_cnnp_model_add_to_array_context_t;

static void _ccv_cnnp_add_to_array(void* const context, const ccv_nnc_tensor_symbol_t symbol, const int is_trainable)
{
	ccv_cnnp_model_add_to_array_context_t* const add_to_array_context = (ccv_cnnp_model_add_to_array_context_t*)context;
	ccv_cnnp_model_t* const model = add_to_array_context->sequence->model;
	int i;
	if (!model->parameter_indices)
		model->parameter_indices = ccv_array_new(sizeof(int), 0, 0);
	for (i = 0; i < add_to_array_context->symbols->rnum; i++)
	{
		const ccv_nnc_tensor_symbol_t other_symbol = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(add_to_array_context->symbols, i);
		if (other_symbol.d == symbol.d && other_symbol.graph == symbol.graph)
		{
			// Only add to parameter_indices if it is trainable.
			if (add_to_array_context->prefix == 't')
				ccv_array_add_unique_int(model->parameter_indices, i);
			// Found it, return, don't add it.
			return;
		}
	}
	// Only add to parameter_indices if it is trainable.
	if (add_to_array_context->prefix == 't')
		ccv_array_push(model->parameter_indices, &add_to_array_context->symbols->rnum);
	// This is a new one, no need to add_unique_int, it is unique.
	ccv_array_push(add_to_array_context->symbols, &symbol);
	if (add_to_array_context->trainables)
		ccv_array_push(add_to_array_context->trainables, &is_trainable);
	char id[2048];
	id[0] = add_to_array_context->prefix;
	id[1] = '-';
	int total_len = 2;
	for (i = 0; i < add_to_array_context->sequence->sequences->rnum; i++)
	{
		const ccv_cnnp_model_name_t* const name = (ccv_cnnp_model_name_t*)ccv_array_get(add_to_array_context->sequence->sequences, i);
		int len;
		if (name->name && name->name[0] != '\0')
			len = snprintf(id + total_len, 2048 - total_len, "%s-%d-", name->name, name->sequence);
		else
			len = snprintf(id + total_len, 2048 - total_len, "%d-", name->sequence);
		total_len += len;
		if (total_len >= 2047)
			break;
	}
	if (total_len < 2047)
		total_len += snprintf(id + total_len, 2048 - total_len, "%d", add_to_array_context->sequence->it);
	assert(total_len < 2048);
	char *heap_id = (char*)ccmalloc(total_len + 1);
	memcpy(heap_id, id, total_len + 1);
	ccv_array_push(add_to_array_context->ids, &heap_id);
	++add_to_array_context->sequence->it;
}

static void _ccv_cnnp_compiled_data_init(ccv_cnnp_compiled_data_t* const compiled_data, const int output_size)
{
	compiled_data->f = compiled_data->fits + output_size;
	compiled_data->xpu_alloc.mp_hdr = -1;
	compiled_data->xpu_alloc.freed = kh_init(dy_str);
	compiled_data->xpu_alloc.allocd = kh_init(dy_alloc);
}

static void _ccv_cnnp_model_compile(ccv_cnnp_model_t* const model, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_cmd_t loss)
{
	assert(model->graph);
	model->inputs = ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * input_size);
	int i;
	for (i = 0; i < input_size; i++)
		model->inputs[i] = ccv_nnc_tensor_symbol_new(model->graph, inputs[i], 0);
	ccv_array_t* const parameters = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_t* const parameter_ids = ccv_array_new(sizeof(char*), 0, 0);
	ccv_array_t* const parameter_trainables = ccv_array_new(sizeof(int), 0, 0);
	ccv_cnnp_model_sequence_t model_sequence = {
		.bank = kh_init(ccv_cnnp_model_name_bank)
	};
	ccv_cnnp_model_add_to_array_context_t add_to_parameter_context = {
		.sequence = &model_sequence,
		.prefix = 't',
		.symbols = parameters,
		.ids = parameter_ids,
		.trainables = parameter_trainables,
	};
	ccv_array_t* const internals = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_t* const internal_ids = ccv_array_new(sizeof(char*), 0, 0);
	ccv_cnnp_model_add_to_array_context_t add_to_output_context = {
		.sequence = &model_sequence,
		.prefix = 'r',
		.symbols = internals,
		.ids = internal_ids,
		.trainables = 0,
	};
	ccv_cnnp_model_build_data_t build_data = {
		.is_trainable = model->is_trainable >= 0 ? model->is_trainable : 1,
		.model_sequence = &model_sequence,
		.add_to_array = _ccv_cnnp_add_to_array,
		.parameters = parameters,
		.context = {
			.add_to_parameter = &add_to_parameter_context,
			.add_to_output = &add_to_output_context,
		},
	};
	model->data = &build_data;
	ccv_cnnp_model_build(model, model->graph, model->inputs, input_size, 0, 0);
	model->data = 0;
	kh_destroy(ccv_cnnp_model_name_bank, model_sequence.bank);
	if (model_sequence.sequences)
		ccv_array_free(model_sequence.sequences);
	// Check if there are parameters that are not trainables. If there are, we will allocate uint64 bitmap to record that.
	int not_trainables = 0;
	// Assert no parameter is alias.
	for (i = 0; i < parameters->rnum; i++)
	{
		const ccv_nnc_tensor_symbol_t parameter = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(parameters, i);
		const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(parameter.graph, parameter);
		assert(alias_to.graph == 0); // Cannot find the one alias to.
		if (*(int*)ccv_array_get(parameter_trainables, i) == 0)
			not_trainables = 1;
	}
	assert(parameters->rnum == parameter_trainables->rnum);
	uint64_t* parameter_flags = 0;
	if (not_trainables)
	{
		parameter_flags = (uint64_t*)cccalloc(((parameters->rnum + 63) >> 6), sizeof(uint64_t));
		for (i = 0; i < parameter_trainables->rnum; i++)
			if (*(int*)ccv_array_get(parameter_trainables, i))
				parameter_flags[i >> 6] |= ((uint64_t)1 << (i & 63));
	}
	ccv_array_free(parameter_trainables);
	// Assert no internal is alias.
	for (i = 0; i < internals->rnum; i++)
	{
		const ccv_nnc_tensor_symbol_t internal = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(internals, i);
		const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(internal.graph, internal);
		assert(alias_to.graph == 0); // Cannot find the one alias to.
	}
	const int output_size = model->output_size;
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(model->graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
			CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
			CCV_NNC_SIMPLIFY_OPS_FUSION,
			CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		model->inputs, input_size,
		model->outputs, output_size,
		SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
	ccv_cnnp_compiled_data_t* compiled_data = model->compiled_data = cccalloc(1, sizeof(ccv_cnnp_compiled_data_t) + sizeof(ccv_nnc_tensor_symbol_t) * (output_size * 2 - 1));
	_ccv_cnnp_compiled_data_init(compiled_data, output_size);
	const int evaluate_to_size = compiled_data->evaluate.to_size = ccv_nnc_symbolic_graph_destination_size(model->graph);
	assert(evaluate_to_size > 0);
	compiled_data->evaluate.tos = ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * evaluate_to_size);
	memcpy(compiled_data->evaluate.tos, ccv_nnc_symbolic_graph_destinations(model->graph), sizeof(ccv_nnc_graph_exec_symbol_t) * evaluate_to_size);
	compiled_data->loss = loss;
	if (loss.cmd == CCV_NNC_NOOP)
	{
		// If no loss function provided, there is no fits.
		for (i = 0; i < output_size; i++)
		{
			compiled_data->fits[i] = NO_TENSOR_SYMBOL;
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(model->graph, model->outputs[i]);
			if (alias_to.d < 0)
				compiled_data->f[i] = model->outputs[i];
			else { // We cannot differentiate against an alias, therefore, we have to verify this output is full, and we can diff against the original.
				int ofs[CCV_NNC_MAX_DIM_ALLOC];
				int inc[CCV_NNC_MAX_DIM_ALLOC];
				ccv_nnc_tensor_symbol_alias_params(model->graph, model->outputs[i], ofs, inc);
				int j;
				for (j = 0; j < CCV_NNC_MAX_DIM_ALLOC; j++)
					{ assert(ofs[j] == 0); } // There is no ofs.
				compiled_data->f[i] = alias_to; // Unfortunately, I cannot assert the size yet.
			}
		}
	} else {
		for (i = 0; i < output_size; i++)
		{
			const ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(model->graph, model->outputs[i]);
			const ccv_nnc_tensor_symbol_t fit = compiled_data->fits[i] = ccv_nnc_tensor_symbol_new(model->graph, info, 0);
			compiled_data->f[i] = ccv_nnc_tensor_symbol_new(model->graph, ccv_nnc_tensor_auto, 0);
			ccv_nnc_graph_exec_symbol_new(model->graph, loss, TENSOR_SYMBOL_LIST(model->outputs[i], fit), TENSOR_SYMBOL_LIST(compiled_data->f[i]), 0);
		}
	}
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(model->graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_OPS_FUSION), // Only do Ops fusion, in this way, we can fuse the loss function.
		0, 0, // No need to provide binds at this point.
		compiled_data->f, model->output_size,
		SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
	// If inputs are from GPU, stream type is GPU.
	compiled_data->parameters = parameters;
	compiled_data->parameter_flags = parameter_flags;
	compiled_data->internals = internals;
	compiled_data->ids.parameters = parameter_ids;
	compiled_data->ids.internals = internal_ids;
}

static void _ccv_cnnp_graph_push_graph_exec_symbol(void* context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
{
	ccv_array_t* const stack = (ccv_array_t*)context;
	ccv_array_push(stack, &symbol.d);
}

static void _ccv_nnc_tensor_symbol_reinit(const ccv_nnc_symbolic_graph_t* const src_graph, ccv_nnc_symbolic_graph_t* const dest_graph, const int src_index, const int dest_index)
{
	const ccv_nnc_tensor_symbol_t src_symbol = {
		.d = src_index,
		.graph = src_graph
	};
	const ccv_nnc_tensor_symbol_t dest_symbol = {
		.d = dest_index,
		.graph = dest_graph
	};
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(src_graph, src_symbol);
	ccv_nnc_tensor_symbol_set(dest_graph, dest_symbol, params);
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
	if (0 == ccv_nnc_tensor_symbol_alias_params(src_graph, src_symbol, ofs, inc))
		ccv_nnc_tensor_symbol_alias_set(dest_graph, dest_symbol, ofs, inc);
}

static int _ccv_nnc_tensor_symbol_check_dim(const ccv_nnc_symbolic_graph_t* const src_graph, ccv_nnc_symbolic_graph_t* const dest_graph, const int src_index, const int dest_index)
{
	const ccv_nnc_tensor_symbol_t src_symbol = {
		.d = src_index,
		.graph = src_graph
	};
	const ccv_nnc_tensor_param_t src_params = ccv_nnc_tensor_symbol_params(src_graph, src_symbol);
	const ccv_nnc_tensor_symbol_t dest_symbol = {
		.d = dest_index,
		.graph = dest_graph
	};
	const ccv_nnc_tensor_param_t dest_params = ccv_nnc_tensor_symbol_params(dest_graph, dest_symbol);
	return memcmp(src_params.dim, dest_params.dim, sizeof(src_params.dim)) == 0;
}

static void _ccv_cnnp_model_gradient_init(ccv_cnnp_model_t* const model, const int gradient_mode, const uint64_t disable_outgrad, ccv_nnc_tensor_t* const* const fits, const int fit_size);
static void _ccv_cnnp_compiled_data_graph_free(ccv_cnnp_compiled_data_t* const compiled_data);

typedef struct {
	int parallel_count;
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
} ccv_nnc_graph_exec_update_t;

static void _ccv_cnnp_cmd_update_for_execs(void* const context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint)
{
	ccv_nnc_graph_exec_update_t* const graph_exec_update = (ccv_nnc_graph_exec_update_t*)context;
	ccv_nnc_graph_exec_arena_t* const graph_exec_arena = graph_exec_update->graph_exec_arena;
	ccv_nnc_graph_exec_t graph_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, symbol);
	ccv_nnc_graph_exec_set(graph_exec.graph, graph_exec, cmd);
	ccv_nnc_graph_exec_set_hint(graph_exec.graph, graph_exec, hint);
	const ccv_nnc_symbolic_graph_t* const graph = graph_exec_update->graph;
	const int parallel_count = graph_exec_update->parallel_count;
	int i;
	for (i = 1; i < parallel_count; i++)
	{
		const ccv_nnc_graph_exec_t copy = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, ccv_nnc_graph_exec_symbol_copy(graph, symbol, i));
		if (!CCV_NO_GRAPH_EXEC(copy))
		{
			ccv_nnc_graph_exec_set(copy.graph, copy, cmd);
			ccv_nnc_graph_exec_set_hint(copy.graph, copy, hint);
		}
	}
}

void ccv_cnnp_model_absorb(ccv_cnnp_model_t* const model, ccv_cnnp_model_t* const init, const ccv_nnc_tensor_param_t* const inputs, const int input_size)
{
	assert(model->graph);
	assert(model->compiled_data);
	assert(!init->graph);
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	init->graph = ccv_nnc_symbolic_graph_new();
	ccv_array_t* const stack = ccv_array_new(sizeof(int), 0, 0);
	ccv_nnc_graph_exec_symbol_new_hook(init->graph, _ccv_cnnp_graph_push_graph_exec_symbol, stack);
	_ccv_cnnp_model_compile(init, inputs, input_size, compiled_data->loss);
	init->parallel_count = model->parallel_count;
	init->memory_compression = model->memory_compression;
	init->compiled_data->stream_type = model->compiled_data->stream_type;
	init->compiled_data->minimize.minimizer = model->compiled_data->minimize.minimizer;
	init->compiled_data->minimize.max_saved_aux_size = model->compiled_data->minimize.max_saved_aux_size;
	if (model->compiled_data->gradient_mode != CCV_CNNP_COMPILED_DATA_GRADIENT_NONE)
		_ccv_cnnp_model_gradient_init(init, model->compiled_data->gradient_mode, model->compiled_data->disable_outgrad, 0, 0);
	ccv_nnc_graph_exec_symbol_new_hook(init->graph, 0, 0);
	ccv_nnc_symbolic_graph_tensor_auto(init->graph, TRAVERSE_FULL);
	int i, j;
	// Verify parameters, internals and saved_aux in both graph has the same dimensionality.
	for (i = 0; i < compiled_data->parameters->rnum; i++)
	{
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i))->d;
		assert(_ccv_nnc_tensor_symbol_check_dim(model->graph, init->graph, d, d));
	}
	for (i = 0; i < compiled_data->internals->rnum; i++)
	{
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, i))->d;
		assert(_ccv_nnc_tensor_symbol_check_dim(model->graph, init->graph, d, d));
	}
	// Update inputs.
	assert(model->input_size == init->input_size);
	for (i = 0; i < model->input_size; i++)
		if (model->inputs[i].d >= 0)
		{
			assert(init->inputs[i].d >= 0);
			_ccv_nnc_tensor_symbol_reinit(init->graph, model->graph, init->inputs[i].d, model->inputs[i].d);
		}
	// Update outputs.
	assert(model->output_size == init->output_size);
	for (i = 0; i < model->output_size; i++)
	{
		if (model->outputs[i].d >= 0)
		{
			assert(init->outputs[i].d >= 0);
			_ccv_nnc_tensor_symbol_reinit(init->graph, model->graph, init->outputs[i].d, model->outputs[i].d);
		}
		if (model->outputs[i].d != model->compiled_data->f[i].d)
		{
			assert(init->outputs[i].d != init->compiled_data->f[i].d);
			if (model->compiled_data->f[i].d >= 0)
			{
				assert(init->compiled_data->f[i].d >= 0);
				_ccv_nnc_tensor_symbol_reinit(init->graph, model->graph, init->compiled_data->f[i].d, model->compiled_data->f[i].d);
			}
		}
	}
	// Go through the graph to set tensor on matching symbols
	for (i = 0; i < stack->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(stack, i);
		// If exceed range, skip.
		if (d >= ccv_nnc_graph_exec_symbol_count(init->graph) ||
			d >= ccv_nnc_graph_exec_symbol_count(model->graph))
			continue;
		const ccv_nnc_graph_exec_symbol_t src_symbol = {
			.d = d,
			.graph = init->graph
		};
		const ccv_nnc_graph_exec_symbol_t dest_symbol = {
			.d = d,
			.graph = model->graph
		};
		const ccv_nnc_cmd_t src_cmd = ccv_nnc_graph_exec_symbol_cmd(init->graph, src_symbol);
		const ccv_nnc_cmd_t dest_cmd = ccv_nnc_graph_exec_symbol_cmd(model->graph, dest_symbol);
		// If the name doesn't match, skip.
		if (dest_cmd.cmd != src_cmd.cmd && src_cmd.cmd != CCV_NNC_NOOP)
			continue;
		// Now get all the inputs and outputs, if matches, set them.
		const int* src_inputs;
		int src_input_size;
		const int* src_outputs;
		int src_output_size;
		ccv_nnc_graph_exec_symbol_io(init->graph, src_symbol, &src_inputs, &src_input_size, &src_outputs, &src_output_size);
		const int* dest_inputs;
		int dest_input_size;
		const int* dest_outputs;
		int dest_output_size;
		ccv_nnc_graph_exec_symbol_io(model->graph, dest_symbol, &dest_inputs, &dest_input_size, &dest_outputs, &dest_output_size);
		// We may have unmatched input / output size because this is the minimizer and it has
		// different saved_aux (for example, when we shrunk with CMD_NOOP).
		if (src_input_size != dest_input_size)
			continue;
		if (src_output_size != dest_output_size)
			continue;
		ccv_nnc_graph_exec_symbol_set(model->graph, dest_symbol, src_cmd);
		// There may be mismatches of the source tensor symbols and destination tensor symbols. The reason is because
		// we may later passed-in the minimizer, therefore, we may allocate tensors for minimizer later in the original
		// graph whereas in the newly created graph, it is streamlined (the minimizer exists from the beginning). That
		// will make the order of tensor symbols creation different, therefore, exact which tensor is which wrong as
		// well. However, set a new minimizer won't change the exec symbol ordering, because we never create new exec
		// symbols after gradient init step. Changing a new minimizer just updated that exec symbols setting, it is not
		// a new exec symbol.
		for (j = 0; j < src_input_size; j++)
			if (src_inputs[j] >= 0)
				_ccv_nnc_tensor_symbol_reinit(init->graph, model->graph, src_inputs[j], dest_inputs[j]);
		for (j = 0; j < src_output_size; j++)
			if (src_outputs[j] >= 0)
				_ccv_nnc_tensor_symbol_reinit(init->graph, model->graph, src_outputs[j], dest_outputs[j]);
	}
	ccv_array_free(stack);
	// After this, we get all tensors in the model graph resolved through tensor_auto.
	ccv_nnc_symbolic_graph_tensor_auto(model->graph, TRAVERSE_FULL);
	// Verify symbols we get matches.
	const int parameter_size = compiled_data->parameters->rnum;
	for (i = 0; i < parameter_size; i++)
		{ assert(((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i))->d == ((ccv_nnc_tensor_symbol_t*)ccv_array_get(init->compiled_data->parameters, i))->d); }
	const int internal_size = compiled_data->internals->rnum;
	for (i = 0; i < internal_size; i++)
		{ assert(((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, i))->d == ((ccv_nnc_tensor_symbol_t*)ccv_array_get(init->compiled_data->internals, i))->d); }
	// Go through compiled data.
	if (compiled_data->tensor_arena)
	{
		const int flag = ccv_nnc_tensor_arena_reinit(compiled_data->tensor_arena, model->graph);
		if (flag == 0 && compiled_data->graph_exec_arena)
		{
			ccv_nnc_graph_exec_reinit(compiled_data->graph_exec_arena, compiled_data->graph, model->graph);
			// Since we will reinit, if we previously set is_test, we need to set it again.
			if (compiled_data->is_test)
			{
				const int parallel_count = ccv_max(model->parallel_count, 1);
				ccv_nnc_graph_exec_update_t update = {
					.parallel_count = parallel_count,
					.graph = model->graph,
					.graph_exec_arena = compiled_data->graph_exec_arena,
				};
				ccv_cnnp_model_set_is_test(model, 1, _ccv_cnnp_cmd_update_for_execs, &update);
			}
		} else
			// Free-up tensor arena & graph exec arena.
			_ccv_cnnp_compiled_data_graph_free(compiled_data);
	}
	// There are other compiled graphs, for accum and apply gradients.
	// However, the main conclusion is, these absorb operations shouldn't impact parameters.
	// Thus, it won't impact the shape of gradients (only outgrad). Since for outgrad, we
	// don't allocate ourselves, it is not a concern. For normal gradients, the shape cannot
	// be changed otherwise parameters' shape will be meaningless. The same goes to internals.
	// That is why we don't update these compiled graphs at all this point.
	// Free the model, we've already "absorbed" it.
	ccv_cnnp_model_free(init);
}

void ccv_cnnp_model_compile(ccv_cnnp_model_t* const model, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_cmd_t minimizer, const ccv_nnc_cmd_t loss)
{
	assert(input_size == model->input_size || model->input_size == 0);
	if (model->input_size == 0)
		model->input_size = input_size;
	if (!model->graph) // The graph is not compiled yet.
	{
		model->graph = ccv_nnc_symbolic_graph_new();
		_ccv_cnnp_model_compile(model, inputs, input_size, loss);
		assert(model->compiled_data);
		int i, flag = 0;
		for (i = 0; !flag && i < input_size; i++)
			flag = (CCV_TENSOR_GET_MEMORY(inputs[i].type) == CCV_TENSOR_GPU_MEMORY);
		// If inputs are from GPU, stream type is GPU.
		model->compiled_data->stream_type = flag ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
		model->compiled_data->minimize.minimizer = minimizer;
		model->compiled_data->minimize.max_saved_aux_size = ccv_nnc_minimizer_saved_aux_size(minimizer);
	} else {
		// Now, finally fill in this part. If the graph is already compiled, we make a copy of the model.
		// And then absorb the "new model" to the old one.
		ccv_cnnp_model_t* const init = ccv_cnnp_model_copy(model, model->is_trainable);
		ccv_cnnp_model_absorb(model, init, inputs, input_size);
		// Reset minimizer.
		ccv_cnnp_model_set_minimizer(model, minimizer, 1, 0, 0);
	}
}

ccv_cnnp_model_t* ccv_cnnp_model_copy(const ccv_cnnp_model_t* const model, const int is_trainable)
{
	ccv_cnnp_model_t* const new_model = _ccv_cnnp_model_copy(model, 0);
	new_model->is_trainable = is_trainable;
	return new_model;
}

void ccv_cnnp_model_tensor_auto(ccv_cnnp_model_t* const model, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(model->graph);
	assert(output_size == model->output_size);
	ccv_nnc_symbolic_graph_t* const graph = model->graph;
	ccv_nnc_symbolic_graph_tensor_auto(graph, TRAVERSE_FULL);
	int i;
	for (i = 0; i < output_size; i++)
	{
		assert(model->outputs[i].d != CCV_NNC_NO_TENSOR_SYMBOL);
		outputs[i] = ccv_nnc_tensor_symbol_params(graph, model->outputs[i]);
	}
}

void ccv_cnnp_model_set_workspace_size(ccv_cnnp_model_t* const model, size_t workspace_size)
{
	if (workspace_size == model->workspace_size)
		return;
	model->workspace_size = workspace_size;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	if (compiled_data && compiled_data->graph)
		ccv_nnc_graph_autotune(compiled_data->graph, workspace_size, 0, TRAVERSE_FULL);
}

size_t ccv_cnnp_model_workspace_size(ccv_cnnp_model_t* const model)
{
	return model->workspace_size;
}

void ccv_cnnp_model_set_data_parallel(ccv_cnnp_model_t* const model, const int parallel)
{
	if (parallel == 0)
		model->parallel_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	else
		model->parallel_count = parallel;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	if (compiled_data)
		{ assert(!compiled_data->graph); }
}

void ccv_cnnp_model_set_memory_compression(ccv_cnnp_model_t* const model, const int memory_compression)
{
	model->memory_compression = memory_compression;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	if (compiled_data)
		{ assert(!compiled_data->graph); }
}

void ccv_cnnp_model_set_file_backed_parameters(ccv_cnnp_model_t* const model, const char* const dir)
{
	if (dir)
	{
		const size_t len = strnlen(dir, 1023);
		const size_t n = len + 1;
		model->file_backed_dir = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		memcpy(model->file_backed_dir, dir, n);
		model->file_backed_dir[len] = 0;
	} else {
		if (model->file_backed_dir)
			ccfree(model->file_backed_dir);
		model->file_backed_dir = 0;
	}
}

typedef struct {
	int parallel_count;
	ccv_nnc_symbolic_graph_t* graph;
	ccv_cnnp_compiled_data_t* compiled_data;
	ccv_nnc_tensor_arena_t* tensor_arena;
} ccv_nnc_tensor_init_states_t;

static int _ccv_cnnp_any_to_init(const ccv_cnnp_compiled_data_t* const compiled_data)
{
	int i;
	for (i = 0; i < compiled_data->parameters->rnum; i++)
	{
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i))->d;
		if (!(compiled_data->tensors_init.v[d >> 5] & (1u << (d & 0x1f))))
			return 1;
	}
	for (i = 0; i < compiled_data->internals->rnum; i++)
	{
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, i))->d;
		if (!(compiled_data->tensors_init.v[d >> 5] & (1u << (d & 0x1f))))
			return 1;
	}
	return 0;
}

static void _ccv_cnnp_init_states_for_tensors(void* const context, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const input, const ccv_nnc_tensor_symbol_t output_symbol)
{
	ccv_nnc_tensor_init_states_t* const tensor_init_states = (ccv_nnc_tensor_init_states_t*)context;
	ccv_nnc_tensor_arena_t* const tensor_arena = tensor_init_states->tensor_arena;
	ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, output_symbol);
	if (!output_tensor)
		return;
	const int d = output_symbol.d;
	assert(d < tensor_init_states->compiled_data->tensors_init.size);
	if (tensor_init_states->compiled_data->tensors_init.v[d >> 5] & (1u << (d & 0x1f)))
		return;
	tensor_init_states->compiled_data->tensors_init.v[d >> 5] |= (1u << (d & 0x1f));
	ccv_nnc_cmd_exec(cmd, hint, flags, &input, input ? 1 : 0, &output_tensor, 1, 0);
	const ccv_nnc_symbolic_graph_t* const graph = tensor_init_states->graph;
	const int parallel_count = tensor_init_states->parallel_count;
	int i;
	for (i = 1; i < parallel_count; i++)
	{
		ccv_nnc_tensor_t* const copy = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(graph, output_symbol, i));
		if (copy)
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &output_tensor, 1, &copy, 1, 0);
	}
}

// This method can only handle cases we added new tensors and exec, never delete. This invariant is true because
// we setup everything (including calling simplify method) in ccv_cnnp_model_compile method, before this rewind setup.
static void _ccv_cnnp_model_rewind_graph(ccv_cnnp_model_t* const model)
{
	assert(model->graph);
	assert(model->compiled_data);
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data->rewindables);
	int i;
	for (i = 0; i < compiled_data->rewindables->rnum; i++)
	{
		const ccv_cnnp_rewind_symbol_t* const rewind_symbol = (ccv_cnnp_rewind_symbol_t*)ccv_array_get(compiled_data->rewindables, i);
		if (rewind_symbol->type == CCV_CNNP_REWIND_GRAPH_EXEC)
			ccv_nnc_graph_exec_symbol_free(model->graph, rewind_symbol->graph_exec);
		else if (rewind_symbol->type == CCV_CNNP_REWIND_TENSOR)
			ccv_nnc_tensor_symbol_free(model->graph, rewind_symbol->tensor);
	}
	ccv_array_clear(compiled_data->rewindables);
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
}


static void _ccv_cnnp_model_tensor_symbol_new_hook(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_param_t info, const char* const name)
{
	const ccv_cnnp_rewind_symbol_t rewind_symbol = {
		.type = CCV_CNNP_REWIND_TENSOR,
		.tensor = symbol
	};
	ccv_array_t* const rewind_symbols = (ccv_array_t*)context;
	ccv_array_push(rewind_symbols, &rewind_symbol);
}

static void _ccv_cnnp_model_tensor_symbol_alias_new_hook(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_symbol_t from_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* const name)
{
	const ccv_cnnp_rewind_symbol_t rewind_symbol = {
		.type = CCV_CNNP_REWIND_TENSOR,
		.tensor = symbol
	};
	ccv_array_t* const rewind_symbols = (ccv_array_t*)context;
	ccv_array_push(rewind_symbols, &rewind_symbol);
}

static void _ccv_cnnp_model_graph_exec_symbol_new_hook(void* context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
{
	const ccv_cnnp_rewind_symbol_t rewind_symbol = {
		.type = CCV_CNNP_REWIND_GRAPH_EXEC,
		.graph_exec = symbol
	};
	ccv_array_t* const rewind_symbols = (ccv_array_t*)context;
	ccv_array_push(rewind_symbols, &rewind_symbol);
}

static void _ccv_cnnp_model_graph_symbol_exec_set_for_graph_exec_arena(const ccv_nnc_graph_exec_arena_t* const graph_exec_arena, const int parallel_count, const ccv_nnc_graph_exec_symbol_t exec_symbol, const ccv_nnc_cmd_t cmd, ccv_nnc_symbolic_graph_t* const symbolic_graph)
{
	ccv_nnc_graph_exec_t const update_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, exec_symbol);
	if (!CCV_NO_GRAPH_EXEC(update_exec))
		ccv_nnc_graph_exec_set(update_exec.graph, update_exec, cmd);
	int i;
	for (i = 1; i < parallel_count; i++)
	{
		ccv_nnc_graph_exec_symbol_t copy_symbol = ccv_nnc_graph_exec_symbol_copy(symbolic_graph, exec_symbol, i);
		const ccv_nnc_graph_exec_t copy = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, copy_symbol);
		if (!CCV_NO_GRAPH_EXEC(copy))
			ccv_nnc_graph_exec_set(copy.graph, copy, cmd);
	}
}

static void _ccv_cnnp_model_graph_exec_symbol_set(ccv_nnc_symbolic_graph_t* const symbolic_graph, ccv_cnnp_compiled_data_t* const compiled_data, const int parallel_count, const ccv_nnc_graph_exec_symbol_t exec_symbol, const ccv_nnc_cmd_t cmd)
{
	assert(compiled_data);
	assert(symbolic_graph);
	ccv_nnc_graph_exec_symbol_set(symbolic_graph, exec_symbol, cmd);
	int i;
	for (i = 1; i < parallel_count; i++)
	{
		ccv_nnc_graph_exec_symbol_t copy_symbol = ccv_nnc_graph_exec_symbol_copy(symbolic_graph, exec_symbol, i);
		if (copy_symbol.graph)
			ccv_nnc_graph_exec_symbol_set(symbolic_graph, copy_symbol, cmd);
	}
	ccv_nnc_graph_exec_arena_t* const graph_exec_arena = compiled_data->graph_exec_arena;
	if (graph_exec_arena)
		_ccv_cnnp_model_graph_symbol_exec_set_for_graph_exec_arena(graph_exec_arena, parallel_count, exec_symbol, cmd, symbolic_graph);
	// Skip backward graph exec arena because it is for a specific accum symbolic graph, not the main graph (model->graph)
	ccv_nnc_graph_exec_arena_t* const gradient_graph_exec_arena = compiled_data->apply_gradients.graph_exec_arena;
	if (gradient_graph_exec_arena)
		_ccv_cnnp_model_graph_symbol_exec_set_for_graph_exec_arena(gradient_graph_exec_arena, parallel_count, exec_symbol, cmd, symbolic_graph);
}

static int _ccv_cnnp_set_minimizer_for_parameter(ccv_nnc_symbolic_graph_t* const graph, ccv_cnnp_compiled_data_t* const compiled_data, ccv_nnc_graph_exec_symbol_t* const update_nodes, ccv_nnc_tensor_symbol_t* const updated_parameters, ccv_nnc_tensor_symbol_map_t* const saved_aux, const int parallel_count, const ccv_nnc_cmd_t minimizer, const int saved_aux_size, const int max_saved_aux_size, const int parameter_indice)
{
	int this_parameter_flag = 0;
	if (update_nodes[parameter_indice].d == CCV_NNC_NO_TENSOR_SYMBOL)
		return this_parameter_flag;
	const ccv_nnc_cmd_t old_minimizer = ccv_nnc_graph_exec_symbol_cmd(graph, update_nodes[parameter_indice]);
	int j, k;
	// For no-op, we can preserve previous saved_aux_size.
	if (old_minimizer.cmd != minimizer.cmd && minimizer.cmd != CCV_NNC_NOOP)
	{
		// If the old minimizer is a noop, then the old_saved_aux_size should be whatever its previous
		// saved_aux_size is, otherwise we will reinit the saved_aux repeatedly if you switch between
		// noop and a minimizer. We don't want that because we do that in high-level frameworks to
		// make sure some model parameters don't update if we don't want them to.
		int old_saved_aux_size;
		if (old_minimizer.cmd == CCV_NNC_NOOP)
		{
			int input_size;
			ccv_nnc_graph_exec_symbol_io(graph, update_nodes[parameter_indice], 0, &input_size, 0, 0);
			if (input_size < 2) // This is not legit.
				old_saved_aux_size = ccv_nnc_minimizer_saved_aux_size(old_minimizer);
			else // See ccv_nnc_minimizer_saved_aux_size, the saved_aux is inputs excluding gradients and parameters.
				old_saved_aux_size = input_size - 2;
		} else
			old_saved_aux_size = ccv_nnc_minimizer_saved_aux_size(old_minimizer);
		if (old_saved_aux_size != saved_aux_size)
		{
			this_parameter_flag = 1;
			if (saved_aux_size > old_saved_aux_size)
			{
				// Allocate new tensor symbols.
				const ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(graph, updated_parameters[parameter_indice]);
				for (j = old_saved_aux_size; j < saved_aux_size; j++)
				{
					saved_aux[parameter_indice * max_saved_aux_size + j].source = ccv_nnc_tensor_symbol_new(graph, info, 0);
					saved_aux[parameter_indice * max_saved_aux_size + j].destination = ccv_nnc_tensor_symbol_new(graph, info, 0);
					const int device_id = CCV_TENSOR_GET_DEVICE_ID(info.type);
					for (k = 1; k < parallel_count; k++)
					{
						ccv_nnc_tensor_param_t dev_info = info;
						if (k != device_id)
							CCV_TENSOR_SET_DEVICE_ID(dev_info.type, k);
						else
							CCV_TENSOR_SET_DEVICE_ID(dev_info.type, 0);
						const ccv_nnc_tensor_symbol_t src_copy = ccv_nnc_tensor_symbol_new(graph, dev_info, 0);
						const ccv_nnc_tensor_symbol_t dest_copy = ccv_nnc_tensor_symbol_new(graph, dev_info, 0);
						ccv_nnc_tensor_symbol_set_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].source, k, src_copy);
						ccv_nnc_tensor_symbol_set_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].destination, k, dest_copy);
					}
				}
			} else {
				for (j = saved_aux_size; j < old_saved_aux_size; j++)
				{
					for (k = 1; k < parallel_count; k++)
					{
						const ccv_nnc_tensor_symbol_t src_copy = ccv_nnc_tensor_symbol_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].source, k);
						if (src_copy.d >= 0)
						{
							ccv_nnc_tensor_symbol_free(graph, src_copy);
							ccv_nnc_tensor_symbol_set_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].source, k, NO_TENSOR_SYMBOL);
						}
						const ccv_nnc_tensor_symbol_t dest_copy = ccv_nnc_tensor_symbol_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].destination, k);
						if (dest_copy.d >= 0)
						{
							ccv_nnc_tensor_symbol_free(graph, dest_copy);
							ccv_nnc_tensor_symbol_set_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].destination, k, NO_TENSOR_SYMBOL);
						}
					}
					ccv_nnc_tensor_symbol_free(graph, saved_aux[parameter_indice * max_saved_aux_size + j].source);
					ccv_nnc_tensor_symbol_free(graph, saved_aux[parameter_indice * max_saved_aux_size + j].destination);
					saved_aux[parameter_indice * max_saved_aux_size + j].source = saved_aux[parameter_indice * max_saved_aux_size + j].destination = NO_TENSOR_SYMBOL;
				}
			}
		}
	}
	_ccv_cnnp_model_graph_exec_symbol_set(graph, compiled_data, parallel_count, update_nodes[parameter_indice], minimizer);
	if (this_parameter_flag)
	{
		ccv_nnc_tensor_symbol_t update_inputs[saved_aux_size + 2];
		ccv_nnc_tensor_symbol_t update_outputs[saved_aux_size + 1];
		const int* inputs = 0;
		int input_size = 0;
		ccv_nnc_graph_exec_symbol_io(graph, update_nodes[parameter_indice], &inputs, &input_size, 0, 0);
		assert(input_size >= 1);
		update_inputs[0].d = inputs[0];
		update_inputs[0].graph = graph;
		update_inputs[1].d = inputs[1];
		update_inputs[1].graph = graph;
		update_outputs[0] = updated_parameters[parameter_indice];
		for (j = 0; j < saved_aux_size; j++)
		{
			update_inputs[j + 2] = saved_aux[parameter_indice * max_saved_aux_size + j].source;
			update_outputs[j + 1] = saved_aux[parameter_indice * max_saved_aux_size + j].destination;
		}
		ccv_nnc_graph_exec_symbol_set_io(graph, update_nodes[parameter_indice], update_inputs, saved_aux_size + 2, update_outputs, saved_aux_size + 1);
		for (k = 1; k < parallel_count; k++)
		{
			const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_copy(graph, update_nodes[parameter_indice], k);
			assert(copy.d >= 0);
			ccv_nnc_graph_exec_symbol_io(graph, copy, &inputs, &input_size, 0, 0);
			assert(input_size >= 1);
			update_inputs[0].d = inputs[0];
			update_inputs[0].graph = graph;
			update_inputs[1].d = inputs[1];
			update_inputs[1].graph = graph;
			update_outputs[0] = ccv_nnc_tensor_symbol_copy(graph, updated_parameters[parameter_indice], k);
			for (j = 0; j < saved_aux_size; j++)
			{
				update_inputs[j + 2] = ccv_nnc_tensor_symbol_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].source, k);
				update_outputs[j + 1] = ccv_nnc_tensor_symbol_copy(graph, saved_aux[parameter_indice * max_saved_aux_size + j].destination, k);
			}
			ccv_nnc_graph_exec_symbol_set_io(graph, copy, update_inputs, saved_aux_size + 2, update_outputs, saved_aux_size + 1);
		}
	}
	return this_parameter_flag;
}

typedef struct {
	int parameter_size;
	ccv_nnc_cmd_t minimizer;
	ccv_cnnp_model_io_t parameters[1];
} ccv_cnnp_set_minimizer_for_parameter_t;

static int _ccv_cnnp_apply_parameters_with_minimizer(ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int max_saved_aux_size = compiled_data->minimize.max_saved_aux_size;
	// We update all parameters, at this point, we have one minimizer.
	const int parameter_size = compiled_data->parameters->rnum;
	ccv_nnc_graph_exec_symbol_t* const update_nodes = compiled_data->update_nodes;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = model->graph;
	assert(symbolic_graph);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	ccv_array_t* const parameters = compiled_data->minimize.parameters;
	ccv_array_t* const parameter_indices = ccv_array_new(sizeof(int), 0, 0);
	int i, j, flag = 0;
	for (i = 0; i < parameters->rnum; i++)
	{
		ccv_cnnp_set_minimizer_for_parameter_t* const set_minimizer_for_parameter = *(ccv_cnnp_set_minimizer_for_parameter_t**)ccv_array_get(parameters, i);
		for (j = 0; j < set_minimizer_for_parameter->parameter_size; j++)
		{
			const int param_sel = set_minimizer_for_parameter->parameters[j]->param_sel > 0 ? set_minimizer_for_parameter->parameters[j]->param_sel - 1 : set_minimizer_for_parameter->parameters[j]->param_sel;
			assert(set_minimizer_for_parameter->parameters[j]->param_sel != 0);
			const int old_rnum = parameter_indices->rnum;
			ccv_cnnp_model_add_to_parameter_indices(set_minimizer_for_parameter->parameters[j]->model, param_sel, parameter_indices);
			const int param_ref = set_minimizer_for_parameter->parameters[j]->param_ref > 0 ? set_minimizer_for_parameter->parameters[j]->param_ref - 1 : set_minimizer_for_parameter->parameters[j]->param_ref;
			assert(set_minimizer_for_parameter->parameters[j]->param_ref != 0);
			if (param_ref >= 0)
			{
				assert(param_ref + old_rnum < parameter_indices->rnum);
				*(int*)ccv_array_get(parameter_indices, old_rnum) = *(int*)ccv_array_get(parameter_indices, param_ref + old_rnum);
				parameter_indices->rnum = old_rnum + 1;
			}
		}
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(set_minimizer_for_parameter->minimizer);
		// We may have duplicated indices, but that is OK, we will set it twice.
		for (j = 0; j < parameter_indices->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(parameter_indices, j);
			assert(d <= parameter_size);
			if (_ccv_cnnp_set_minimizer_for_parameter(symbolic_graph, compiled_data, update_nodes, compiled_data->updated_parameters, compiled_data->saved_aux, parallel_count, set_minimizer_for_parameter->minimizer, saved_aux_size, max_saved_aux_size, d))
				flag = 1;
		}
		ccv_array_clear(parameter_indices);
	}
	ccv_array_free(parameter_indices);
	return flag;
}

static void _ccv_cnnp_scatter_saved_aux(ccv_nnc_tensor_symbol_map_t* const saved_aux, const int parameter_size, const int old_saved_aux_size, const int new_saved_aux_size)
{
	if (new_saved_aux_size == old_saved_aux_size)
		return;
	assert(new_saved_aux_size > old_saved_aux_size);
	int i, j;
	for (i = parameter_size - 1; i >= 0; i--)
	{
		for (j = new_saved_aux_size - 1; j >= old_saved_aux_size; j--)
			saved_aux[i * new_saved_aux_size + j].source = saved_aux[i * new_saved_aux_size + j].destination = NO_TENSOR_SYMBOL;
		for (j = old_saved_aux_size - 1; j >= 0; j--)
			saved_aux[i * new_saved_aux_size + j] = saved_aux[i * old_saved_aux_size + j];
	}
}

static void _ccv_cnnp_model_set_rewindables(ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	if (!compiled_data->rewindables)
		compiled_data->rewindables = ccv_array_new(sizeof(ccv_cnnp_rewind_symbol_t), 0, 0);
	ccv_nnc_tensor_symbol_new_hook(model->graph, _ccv_cnnp_model_tensor_symbol_new_hook, compiled_data->rewindables);
	ccv_nnc_tensor_symbol_alias_new_hook(model->graph, _ccv_cnnp_model_tensor_symbol_alias_new_hook, compiled_data->rewindables);
	ccv_nnc_graph_exec_symbol_new_hook(model->graph, _ccv_cnnp_model_graph_exec_symbol_new_hook, compiled_data->rewindables);
}

static void _ccv_cnnp_model_gradient_init(ccv_cnnp_model_t* const model, const int gradient_mode, const uint64_t disable_outgrad, ccv_nnc_tensor_t* const* const fits, const int fit_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_NONE);
	assert(gradient_mode != CCV_CNNP_COMPILED_DATA_GRADIENT_NONE);
	const int evaluate_to_size = compiled_data->evaluate.to_size;
	assert(evaluate_to_size > 0);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	compiled_data->evaluate.tos = ccrealloc(compiled_data->evaluate.tos, sizeof(ccv_nnc_graph_exec_symbol_t) * evaluate_to_size * parallel_count + sizeof(ccv_nnc_graph_exec_t) * evaluate_to_size * parallel_count);
	compiled_data->evaluate.to_ops = (ccv_nnc_graph_exec_t*)(compiled_data->evaluate.tos + evaluate_to_size * parallel_count);
	int i, j;
	const int output_size = model->output_size;
	assert(!fits || fit_size == output_size * parallel_count);
	if (fits)
		for (i = 0; i < output_size; i++)
			ccv_nnc_tensor_symbol_set(model->graph, compiled_data->fits[i], fits[i]->info);
	const int max_saved_aux_size = compiled_data->minimize.max_saved_aux_size;
	const int parameter_size = compiled_data->parameters->rnum;
	compiled_data->updated_parameters = (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * parameter_size + sizeof(ccv_nnc_graph_exec_symbol_t) * parameter_size + sizeof(ccv_nnc_tensor_symbol_map_t) * max_saved_aux_size * parameter_size);
	compiled_data->update_nodes = (ccv_nnc_graph_exec_symbol_t*)(compiled_data->updated_parameters + parameter_size);
	compiled_data->saved_aux = (ccv_nnc_tensor_symbol_map_t*)(compiled_data->update_nodes + parameter_size);
	int parameter_size_maybe_more = parameter_size;
	compiled_data->disable_outgrad = disable_outgrad;
	int outgrad_size;
	if (gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES || model->input_size == 0)
		outgrad_size = 0;
	else if (disable_outgrad == CCV_CNNP_DISABLE_OUTGRAD_NONE) // Compute minimize with gradients including inputs.
		outgrad_size = model->input_size;
	else {
		assert(disable_outgrad != CCV_CNNP_DISABLE_OUTGRAD_ALL); // If it is disable all, gradient mode won't be this.
		outgrad_size = 0;
		for (i = 0; i < model->input_size; i++)
			if (!(disable_outgrad & ((uint64_t)1 << i)))
				++outgrad_size;
	}
	compiled_data->outgrad_size = outgrad_size;
	parameter_size_maybe_more += outgrad_size;
	compiled_data->gradients = (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * parameter_size_maybe_more + sizeof(ccv_nnc_graph_exec_symbol_t) * parameter_size_maybe_more * parallel_count);
	compiled_data->outgrads = parameter_size_maybe_more > parameter_size ? compiled_data->gradients + parameter_size : 0;
	compiled_data->backward.tos = (ccv_nnc_graph_exec_symbol_t*)(compiled_data->gradients + parameter_size_maybe_more);
	compiled_data->backward.to_size = parameter_size_maybe_more;
	ccv_nnc_tensor_symbol_t* parameters = (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0);
	if (compiled_data->parameter_flags)
	{
		parameters = (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * parameter_size);
		for (i = 0; i < parameter_size; i++)
			if (compiled_data->parameter_flags[i >> 6] & ((uint64_t)1 << (i & 63)))
				parameters[i] = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i);
			else
				parameters[i] = NO_TENSOR_SYMBOL;
	}
	if (gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES || model->input_size == 0)
		ccv_nnc_symbolic_graph_minimize(model->graph, compiled_data->minimize.minimizer, compiled_data->f, output_size, parameters, parameter_size, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), compiled_data->gradients, compiled_data->updated_parameters, compiled_data->saved_aux, compiled_data->update_nodes);
	else if (disable_outgrad == CCV_CNNP_DISABLE_OUTGRAD_NONE) // Compute minimize with gradients including inputs.
		ccv_nnc_symbolic_graph_minimize(model->graph, compiled_data->minimize.minimizer, compiled_data->f, output_size, parameters, parameter_size, model->inputs, model->input_size, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), compiled_data->gradients, compiled_data->updated_parameters, compiled_data->saved_aux, compiled_data->update_nodes);
	else { // Compute minimize with gradients including selected inputs.
		assert(model->input_size > 0);
		assert(disable_outgrad != CCV_CNNP_DISABLE_OUTGRAD_ALL); // If it is disable all, gradient mode won't be this.
		assert(outgrad_size > 0);
		ccv_nnc_tensor_symbol_t outgrads[outgrad_size];
		j = 0;
		for (i = 0; i < model->input_size; i++)
			if (!(disable_outgrad & ((uint64_t)1 << i)))
				outgrads[j++] = model->inputs[i];
		ccv_nnc_symbolic_graph_minimize(model->graph, compiled_data->minimize.minimizer, compiled_data->f, output_size, parameters, parameter_size, outgrads, outgrad_size, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), compiled_data->gradients, compiled_data->updated_parameters, compiled_data->saved_aux, compiled_data->update_nodes);
	}
	if (compiled_data->parameter_flags)
		ccfree(parameters);
	_ccv_cnnp_scatter_saved_aux(compiled_data->saved_aux, parameter_size, ccv_nnc_minimizer_saved_aux_size(compiled_data->minimize.minimizer), compiled_data->minimize.max_saved_aux_size);
	if (compiled_data->minimize.parameters)
		_ccv_cnnp_apply_parameters_with_minimizer(model);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t df = ccv_nnc_tensor_symbol_for_backward(model->graph, compiled_data->f[i]);
		// Init this to 1 so we can backprop.
		ccv_nnc_tensor_symbol_set_flags(model->graph, df, CCV_NNC_TENSOR_SYMBOL_INIT_ONES);
	}
	compiled_data->backward.to_size = 0;
	for (i = 0; i < parameter_size_maybe_more; i++)
		if (!compiled_data->parameter_flags || (compiled_data->parameter_flags[i >> 6] & ((uint64_t)1 << (i & 63))))
			compiled_data->backward.tos[compiled_data->backward.to_size++] = ccv_nnc_graph_exec_symbol_for_backward(model->graph, compiled_data->gradients[i]);
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS);
	ccv_nnc_symbolic_graph_set_destinations(model->graph, compiled_data->update_nodes, parameter_size);
	for (i = 0; i < parameter_size_maybe_more - parameter_size; i++)
	{
		const ccv_nnc_graph_exec_symbol_t outgrad = ccv_nnc_graph_exec_symbol_for_backward(model->graph, compiled_data->outgrads[i]);
		const int* tos;
		int to_size;
		ccv_nnc_graph_exec_symbol_to(model->graph, outgrad, &tos, &to_size);
		if (to_size == 0) // If this is the end (no minimizers afterwards). We need to attach this as a destination. Otherwise this is covered in update_nodes.
		{
			const ccv_nnc_graph_exec_symbol_t* destinations = ccv_nnc_symbolic_graph_destinations(model->graph);
			int flag = 0;
			for (j = i - 1; !flag && j >= 0; j--)
				flag = (destinations[j + parameter_size].d == outgrad.d);
			if (!flag) // Only if we cannot find it, we add it.
				ccv_nnc_symbolic_graph_add_destination(model->graph, outgrad);
		}
	}
	if (parallel_count > 1)
	{
		ccv_nnc_symbolic_graph_data_parallel(model->graph, parallel_count,
			0, 0,
			compiled_data->gradients, parameter_size /* No need to deal with outgrads, we don't allreduce outgrads */,
			compiled_data->gradients /* We only care about gradients before allreduce, thus, update our current pointers */,
			0, 0, 0,
			CCV_NNC_PARALLEL_REDUCE_OP_SUM,
			SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
		ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		for (i = 0; i < evaluate_to_size; i++)
			for (j = 1; j < parallel_count; j++)
			{
				const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_copy(model->graph, compiled_data->evaluate.tos[i], j);
				if (copy.d != CCV_NNC_NO_GRAPH_EXEC_SYMBOL)
					compiled_data->evaluate.tos[compiled_data->evaluate.to_size++] = copy;
			}
		const int backward_to_size = compiled_data->backward.to_size;
		for (i = 0; i < backward_to_size; i++)
			for (j = 1; j < parallel_count; j++)
			{
				const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_copy(model->graph, compiled_data->backward.tos[i], j);
				if (copy.d != CCV_NNC_NO_GRAPH_EXEC_SYMBOL)
					compiled_data->backward.tos[compiled_data->backward.to_size++] = copy;
			}
	}
	// Only use memory compression if we are in gradient parameter mode.
	if (gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES && model->memory_compression)
		ccv_nnc_symbolic_graph_memory_compression(model->graph, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
	compiled_data->backward.to_size = _ccv_nnc_array_dedup_graph_exec_symbols(compiled_data->backward.tos, compiled_data->backward.to_size);
	compiled_data->gradient_mode = gradient_mode;
}

void ccv_cnnp_model_tensors_init(const ccv_cnnp_model_t* const model, ccv_cnnp_compiled_data_t* const compiled_data)
{
	assert(!compiled_data->tensors.parameters);
	const int parameter_size = compiled_data->parameters->rnum;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int internal_size = compiled_data->internals->rnum;
	compiled_data->tensors_init.size = ccv_nnc_tensor_symbol_count(model->graph);
	compiled_data->tensors_init.v = cccalloc(((compiled_data->tensors_init.size + 31) >> 5), sizeof(uint32_t));
	compiled_data->tensors.parameters = (ccv_nnc_tensor_t**)ccmalloc((sizeof(ccv_nnc_tensor_t*) * parameter_size + sizeof(ccv_nnc_tensor_t*) * internal_size) * parallel_count);
	compiled_data->tensors.internals = compiled_data->tensors.parameters + parameter_size * parallel_count;
	int i, j;
	for (i = 0; i < parameter_size; i++)
	{
		const ccv_nnc_tensor_symbol_t parameter = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(parameter.graph, parameter);
		if (CCV_TENSOR_GET_DEVICE(info.type) == CCV_COMPUTE_DEVICE_ANY)
			CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(info.type);
		compiled_data->tensors.parameters[i] = ccv_nnc_tensor_new(0, info, 0);
		for (j = 1; j < parallel_count; j++)
		{
			if (j != device_id)
				CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			else
				CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
			compiled_data->tensors.parameters[i + j * parameter_size] = ccv_nnc_tensor_new(0, info, 0);
		}
	}
	for (i = 0; i < internal_size; i++)
	{
		const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(retained.graph, retained);
		if (CCV_TENSOR_GET_DEVICE(info.type) == CCV_COMPUTE_DEVICE_ANY)
			CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(info.type);
		compiled_data->tensors.internals[i] = ccv_nnc_tensor_new(0, info, 0);
		for (j = 1; j < parallel_count; j++)
		{
			if (j != device_id)
				CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			else
				CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
			compiled_data->tensors.internals[i + j * internal_size] = ccv_nnc_tensor_new(0, info, 0);
		}
	}
}

static void _ccv_cnnp_model_copy_tensors(const uint32_t* const tensors_init, const ccv_nnc_tensor_symbol_t* const tensor_symbols, ccv_nnc_tensor_t* const* const tensors, const int tensor_size, const int parallel_count)
{
	assert(parallel_count > 0);
	int i, j;
	for (i = 0; i < tensor_size; i++)
	{
		if (!tensors[i])
			continue;
		const int d = tensor_symbols[i].d;
		if (!(tensors_init[d >> 5] & (1u << (d & 0x1f))))
			continue;
		for (j = 1; j < parallel_count; j++)
			if (tensors[i + j * tensor_size])
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[i], 1, &tensors[i + j * tensor_size], 1, 0);
	}
}

static void _ccv_cnnp_model_remove_nocopies(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const tensor_symbols, ccv_nnc_tensor_t** const tensors, const int tensor_size, const int parallel_count)
{
	assert(parallel_count > 0);
	int i, j;
	for (i = 0; i < tensor_size; i++)
	{
		const ccv_nnc_tensor_symbol_t tensor_symbol = tensor_symbols[i];
		for (j = 1; j < parallel_count; j++)
		{
			const ccv_nnc_tensor_symbol_t copy = ccv_nnc_tensor_symbol_copy(graph, tensor_symbol, j);
			ccv_nnc_tensor_t* copy_tensor = tensors[i + j * tensor_size];
			if (copy_tensor && copy.d == CCV_NNC_NO_TENSOR_SYMBOL)
			{ // We shouldn't allocate this, free it up.
				ccv_nnc_tensor_free(tensors[i + j * tensor_size]);
				tensors[i + j * tensor_size] = 0;
			}
		}
	}
}

static void _ccv_cnnp_model_bind_tensors(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const tensor_symbols, ccv_nnc_tensor_t* const* const tensors, const int tensor_size, const int parallel_count, ccv_array_t* const tensor_binds)
{
	assert(parallel_count > 0);
	int i, j;
	for (i = 0; i < tensor_size; i++)
	{
		ccv_nnc_tensor_symbol_t tensor_symbol = tensor_symbols[i];
		if (tensor_symbol.d == CCV_NNC_NO_TENSOR_SYMBOL)
			continue;
		if (graph)
		{
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(graph, tensor_symbol);
			if (alias_to.d != CCV_NNC_NO_TENSOR_SYMBOL)
				tensor_symbol = alias_to;
		}
		ccv_nnc_tensor_t* const tensor = tensors[i];
		if (tensor && tensor_symbol.d != CCV_NNC_NO_TENSOR_SYMBOL)
		{
			const ccv_nnc_tensor_bind_t retained_bind = {
				.symbol = tensor_symbol,
				.tensor = tensor
			};
			ccv_array_push(tensor_binds, &retained_bind);
		}
		for (j = 1; j < parallel_count; j++)
		{
			const ccv_nnc_tensor_symbol_t copy = ccv_nnc_tensor_symbol_copy(graph, tensor_symbol, j);
			ccv_nnc_tensor_t* copy_tensor = tensors[i + j * tensor_size];
			if (copy_tensor && copy.d != CCV_NNC_NO_TENSOR_SYMBOL)
			{
				const ccv_nnc_tensor_bind_t bind = {
					.symbol = copy,
					.tensor = tensors[i + j * tensor_size]
				};
				ccv_array_push(tensor_binds, &bind);
			}
		}
	}
}

static void _ccv_cnnp_compiled_data_graph_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	if (compiled_data->graph)
		ccv_nnc_graph_free(compiled_data->graph);
	compiled_data->graph = 0;
	compiled_data->is_test = 0;
	if (compiled_data->tensor_arena)
		ccv_nnc_tensor_arena_free(compiled_data->tensor_arena);
	compiled_data->tensor_arena = 0;
	if (compiled_data->graph_exec_arena)
		ccv_nnc_graph_exec_arena_free(compiled_data->graph_exec_arena);
	compiled_data->graph_exec_arena = 0;
	if (compiled_data->backward.from_ops)
		ccfree(compiled_data->backward.from_ops);
	compiled_data->backward.from_ops = 0;
	if (compiled_data->evaluate.schedule)
		ccv_nnc_graph_static_schedule_free(compiled_data->evaluate.schedule);
	compiled_data->evaluate.schedule = 0;
	if (compiled_data->backward.schedule)
		ccv_nnc_graph_static_schedule_free(compiled_data->backward.schedule);
	compiled_data->backward.schedule = 0;
}

static void _ccv_cnnp_compiled_data_gradient_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	if (compiled_data->gradients)
		ccfree(compiled_data->gradients);
	compiled_data->gradients = 0;
	if (compiled_data->updated_parameters)
		ccfree(compiled_data->updated_parameters);
	compiled_data->updated_parameters = 0;
	compiled_data->update_nodes = 0;
	compiled_data->saved_aux = 0;
}

static void _ccv_cnnp_compiled_data_backward_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	if (compiled_data->backward.gradients)
		ccfree(compiled_data->backward.gradients);
	compiled_data->backward.gradients = 0;
	if (compiled_data->backward.accum)
		ccv_nnc_graph_free(compiled_data->backward.accum);
	compiled_data->backward.accum = 0;
	if (compiled_data->backward.tensor_arena)
		ccv_nnc_tensor_arena_free(compiled_data->backward.tensor_arena);
	compiled_data->backward.tensor_arena = 0;
	if (compiled_data->backward.graph_exec_arena)
		ccv_nnc_graph_exec_arena_free(compiled_data->backward.graph_exec_arena);
	compiled_data->backward.graph_exec_arena = 0;
}

static void _ccv_cnnp_compiled_data_apply_gradients_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	if (compiled_data->apply_gradients.graph)
		ccv_nnc_graph_free(compiled_data->apply_gradients.graph);
	compiled_data->apply_gradients.graph = 0;
	if (compiled_data->apply_gradients.tensor_arena)
		ccv_nnc_tensor_arena_free(compiled_data->apply_gradients.tensor_arena);
	compiled_data->apply_gradients.tensor_arena = 0;
	if (compiled_data->apply_gradients.graph_exec_arena)
		ccv_nnc_graph_exec_arena_free(compiled_data->apply_gradients.graph_exec_arena);
	compiled_data->apply_gradients.graph_exec_arena = 0;
}

// Compile the graph to run ccv_cnnp_model_fit
static void _ccv_cnnp_model_fit_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const fits, const int fit_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i, j;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(!compiled_data->graph || compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_FIT_MODE);
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_FIT_MODE;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(!fits || output_size == fit_size);
	assert(output_size > 0);
	if (compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_NONE)
	{
		_ccv_cnnp_model_set_rewindables(model);
		_ccv_cnnp_model_gradient_init(model, CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES, CCV_CNNP_DISABLE_OUTGRAD_ALL, fits, fit_size);
	} else if (compiled_data->gradient_mode != CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES) {
		_ccv_cnnp_model_rewind_graph(model);
		_ccv_cnnp_compiled_data_gradient_free(compiled_data);
		compiled_data->gradient_mode = CCV_CNNP_COMPILED_DATA_GRADIENT_NONE;
		_ccv_cnnp_model_gradient_init(model, CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES, CCV_CNNP_DISABLE_OUTGRAD_ALL, fits, fit_size);
	}
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init(model, compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	assert((input_size % parallel_count) == 0);
	assert((output_size % parallel_count) == 0);
	assert((fit_size % parallel_count) == 0);
	const int input_size_per_p = input_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->inputs, inputs, input_size_per_p, parallel_count, tensor_binds);
	const int output_size_per_p = output_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->outputs, outputs, output_size_per_p, parallel_count, tensor_binds);
	const int fit_size_per_p = fit_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->fits, fits, fit_size_per_p, parallel_count, tensor_binds);
	const int parameter_size = compiled_data->parameters->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0), compiled_data->tensors.parameters, parameter_size, parallel_count, tensor_binds);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->updated_parameters, compiled_data->tensors.parameters, parameter_size, parallel_count, tensor_binds);
	const int internal_size = compiled_data->internals->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, 0), compiled_data->tensors.internals, internal_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, 0), compiled_data->tensors.internals, internal_size, parallel_count, tensor_binds);
	ccv_nnc_symbolic_graph_compile(model->graph, compiled_data->compile_params, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	ccv_array_free(tensor_binds);
	if (tensors_init && parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->tensors_init.v, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0), compiled_data->tensors.parameters, compiled_data->parameters->rnum, parallel_count);
	// If tensor is not init'ed, we need to init states first.
	if (_ccv_cnnp_any_to_init(compiled_data))
	{
		ccv_nnc_tensor_init_states_t tensor_init_states = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.compiled_data = compiled_data,
			.tensor_arena = compiled_data->tensor_arena
		};
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, &tensor_init_states);
	}
	compiled_data->is_test = 0;
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimize.minimizer);
	// No need to set because it is default to training mode.
	// ccv_cnnp_model_set_is_test(model, 0, _ccv_cnnp_cmd_update_for_execs, &update);
	for (i = 0; i < saved_aux_size * parameter_size; i++)
	{
		if (compiled_data->saved_aux[i].source.d == CCV_NNC_NO_TENSOR_SYMBOL)
			continue;
		ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, compiled_data->saved_aux[i].source);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &tensor, 1, 0);
		for (j = 1; j < parallel_count; j++)
		{
			ccv_nnc_tensor_t* const copy = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, compiled_data->saved_aux[i].source, j));
			if (copy)
				ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &copy, 1, 0);
		}
	}
	const int evaluate_to_size = compiled_data->evaluate.to_size;
	compiled_data->evaluate.to_op_size = 0;
	for (i = 0; i < evaluate_to_size; i++)
	{
		ccv_nnc_graph_exec_t const to = ccv_nnc_graph_exec_from_symbol(compiled_data->graph_exec_arena, compiled_data->evaluate.tos[i]);
		if (to.graph)
			compiled_data->evaluate.to_ops[compiled_data->evaluate.to_op_size++] = to;
	}
	ccv_nnc_graph_set_default_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_nnc_graph_autotune(compiled_data->graph, model->workspace_size, 0, TRAVERSE_FULL);
}

ccv_nnc_stream_context_t* ccv_cnnp_model_default_stream(const ccv_cnnp_model_t* const model)
{
	const ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	if (!compiled_data || !compiled_data->graph)
		return 0;
	return ccv_nnc_graph_default_stream(compiled_data->graph);
}

uint64_t ccv_cnnp_model_memory_size(const ccv_cnnp_model_t* const model)
{
	const ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	if (!compiled_data || !compiled_data->tensor_arena)
		return 0;
	return ccv_nnc_tensor_arena_size(compiled_data->tensor_arena);
}

static void _ccv_cnnp_bind_tensors_to_arena(ccv_nnc_tensor_arena_t* const tensor_arena, const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const tensor_symbols, ccv_nnc_tensor_t* const* const tensors, const int tensor_size, const int parallel_count)
{
	int i, j;
	for (i = 0; i < tensor_size; i++)
	{
		ccv_nnc_tensor_symbol_t tensor_symbol = tensor_symbols[i];
		if (graph)
		{
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(graph, tensor_symbol);
			if (alias_to.d != CCV_NNC_NO_TENSOR_SYMBOL)
				tensor_symbol = alias_to;
		}
		ccv_nnc_tensor_bind_symbol(tensor_arena, tensor_symbol, tensors[i]);
		for (j = 1; j < parallel_count; j++)
		{
			const ccv_nnc_tensor_symbol_t copy = ccv_nnc_tensor_symbol_copy(graph, tensor_symbol, j);
			if (copy.d != CCV_NNC_NO_TENSOR_SYMBOL)
				ccv_nnc_tensor_bind_symbol(tensor_arena, copy, tensors[i + tensor_size * j]);
		}
	}
}

void ccv_cnnp_model_fit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const fits, const int fit_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(input_size == model->input_size * parallel_count);
	assert(!fits || fit_size == output_size);
	assert(model->graph);
	if (!compiled_data->graph || compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_FIT_MODE)
	{
		_ccv_cnnp_compiled_data_graph_free(compiled_data);
		_ccv_cnnp_compiled_data_backward_free(compiled_data);
		_ccv_cnnp_compiled_data_apply_gradients_free(compiled_data);
		// Compile the symbolic graph down only when needed.
		_ccv_cnnp_model_fit_jit(model, inputs, input_size, fits, fit_size, outputs, output_size);
	} else {
		assert((input_size % parallel_count) == 0);
		assert((output_size % parallel_count) == 0);
		assert((fit_size % parallel_count) == 0);
		const int input_size_per_p = input_size / parallel_count;
		_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, model->inputs, inputs, input_size_per_p, parallel_count);
		const int output_size_per_p = output_size / parallel_count;
		_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, model->outputs, outputs, output_size_per_p, parallel_count);
		const int fit_size_per_p = fit_size / parallel_count;
		_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, compiled_data->fits, fits, fit_size_per_p, parallel_count);
	}
	if (compiled_data->is_test)
	{
		compiled_data->is_test = 0;
		ccv_nnc_graph_exec_update_t update = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.graph_exec_arena = compiled_data->graph_exec_arena,
		};
		ccv_cnnp_model_set_is_test(model, 0, _ccv_cnnp_cmd_update_for_execs, &update);
	}
	ccv_nnc_graph_run_with_schedule(compiled_data->graph, 0, 0, tensor_tape, stream_context);
}

// Compile the graph to run ccv_cnnp_model_evaluate with require_grad = false (MULTISTAGE_MODE_NO_GRAD).
static void _ccv_cnnp_model_multistage_no_grad_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE_NO_GRAD;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(output_size > 0);
	// If the gradient is not initialized, continue to setup parallel process. We don't init gradient here, but rather,
	// we setup proper rewindables so the graph can be rewinded to previous state before we run data parallel.
	if (parallel_count > 1 && compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_NONE)
	{
		const int evaluate_to_size = compiled_data->evaluate.to_size;
		compiled_data->evaluate.tos = ccrealloc(compiled_data->evaluate.tos, sizeof(ccv_nnc_graph_exec_symbol_t) * evaluate_to_size * parallel_count + sizeof(ccv_nnc_graph_exec_t) * evaluate_to_size * parallel_count);
		_ccv_cnnp_model_set_rewindables(model);
		ccv_nnc_symbolic_graph_data_parallel(model->graph, parallel_count,
			0, 0,
			0, 0, 0,
			0, 0, 0,
			CCV_NNC_PARALLEL_REDUCE_OP_SUM,
			SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
		ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		int i, j;
		for (i = 0; i < evaluate_to_size; i++)
			for (j = 1; j < parallel_count; j++)
			{
				const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_copy(model->graph, compiled_data->evaluate.tos[i], j);
				if (copy.d != CCV_NNC_NO_GRAPH_EXEC_SYMBOL)
					compiled_data->evaluate.tos[compiled_data->evaluate.to_size++] = copy;
			}
	}
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init(model, compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	assert((input_size % parallel_count) == 0);
	assert((output_size % parallel_count) == 0);
	const int input_size_per_p = input_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->inputs, inputs, input_size_per_p, parallel_count, tensor_binds);
	const int output_size_per_p = output_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->outputs, outputs, output_size_per_p, parallel_count, tensor_binds);
	const int parameter_size = compiled_data->parameters->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0), compiled_data->tensors.parameters, parameter_size, parallel_count, tensor_binds);
	const int internal_size = compiled_data->internals->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, 0), compiled_data->tensors.internals, internal_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, 0), compiled_data->tensors.internals, internal_size, parallel_count, tensor_binds);
	// If we generated gradient for the graph, only compile part of the graph because the rest is irrelevant for evaluation.
	ccv_nnc_symbolic_graph_compile(model->graph, compiled_data->compile_params, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), compiled_data->evaluate.tos, compiled_data->evaluate.to_size, &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	ccv_array_free(tensor_binds);
	// If tensor is not init'ed, we need to init states first.
	if (tensors_init && parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->tensors_init.v, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0), compiled_data->tensors.parameters, compiled_data->parameters->rnum, parallel_count);
	if (_ccv_cnnp_any_to_init(compiled_data))
	{
		ccv_nnc_tensor_init_states_t tensor_init_states = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.compiled_data = compiled_data,
			.tensor_arena = compiled_data->tensor_arena
		};
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, &tensor_init_states);
	}
	compiled_data->is_test = 1;
	ccv_nnc_graph_exec_update_t update = {
		.parallel_count = parallel_count,
		.graph = model->graph,
		.graph_exec_arena = compiled_data->graph_exec_arena,
	};
	ccv_cnnp_model_set_is_test(model, 1, _ccv_cnnp_cmd_update_for_execs, &update);
	ccv_nnc_graph_set_default_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_nnc_graph_autotune(compiled_data->graph, model->workspace_size, 0, TRAVERSE_FULL);
}

static void _ccv_cnnp_model_gradient_tensors_init(const ccv_cnnp_model_t* const model, ccv_cnnp_compiled_data_t* const compiled_data)
{
	assert(!compiled_data->tensors.gradients);
	const int parameter_size = compiled_data->parameters->rnum;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	compiled_data->tensors.gradients = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * parameter_size * 2 * parallel_count);
	compiled_data->tensors.accum_gradients = compiled_data->tensors.gradients + parameter_size * parallel_count;
	int i, j;
	for (i = 0; i < parameter_size; i++)
	{
		const ccv_nnc_tensor_symbol_t parameter = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(parameter.graph, parameter);
		if (CCV_TENSOR_GET_DEVICE(info.type) == CCV_COMPUTE_DEVICE_ANY)
			CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(info.type);
		compiled_data->tensors.gradients[i] = ccv_nnc_tensor_new(0, info, 0);
		compiled_data->tensors.accum_gradients[i] = 0; // delay the accumulated gradient allocation until when we need it.
		for (j = 1; j < parallel_count; j++)
		{
			if (j != device_id)
				CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			else
				CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
			compiled_data->tensors.gradients[i + j * parameter_size] = ccv_nnc_tensor_new(0, info, 0);
			compiled_data->tensors.accum_gradients[i + j * parameter_size] = 0;
		}
	}
}

static int _ccv_cnnp_is_disable_outgrad_all(const uint64_t disable_outgrad, const int input_size)
{
	if (disable_outgrad == CCV_CNNP_DISABLE_OUTGRAD_ALL)
		return 1;
	if (disable_outgrad == CCV_CNNP_DISABLE_OUTGRAD_NONE)
		return 0;
	int i;
	for (i = 0; i < input_size; i++)
		if (!(disable_outgrad & ((uint64_t)1 << i)))
			return 0;
	return 1;
}

// Compile the graph to run ccv_cnnp_model_evaluate with requires_grad = true (MULTISTAGE_MODE).
// Particularly, this method compiles the evaluation and backprop graph (the main graph).
static void _ccv_cnnp_model_multistage_jit_0(ccv_cnnp_model_t* const model, const uint64_t disable_outgrad, const int is_test, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i, j;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	const int target_gradient_mode = _ccv_cnnp_is_disable_outgrad_all(disable_outgrad, model->input_size) ? CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES : CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS;
	assert(!compiled_data->graph || compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE || compiled_data->gradient_mode != target_gradient_mode);
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(output_size > 0);
	// There shouldn't be a loss function if we evaluate with multistage jit.
	assert(compiled_data->loss.cmd == CCV_NNC_NOOP);
	if (compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_NONE)
	{
		_ccv_cnnp_model_set_rewindables(model);
		_ccv_cnnp_model_gradient_init(model, target_gradient_mode, disable_outgrad, 0, 0); // The type of outputs and fits should be the same. We only use type here.
	} else if (compiled_data->gradient_mode != target_gradient_mode) {
		_ccv_cnnp_model_rewind_graph(model);
		_ccv_cnnp_compiled_data_gradient_free(compiled_data);
		compiled_data->gradient_mode = CCV_CNNP_COMPILED_DATA_GRADIENT_NONE;
		_ccv_cnnp_model_gradient_init(model, target_gradient_mode, disable_outgrad, 0, 0); // The type of outputs and fits should be the same. We only use type here.
	}
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init(model, compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	assert((input_size % parallel_count) == 0);
	assert((output_size % parallel_count) == 0);
	const int input_size_per_p = input_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->inputs, inputs, input_size_per_p, parallel_count, tensor_binds);
	const int output_size_per_p = output_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->outputs, outputs, output_size_per_p, parallel_count, tensor_binds);
	const int parameter_size = compiled_data->parameters->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0), compiled_data->tensors.parameters, parameter_size, parallel_count, tensor_binds);
	const int internal_size = compiled_data->internals->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, 0), compiled_data->tensors.internals, internal_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, 0), compiled_data->tensors.internals, internal_size, parallel_count, tensor_binds);
	if (!compiled_data->tensors.gradients)
		_ccv_cnnp_model_gradient_tensors_init(model, compiled_data);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->gradients, compiled_data->tensors.gradients, parameter_size, parallel_count, tensor_binds);
	ccv_nnc_symbolic_graph_compile(model->graph, compiled_data->compile_params, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), compiled_data->backward.tos, compiled_data->backward.to_size, &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	ccv_array_free(tensor_binds);
	if (tensors_init && parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->tensors_init.v, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0), compiled_data->tensors.parameters, compiled_data->parameters->rnum, parallel_count);
	// If tensor is not init'ed, we need to init states first.
	if (_ccv_cnnp_any_to_init(compiled_data))
	{
		ccv_nnc_tensor_init_states_t tensor_init_states = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.compiled_data = compiled_data,
			.tensor_arena = compiled_data->tensor_arena
		};
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, &tensor_init_states);
	}
	compiled_data->is_test = is_test;
	ccv_nnc_graph_exec_update_t update = {
		.parallel_count = parallel_count,
		.graph = model->graph,
		.graph_exec_arena = compiled_data->graph_exec_arena,
	};
	ccv_cnnp_model_set_is_test(model, is_test, _ccv_cnnp_cmd_update_for_execs, &update);
	const int evaluate_to_size = compiled_data->evaluate.to_size;
	compiled_data->evaluate.to_op_size = 0;
	ccv_array_t* const backward_from = ccv_array_new(sizeof(int), 0, 0);
	for (i = 0; i < evaluate_to_size; i++)
	{
		ccv_nnc_graph_exec_t const to_op = ccv_nnc_graph_exec_from_symbol(compiled_data->graph_exec_arena, compiled_data->evaluate.tos[i]);
		if (to_op.graph)
			compiled_data->evaluate.to_ops[compiled_data->evaluate.to_op_size++] = to_op;
		const int* tos;
		int to_size;
		ccv_nnc_graph_exec_symbol_to(model->graph, compiled_data->evaluate.tos[i], &tos, &to_size);
		for (j = 0; j < to_size; j++)
		{
			ccv_nnc_graph_exec_t const to_op = ccv_nnc_graph_exec_from_symbol(compiled_data->graph_exec_arena, (ccv_nnc_graph_exec_symbol_t){
				.d = tos[j],
				.graph = model->graph
			});
			if (to_op.graph)
				ccv_array_add_unique_int(backward_from, to_op.d);
		}
	}
	assert(backward_from->rnum > 0);
	compiled_data->backward.from_op_size = backward_from->rnum;
	compiled_data->backward.from_ops = (ccv_nnc_graph_exec_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_t) * backward_from->rnum);
	for (i = 0; i < backward_from->rnum; i++)
		compiled_data->backward.from_ops[i] = (ccv_nnc_graph_exec_t){
			.d = *(int*)ccv_array_get(backward_from, i),
			.graph = compiled_data->graph,
		};
	ccv_array_free(backward_from);
	ccv_nnc_graph_set_default_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_nnc_graph_autotune(compiled_data->graph, model->workspace_size, 0, TRAVERSE_FULL);
}

void ccv_cnnp_model_evaluate(ccv_cnnp_model_t* const model, const ccv_cnnp_evaluate_param_t params, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(input_size == model->input_size * parallel_count);
	assert(model->graph);
	const int target_gradient_mode = _ccv_cnnp_is_disable_outgrad_all(params.disable_outgrad, model->input_size) ? CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES : CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS;
	const int mode_mismatch = (params.requires_grad && (compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE || compiled_data->gradient_mode != target_gradient_mode || compiled_data->disable_outgrad != params.disable_outgrad));
	if (!compiled_data->graph || mode_mismatch)
	{
		_ccv_cnnp_compiled_data_graph_free(compiled_data);
		if (mode_mismatch) // If mode mismatch, we need to redo the backward as well (no need to redo apply_gradients, it doesn't require target_gradient_mode or disable_outgrad.
			_ccv_cnnp_compiled_data_backward_free(compiled_data);
		if (params.requires_grad)
			_ccv_cnnp_model_multistage_jit_0(model, params.disable_outgrad, params.is_test, inputs, input_size, outputs, output_size);
		else
			_ccv_cnnp_model_multistage_no_grad_jit(model, inputs, input_size, outputs, output_size);
	} else {
		ccv_nnc_tensor_arena_clear_bindings(compiled_data->tensor_arena);
		assert((input_size % parallel_count) == 0);
		const int input_size_per_p = input_size / parallel_count;
		_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, model->inputs, inputs, input_size_per_p, parallel_count);
		assert((output_size % parallel_count) == 0);
		const int output_size_per_p = output_size / parallel_count;
		_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, model->outputs, outputs, output_size_per_p, parallel_count);
	}
	if (compiled_data->is_test != params.is_test)
	{
		compiled_data->is_test = params.is_test;
		ccv_nnc_graph_exec_update_t update = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.graph_exec_arena = compiled_data->graph_exec_arena,
		};
		ccv_cnnp_model_set_is_test(model, params.is_test, _ccv_cnnp_cmd_update_for_execs, &update);
	}
	if (compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE_NO_GRAD)
		ccv_nnc_graph_run_with_schedule(compiled_data->graph, 0, 0, tensor_tape, stream_context);
	else {
		if (!compiled_data->evaluate.schedule)
			compiled_data->evaluate.schedule = ccv_nnc_graph_static_schedule_new(compiled_data->graph, compiled_data->stream_type, 0, 0, compiled_data->evaluate.to_ops, compiled_data->evaluate.to_op_size);
		ccv_nnc_graph_run_with_schedule(compiled_data->graph, 0, compiled_data->evaluate.schedule, tensor_tape, stream_context);
	}
}

// Compile the graph to run ccv_cnnp_model_backward after ccv_cnnp_model_evaluate with requires_grad = true (MULTISTAGE_MODE).
// Particularly, this method compiles the accumulator graph.
static void _ccv_cnnp_model_multistage_jit_1(ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	ccv_nnc_symbolic_graph_t* accum = ccv_nnc_symbolic_graph_new();
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int parameter_size = compiled_data->parameters->rnum;
	int i, j;
	compiled_data->backward.gradients = (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * parameter_size * parallel_count * 3);
	compiled_data->backward.accum_gradients = compiled_data->backward.gradients + parameter_size * parallel_count;
	compiled_data->backward.updated_accum_gradients = compiled_data->backward.accum_gradients + parameter_size * parallel_count;
	for (i = 0; i < parameter_size; i++)
		for (j = 0; j < parallel_count; j++)
		{
			const ccv_nnc_tensor_param_t info = compiled_data->tensors.gradients[i + j * parameter_size]->info;
			// Now, the old gradient is the accumulated gradient, getting new gradient tensor setup so we can collect them.
			compiled_data->tensors.accum_gradients[i + j * parameter_size] = compiled_data->tensors.gradients[i + j * parameter_size];
			compiled_data->tensors.gradients[i + j * parameter_size] = ccv_nnc_tensor_new(0, info, 0);
			ccv_nnc_tensor_symbol_t inputs[2];
			inputs[0] = compiled_data->backward.accum_gradients[i + j * parameter_size] = ccv_nnc_tensor_symbol_new(accum, info, 0);
			inputs[1] = compiled_data->backward.gradients[i + j * parameter_size] = ccv_nnc_tensor_symbol_new(accum, info, 0);
			ccv_nnc_tensor_symbol_t output = compiled_data->backward.updated_accum_gradients[i + j * parameter_size] = ccv_nnc_tensor_symbol_new(accum, info, 0);
			ccv_nnc_graph_exec_symbol_new(accum, CMD_EWSUM_FORWARD(), inputs, 2, &output, 1, 0);
		}
	ccv_nnc_graph_exec_symbol_autogen(accum, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	_ccv_cnnp_model_bind_tensors(accum, compiled_data->backward.accum_gradients, compiled_data->tensors.accum_gradients, parameter_size * parallel_count, 1, tensor_binds);
	_ccv_cnnp_model_bind_tensors(accum, compiled_data->backward.gradients, compiled_data->tensors.gradients, parameter_size * parallel_count, 1, tensor_binds);
	_ccv_cnnp_model_bind_tensors(accum, compiled_data->backward.updated_accum_gradients, compiled_data->tensors.accum_gradients, parameter_size * parallel_count, 1, tensor_binds);
	ccv_nnc_symbolic_graph_compile(accum, compiled_data->compile_params, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(accum), SYMBOLIC_GRAPH_DESTINATIONS(accum), &compiled_data->backward.accum, &compiled_data->backward.tensor_arena, &compiled_data->backward.graph_exec_arena);
	ccv_nnc_symbolic_graph_free(accum);
	ccv_nnc_graph_set_default_static_schedule(compiled_data->backward.accum, compiled_data->stream_type);
	ccv_array_free(tensor_binds);
}

void ccv_cnnp_model_backward(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const ingrads, const int ingrad_size, ccv_nnc_tensor_t* const* const outgrads, const int outgrad_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	assert(ingrad_size == 0 || ingrad_size == model->output_size * parallel_count);
	if (outgrad_size > 0)
		{ assert(outgrad_size == compiled_data->outgrad_size * parallel_count); }
	assert(model->graph);
	assert(compiled_data->graph);
	const int parameter_size = compiled_data->parameters->rnum;
	// If we need to accumulate the gradients now, do jit on accumulator.
	if (compiled_data->backward.count > 0)
	{
		if (!compiled_data->backward.accum)
			_ccv_cnnp_model_multistage_jit_1(model);
		else if (compiled_data->backward.count == 1) {
			//  On this round, we need to switch accumulated gradients with gradients (so we can do accumulation properly).
			int i;
			ccv_nnc_tensor_arena_clear_bindings(compiled_data->backward.tensor_arena);
			for (i = 0; i < parameter_size * parallel_count; i++)
			{
				ccv_nnc_tensor_t* tensor;
				CCV_SWAP(compiled_data->tensors.accum_gradients[i], compiled_data->tensors.gradients[i], tensor);
			}
			// Do rebind in case we messed up the binding (we switch accum_gradients and gradients).
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->backward.tensor_arena, 0, compiled_data->backward.gradients, compiled_data->tensors.gradients, parameter_size * parallel_count, 1);
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->backward.tensor_arena, 0, compiled_data->backward.accum_gradients, compiled_data->tensors.accum_gradients, parameter_size * parallel_count, 1);
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->backward.tensor_arena, 0, compiled_data->backward.updated_accum_gradients, compiled_data->tensors.accum_gradients, parameter_size * parallel_count, 1);
		}
	}
	const int ingrad_size_per_p = model->output_size;
	const int outgrad_size_per_p = compiled_data->outgrad_size;
	int i, j;
	for (i = 0; i < ingrad_size_per_p; i++)
	{
		const ccv_nnc_tensor_symbol_t ingrad = ccv_nnc_tensor_symbol_for_backward(model->graph, compiled_data->f[i]);
		if (!ingrad_size || !ingrads || ingrads[i] == 0)
		{
			// Set it to 1 if it is not specified.
			ccv_nnc_tensor_t* const ingrad_tensor = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, ingrad);
			if (ingrad_tensor)
				ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(ingrad_tensor), stream_context);
			for (j = 1; j < parallel_count; j++)
			{
				ccv_nnc_tensor_t* const ingrad_tensor = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, ingrad, j));
				if (ingrad_tensor)
					ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(ingrad_tensor), stream_context);
			}
		} else {
			// Make sure the length matches, in case it is an alias.
			assert(ccv_nnc_tensor_count(ingrads[i]->info) == ccv_nnc_tensor_count(ccv_nnc_tensor_symbol_params(model->graph, ingrad)));
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ingrad, ingrads[i]);
			for (j = 1; j < parallel_count; j++)
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, ingrad, j), ingrads[i + ingrad_size_per_p * j]);
		}
	}
	if (outgrad_size > 0)
	{
		assert(compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS && "shouldn't pass disable_outgrad to ccv_cnnp_model_evaluate before if you plan to compute outgrad");
		for (i = 0; i < outgrad_size_per_p; i++)
			if (outgrads[i])
			{
				const ccv_nnc_tensor_symbol_t outgrad = compiled_data->outgrads[i];
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, outgrad, outgrads[i]);
				for (j = 1; j < parallel_count; j++)
					ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, outgrad, j), outgrads[i + outgrad_size_per_p * j]);
			}
	} else {
		assert(compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES ||
			compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS);
	}
	// We need to rebind here because in ccv_cnnp_evaluate, we clear bindings, that will reset all bindings for the gradients.
	// For parameters and internals these are fine because when we clear bindings, it restores to original bindings, which are these
	// parameters and internals. The same cannot be said for gradients due to the accum_gradients switching.
	_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, compiled_data->gradients, compiled_data->tensors.gradients, parameter_size, parallel_count);
	if (!compiled_data->backward.schedule)
		compiled_data->backward.schedule = ccv_nnc_graph_static_schedule_new(compiled_data->graph, compiled_data->stream_type, compiled_data->backward.from_ops, compiled_data->backward.from_op_size, 0, 0);
	// Run the backward pass.
	ccv_nnc_graph_run_with_schedule(compiled_data->graph, 0, compiled_data->backward.schedule, tensor_tape, stream_context);
	// If we need to run accumulation round, do that now.
	if (compiled_data->backward.count > 0)
		ccv_nnc_graph_run_with_schedule(compiled_data->backward.accum, 0, 0, 0, stream_context);
	// Update the count, this determines whether we need to accumulate or not.
	++compiled_data->backward.count;
}

// Compile the graph to run ccv_cnnp_model_apply_gradients after ccv_cnnp_model_backward (MULTISTAGE_MODE).
// Particularly, this method compiles the parameter update graph.
static void _ccv_cnnp_model_multistage_jit_2(ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int parameter_size = compiled_data->parameters->rnum;
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, 0), compiled_data->tensors.parameters, parameter_size, parallel_count, tensor_binds);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->updated_parameters, compiled_data->tensors.parameters, parameter_size, parallel_count, tensor_binds);
	// Bind accumulated gradients.
	if (compiled_data->backward.count > 1)
		_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->gradients, compiled_data->tensors.accum_gradients, parameter_size, parallel_count, tensor_binds);
	else
		_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->gradients, compiled_data->tensors.gradients, parameter_size, parallel_count, tensor_binds);
	ccv_array_t* const apply_gradients_from = ccv_array_new(sizeof(int), 0, 0);
	int i, j;
	for (i = 0; i < compiled_data->backward.to_size; i++)
	{
		const int* tos;
		int to_size;
		ccv_nnc_graph_exec_symbol_to(model->graph, compiled_data->backward.tos[i], &tos, &to_size);
		for (j = 0; j < to_size; j++)
		{
			// Check if this is already show up in the backward graph, if that is the case, it won't be in the apply
			// gradients graph.
			const ccv_nnc_graph_exec_t exec = ccv_nnc_graph_exec_from_symbol(compiled_data->graph_exec_arena, (ccv_nnc_graph_exec_symbol_t){
				.d = tos[j],
				.graph = model->graph,
			});
			if (!exec.graph)
				ccv_array_add_unique_int(apply_gradients_from, tos[j]);
		}
	}
	const int from_size = apply_gradients_from->rnum;
	ccv_nnc_graph_exec_symbol_t* const froms = (ccv_nnc_graph_exec_symbol_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * from_size);
	for (i = 0; i < from_size; i++)
		froms[i] = (ccv_nnc_graph_exec_symbol_t){
			.d = *(int*)ccv_array_get(apply_gradients_from, i),
			.graph = model->graph
		};
	ccv_array_free(apply_gradients_from);
	// It can only ends with updates on the parameters.
	ccv_array_t* const tos = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), parameter_size * parallel_count, 0);
	for (i = 0;  i < parameter_size; i++)
	{
		if (compiled_data->update_nodes[i].d == CCV_NNC_NO_TENSOR_SYMBOL)
			continue;
		ccv_array_push(tos, &compiled_data->update_nodes[i]);
		for (j = 1; j < parallel_count; j++)
		{
			const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_copy(model->graph, compiled_data->update_nodes[i], j);
			ccv_array_push(tos, &copy);
		}
	}
	ccv_nnc_symbolic_graph_compile(model->graph, compiled_data->compile_params, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, froms, from_size, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(tos, 0), tos->rnum, &compiled_data->apply_gradients.graph, &compiled_data->apply_gradients.tensor_arena, &compiled_data->apply_gradients.graph_exec_arena);
	ccv_array_free(tos);
	ccv_array_free(tensor_binds);
	ccfree(froms);
	const int max_saved_aux_size = compiled_data->minimize.max_saved_aux_size;
	for (i = 0; i < max_saved_aux_size * parameter_size; i++)
	{
		// Skip on no tensor.
		if (compiled_data->saved_aux[i].source.d == CCV_NNC_NO_TENSOR_SYMBOL)
			continue;
		ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_symbol(compiled_data->apply_gradients.tensor_arena, compiled_data->saved_aux[i].source);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &tensor, 1, 0);
		for (j = 1; j < parallel_count; j++)
		{
			ccv_nnc_tensor_t* const copy = ccv_nnc_tensor_from_symbol(compiled_data->apply_gradients.tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, compiled_data->saved_aux[i].source, j));
			if (copy)
				ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &copy, 1, 0);
		}
	}
	ccv_nnc_graph_set_default_static_schedule(compiled_data->apply_gradients.graph, compiled_data->stream_type);
}

void ccv_cnnp_model_apply_gradients(ccv_cnnp_model_t* const model, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	assert(model->graph);
	assert(compiled_data->graph);
	// Skip if there is no backward pass.
	if (compiled_data->backward.count <= 0)
		return;
	// Skip if there is no parameters.
	if (compiled_data->parameters->rnum == 0)
	{
		compiled_data->backward.count = 0;
		return;
	}
	if (!compiled_data->apply_gradients.graph)
		_ccv_cnnp_model_multistage_jit_2(model);
	else {
		const int parameter_size = compiled_data->parameters->rnum;
		ccv_nnc_tensor_arena_clear_bindings(compiled_data->apply_gradients.tensor_arena);
		// Change to bind accum_gradients if we do gradient accumulation (run backward more than once).
		if (compiled_data->backward.count > 1)
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->apply_gradients.tensor_arena, model->graph, compiled_data->gradients, compiled_data->tensors.accum_gradients, parameter_size, parallel_count);
		else
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->apply_gradients.tensor_arena, model->graph, compiled_data->gradients, compiled_data->tensors.gradients, parameter_size, parallel_count);
	}
	ccv_nnc_graph_run_with_schedule(compiled_data->apply_gradients.graph, 0, 0, 0, stream_context);
	// Reset backward count to 0.
	compiled_data->backward.count = 0;
}

void ccv_cnnp_model_set_parameter(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameter, const ccv_nnc_tensor_t* const tensor)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	const int param_sel = parameter->param_sel > 0 ? parameter->param_sel - 1 : parameter->param_sel;
	assert(parameter->param_sel != 0);
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init(model, compiled_data);
	ccv_array_t* const parameter_indices = ccv_array_new(sizeof(int), 0, 0);
	ccv_cnnp_model_add_to_parameter_indices(parameter->model, param_sel, parameter_indices);
	const int param_ref = parameter->param_ref > 0 ? parameter->param_ref - 1 : parameter->param_ref;
	if (param_ref < 0)
		{ assert(parameter_indices->rnum == 1); }
	else
		{ assert(param_ref < parameter_indices->rnum); }
	const int d = *(int*)ccv_array_get(parameter_indices, param_ref >= 0 ? param_ref : 0);
	ccv_array_free(parameter_indices);
	const int parameter_size = compiled_data->parameters->rnum;
	assert(d >= 0);
	assert(d < parameter_size);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	ccv_nnc_tensor_t* const dest = compiled_data->tensors.parameters[d];
	assert(dest);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)tensor), TENSOR_LIST(dest), 0);
	int i;
	for (i = 1; i < parallel_count; i++)
	{
		ccv_nnc_tensor_t* const copy_tensor = compiled_data->tensors.parameters[d + i * parameter_size];
		if (copy_tensor)
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dest), TENSOR_LIST(copy_tensor), 0);
	}
	// Mark this symbol as init'ed.
	const int s = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, d))->d;
	compiled_data->tensors_init.v[s >> 5] |= (1u << (s & 0x1f));
}

void ccv_cnnp_model_parameter_copy(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameter, ccv_nnc_tensor_t* const tensor)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	const int param_sel = parameter->param_sel > 0 ? parameter->param_sel - 1 : parameter->param_sel;
	assert(parameter->param_sel != 0);
	assert(compiled_data->tensors.parameters);
	ccv_array_t* const parameter_indices = ccv_array_new(sizeof(int), 0, 0);
	ccv_cnnp_model_add_to_parameter_indices(parameter->model, param_sel, parameter_indices);
	const int param_ref = parameter->param_ref > 0 ? parameter->param_ref - 1 : parameter->param_ref;
	if (param_ref < 0)
		{ assert(parameter_indices->rnum == 1); }
	else
		{ assert(param_ref < parameter_indices->rnum); }
	const int d = *(int*)ccv_array_get(parameter_indices, param_ref >= 0 ? param_ref : 0);
	ccv_array_free(parameter_indices);
	const int parameter_size = compiled_data->parameters->rnum;
	assert(d >= 0);
	assert(d < parameter_size);
	// We don't need to consider parallel_count, every parameter on each device is identical.
	ccv_nnc_tensor_t* const src = compiled_data->tensors.parameters[d];
	assert(src);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(src), TENSOR_LIST(tensor), 0);
}

ccv_nnc_tensor_param_t ccv_cnnp_model_parameter_tensor_params(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameter)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	const int param_sel = parameter->param_sel > 0 ? parameter->param_sel - 1 : parameter->param_sel;
	assert(parameter->param_sel != 0);
	assert(compiled_data->tensors.parameters);
	ccv_array_t* const parameter_indices = ccv_array_new(sizeof(int), 0, 0);
	ccv_cnnp_model_add_to_parameter_indices(parameter->model, param_sel, parameter_indices);
	const int param_ref = parameter->param_ref > 0 ? parameter->param_ref - 1 : parameter->param_ref;
	if (param_ref < 0)
		{ assert(parameter_indices->rnum == 1); }
	else
		{ assert(param_ref < parameter_indices->rnum); }
	const int d = *(int*)ccv_array_get(parameter_indices, param_ref >= 0 ? param_ref : 0);
	ccv_array_free(parameter_indices);
	const int parameter_size = compiled_data->parameters->rnum;
	assert(d >= 0);
	assert(d < parameter_size);
	// We don't need to consider parallel_count, every parameter on each device is identical.
	ccv_nnc_tensor_t* const tensor = compiled_data->tensors.parameters[d];
	assert(tensor);
	return tensor->info;
}

const char* ccv_cnnp_model_parameter_name(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameter)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	const int param_sel = parameter->param_sel > 0 ? parameter->param_sel - 1 : parameter->param_sel;
	assert(parameter->param_sel != 0);
	ccv_array_t* const parameter_indices = ccv_array_new(sizeof(int), 0, 0);
	ccv_cnnp_model_add_to_parameter_indices(parameter->model, param_sel, parameter_indices);
	const int param_ref = parameter->param_ref > 0 ? parameter->param_ref - 1 : parameter->param_ref;
	if (param_ref < 0)
		{ assert(parameter_indices->rnum == 1); }
	else
		{ assert(param_ref < parameter_indices->rnum); }
	const int d = *(int*)ccv_array_get(parameter_indices, param_ref >= 0 ? param_ref : 0);
	ccv_array_free(parameter_indices);
	const int parameter_size = compiled_data->parameters->rnum;
	assert(d >= 0);
	assert(d < parameter_size);
	return *(char**)ccv_array_get(compiled_data->ids.parameters, d);
}

static ccv_array_t* _ccv_cnnp_model_parameter_indices(const ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, int* const param_ref)
{
	const int to_param_sel = parameters->param_sel > 0 ? parameters->param_sel - 1 : parameters->param_sel;
	assert(parameters->param_sel != 0);
	ccv_array_t* const to_parameter_indices = ccv_array_new(sizeof(int), 0, 0);
	ccv_cnnp_model_add_to_parameter_indices(parameters->model, to_param_sel, to_parameter_indices);
	*param_ref = parameters->param_ref > 0 ? parameters->param_ref - 1 : parameters->param_ref;
	return to_parameter_indices;
}

static void _ccv_cnnp_model_to_parameter_indices_and_from_parameter_indices(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, const ccv_cnnp_model_t* const from_model, const ccv_cnnp_model_io_t from_parameters, ccv_array_t** const parameter_indices, int* const param_ref, ccv_array_t** const from_parameter_indices, int* const from_param_ref)
{
	// If the model is not compiled yet. Compile them now.
	if (!model->graph)
	{
		model->graph = ccv_nnc_symbolic_graph_new();
		assert(from_model->compiled_data);
		const int input_size = from_model->input_size;
		ccv_nnc_tensor_param_t input_params[input_size];
		int i;
		for (i = 0; i < input_size; i++)
			input_params[i] = ccv_nnc_tensor_symbol_params(from_model->graph, from_model->inputs[i]);
		_ccv_cnnp_model_compile(model, input_params, input_size, from_model->compiled_data->loss);
		model->parallel_count = from_model->parallel_count;
		model->memory_compression = from_model->memory_compression;
		model->compiled_data->stream_type = from_model->compiled_data->stream_type;
		model->compiled_data->minimize.minimizer = from_model->compiled_data->minimize.minimizer;
		model->compiled_data->minimize.max_saved_aux_size = from_model->compiled_data->minimize.max_saved_aux_size;
	}
	ccv_cnnp_compiled_data_t* const to_compiled_data = model->compiled_data;
	assert(to_compiled_data);
	const int to_tensors_init = !!to_compiled_data->tensors_init.v;
	if (!to_tensors_init)
		ccv_cnnp_model_tensors_init(model, to_compiled_data);
	assert(to_compiled_data->tensors.parameters);
	*parameter_indices = _ccv_cnnp_model_parameter_indices(model, parameters, param_ref);
	*from_parameter_indices = _ccv_cnnp_model_parameter_indices(from_model, from_parameters, from_param_ref);
	if (*from_param_ref < 0 && *param_ref >= 0)
		{ assert((*from_parameter_indices)->rnum == 1); }
	else if (*from_param_ref >= 0)
		{ assert(*from_param_ref < (*from_parameter_indices)->rnum); }
	if (*param_ref < 0 && *from_param_ref >= 0)
		{ assert((*parameter_indices)->rnum == 1); }
	else if (*param_ref >= 0)
		{ assert(*param_ref < (*parameter_indices)->rnum); }
	// Should be exactly the same tensor.
	if (*param_ref < 0 && *from_param_ref < 0)
		{ assert((*from_parameter_indices)->rnum == (*parameter_indices)->rnum); }
}

void ccv_cnnp_model_set_parameters(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, const ccv_cnnp_model_t* const from_model, const ccv_cnnp_model_io_t from_parameters)
{
	ccv_array_t* to_parameter_indices;
	int to_param_ref;
	ccv_array_t* from_parameter_indices;
	int from_param_ref;
	_ccv_cnnp_model_to_parameter_indices_and_from_parameter_indices(model, parameters, from_model, from_parameters, &to_parameter_indices, &to_param_ref, &from_parameter_indices, &from_param_ref);
	// To models.
	ccv_cnnp_compiled_data_t* const to_compiled_data = model->compiled_data;
	assert(to_compiled_data);
	// From models.
	const ccv_cnnp_compiled_data_t* const from_compiled_data = from_model->compiled_data;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int to_parameter_size = to_compiled_data->parameters->rnum;
	const int rnum = (to_param_ref < 0 && from_param_ref < 0) ? from_parameter_indices->rnum : 1;
	int i, j;
	for (i = 0; i < rnum; i++)
	{
		const int src_d = *(int*)ccv_array_get(from_parameter_indices,from_param_ref >= 0 ? from_param_ref : i);
		assert(src_d >= 0);
		assert(src_d < from_compiled_data->parameters->rnum);
		const int s = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(from_compiled_data->parameters, src_d))->d;
		// If the original is not init'ed. We cannot copy from.
		if (!(from_compiled_data->tensors_init.v[s >> 5] & (1u << (s & 0x1f))))
			continue;
		const int dest_d = *(int*)ccv_array_get(to_parameter_indices, to_param_ref >= 0 ? to_param_ref : i);
		assert(dest_d >= 0);
		assert(dest_d < to_compiled_data->parameters->rnum);
		ccv_nnc_tensor_t* const src = from_compiled_data->tensors.parameters[src_d];
		assert(src);
		ccv_nnc_tensor_t* const dest = to_compiled_data->tensors.parameters[dest_d];
		assert(dest);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(src), TENSOR_LIST(dest), 0);
		for (j = 1; j < parallel_count; j++)
		{
			ccv_nnc_tensor_t* const copy_tensor = to_compiled_data->tensors.parameters[dest_d + j * to_parameter_size];
			if (copy_tensor)
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dest), TENSOR_LIST(copy_tensor), 0);
		}
		// Mark this symbol as init'ed.
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(to_compiled_data->parameters, dest_d))->d;
		to_compiled_data->tensors_init.v[d >> 5] |= (1u << (d & 0x1f));
	}
	ccv_array_free(to_parameter_indices);
	ccv_array_free(from_parameter_indices);
}

ccv_nnc_stream_context_t* ccv_cnnp_compiled_data_get_stream(ccv_cnnp_compiled_data_t* const compiled_data, const int type)
{
	if (!compiled_data->stream_map)
		compiled_data->stream_map = kh_init(stream_map);
	int ret = 0;
	khiter_t k = kh_put(stream_map, compiled_data->stream_map, type, &ret);
	assert(ret >= 0);
	ccv_nnc_stream_context_t* stream = kh_val(compiled_data->stream_map, k);
	// If ret == 0, the key already exist, we can return directly, otherwise, create and return.
	if (ret != 0)
	{
		stream = ccv_nnc_stream_context_new(type);
		kh_val(compiled_data->stream_map, k) = stream;
	}
	return stream;
}

void ccv_cnnp_model_parameters_zip_map(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const aux_ins, const int aux_in_size, ccv_nnc_tensor_t* const* const aux_outs, const int aux_out_size, ccv_nnc_stream_context_t* const stream_context, const ccv_cnnp_model_t* const from_model, const ccv_cnnp_model_io_t from_parameters)
{
	ccv_array_t* to_parameter_indices;
	int to_param_ref;
	ccv_array_t* from_parameter_indices;
	int from_param_ref;
	_ccv_cnnp_model_to_parameter_indices_and_from_parameter_indices(model, parameters, from_model, from_parameters, &to_parameter_indices, &to_param_ref, &from_parameter_indices, &from_param_ref);
	// To models.
	ccv_cnnp_compiled_data_t* const to_compiled_data = model->compiled_data;
	assert(to_compiled_data);
	// From models.
	const ccv_cnnp_compiled_data_t* const from_compiled_data = from_model->compiled_data;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int to_parameter_size = to_compiled_data->parameters->rnum;
	const int rnum = (to_param_ref < 0 && from_param_ref < 0) ? from_parameter_indices->rnum : 1;
	assert(aux_in_size >= 0);
	assert(aux_out_size >= 0);
	int i, j;
	ccv_nnc_tensor_t* inputs[aux_in_size + 2];
	ccv_nnc_tensor_t* outputs[aux_out_size + 1];
	for (i = 0; i < aux_in_size; i++)
		inputs[i + 2] = aux_ins[i];
	for (i = 0; i < aux_out_size; i++)
		outputs[i + 1] = aux_outs[i];
	for (i = 0; i < rnum; i++)
	{
		const int src_d = *(int*)ccv_array_get(from_parameter_indices,from_param_ref >= 0 ? from_param_ref : i);
		assert(src_d >= 0);
		assert(src_d < from_compiled_data->parameters->rnum);
		const int s = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(from_compiled_data->parameters, src_d))->d;
		// If the original is not init'ed. We cannot copy from.
		if (!(from_compiled_data->tensors_init.v[s >> 5] & (1u << (s & 0x1f))))
			continue;
		const int dest_d = *(int*)ccv_array_get(to_parameter_indices, to_param_ref >= 0 ? to_param_ref : i);
		assert(dest_d >= 0);
		assert(dest_d < to_compiled_data->parameters->rnum);
		if (parallel_count > 1)
		{
			ccv_nnc_stream_context_t* streams[parallel_count];
			ccv_nnc_stream_signal_t* signal;
			if (stream_context)
				signal = ccv_nnc_stream_context_emit_signal_new(stream_context);
			for (j = 0; j < parallel_count; j++)
			{
				ccv_nnc_tensor_t* const src = from_compiled_data->tensors.parameters[src_d + j * to_parameter_size];
				ccv_nnc_tensor_t* const dest = to_compiled_data->tensors.parameters[dest_d + j * to_parameter_size];
				if (!dest || !src)
				{
					streams[j] = 0;
					continue;
				}
				// At the moment, can only handle them on the same device.
				assert(CCV_TENSOR_GET_MEMORY(src->info.type) == CCV_TENSOR_GET_MEMORY(dest->info.type));
				assert(CCV_TENSOR_GET_DEVICE_ID(src->info.type) == CCV_TENSOR_GET_DEVICE_ID(dest->info.type));
				const int stream_type = CCV_TENSOR_GET_MEMORY(src->info.type) == CCV_TENSOR_GPU_MEMORY ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
				const int device_id = CCV_TENSOR_GET_DEVICE_ID(src->info.type);
				int type = stream_type;
				CCV_STREAM_SET_DEVICE_ID(type, device_id);
				ccv_nnc_stream_context_t* const stream_0 = ccv_cnnp_compiled_data_get_stream(to_compiled_data, type);
				// Wait signal to finish.
				if (stream_context)
					ccv_nnc_stream_context_wait_signal(stream_0, signal);
				inputs[0] = outputs[0] = dest;
				inputs[1] = src;
				ccv_nnc_cmd_exec(cmd, hint, flags, inputs, aux_in_size + 2, outputs, aux_out_size + 1, stream_0);
				if (stream_context)
				{
					ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_emit_signal_new(stream_0);
					ccv_nnc_stream_context_wait_signal(stream_context, signal);
				}
				streams[j] = stream_0;
			}
			// If this should be blocking, blocking it.
			if (!stream_context)
				for (j = 0; j < parallel_count; j++)
					if (streams[j])
						ccv_nnc_stream_context_wait(streams[j]);
		} else {
			ccv_nnc_tensor_t* const src = from_compiled_data->tensors.parameters[src_d];
			assert(src);
			ccv_nnc_tensor_t* const dest = to_compiled_data->tensors.parameters[dest_d];
			assert(dest);
			inputs[0] = outputs[0] = dest;
			inputs[1] = src;
			ccv_nnc_cmd_exec(cmd, hint, flags, inputs, aux_in_size + 2, outputs, aux_out_size + 1, stream_context);
		}
		// Mark this symbol as init'ed.
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(to_compiled_data->parameters, dest_d))->d;
		to_compiled_data->tensors_init.v[d >> 5] |= (1u << (d & 0x1f));
	}
	ccv_array_free(to_parameter_indices);
	ccv_array_free(from_parameter_indices);
}

void ccv_cnnp_model_parameters_map(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const aux_ins, const int aux_in_size, ccv_nnc_tensor_t* const* const aux_outs, const int aux_out_size, ccv_nnc_stream_context_t* const stream_context)
{
	int to_param_ref;
	ccv_array_t* const to_parameter_indices = _ccv_cnnp_model_parameter_indices(model, parameters, &to_param_ref);
	// To models.
	ccv_cnnp_compiled_data_t* const to_compiled_data = model->compiled_data;
	assert(to_compiled_data);
	// Tensor has to be inited already.
	assert(!!to_compiled_data->tensors_init.v);
	assert(to_compiled_data->tensors.parameters);
	// From models.
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int to_parameter_size = to_compiled_data->parameters->rnum;
	const int rnum = (to_param_ref < 0) ? to_parameter_indices->rnum : 1;
	assert(aux_in_size >= 0);
	assert(aux_out_size >= 0);
	int i, j;
	ccv_nnc_tensor_t* inputs[aux_in_size + 1];
	ccv_nnc_tensor_t* outputs[aux_out_size + 1];
	for (i = 0; i < aux_in_size; i++)
		inputs[i + 1] = aux_ins[i];
	for (i = 0; i < aux_out_size; i++)
		outputs[i + 1] = aux_outs[i];
	for (i = 0; i < rnum; i++)
	{
		const int dest_d = *(int*)ccv_array_get(to_parameter_indices, to_param_ref >= 0 ? to_param_ref : i);
		assert(dest_d >= 0);
		assert(dest_d < to_compiled_data->parameters->rnum);
		if (parallel_count > 1)
		{
			ccv_nnc_stream_context_t* streams[parallel_count];
			ccv_nnc_stream_signal_t* signal;
			if (stream_context)
				signal = ccv_nnc_stream_context_emit_signal_new(stream_context);
			for (j = 0; j < parallel_count; j++)
			{
				ccv_nnc_tensor_t* const dest = to_compiled_data->tensors.parameters[dest_d + j * to_parameter_size];
				if (!dest)
				{
					streams[j] = 0;
					continue;
				}
				const int stream_type = CCV_TENSOR_GET_MEMORY(dest->info.type) == CCV_TENSOR_GPU_MEMORY ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
				const int device_id = CCV_TENSOR_GET_DEVICE_ID(dest->info.type);
				int type = stream_type;
				CCV_STREAM_SET_DEVICE_ID(type, device_id);
				ccv_nnc_stream_context_t* const stream_0 = ccv_cnnp_compiled_data_get_stream(to_compiled_data, type);
				// Wait signal to finish.
				if (stream_context)
					ccv_nnc_stream_context_wait_signal(stream_0, signal);
				inputs[0] = outputs[0] = dest;
				ccv_nnc_cmd_exec(cmd, hint, flags, inputs, aux_in_size + 1, outputs, aux_out_size + 1, stream_0);
				if (stream_context)
				{
					ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_emit_signal_new(stream_0);
					ccv_nnc_stream_context_wait_signal(stream_context, signal);
				}
				streams[j] = stream_0;
			}
			// If this should be blocking, blocking it.
			if (!stream_context)
				for (j = 0; j < parallel_count; j++)
					if (streams[j])
						ccv_nnc_stream_context_wait(streams[j]);
		} else {
			ccv_nnc_tensor_t* const dest = to_compiled_data->tensors.parameters[dest_d];
			assert(dest);
			inputs[0] = outputs[0] = dest;
			ccv_nnc_cmd_exec(cmd, hint, flags, inputs, aux_in_size + 1, outputs, aux_out_size + 1, stream_context);
		}
		// No need to mark this symbol as init'ed, it is already.
	}
	ccv_array_free(to_parameter_indices);
}

void ccv_cnnp_model_parameter_gradients_map(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const aux_ins, const int aux_in_size, ccv_nnc_tensor_t* const* const aux_outs, const int aux_out_size, ccv_nnc_stream_context_t* const stream_context)
{
	int to_param_ref;
	ccv_array_t* const to_parameter_indices = _ccv_cnnp_model_parameter_indices(model, parameters, &to_param_ref);
	// To models.
	ccv_cnnp_compiled_data_t* const to_compiled_data = model->compiled_data;
	assert(to_compiled_data);
	// Tensor has to be inited already.
	assert(!!to_compiled_data->tensors_init.v);
	ccv_nnc_tensor_t** tensor_gradients;
	if (to_compiled_data->backward.count > 1)
		tensor_gradients = to_compiled_data->tensors.accum_gradients;
	else
		tensor_gradients = to_compiled_data->tensors.gradients;
	assert(tensor_gradients);
	// From models.
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int to_parameter_size = to_compiled_data->parameters->rnum;
	const int rnum = (to_param_ref < 0) ? to_parameter_indices->rnum : 1;
	assert(aux_in_size >= 0);
	assert(aux_out_size >= 0);
	int i, j;
	ccv_nnc_tensor_t* inputs[aux_in_size + 1];
	ccv_nnc_tensor_t* outputs[aux_out_size + 1];
	for (i = 0; i < aux_in_size; i++)
		inputs[i + 1] = aux_ins[i];
	for (i = 0; i < aux_out_size; i++)
		outputs[i + 1] = aux_outs[i];
	for (i = 0; i < rnum; i++)
	{
		const int dest_d = *(int*)ccv_array_get(to_parameter_indices, to_param_ref >= 0 ? to_param_ref : i);
		assert(dest_d >= 0);
		assert(dest_d < to_compiled_data->parameters->rnum);
		if (parallel_count > 1)
		{
			ccv_nnc_stream_context_t* streams[parallel_count];
			ccv_nnc_stream_signal_t* signal;
			if (stream_context)
				signal = ccv_nnc_stream_context_emit_signal_new(stream_context);
			for (j = 0; j < parallel_count; j++)
			{
				ccv_nnc_tensor_t* const dest = tensor_gradients[dest_d + j * to_parameter_size];
				if (!dest)
				{
					streams[j] = 0;
					continue;
				}
				const int stream_type = CCV_TENSOR_GET_MEMORY(dest->info.type) == CCV_TENSOR_GPU_MEMORY ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
				const int device_id = CCV_TENSOR_GET_DEVICE_ID(dest->info.type);
				int type = stream_type;
				CCV_STREAM_SET_DEVICE_ID(type, device_id);
				ccv_nnc_stream_context_t* const stream_0 = ccv_cnnp_compiled_data_get_stream(to_compiled_data, type);
				// Wait signal to finish.
				if (stream_context)
					ccv_nnc_stream_context_wait_signal(stream_0, signal);
				inputs[0] = outputs[0] = dest;
				ccv_nnc_cmd_exec(cmd, hint, flags, inputs, aux_in_size + 1, outputs, aux_out_size + 1, stream_0);
				if (stream_context)
				{
					ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_emit_signal_new(stream_0);
					ccv_nnc_stream_context_wait_signal(stream_context, signal);
				}
				streams[j] = stream_0;
			}
			// If this should be blocking, blocking it.
			if (!stream_context)
				for (j = 0; j < parallel_count; j++)
					if (streams[j])
						ccv_nnc_stream_context_wait(streams[j]);
		} else {
			ccv_nnc_tensor_t* const dest = tensor_gradients[dest_d];
			assert(dest);
			inputs[0] = outputs[0] = dest;
			ccv_nnc_cmd_exec(cmd, hint, flags, inputs, aux_in_size + 1, outputs, aux_out_size + 1, stream_context);
		}
		// No need to mark this symbol as init'ed, it is already.
	}
	ccv_array_free(to_parameter_indices);
}

ccv_nnc_cmd_t ccv_cnnp_model_minimizer(ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	return compiled_data->minimize.minimizer;
}

void ccv_cnnp_model_set_minimizer(ccv_cnnp_model_t* const model, const ccv_nnc_cmd_t minimizer, const int reset, const ccv_cnnp_model_io_t* const set_parameters, const int set_parameter_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parameter_size = compiled_data->parameters->rnum;
	if (parameter_size == 0)
		return;
	if (reset)
		{ assert(set_parameters == 0 && set_parameter_size == 0); }
	const int old_max_saved_aux_size = compiled_data->minimize.max_saved_aux_size;
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(minimizer);
	if (saved_aux_size > compiled_data->minimize.max_saved_aux_size)
		compiled_data->minimize.max_saved_aux_size = saved_aux_size;
	const int max_saved_aux_size = compiled_data->minimize.max_saved_aux_size;
	// We update all parameters, at this point, we have one minimizer.
	if (set_parameters == 0 || set_parameter_size == 0)
		compiled_data->minimize.minimizer = minimizer;
	int i;
	if (set_parameters && set_parameter_size)
	{
		// I need to save what's the minimizer along with this.
		if (!compiled_data->minimize.parameters)
			compiled_data->minimize.parameters = ccv_array_new(sizeof(ccv_cnnp_set_minimizer_for_parameter_t*), 1, 0);
		ccv_cnnp_set_minimizer_for_parameter_t* const set_minimizer_for_parameter = ccmalloc(sizeof(ccv_cnnp_set_minimizer_for_parameter_t) + (set_parameter_size - 1) * sizeof(ccv_cnnp_model_io_t));
		set_minimizer_for_parameter->minimizer = minimizer;
		set_minimizer_for_parameter->parameter_size = set_parameter_size;
		memcpy(set_minimizer_for_parameter->parameters, set_parameters, sizeof(ccv_cnnp_model_io_t) * set_parameter_size);
		ccv_array_push(compiled_data->minimize.parameters, &set_minimizer_for_parameter);
	}
	// If reset is true, clear the parameters array.
	if (reset && compiled_data->minimize.parameters)
	{
		for (i = 0; i < compiled_data->minimize.parameters->rnum; i++)
			ccfree(*(ccv_cnnp_set_minimizer_for_parameter_t**)ccv_array_get(compiled_data->minimize.parameters, i));
		ccv_array_clear(compiled_data->minimize.parameters);
	}
	if (!compiled_data->update_nodes)
		return;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = model->graph;
	assert(symbolic_graph);
	if (saved_aux_size > old_max_saved_aux_size)
	{
		assert(compiled_data->updated_parameters);
		// Reallocate first, move them around later.
		compiled_data->updated_parameters = (ccv_nnc_tensor_symbol_t*)ccrealloc(compiled_data->updated_parameters, sizeof(ccv_nnc_tensor_symbol_t) * parameter_size + sizeof(ccv_nnc_graph_exec_symbol_t) * parameter_size + sizeof(ccv_nnc_tensor_symbol_map_t) * saved_aux_size * parameter_size);
		compiled_data->update_nodes = (ccv_nnc_graph_exec_symbol_t*)(compiled_data->updated_parameters + parameter_size);
		compiled_data->saved_aux = (ccv_nnc_tensor_symbol_map_t*)(compiled_data->update_nodes + parameter_size);
		// We need to do this from back to front because saved_aux_size > old_saved_aux_size, it could overlap.
		_ccv_cnnp_scatter_saved_aux(compiled_data->saved_aux, parameter_size, old_max_saved_aux_size, saved_aux_size);
	}
	int flag = 0;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	if (set_parameters && set_parameter_size)
	{
		ccv_array_t* const parameter_indices = ccv_array_new(sizeof(int), 0, 0);
		for (i = 0; i < set_parameter_size; i++)
		{
			const int param_sel = set_parameters[i]->param_sel > 0 ? set_parameters[i]->param_sel - 1 : set_parameters[i]->param_sel;
			assert(set_parameters[i]->param_sel != 0);
			const int old_rnum = parameter_indices->rnum;
			ccv_cnnp_model_add_to_parameter_indices(set_parameters[i]->model, param_sel, parameter_indices);
			const int param_ref = set_parameters[i]->param_ref > 0 ? set_parameters[i]->param_ref - 1 : set_parameters[i]->param_ref;
			assert(set_parameters[i]->param_ref != 0);
			if (param_ref >= 0)
			{
				assert(param_ref + old_rnum < parameter_indices->rnum);
				*(int*)ccv_array_get(parameter_indices, old_rnum) = *(int*)ccv_array_get(parameter_indices, param_ref + old_rnum);
				parameter_indices->rnum = old_rnum + 1;
			}
		}
		// We may have duplicated indices, but that is OK, we will set it twice.
		for (i = 0; i < parameter_indices->rnum; i++)
		{
			const int d = *(int*)ccv_array_get(parameter_indices, i);
			if (_ccv_cnnp_set_minimizer_for_parameter(symbolic_graph, compiled_data, compiled_data->update_nodes, compiled_data->updated_parameters, compiled_data->saved_aux, parallel_count, minimizer, saved_aux_size, max_saved_aux_size, d))
				flag = 1;
		}
		ccv_array_free(parameter_indices);
	} else {
		for (i = 0; i < parameter_size; i++)
			if (_ccv_cnnp_set_minimizer_for_parameter(symbolic_graph, compiled_data, compiled_data->update_nodes, compiled_data->updated_parameters, compiled_data->saved_aux, parallel_count, minimizer, saved_aux_size, max_saved_aux_size, i))
				flag = 1;
		if (compiled_data->minimize.parameters)
			if (_ccv_cnnp_apply_parameters_with_minimizer(model))
				flag = 1;
	}
	if (flag)
	{
		// If saved_aux_size doesn't match, we need to remove / add new saved_aux to the graph. But first, free up apply gradients graph.
		if (compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_FIT_MODE)
			_ccv_cnnp_compiled_data_graph_free(compiled_data);
		_ccv_cnnp_compiled_data_apply_gradients_free(compiled_data);
	}
}

void ccv_cnnp_model_set_compile_params(ccv_cnnp_model_t* const model, const ccv_nnc_symbolic_graph_compile_param_t compile_params)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	compiled_data->compile_params = compile_params;
}

void ccv_cnnp_model_dot(const ccv_cnnp_model_t* const model, const int flags, FILE** const outs, const int out_size)
{
	if (model->graph && out_size > 0)
		ccv_nnc_symbolic_graph_dot(model->graph, flags, outs[0]);
	if (model->compiled_data && model->compiled_data->graph && out_size > 1)
		ccv_nnc_graph_dot(model->compiled_data->graph, flags, outs[1]);
	if (model->compiled_data && model->compiled_data->backward.accum && out_size > 2)
		ccv_nnc_graph_dot(model->compiled_data->backward.accum, flags, outs[2]);
	if (model->compiled_data && model->compiled_data->apply_gradients.graph && out_size > 3)
		ccv_nnc_graph_dot(model->compiled_data->apply_gradients.graph, flags, outs[3]);
}

void ccv_cnnp_model_format(const ccv_cnnp_model_t* const model, const ccv_nnc_symbolic_graph_format_f format_fn, void* const context)
{
	if (model->graph)
		ccv_nnc_symbolic_graph_format(model->graph, 0, 0, 0, 0, format_fn, context);
}

static void _ccv_cnnp_compiled_data_free(const ccv_cnnp_model_t* const model, ccv_cnnp_compiled_data_t* const compiled_data)
{
	int i;
	const int parameter_size = compiled_data->parameters->rnum;
	ccv_array_free(compiled_data->parameters);
	if (compiled_data->parameter_flags)
		ccfree(compiled_data->parameter_flags);
	const int internal_size = compiled_data->internals->rnum;
	ccv_array_free(compiled_data->internals);
	assert(compiled_data->ids.parameters->rnum == parameter_size);
	assert(compiled_data->ids.internals->rnum == internal_size);
	for (i = 0; i < parameter_size; i++)
		ccfree(*(char**)ccv_array_get(compiled_data->ids.parameters, i));
	ccv_array_free(compiled_data->ids.parameters);
	for (i = 0; i < internal_size; i++)
		ccfree(*(char**)ccv_array_get(compiled_data->ids.internals, i));
	ccv_array_free(compiled_data->ids.internals);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	if (compiled_data->tensors.parameters)
	{
		for (i = 0; i < parameter_size * parallel_count; i++)
			ccv_nnc_tensor_free(compiled_data->tensors.parameters[i]);
		for (i = 0; i < internal_size * parallel_count; i++)
			if (compiled_data->tensors.internals[i])
				ccv_nnc_tensor_free(compiled_data->tensors.internals[i]);
		ccfree(compiled_data->tensors.parameters);
	}
	if (compiled_data->tensors.gradients)
	{
		for (i = 0; i < parameter_size * parallel_count; i++)
		{
			ccv_nnc_tensor_free(compiled_data->tensors.gradients[i]);
			if (compiled_data->tensors.accum_gradients[i])
				ccv_nnc_tensor_free(compiled_data->tensors.accum_gradients[i]);
		}
		ccfree(compiled_data->tensors.gradients);
	}
	if (compiled_data->minimize.parameters)
	{
		for (i = 0; i < compiled_data->minimize.parameters->rnum; i++)
			ccfree(*(ccv_cnnp_set_minimizer_for_parameter_t**)ccv_array_get(compiled_data->minimize.parameters, i));
		ccv_array_free(compiled_data->minimize.parameters);
	}
	if (compiled_data->rewindables)
		ccv_array_free(compiled_data->rewindables);
	if (compiled_data->tensors_init.v)
		ccfree(compiled_data->tensors_init.v);
	if (compiled_data->evaluate.tos)
		ccfree(compiled_data->evaluate.tos);
	compiled_data->evaluate.tos = 0;
	if (compiled_data->stream_map)
	{
		khiter_t k;
		for (k = kh_begin(compiled_data->stream_map); k != kh_end(compiled_data->stream_map); ++k)
		{
			if (!kh_exist(compiled_data->stream_map, k))
				continue;
			ccv_nnc_stream_context_t* const stream = kh_val(compiled_data->stream_map, k);
			ccv_nnc_stream_context_free(stream);
		}
		kh_destroy(stream_map, compiled_data->stream_map);
	}
	_ccv_cnnp_compiled_data_graph_free(compiled_data);
	_ccv_cnnp_compiled_data_gradient_free(compiled_data);
	_ccv_cnnp_compiled_data_backward_free(compiled_data);
	_ccv_cnnp_compiled_data_apply_gradients_free(compiled_data);
	ccv_nnc_xpu_alloc_destroy(&compiled_data->xpu_alloc);
	ccfree(compiled_data);
}

void ccv_cnnp_model_free(ccv_cnnp_model_t* const model)
{
	if (model->isa->deinit)
		model->isa->deinit(model);
	if (model->io)
	{
		int i;
		for (i = 0; i < model->io->rnum; i++)
		{
			ccv_cnnp_model_io_t model_io = *(ccv_cnnp_model_io_t*)ccv_array_get(model->io, i);
			if (model_io->outgoings)
				ccv_array_free(model_io->outgoings);
			if (model_io->incomings)
				ccv_array_free(model_io->incomings);
			if (model_io->dependencies)
				ccv_array_free(model_io->dependencies);
			ccfree(model_io);
		}
		ccv_array_free(model->io);
	}
	if (model->parameter_indices)
		ccv_array_free(model->parameter_indices);
	if (model->inputs)
		ccfree(model->inputs);
	if (model->graph)
		ccv_nnc_symbolic_graph_free(model->graph);
	if (model->compiled_data)
		_ccv_cnnp_compiled_data_free(model, model->compiled_data);
	if (model->name)
		ccfree(model->name);
	if (model->file_backed_dir)
		ccfree(model->file_backed_dir);
	ccfree(model);
}
