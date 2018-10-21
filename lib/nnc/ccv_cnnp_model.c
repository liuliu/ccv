#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"

static const ccv_cnnp_model_vtab_t ccv_cnnp_input_isa;

struct ccv_cnnp_model_io_s {
	uint8_t tbits; // Temporary bits stored in the ccv_cnnp_model_io_t object, whoever uses it should clean it up.
	ccv_cnnp_model_t* model; // Reference back to the model who holds it. This is required because the model is the one whole holds the io.
	ccv_array_t* incomings; // Array of ccv_cnnp_model_io_t. The order is important because it impacts the order of symbols.
	ccv_array_t* outgoings; // Array of ccv_cnnp_model_io_t.
};

typedef struct {
	ccv_cnnp_model_t super;
	int sequence_size;
	ccv_cnnp_model_t* sequence[1];
} ccv_cnnp_sequential_model_t;

static void _ccv_cnnp_sequential_model_deinit(ccv_cnnp_model_t* const super)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_free(self->sequence[i]);
}

static void _ccv_cnnp_sequential_model_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	ccv_nnc_tensor_symbol_t input = inputs[0];
	assert(input_size == 1);
	for (i = 0; i < self->sequence_size; i++)
	{
		ccv_nnc_tensor_symbol_t output;
		ccv_cnnp_model_t* const sub_model = self->sequence[i];
		// Go through each sub model to build the graph.
		ccv_cnnp_model_build(sub_model, graph, &input, 1, &output, 1);
		input = output;
	}
	outputs[0] = input;
}

static void _ccv_cnnp_sequential_model_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_init_states(self->sequence[i], graph, initializer, context);
}

static void _ccv_cnnp_sequential_model_add_to_trainable(ccv_cnnp_model_t* const super, ccv_array_t* const trainables)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_trainable(self->sequence[i], trainables);
}

static void _ccv_cnnp_sequential_model_add_to_output(ccv_cnnp_model_t* const super, ccv_array_t* const outputs)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_output(self->sequence[i], outputs);
}

static void _ccv_cnnp_sequential_model_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_set_is_test(self->sequence[i], is_test, updater, context);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_sequential_model_isa = {
	.deinit = _ccv_cnnp_sequential_model_deinit,
	.build = _ccv_cnnp_sequential_model_build,
	.init_states = _ccv_cnnp_sequential_model_init_states,
	.add_to_trainable = _ccv_cnnp_sequential_model_add_to_trainable,
	.add_to_output = _ccv_cnnp_sequential_model_add_to_output,
	.set_is_test = _ccv_cnnp_sequential_model_set_is_test,
};

ccv_cnnp_model_t* ccv_cnnp_sequential_new(ccv_cnnp_model_t* const* const models, const int model_size)
{
	assert(model_size > 0);
	ccv_cnnp_sequential_model_t* const sequential_model = (ccv_cnnp_sequential_model_t*)cccalloc(1, sizeof(ccv_cnnp_sequential_model_t) + sizeof(ccv_cnnp_model_t*) * (model_size - 1) + sizeof(ccv_nnc_tensor_symbol_t));
	sequential_model->super.isa = &ccv_cnnp_sequential_model_isa;
	sequential_model->super.input_size = 1;
	sequential_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(sequential_model->sequence + model_size);
	sequential_model->super.output_size = 1;
	sequential_model->sequence_size = model_size;
	memcpy(sequential_model->sequence, models, sizeof(ccv_cnnp_model_t*) * model_size);
	return (ccv_cnnp_model_t*)sequential_model;
}

typedef struct {
	ccv_cnnp_model_t super;
	// The name is similar to sequential model, but it is just topological sorted models.
	int sequence_size;
	ccv_cnnp_model_io_t sequence[1];
} ccv_cnnp_functional_model_t;

static void _ccv_cnnp_functional_model_deinit(ccv_cnnp_model_t* const super)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_free(self->sequence[i]->model);
}

static void _ccv_cnnp_functional_model_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	assert(self->super.input_size == input_size);
	assert(self->super.output_size == output_size);
	int i, j, k;
	for (i = 0; i < self->super.input_size; i++)
		self->sequence[i]->model->outputs[0] = inputs[i]; // Assigning the output symbol of input layer to be the input symbol.
	ccv_array_t* input_symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 1, 0);
	for (i = self->super.input_size; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const sub_model = self->sequence[i]->model;
		assert(sub_model->io);
		ccv_array_clear(input_symbols);
		const ccv_array_t* const incomings = sub_model->io->incomings;
		for (j = 0; j < incomings->rnum; j++)
		{
			const ccv_cnnp_model_io_t input = *(ccv_cnnp_model_io_t*)ccv_array_get(incomings, j);
			for (k = 0; k < input->model->output_size; k++)
				ccv_array_push(input_symbols, &input->model->outputs[k]);
		}
		// Go through each sub model to build the graph.
		ccv_cnnp_model_build(sub_model, graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(input_symbols, 0), input_symbols->rnum, 0, 0);
	}
	ccv_array_free(input_symbols);
	for (i = output_size, k = self->sequence_size - 1; k >= 0; k--)
	{
		ccv_cnnp_model_t* const sub_model = self->sequence[k]->model;
		assert(sub_model->io);
		i -= sub_model->output_size;
		if (i < 0)
			break;
		for (j = 0; j < sub_model->output_size; j++)
			outputs[i + j] = sub_model->outputs[j];
	}
}

static void _ccv_cnnp_functional_model_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_init_states(self->sequence[i]->model, graph, initializer, context);
}

static void _ccv_cnnp_functional_model_add_to_trainable(ccv_cnnp_model_t* const super, ccv_array_t* const trainables)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_trainable(self->sequence[i]->model, trainables);
}

static void _ccv_cnnp_functional_model_add_to_output(ccv_cnnp_model_t* const super, ccv_array_t* const outputs)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_output(self->sequence[i]->model, outputs);
}

static void _ccv_cnnp_functional_model_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_set_is_test(self->sequence[i]->model, is_test, updater, context);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_functional_model_isa = {
	.deinit = _ccv_cnnp_functional_model_deinit,
	.build = _ccv_cnnp_functional_model_build,
	.init_states = _ccv_cnnp_functional_model_init_states,
	.add_to_trainable = _ccv_cnnp_functional_model_add_to_trainable,
	.add_to_output = _ccv_cnnp_functional_model_add_to_output,
	.set_is_test = _ccv_cnnp_functional_model_set_is_test,
};

#define CCV_CNNP_IS_MODEL_INPUT(x) ((x)->isa == &ccv_cnnp_input_isa)

ccv_cnnp_model_t* ccv_cnnp_model_new(const ccv_cnnp_model_io_t* const inputs, const int input_size, const ccv_cnnp_model_io_t* const outputs, const int output_size)
{
	assert(output_size > 0);
	// Do topological sort.
	ccv_array_t* const reverse_top = ccv_array_new(sizeof(ccv_cnnp_model_io_t), output_size, 0);
	ccv_array_resize(reverse_top, output_size);
	memcpy(ccv_array_get(reverse_top, 0), outputs, sizeof(ccv_cnnp_model_io_t) * output_size);
	// Go from the output, until we meet inputs.
	int i, j, input_count = 0;
	int tensor_output_size = 0; // io can be mapped to multiple tensor outputs, therefore, need to compute the exact tensor output size.
	for (i = 0; i < output_size; i++)
		tensor_output_size += outputs[i]->model->output_size;
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_cnnp_model_io_t output = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, i);
		assert(!CCV_CNNP_IS_MODEL_INPUT(output->model));
		// If it is input, push it here.
		if (output->incomings)
			for (j = 0; j < output->incomings->rnum; j++)
			{
				const ccv_cnnp_model_io_t input = *(ccv_cnnp_model_io_t*)ccv_array_get(output->incomings, j);
				if (input->tbits) // visited.
					continue;
				input->tbits = 1; // Mark it as visited.
				if (!CCV_CNNP_IS_MODEL_INPUT(input->model))
					ccv_array_push(reverse_top, &input);
				else
					++input_count;
			}
	}
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_cnnp_model_io_t output = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, i);
		output->tbits = 0; // Clean the tbits back.
	}
	for (i = 0; i < input_size; i++)
		inputs[i]->tbits = 0; // Clean the tbits back.
	assert(input_count == input_size); // Assuming they all match.
	const int sequence_size = reverse_top->rnum + input_size;
	ccv_cnnp_functional_model_t* const functional_model = (ccv_cnnp_functional_model_t*)cccalloc(1, sizeof(ccv_cnnp_functional_model_t) + sizeof(ccv_cnnp_model_t*) * (sequence_size - 1) + sizeof(ccv_nnc_tensor_symbol_t) * tensor_output_size);
	functional_model->super.isa = &ccv_cnnp_functional_model_isa;
	functional_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(functional_model->sequence + sequence_size);
	functional_model->super.output_size = tensor_output_size;
	functional_model->super.input_size = input_size;
	functional_model->sequence_size = sequence_size;
	memcpy(functional_model->sequence, inputs, sizeof(ccv_cnnp_model_io_t) * input_size);
	for (i = 0; i < reverse_top->rnum; i++)
		functional_model->sequence[input_size + i] = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, reverse_top->rnum - 1 - i);
	ccv_array_free(reverse_top);
	return (ccv_cnnp_model_t*)functional_model;
}

ccv_cnnp_model_io_t ccv_cnnp_model_apply(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t* const inputs, const int input_size)
{
	assert(input_size > 0);
	if (!model->io)
	{
		model->io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
		model->io->tbits = 0;
		model->io->model = model;
		model->io->incomings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
		model->io->outgoings = 0;
	}
	if (model->io->outgoings)
		ccv_array_clear(model->io->outgoings); // New outputs.
	int i;
	ccv_array_resize(model->io->incomings, input_size);
	memcpy(ccv_array_get(model->io->incomings, 0), inputs, sizeof(ccv_cnnp_model_io_t) * input_size);
	for (i = 0; i < input_size; i++)
	{
		if (!inputs[i]->outgoings)
			inputs[i]->outgoings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
		ccv_array_push(inputs[i]->outgoings, &model->io);
	}
	return model->io;
}

static void _ccv_nnc_array_dedup_tensor_symbols(ccv_array_t* const tensor_symbols)
{
	int i, j;
	for (i = 0; i < tensor_symbols->rnum; i++)
	{
		ccv_nnc_tensor_symbol_t* const tensor_symbol = (ccv_nnc_tensor_symbol_t*)ccv_array_get(tensor_symbols, i);
		// Check whether this tensor symbol has any duplicate.
		for (j = i + 1; j < tensor_symbols->rnum;)
		{
			ccv_nnc_tensor_symbol_t* const other_symbol = (ccv_nnc_tensor_symbol_t*)ccv_array_get(tensor_symbols, j);
			// If there is a same tensor symbol, remove it.
			if (other_symbol->d == tensor_symbol->d && other_symbol->graph == tensor_symbol->graph)
			{
				if (j + 1 < tensor_symbols->rnum)
					*other_symbol = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(tensor_symbols, tensor_symbols->rnum - 1);
				--tensor_symbols->rnum;
				continue;
			}
			++j;
		}
	}
}

void ccv_cnnp_model_compile(ccv_cnnp_model_t* const model, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_cmd_t minimizer, const ccv_nnc_cmd_t loss)
{
	assert(input_size == model->input_size);
	if (!model->graph) // The graph is not compiled yet.
	{
		model->graph = ccv_nnc_symbolic_graph_new();
		model->inputs = ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * input_size);
		int i;
		for (i = 0; i < input_size; i++)
			model->inputs[i] = ccv_nnc_tensor_symbol_new(model->graph, inputs[i], 0);
		ccv_cnnp_model_build(model, model->graph, model->inputs, input_size, 0, 0);
		ccv_array_t* const trainables = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
		ccv_cnnp_model_add_to_trainable(model, trainables);
		_ccv_nnc_array_dedup_tensor_symbols(trainables);
		// Assert no trainable is alias.
		for (i = 0; i < trainables->rnum; i++)
		{
			const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(trainables, i);
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(trainable.graph, trainable);
			assert(alias_to.graph == 0); // Cannot find the one alias to.
		}
		ccv_array_t* const retains = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
		ccv_cnnp_model_add_to_output(model, retains);
		_ccv_nnc_array_dedup_tensor_symbols(retains);
		// Assert no trainable is alias.
		for (i = 0; i < retains->rnum; i++)
		{
			const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(retains, i);
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(retained.graph, retained);
			assert(alias_to.graph == 0); // Cannot find the one alias to.
		}
		ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_symbolic_graph_simplify(model->graph,
			SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
				CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
				CCV_NNC_SIMPLIFY_OPS_FUSION,
				CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
			model->outputs, model->output_size,
			SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
		int flag = 0;
		for (i = 0; !flag && i < input_size; i++)
			flag = (CCV_TENSOR_GET_MEMORY(inputs[i].type) == CCV_TENSOR_GPU_MEMORY);
		model->compiled_data = cccalloc(1, sizeof(ccv_cnnp_compiled_data_t) + sizeof(ccv_nnc_tensor_symbol_t) * (model->output_size - 1));
		// If inputs are from GPU, stream type is GPU.
		model->compiled_data->stream_type = flag ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
		model->compiled_data->trainables = trainables;
		model->compiled_data->retains = retains;
		model->compiled_data->minimizer = minimizer;
		model->compiled_data->loss = loss;
	}
}

static void _ccv_cnnp_init_states_for_tensors(void* const context, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_symbol_t symbol)
{
	ccv_nnc_tensor_arena_t* const tensor_arena = (ccv_nnc_tensor_arena_t*)context;
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_symbol(tensor_arena, symbol);
	if (!tensor)
		return;
	ccv_nnc_cmd_exec(cmd, hint, flags, 0, 0, &tensor, 1, 0);
}

static void _ccv_cnnp_cmd_update_for_execs(void* const context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint)
{
	ccv_nnc_graph_exec_arena_t* const graph_exec_arena = (ccv_nnc_graph_exec_arena_t*)context;
	ccv_nnc_graph_exec_t graph_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, symbol);
	ccv_nnc_graph_exec_set(graph_exec.graph, graph_exec, cmd);
	ccv_nnc_graph_exec_set_hint(graph_exec.graph, graph_exec, hint);
}

static void _ccv_cnnp_model_gradient_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const fits, const int fit_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(!compiled_data->gradient_init);
	const int dest_to_eval_size = compiled_data->dest_to_eval_size = ccv_nnc_symbolic_graph_destination_size(model->graph);
	assert(dest_to_eval_size > 0);
	compiled_data->dest_to_evals = ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * dest_to_eval_size + sizeof(ccv_nnc_graph_exec_t) * dest_to_eval_size);
	compiled_data->dest_to_eval_execs = (ccv_nnc_graph_exec_t*)(compiled_data->dest_to_evals + dest_to_eval_size);
	memcpy(compiled_data->dest_to_evals, ccv_nnc_symbolic_graph_destinations(model->graph), sizeof(ccv_nnc_graph_exec_symbol_t) * dest_to_eval_size);
	int i;
	const int output_size = model->output_size;
	assert(fit_size == output_size);
	ccv_nnc_tensor_symbol_t f[output_size];
	ccv_nnc_graph_exec_symbol_t loss_func[output_size];
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t fit = compiled_data->fits[i] = ccv_nnc_tensor_symbol_new(model->graph, fits[i]->info, 0);
		f[i] = ccv_nnc_tensor_symbol_new(model->graph, ccv_nnc_tensor_auto, 0);
		loss_func[i] = ccv_nnc_graph_exec_symbol_new(model->graph, compiled_data->loss, TENSOR_SYMBOL_LIST(model->outputs[i], fit), TENSOR_SYMBOL_LIST(f[i]), 0);
	}
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const int trainable_size = compiled_data->trainables->rnum;
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimizer);
	compiled_data->saved_aux = (ccv_nnc_tensor_symbol_map_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_map_t) * saved_aux_size * trainable_size + sizeof(ccv_nnc_tensor_symbol_t) * trainable_size + sizeof(ccv_nnc_graph_exec_symbol_t) * trainable_size);
	compiled_data->updated_trainables = (ccv_nnc_tensor_symbol_t*)(compiled_data->saved_aux + saved_aux_size * trainable_size);
	compiled_data->update_execs = (ccv_nnc_graph_exec_symbol_t*)(compiled_data->updated_trainables + trainable_size);
	ccv_nnc_symbolic_graph_minimize(model->graph, compiled_data->minimizer, f, output_size, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), trainable_size, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), compiled_data->updated_trainables, compiled_data->saved_aux, compiled_data->update_execs);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t df = ccv_nnc_tensor_symbol_for_backward(model->graph, f[i]);
		const ccv_nnc_graph_exec_symbol_t set = ccv_nnc_graph_exec_symbol_new(model->graph, CMD_SET_FORWARD(1), 0, 0, TENSOR_SYMBOL_LIST(df), 0);
		ccv_nnc_graph_exec_symbol_concat(model->graph, loss_func[i], set);
		// Relies on autogen to find the output execs.
	}
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS);
	ccv_nnc_symbolic_graph_set_destinations(model->graph, compiled_data->update_execs, trainable_size);
	compiled_data->gradient_init = 1;
}

static void _ccv_cnnp_model_tensors_init(ccv_cnnp_compiled_data_t* const compiled_data)
{
	assert(!compiled_data->trainable_tensors);
	const int trainable_size = compiled_data->trainables->rnum;
	const int retain_size = compiled_data->retains->rnum;
	compiled_data->trainable_tensors = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * trainable_size + sizeof(ccv_nnc_tensor_t*) * retain_size);
	compiled_data->retain_tensors = compiled_data->trainable_tensors + trainable_size;
	int i;
	for (i = 0; i < trainable_size; i++)
	{
		const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i);
		compiled_data->trainable_tensors[i] = ccv_nnc_tensor_new(0, ccv_nnc_tensor_symbol_params(trainable.graph, trainable), 0);
	}
	for (i = 0; i < retain_size; i++)
	{
		const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, i);
		compiled_data->retain_tensors[i] = ccv_nnc_tensor_new(0, ccv_nnc_tensor_symbol_params(retained.graph, retained), 0);
	}
}

static void _ccv_cnnp_model_fit_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const fits, const int fit_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(!compiled_data->graph || compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_EVALUATE_MODE);
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_FIT_MODE;
	assert(output_size == model->output_size);
	assert(output_size == fit_size);
	assert(output_size > 0);
	if (!compiled_data->gradient_init)
		_ccv_cnnp_model_gradient_jit(model, fits, fit_size);
	const int tensors_init = !!compiled_data->trainable_tensors;
	if (!compiled_data->trainable_tensors)
		_ccv_cnnp_model_tensors_init(compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_bind_t bind = {
			.symbol = model->inputs[i],
			.tensor = inputs[i]
		};
		ccv_array_push(tensor_binds, &bind);
	}
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_bind_t bind = {
			.symbol = model->outputs[i],
			.tensor = outputs[i]
		};
		ccv_array_push(tensor_binds, &bind);
	}
	for (i = 0; i < fit_size; i++)
	{
		const ccv_nnc_tensor_bind_t bind = {
			.symbol = compiled_data->fits[i],
			.tensor = fits[i]
		};
		ccv_array_push(tensor_binds, &bind);
	}
	const int trainable_size = compiled_data->trainables->rnum;
	const int retain_size = compiled_data->retains->rnum;
	for (i = 0; i < trainable_size; i++)
	{
		const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i);
		ccv_nnc_tensor_t* const tensor = compiled_data->trainable_tensors[i];
		const ccv_nnc_tensor_bind_t trainable_bind = {
			.symbol = trainable,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &trainable_bind);
		const ccv_nnc_tensor_bind_t updated_trainable_bind = {
			.symbol = compiled_data->updated_trainables[i],
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &updated_trainable_bind);
	}
	for (i = 0; i < retain_size; i++)
	{
		const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, i);
		ccv_nnc_tensor_t* const tensor = compiled_data->retain_tensors[i];
		const ccv_nnc_tensor_bind_t retained_bind = {
			.symbol = retained,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &retained_bind);
	}
	ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	// If tensor is not init'ed, we need to init states first.
	if (!tensors_init)
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, compiled_data->tensor_arena);
	compiled_data->is_test = 0;
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimizer);
	// No need to set because it is default to training mode.
	// ccv_cnnp_model_set_is_test(model, 0, _ccv_cnnp_cmd_update_for_execs, compiled_data->graph_exec_arena);
	for (i = 0; i < saved_aux_size * trainable_size; i++)
	{
		ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, compiled_data->saved_aux[i].source);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &tensor, 1, 0);
	}
	const int dest_to_eval_size = compiled_data->dest_to_eval_size;
	compiled_data->dest_to_eval_exec_size = 0;
	for (i = 0; i < dest_to_eval_size; i++)
	{
		ccv_nnc_graph_exec_t const dest_to_eval = ccv_nnc_graph_exec_from_symbol(compiled_data->graph_exec_arena, compiled_data->dest_to_evals[i]);
		if (dest_to_eval.graph)
			compiled_data->dest_to_eval_execs[compiled_data->dest_to_eval_exec_size++] = dest_to_eval;
	}
	ccv_nnc_graph_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_array_free(tensor_binds);
}

void ccv_cnnp_model_fit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const fits, const int fit_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == model->output_size);
	assert(fit_size == output_size);
	assert(model->graph);
	int i;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	if (!compiled_data->graph || compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_FIT_MODE)
	{
		if (compiled_data->graph)
			ccv_nnc_graph_free(compiled_data->graph);
		if (compiled_data->tensor_arena)
			ccv_nnc_tensor_arena_free(compiled_data->tensor_arena);
		if (compiled_data->graph_exec_arena)
			ccv_nnc_graph_exec_arena_free(compiled_data->graph_exec_arena);
		// Compile the symbolic graph down only when needed.
		_ccv_cnnp_model_fit_jit(model, inputs, input_size, fits, fit_size, outputs, output_size);
	} else {
		for (i = 0; i < input_size; i++)
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->inputs[i], inputs[i]);
		for (i = 0; i < output_size; i++)
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->outputs[i], outputs[i]);
		for (i = 0; i < fit_size; i++)
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, compiled_data->fits[i], fits[i]);
	}
	if (compiled_data->is_test)
	{
		compiled_data->is_test = 0;
		ccv_cnnp_model_set_is_test(model, 0, _ccv_cnnp_cmd_update_for_execs, compiled_data->graph_exec_arena);
	}
	ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, TRAVERSE_FULL);
}

static void _ccv_cnnp_model_evaluate_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_EVALUATE_MODE;
	assert(output_size == model->output_size);
	assert(output_size > 0);
	const int tensors_init = !!compiled_data->trainable_tensors;
	if (!compiled_data->trainable_tensors)
		_ccv_cnnp_model_tensors_init(compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_bind_t bind = {
			.symbol = model->inputs[i],
			.tensor = inputs[i]
		};
		ccv_array_push(tensor_binds, &bind);
	}
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_bind_t bind = {
			.symbol = model->outputs[i],
			.tensor = outputs[i]
		};
		ccv_array_push(tensor_binds, &bind);
	}
	// Rebind trainable.
	const int trainable_size = compiled_data->trainables->rnum;
	for (i = 0; i < trainable_size; i++)
	{
		const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i);
		ccv_nnc_tensor_t* const tensor = compiled_data->trainable_tensors[i];
		const ccv_nnc_tensor_bind_t trainable_bind = {
			.symbol = trainable,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &trainable_bind);
	}
	const int retain_size = compiled_data->retains->rnum;
	for (i = 0; i < retain_size; i++)
	{
		const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, i);
		ccv_nnc_tensor_t* const tensor = compiled_data->retain_tensors[i];
		const ccv_nnc_tensor_bind_t retained_bind = {
			.symbol = retained,
			.tensor = tensor
		};
		ccv_array_push(tensor_binds, &retained_bind);
	}
	// If we generated gradient for the graph, only compile part of the graph because the rest is irrelevant for evaluation.
	if (compiled_data->gradient_init)
		ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), compiled_data->dest_to_evals, compiled_data->dest_to_eval_size, &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	else
		ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	// If tensor is not init'ed, we need to init states first.
	if (!tensors_init)
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, compiled_data->tensor_arena);
	compiled_data->is_test = 1;
	ccv_cnnp_model_set_is_test(model, 1, _ccv_cnnp_cmd_update_for_execs, compiled_data->graph_exec_arena);
	ccv_nnc_graph_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_array_free(tensor_binds);
}

void ccv_cnnp_model_evaluate(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == model->output_size);
	assert(model->graph);
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	int i;
	if (!compiled_data->graph ||
		// If a stream context is provided, we need to recompile because we cannot run them efficiently in FIT_MODE.
		(stream_context && compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_EVALUATE_MODE))
	{
		if (compiled_data->graph)
			ccv_nnc_graph_free(compiled_data->graph);
		if (compiled_data->tensor_arena)
			ccv_nnc_tensor_arena_free(compiled_data->tensor_arena);
		if (compiled_data->graph_exec_arena)
			ccv_nnc_graph_exec_arena_free(compiled_data->graph_exec_arena);
		// Compile the symbolic graph down only when needed.
		_ccv_cnnp_model_evaluate_jit(model, inputs, input_size, outputs, output_size);
	} else {
		for (i = 0; i < input_size; i++)
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->inputs[i], inputs[i]);
		for (i = 0; i < output_size; i++)
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->outputs[i], outputs[i]);
	}
	if (!compiled_data->is_test)
	{
		compiled_data->is_test = 1;
		ccv_cnnp_model_set_is_test(model, 1, _ccv_cnnp_cmd_update_for_execs, compiled_data->graph_exec_arena);
	}
	if (compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_EVALUATE_MODE)
		ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, TRAVERSE_FULL);
	else
		ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, 0, 0,
			compiled_data->dest_to_eval_execs, compiled_data->dest_to_eval_exec_size);
}

void ccv_cnnp_model_set_minimizer(ccv_cnnp_model_t* const model, const ccv_nnc_cmd_t minimizer)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	compiled_data->minimizer = minimizer;
	if (compiled_data->update_execs)
	{
		int i;
		const int trainable_size = compiled_data->trainables->rnum;
		ccv_nnc_graph_exec_symbol_t* const update_execs = compiled_data->update_execs;
		ccv_nnc_symbolic_graph_t* const symbolic_graph = model->graph;
		assert(symbolic_graph);
		ccv_nnc_graph_exec_arena_t* const graph_exec_arena = compiled_data->graph_exec_arena;
		for (i = 0; i < trainable_size; i++)
		{
			ccv_nnc_graph_exec_symbol_set(symbolic_graph, update_execs[i], minimizer);
			ccv_nnc_graph_exec_t const update_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, update_execs[i]);
			if (update_exec.graph)
				ccv_nnc_graph_exec_set(update_exec.graph, update_exec, minimizer);
		}
	}
}

void ccv_cnnp_model_dot(const ccv_cnnp_model_t* const model, const int flags, FILE* out)
{
	if (model->graph)
		ccv_nnc_symbolic_graph_dot(model->graph, flags, out);
	if (model->compiled_data && model->compiled_data->graph)
		ccv_nnc_graph_dot(model->compiled_data->graph, flags, out);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_input_isa = {};

ccv_cnnp_model_io_t ccv_cnnp_input(void)
{
	ccv_cnnp_model_t* const input = (ccv_cnnp_model_t*)cccalloc(1, sizeof(ccv_cnnp_model_t) + sizeof(ccv_nnc_tensor_symbol_t));
	input->isa = &ccv_cnnp_input_isa;
	input->io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
	input->io->tbits = 0;
	input->io->incomings = 0;
	input->io->outgoings = 0;
	input->io->model = input;
	input->outputs = (ccv_nnc_tensor_symbol_t*)(input + 1);
	input->output_size = 1;
	return input->io;
}

static void _ccv_cnnp_compiled_data_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	int i;
	if (compiled_data->trainable_tensors)
		for (i = 0; i < compiled_data->trainables->rnum; i++)
			ccv_nnc_tensor_free(compiled_data->trainable_tensors[i]);
	ccv_array_free(compiled_data->trainables);
	if (compiled_data->retain_tensors)
		for (i = 0; i < compiled_data->retains->rnum; i++)
			ccv_nnc_tensor_free(compiled_data->retain_tensors[i]);
	ccv_array_free(compiled_data->retains);
	if (compiled_data->graph)
		ccv_nnc_graph_free(compiled_data->graph);
	if (compiled_data->tensor_arena)
		ccv_nnc_tensor_arena_free(compiled_data->tensor_arena);
	if (compiled_data->graph_exec_arena)
		ccv_nnc_graph_exec_arena_free(compiled_data->graph_exec_arena);
	if (compiled_data->saved_aux)
		ccfree(compiled_data->saved_aux);
	if (compiled_data->trainable_tensors)
		ccfree(compiled_data->trainable_tensors);
	if (compiled_data->dest_to_evals)
		ccfree(compiled_data->dest_to_evals);
	ccfree(compiled_data);
}

void ccv_cnnp_model_free(ccv_cnnp_model_t* const model)
{
	if (model->isa->deinit)
		model->isa->deinit(model);
	if (model->io)
	{
		if (model->io->outgoings)
			ccv_array_free(model->io->outgoings);
		if (model->io->incomings)
			ccv_array_free(model->io->incomings);
		ccfree(model->io);
	}
	if (model->inputs)
		ccfree(model->inputs);
	if (model->graph)
		ccv_nnc_symbolic_graph_free(model->graph);
	if (model->compiled_data)
		_ccv_cnnp_compiled_data_free(model->compiled_data);
	ccfree(model);
}
