#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"

#pragma mark - Level-5 API

static const ccv_cnnp_model_vtab_t ccv_cnnp_input_isa;

#define CCV_CNNP_IS_MODEL_INPUT(x) ((x)->isa == &ccv_cnnp_input_isa)

struct ccv_cnnp_model_io_s {
	int visit; // Temporary bits stored in the ccv_cnnp_model_io_t object, whoever uses it should clean it up.
	ccv_cnnp_model_t* model; // Reference back to the model who holds it. This is required because the model is the one whole holds the io.
	ccv_array_t* incomings; // Array of ccv_cnnp_model_io_t. The order is important because it impacts the order of symbols.
	ccv_array_t* outgoings; // Array of ccv_cnnp_model_io_t.
	ccv_nnc_tensor_symbol_t* outputs; // This is different from the outputs from a model. A model could be reused, causing the outputs on that model to be the most recent one. This keeps the outputs of each.
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
	int i, j = 0, k;
	for (i = 0; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const model = self->sequence[i]->model;
		if (!model)
			continue;
		self->sequence[j++] = (ccv_cnnp_model_io_t)model;
		// Go through all their IO to remove itself as model.
		assert(model->io);
		for (k = 0; k < model->io->rnum; k++)
		{
			ccv_cnnp_model_io_t model_io = *(ccv_cnnp_model_io_t*)ccv_array_get(model->io, k);
			model_io->model = 0;
		}
	}
	for (i = 0; i < j; i++)
		ccv_cnnp_model_free((ccv_cnnp_model_t*)self->sequence[i]);
}

static void _ccv_cnnp_functional_model_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	assert(self->super.input_size == input_size);
	assert(self->super.output_size == output_size);
	int i, j, k;
	for (i = 0; i < self->super.input_size; i++)
		self->sequence[i]->outputs[0] = self->sequence[i]->model->outputs[0] = inputs[i]; // Assigning the output symbol of input layer to be the input symbol.
	ccv_array_t* input_symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 1, 0);
	for (i = self->super.input_size; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const sub_model = self->sequence[i]->model;
		ccv_array_clear(input_symbols);
		const ccv_array_t* const incomings = self->sequence[i]->incomings;
		for (j = 0; j < incomings->rnum; j++)
		{
			const ccv_cnnp_model_io_t input = *(ccv_cnnp_model_io_t*)ccv_array_get(incomings, j);
			for (k = 0; k < input->model->output_size; k++)
				ccv_array_push(input_symbols, &input->outputs[k]);
		}
		// Go through each sub model to build the graph.
		ccv_cnnp_model_build(sub_model, graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(input_symbols, 0), input_symbols->rnum, self->sequence[i]->outputs, sub_model->output_size);
	}
	ccv_array_free(input_symbols);
	for (i = output_size, k = self->sequence_size - 1; k >= 0; k--)
	{
		ccv_cnnp_model_t* const sub_model = self->sequence[k]->model;
		i -= sub_model->output_size;
		if (i < 0)
			break;
		for (j = 0; j < sub_model->output_size; j++)
			outputs[i + j] = self->sequence[k]->outputs[j];
	}
	assert(i <= 0);
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
				++input->visit; // Mark it as visited.
				if (input->visit != input->outgoings->rnum) // Not all dependencies visited.
					continue;
				if (!CCV_CNNP_IS_MODEL_INPUT(input->model))
					ccv_array_push(reverse_top, &input);
				else
					++input_count;
			}
	}
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_cnnp_model_io_t output = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, i);
		output->visit = 0; // Clean the visit back.
	}
	for (i = 0; i < input_size; i++)
		inputs[i]->visit = 0; // Clean the visit back.
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
		model->io = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
	ccv_cnnp_model_io_t model_io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s) + sizeof(ccv_nnc_tensor_symbol_t) * model->output_size);
	model_io->visit = 0;
	model_io->model = model;
	model_io->incomings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
	model_io->outgoings = 0;
	model_io->outputs = (ccv_nnc_tensor_symbol_t*)(model_io + 1);
	ccv_array_push(model->io, &model_io);
	int i;
	ccv_array_resize(model_io->incomings, input_size);
	memcpy(ccv_array_get(model_io->incomings, 0), inputs, sizeof(ccv_cnnp_model_io_t) * input_size);
	for (i = 0; i < input_size; i++)
	{
		if (!inputs[i]->outgoings)
			inputs[i]->outgoings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
		ccv_array_push(inputs[i]->outgoings, &model_io);
	}
	return model_io;
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

void ccv_cnnp_model_set_workspace_size(ccv_cnnp_model_t* const model, size_t workspace_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	if (workspace_size == compiled_data->workspace_size)
		return;
	compiled_data->workspace_size = workspace_size;
	if (compiled_data->graph)
		ccv_nnc_graph_autotune(compiled_data->graph, workspace_size, 0, TRAVERSE_FULL);
}

void ccv_cnnp_model_set_data_parallel(ccv_cnnp_model_t* const model, const int parallel)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(!compiled_data->graph);
	if (parallel== 0)
		compiled_data->parallel_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	else
		compiled_data->parallel_count = parallel;
}

typedef struct {
	int parallel_count;
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
} ccv_nnc_tensor_init_states_t;

static void _ccv_cnnp_init_states_for_tensors(void* const context, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const input, const ccv_nnc_tensor_symbol_t output_symbol)
{
	ccv_nnc_tensor_init_states_t* const tensor_init_states = (ccv_nnc_tensor_init_states_t*)context;
	ccv_nnc_tensor_arena_t* const tensor_arena = tensor_init_states->tensor_arena;
	ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, output_symbol);
	if (!output_tensor)
		return;
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

static void _ccv_cnnp_model_gradient_init(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const fits, const int fit_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(!compiled_data->gradient_init);
	const int dest_to_eval_size = compiled_data->dest_to_eval_size = ccv_nnc_symbolic_graph_destination_size(model->graph);
	assert(dest_to_eval_size > 0);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	compiled_data->dest_to_evals = ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * dest_to_eval_size * parallel_count + sizeof(ccv_nnc_graph_exec_t) * dest_to_eval_size * parallel_count);
	compiled_data->dest_to_eval_execs = (ccv_nnc_graph_exec_t*)(compiled_data->dest_to_evals + dest_to_eval_size * parallel_count);
	memcpy(compiled_data->dest_to_evals, ccv_nnc_symbolic_graph_destinations(model->graph), sizeof(ccv_nnc_graph_exec_symbol_t) * dest_to_eval_size);
	int i, j;
	const int output_size = model->output_size;
	assert(!fits || fit_size == output_size * parallel_count);
	ccv_nnc_tensor_symbol_t f[output_size];
	if (compiled_data->loss.cmd == CCV_NNC_NOOP)
	{
		// If no loss function provided, there is no fits.
		for (i = 0; i < output_size; i++)
		{
			compiled_data->fits[i] = NO_TENSOR_SYMBOL;
			f[i] = model->outputs[i];
		}
	} else {
		for (i = 0; i < output_size; i++)
		{
			const ccv_nnc_tensor_symbol_t fit = compiled_data->fits[i] = ccv_nnc_tensor_symbol_new(model->graph, fits[i]->info, 0);
			f[i] = ccv_nnc_tensor_symbol_new(model->graph, ccv_nnc_tensor_auto, 0);
			ccv_nnc_graph_exec_symbol_new(model->graph, compiled_data->loss, TENSOR_SYMBOL_LIST(model->outputs[i], fit), TENSOR_SYMBOL_LIST(f[i]), 0);
		}
	}
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_simplify(model->graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_OPS_FUSION), // Only do Ops fusion, in this way, we can fuse the loss function.
		f, model->output_size,
		SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
	const int trainable_size = compiled_data->trainables->rnum;
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimizer);
	compiled_data->saved_aux = (ccv_nnc_tensor_symbol_map_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_map_t) * saved_aux_size * trainable_size + sizeof(ccv_nnc_tensor_symbol_t) * trainable_size + sizeof(ccv_nnc_graph_exec_symbol_t) * trainable_size);
	compiled_data->updated_trainables = (ccv_nnc_tensor_symbol_t*)(compiled_data->saved_aux + saved_aux_size * trainable_size);
	compiled_data->update_execs = (ccv_nnc_graph_exec_symbol_t*)(compiled_data->updated_trainables + trainable_size);
	ccv_nnc_tensor_symbol_t* const gradients = parallel_count > 1 ? (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * trainable_size) : 0;
	ccv_nnc_symbolic_graph_minimize(model->graph, compiled_data->minimizer, f, output_size, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), trainable_size, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), gradients, compiled_data->updated_trainables, compiled_data->saved_aux, compiled_data->update_execs);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t df = ccv_nnc_tensor_symbol_for_backward(model->graph, f[i]);
		// Init this to 1 so we can backprop.
		ccv_nnc_tensor_symbol_set_flags(model->graph, df, CCV_NNC_TENSOR_SYMBOL_INIT_ONES);
	}
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS);
	ccv_nnc_symbolic_graph_set_destinations(model->graph, compiled_data->update_execs, trainable_size);
	if (parallel_count > 1)
	{
		ccv_nnc_symbolic_graph_data_parallel(model->graph, compiled_data->parallel_count,
			0, 0,
			gradients, trainable_size,
			0, 0,
			CCV_NNC_PARALLEL_REDUCE_OP_SUM,
			SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
		ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		for (i = 0; i < dest_to_eval_size; i++)
			for (j = 1; j < parallel_count; j++)
			{
				const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_copy(model->graph, compiled_data->dest_to_evals[i], j);
				if (copy.d != CCV_NNC_NO_GRAPH_EXEC_SYMBOL)
					compiled_data->dest_to_evals[compiled_data->dest_to_eval_size++] = copy;
			}
		ccfree(gradients);
	}
	compiled_data->gradient_init = 1;
}

void ccv_cnnp_model_tensors_init(const ccv_nnc_symbolic_graph_t* const graph, ccv_cnnp_compiled_data_t* const compiled_data)
{
	assert(!compiled_data->trainable_tensors);
	const int trainable_size = compiled_data->trainables->rnum;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	const int retain_size = compiled_data->retains->rnum;
	compiled_data->trainable_tensors = (ccv_nnc_tensor_t**)ccmalloc((sizeof(ccv_nnc_tensor_t*) * trainable_size + sizeof(ccv_nnc_tensor_t*) * retain_size) * parallel_count);
	compiled_data->retain_tensors = compiled_data->trainable_tensors + trainable_size * parallel_count;
	int i, j;
	for (i = 0; i < trainable_size; i++)
	{
		const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(trainable.graph, trainable);
		compiled_data->trainable_tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		for (j = 1; j < parallel_count; j++)
		{
			CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			compiled_data->trainable_tensors[i + j * trainable_size] = ccv_nnc_tensor_new(0, info, 0);
		}
	}
	for (i = 0; i < retain_size; i++)
	{
		const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(retained.graph, retained);
		compiled_data->retain_tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		for (j = 1; j < parallel_count; j++)
		{
			CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			compiled_data->retain_tensors[i + j * retain_size] = ccv_nnc_tensor_new(0, info, 0);
		}
	}
}

static void _ccv_cnnp_model_copy_tensors(ccv_nnc_tensor_t* const* const tensors, const int tensor_size, const int parallel_count)
{
	assert(parallel_count > 0);
	int i, j;
	for (i = 0; i < tensor_size; i++)
		for (j = 1; j < parallel_count; j++)
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[i], 1, &tensors[i + j * tensor_size], 1, 0);
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
		const ccv_nnc_tensor_symbol_t tensor_symbol = tensor_symbols[i];
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

// Compile the graph to run ccv_cnnp_model_fit
static void _ccv_cnnp_model_fit_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const fits, const int fit_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i, j;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(!compiled_data->graph || compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_FIT_MODE);
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_FIT_MODE;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(!fits || output_size == fit_size);
	assert(output_size > 0);
	if (!compiled_data->gradient_init)
		_ccv_cnnp_model_gradient_init(model, fits, fit_size);
	const int tensors_init = !!compiled_data->trainable_tensors;
	if (!compiled_data->trainable_tensors)
		ccv_cnnp_model_tensors_init(model->graph, compiled_data);
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
	const int trainable_size = compiled_data->trainables->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->trainable_tensors, trainable_size, parallel_count, tensor_binds);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->updated_trainables, compiled_data->trainable_tensors, trainable_size, parallel_count, tensor_binds);
	const int retain_size = compiled_data->retains->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, 0), compiled_data->retain_tensors, retain_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, 0), compiled_data->retain_tensors, retain_size, parallel_count, tensor_binds);
	ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	// If tensor is not init'ed, we need to init states first.
	if (!tensors_init)
	{
		ccv_nnc_tensor_init_states_t tensor_init_states = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.tensor_arena = compiled_data->tensor_arena
		};
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, &tensor_init_states);
	} else if (parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->trainable_tensors, compiled_data->trainables->rnum, parallel_count);
	compiled_data->is_test = 0;
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimizer);
	// No need to set because it is default to training mode.
	// ccv_cnnp_model_set_is_test(model, 0, _ccv_cnnp_cmd_update_for_execs, compiled_data->graph_exec_arena);
	for (i = 0; i < saved_aux_size * trainable_size; i++)
	{
		ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, compiled_data->saved_aux[i].source);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &tensor, 1, 0);
		for (j = 1; j < parallel_count; j++)
		{
			ccv_nnc_tensor_t* const copy = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, compiled_data->saved_aux[i].source, j));
			if (copy)
				ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &copy, 1, 0);
		}
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
	ccv_nnc_graph_autotune(compiled_data->graph, compiled_data->workspace_size, 0, TRAVERSE_FULL);
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

void ccv_cnnp_model_fit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const fits, const int fit_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(!fits || fit_size == output_size);
	assert(model->graph);
	int i, j;
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
		assert((input_size % parallel_count) == 0);
		assert((output_size % parallel_count) == 0);
		assert((fit_size % parallel_count) == 0);
		const int input_size_per_p = input_size / parallel_count;
		for (i = 0; i < input_size_per_p; i++)
		{
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->inputs[i], inputs[i]);
			for (j = 1; j < parallel_count; j++)
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, model->inputs[i], j), inputs[i + input_size_per_p * j]);
		}
		const int output_size_per_p = output_size / parallel_count;
		for (i = 0; i < output_size_per_p; i++)
		{
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->outputs[i], outputs[i]);
			for (j = 1; j < parallel_count; j++)
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, model->outputs[i], j), outputs[i + output_size_per_p * j]);
		}
		const int fit_size_per_p = fit_size / parallel_count;
		for (i = 0; i < fit_size_per_p; i++)
			if (compiled_data->fits[i].d >= 0)
			{
				assert(fits);
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, compiled_data->fits[i], fits[i]);
				for (j = 1; j < parallel_count; j++)
					ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, compiled_data->fits[i], j), fits[i + fit_size_per_p * j]);
			}
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
	ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, TRAVERSE_FULL);
}

// Compile the graph to run ccv_cnnp_model_evaluate with require_grad = false (MULTISTAGE_MODE_NO_GRAD).
static void _ccv_cnnp_model_multistage_no_grad_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE_NO_GRAD;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(output_size > 0);
	const int tensors_init = !!compiled_data->trainable_tensors;
	if (!compiled_data->trainable_tensors)
		ccv_cnnp_model_tensors_init(model->graph, compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	assert((input_size % parallel_count) == 0);
	assert((output_size % parallel_count) == 0);
	const int input_size_per_p = input_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->inputs, inputs, input_size_per_p, parallel_count, tensor_binds);
	const int output_size_per_p = output_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->outputs, outputs, output_size_per_p, parallel_count, tensor_binds);
	const int trainable_size = compiled_data->trainables->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->trainable_tensors, trainable_size, parallel_count, tensor_binds);
	const int retain_size = compiled_data->retains->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, 0), compiled_data->retain_tensors, retain_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, 0), compiled_data->retain_tensors, retain_size, parallel_count, tensor_binds);
	// If we generated gradient for the graph, only compile part of the graph because the rest is irrelevant for evaluation.
	if (compiled_data->gradient_init)
		ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), compiled_data->dest_to_evals, compiled_data->dest_to_eval_size, &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	else {
		assert(compiled_data->parallel_count <= 1); // I don't know how to handle parallel_count larger than 1.
		ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	}
	// If tensor is not init'ed, we need to init states first.
	if (!tensors_init)
	{
		ccv_nnc_tensor_init_states_t tensor_init_states = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.tensor_arena = compiled_data->tensor_arena
		};
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, &tensor_init_states);
	} else if (parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->trainable_tensors, compiled_data->trainables->rnum, parallel_count);
	compiled_data->is_test = 1;
	ccv_nnc_graph_exec_update_t update = {
		.parallel_count = parallel_count,
		.graph = model->graph,
		.graph_exec_arena = compiled_data->graph_exec_arena,
	};
	ccv_cnnp_model_set_is_test(model, 1, _ccv_cnnp_cmd_update_for_execs, &update);
	ccv_nnc_graph_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_array_free(tensor_binds);
	ccv_nnc_graph_autotune(compiled_data->graph, compiled_data->workspace_size, 0, TRAVERSE_FULL);
}

// Compile the graph to run ccv_cnnp_model_evaluate with requires_grad = true (MULTISTAGE_MODE).
static void _ccv_cnnp_model_multistage_jit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i, j;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(!compiled_data->graph || compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(output_size > 0);
	// There shouldn't be a loss function if we evaluate with multistage jit.
	assert(compiled_data->loss.cmd == CCV_NNC_NOOP);
	if (!compiled_data->gradient_init)
		_ccv_cnnp_model_gradient_init(model, 0, 0); // The type of outputs and fits should be the same. We only use type here.
	const int tensors_init = !!compiled_data->trainable_tensors;
	if (!compiled_data->trainable_tensors)
		ccv_cnnp_model_tensors_init(model->graph, compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	assert((input_size % parallel_count) == 0);
	assert((output_size % parallel_count) == 0);
	const int input_size_per_p = input_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->inputs, inputs, input_size_per_p, parallel_count, tensor_binds);
	const int output_size_per_p = output_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->outputs, outputs, output_size_per_p, parallel_count, tensor_binds);
	const int trainable_size = compiled_data->trainables->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->trainable_tensors, trainable_size, parallel_count, tensor_binds);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->updated_trainables, compiled_data->trainable_tensors, trainable_size, parallel_count, tensor_binds);
	const int retain_size = compiled_data->retains->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, 0), compiled_data->retain_tensors, retain_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retains, 0), compiled_data->retain_tensors, retain_size, parallel_count, tensor_binds);
	ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	// If tensor is not init'ed, we need to init states first.
	if (!tensors_init)
	{
		ccv_nnc_tensor_init_states_t tensor_init_states = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.tensor_arena = compiled_data->tensor_arena
		};
		ccv_cnnp_model_init_states(model, model->graph, _ccv_cnnp_init_states_for_tensors, &tensor_init_states);
	} else if (parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->trainable_tensors, compiled_data->trainables->rnum, parallel_count);
	compiled_data->is_test = 0;
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimizer);
	// No need to set because it is default to training mode.
	// ccv_cnnp_model_set_is_test(model, 0, _ccv_cnnp_cmd_update_for_execs, compiled_data->graph_exec_arena);
	for (i = 0; i < saved_aux_size * trainable_size; i++)
	{
		ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, compiled_data->saved_aux[i].source);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &tensor, 1, 0);
		for (j = 1; j < parallel_count; j++)
		{
			ccv_nnc_tensor_t* const copy = ccv_nnc_tensor_from_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, compiled_data->saved_aux[i].source, j));
			if (copy)
				ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &copy, 1, 0);
		}
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
	ccv_nnc_graph_autotune(compiled_data->graph, compiled_data->workspace_size, 0, TRAVERSE_FULL);
}

void ccv_cnnp_model_evaluate(ccv_cnnp_model_t* const model, const int requires_grad, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(model->graph);
	int i, j;
	if (!compiled_data->graph ||
		// If a stream context is provided, we need to recompile because we cannot run them efficiently in FIT_MODE.
		(stream_context &&
		 !((requires_grad && compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE) ||
		   (!requires_grad && compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE_NO_GRAD))))
	{
		if (compiled_data->graph)
			ccv_nnc_graph_free(compiled_data->graph);
		if (compiled_data->tensor_arena)
			ccv_nnc_tensor_arena_free(compiled_data->tensor_arena);
		if (compiled_data->graph_exec_arena)
			ccv_nnc_graph_exec_arena_free(compiled_data->graph_exec_arena);
		if (requires_grad)
			_ccv_cnnp_model_multistage_jit(model, inputs, input_size, outputs, output_size);
		else
			_ccv_cnnp_model_multistage_no_grad_jit(model, inputs, input_size, outputs, output_size);
	} else {
		assert((input_size % parallel_count) == 0);
		assert((output_size % parallel_count) == 0);
		const int input_size_per_p = input_size / parallel_count;
		for (i = 0; i < input_size_per_p; i++)
		{
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->inputs[i], inputs[i]);
			for (j = 1; j < parallel_count; j++)
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, model->inputs[i], j), inputs[i + input_size_per_p * j]);
		}
		const int output_size_per_p = output_size / parallel_count;
		for (i = 0; i < output_size_per_p; i++)
		{
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, model->outputs[i], outputs[i]);
			for (j = 1; j < parallel_count; j++)
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, model->outputs[i], j), outputs[i + input_size_per_p * j]);
		}
	}
	if (!compiled_data->is_test)
	{
		compiled_data->is_test = 1;
		ccv_nnc_graph_exec_update_t update = {
			.parallel_count = parallel_count,
			.graph = model->graph,
			.graph_exec_arena = compiled_data->graph_exec_arena,
		};
		ccv_cnnp_model_set_is_test(model, 1, _ccv_cnnp_cmd_update_for_execs, &update);
	}
	if (compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE)
		ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, TRAVERSE_FULL);
	else
		ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, 0, 0,
			compiled_data->dest_to_eval_execs, compiled_data->dest_to_eval_exec_size);
}

void ccv_cnnp_model_backward(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const ingrads, const int ingrad_size, ccv_nnc_tensor_t* const* const outgrads, const int outgrad_size, ccv_nnc_stream_context_t* const stream_context)
{
}

void ccv_cnnp_model_apply_gradients(ccv_cnnp_model_t* const model, ccv_nnc_stream_context_t* const stream_context)
{
}

void ccv_cnnp_model_set_minimizer(ccv_cnnp_model_t* const model, const ccv_nnc_cmd_t minimizer)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	compiled_data->minimizer = minimizer;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	if (compiled_data->update_execs)
	{
		int i, j;
		const int trainable_size = compiled_data->trainables->rnum;
		ccv_nnc_graph_exec_symbol_t* const update_execs = compiled_data->update_execs;
		ccv_nnc_symbolic_graph_t* const symbolic_graph = model->graph;
		assert(symbolic_graph);
		ccv_nnc_graph_exec_arena_t* const graph_exec_arena = compiled_data->graph_exec_arena;
		for (i = 0; i < trainable_size; i++)
		{
			ccv_nnc_graph_exec_symbol_set(symbolic_graph, update_execs[i], minimizer);
			ccv_nnc_graph_exec_t const update_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, update_execs[i]);
			if (!CCV_NO_GRAPH_EXEC(update_exec))
				ccv_nnc_graph_exec_set(update_exec.graph, update_exec, minimizer);
			for (j = 1; j < parallel_count; j++)
			{
				ccv_nnc_graph_exec_symbol_t copy_symbol = ccv_nnc_graph_exec_symbol_copy(symbolic_graph, update_execs[i], j);
				if (copy_symbol.graph)
					ccv_nnc_graph_exec_symbol_set(symbolic_graph, copy_symbol, minimizer);
				const ccv_nnc_graph_exec_t copy = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, copy_symbol);
				if (!CCV_NO_GRAPH_EXEC(copy))
					ccv_nnc_graph_exec_set(copy.graph, copy, minimizer);
			}
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
	input->io = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
	ccv_cnnp_model_io_t input_io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s) + sizeof(ccv_nnc_tensor_symbol_t));
	input_io->visit = 0;
	input_io->incomings = 0;
	input_io->outgoings = 0;
	input_io->model = input;
	input_io->outputs = (ccv_nnc_tensor_symbol_t*)(input_io + 1);
	ccv_array_push(input->io, &input_io);
	input->outputs = (ccv_nnc_tensor_symbol_t*)(input + 1);
	input->output_size = 1;
	return input_io;
}

static void _ccv_cnnp_compiled_data_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	int i;
	if (compiled_data->trainable_tensors)
		for (i = 0; i < compiled_data->trainables->rnum; i++)
			ccv_nnc_tensor_free(compiled_data->trainable_tensors[i]);
	ccv_array_free(compiled_data->trainables);
	if (compiled_data->retain_tensors)
		for (i = 0; i < compiled_data->retains->rnum * ccv_max(compiled_data->parallel_count, 1); i++)
			if (compiled_data->retain_tensors[i])
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
		int i;
		for (i = 0; i < model->io->rnum; i++)
		{
			ccv_cnnp_model_io_t model_io = *(ccv_cnnp_model_io_t*)ccv_array_get(model->io, i);
			if (model_io->outgoings)
				ccv_array_free(model_io->outgoings);
			if (model_io->incomings)
				ccv_array_free(model_io->incomings);
			ccfree(model_io);
		}
		ccv_array_free(model->io);
	}
	if (model->inputs)
		ccfree(model->inputs);
	if (model->graph)
		ccv_nnc_symbolic_graph_free(model->graph);
	if (model->compiled_data)
		_ccv_cnnp_compiled_data_free(model->compiled_data);
	ccfree(model);
}
