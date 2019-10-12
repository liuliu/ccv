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
	int i, j;
	for (i = 0; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const model = self->sequence[i];
		if (!model)
			continue;
		ccv_cnnp_model_free(model);
		for (j = i + 1; j < self->sequence_size; j++)
			if (self->sequence[j] == model)
				self->sequence[j] = 0;
	}
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

static void _ccv_cnnp_sequential_model_add_to_trainable(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const trainables)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_trainable(self->sequence[i], add_to_array, trainables);
}

static void _ccv_cnnp_sequential_model_add_to_output(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_output(self->sequence[i], add_to_array, outputs);
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

ccv_cnnp_model_t* ccv_cnnp_sequential_new(ccv_cnnp_model_t* const* const models, const int model_size, const char* const name)
{
	assert(model_size > 0);
	ccv_cnnp_sequential_model_t* const sequential_model = (ccv_cnnp_sequential_model_t*)cccalloc(1, sizeof(ccv_cnnp_sequential_model_t) + sizeof(ccv_cnnp_model_t*) * (model_size - 1) + sizeof(ccv_nnc_tensor_symbol_t));
	sequential_model->super.isa = &ccv_cnnp_sequential_model_isa;
	sequential_model->super.input_size = 1;
	sequential_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(sequential_model->sequence + model_size);
	sequential_model->super.output_size = 1;
	ccv_cnnp_model_copy_name(&sequential_model->super, name);
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

static void _ccv_cnnp_functional_model_add_to_trainable(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const trainables)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_trainable(self->sequence[i]->model, add_to_array, trainables);
}

static void _ccv_cnnp_functional_model_add_to_output(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_output(self->sequence[i]->model, add_to_array, outputs);
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

ccv_cnnp_model_t* ccv_cnnp_model_new(const ccv_cnnp_model_io_t* const inputs, const int input_size, const ccv_cnnp_model_io_t* const outputs, const int output_size, const char* const name)
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
	ccv_cnnp_model_copy_name(&functional_model->super, name);
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
	ccv_cnnp_model_sequence_t sequence;
	char prefix;
	ccv_array_t* symbols;
	ccv_array_t* ids;
} ccv_cnnp_model_add_to_array_context_t;

static void _ccv_cnnp_add_to_array(void* const context, const ccv_nnc_tensor_symbol_t symbol)
{
	ccv_cnnp_model_add_to_array_context_t* const add_to_array_context = (ccv_cnnp_model_add_to_array_context_t*)context;
	int i;
	for (i = 0; i < add_to_array_context->symbols->rnum; i++)
	{
		const ccv_nnc_tensor_symbol_t other_symbol = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(add_to_array_context->symbols, i);
		if (other_symbol.d == symbol.d && other_symbol.graph == symbol.graph)
			return;
	}
	ccv_array_push(add_to_array_context->symbols, &symbol);
	char id[1024];
	id[0] = add_to_array_context->prefix;
	id[1] = '-';
	int total_len = 2;
	for (i = 0; i < add_to_array_context->sequence.sequences->rnum; i++)
	{
		const ccv_cnnp_model_name_t* const name = (ccv_cnnp_model_name_t*)ccv_array_get(add_to_array_context->sequence.sequences, i);
		int len;
		if (name->type == CCV_CNNP_MODEL_NAME)
			len = snprintf(id + total_len, 1024 - total_len, "%s-", name->name);
		else
			len = snprintf(id + total_len, 1024 - total_len, "%d-", name->sequence);
		total_len += len;
		if (total_len >= 1023)
			break;
	}
	if (total_len < 1023)
		total_len += snprintf(id + total_len, 1024 - total_len, "%d", add_to_array_context->sequence.it);
	assert(total_len < 1024);
	char *heap_id = (char*)ccmalloc(total_len + 1);
	memcpy(heap_id, id, total_len + 1);
	ccv_array_push(add_to_array_context->ids, &heap_id);
	++add_to_array_context->sequence.it;
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
		ccv_array_t* const trainable_ids = ccv_array_new(sizeof(char*), 0, 0);
		ccv_cnnp_model_add_to_array_context_t context = {
			.sequence = {},
			.prefix = 't',
			.symbols = trainables,
			.ids = trainable_ids,
		};
		ccv_cnnp_model_add_to_trainable(model, _ccv_cnnp_add_to_array, &context);
		// Assert no trainable is alias.
		for (i = 0; i < trainables->rnum; i++)
		{
			const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(trainables, i);
			const ccv_nnc_tensor_symbol_t alias_to = ccv_nnc_tensor_symbol_alias_to(trainable.graph, trainable);
			assert(alias_to.graph == 0); // Cannot find the one alias to.
		}
		ccv_array_t* const retainables = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
		ccv_array_t* const retainable_ids = ccv_array_new(sizeof(char*), 0, 0);
		if (context.sequence.sequences)
			ccv_array_clear(context.sequence.sequences);
		context.prefix = 'r';
		context.symbols = retainables;
		context.ids = retainable_ids;
		ccv_cnnp_model_add_to_output(model, _ccv_cnnp_add_to_array, &context);
		ccv_array_free(context.sequence.sequences);
		// Assert no retainable is alias.
		for (i = 0; i < retainables->rnum; i++)
		{
			const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(retainables, i);
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
		model->compiled_data->retainables = retainables;
		model->compiled_data->ids.trainables = trainable_ids;
		model->compiled_data->ids.retainables = retainable_ids;
		model->compiled_data->minimize.minimizer = minimizer;
		model->compiled_data->loss = loss;
	}
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

void ccv_cnnp_model_set_memory_compression(ccv_cnnp_model_t* const model, const int memory_compression)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(!compiled_data->graph);
	compiled_data->memory_compression = memory_compression;
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
	for (i = 0; i < compiled_data->trainables->rnum; i++)
	{
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i))->d;
		if (!(compiled_data->tensors_init.v[d >> 5] & (1u << (d & 0x1f))))
			return 1;
	}
	for (i = 0; i < compiled_data->retainables->rnum; i++)
	{
		const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, i))->d;
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
		if (copy_symbol.graph)
			ccv_nnc_graph_exec_symbol_set(symbolic_graph, copy_symbol, cmd);
		const ccv_nnc_graph_exec_t copy = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, copy_symbol);
		if (!CCV_NO_GRAPH_EXEC(copy))
			ccv_nnc_graph_exec_set(copy.graph, copy, cmd);
	}
}

static void _ccv_cnnp_model_graph_exec_symbol_set(ccv_cnnp_model_t* const model, const ccv_nnc_graph_exec_symbol_t exec_symbol, const ccv_nnc_cmd_t cmd)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	ccv_nnc_symbolic_graph_t* const symbolic_graph = model->graph;
	assert(symbolic_graph);
	ccv_nnc_graph_exec_symbol_set(symbolic_graph, exec_symbol, cmd);
	ccv_nnc_graph_exec_arena_t* const graph_exec_arena = compiled_data->graph_exec_arena;
	if (graph_exec_arena)
		_ccv_cnnp_model_graph_symbol_exec_set_for_graph_exec_arena(graph_exec_arena, parallel_count, exec_symbol, cmd, symbolic_graph);
	// Skip backward graph exec arena because it is for a specific accum symbolic graph, not the main graph (model->graph)
	ccv_nnc_graph_exec_arena_t* const gradient_graph_exec_arena = compiled_data->apply_gradients.graph_exec_arena;
	if (gradient_graph_exec_arena)
		_ccv_cnnp_model_graph_symbol_exec_set_for_graph_exec_arena(gradient_graph_exec_arena, parallel_count, exec_symbol, cmd, symbolic_graph);
}

static void _ccv_cnnp_model_set_minimizer_setter(ccv_cnnp_model_t* const model, const ccv_cnnp_model_minimizer_set_f minimizer_setter, const void* const context)
{
	int i;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(minimizer_setter);
	const int trainable_size = compiled_data->trainables->rnum;
	ccv_nnc_graph_exec_symbol_t* const update_nodes = compiled_data->update_nodes;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = model->graph;
	// Collect which trainable exists at which node.
	const int tensor_symbol_count = ccv_nnc_tensor_symbol_count(symbolic_graph);
	ccv_array_t** const trainable_pos = (ccv_array_t**)cccalloc(tensor_symbol_count, sizeof(ccv_array_t*));
	for (i = 0; i < trainable_size; i++)
		trainable_pos[((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i))->d] = ccv_array_new(sizeof(ccv_cnnp_trainable_index_t), 0, 0);
	ccv_nnc_symbolic_graph_iter_t* const iter = ccv_nnc_symbolic_graph_iter_new(symbolic_graph, 0, 0, 0, 0);
	while (ccv_nnc_symbolic_graph_iter_next(iter)) {
		ccv_nnc_tensor_symbol_t* inputs;
		int input_size;
		ccv_nnc_tensor_symbol_io_from_iter(iter, &inputs, &input_size, 0, 0);
		for (i = 0; i < input_size; i++)
			if (inputs[i].d >= 0 && trainable_pos[inputs[i].d])
			{
				ccv_nnc_cmd_t cmd;
				ccv_nnc_graph_exec_symbol_from_iter(iter, &cmd, 0, 0, 0);
				const ccv_cnnp_trainable_index_t trainable_index = (ccv_cnnp_trainable_index_t){
					.cmd = cmd,
					.index = i,
				};
				ccv_array_push(trainable_pos[inputs[i].d], &trainable_index);
			}
	}
	ccv_nnc_symbolic_graph_iter_free(iter);
	for (i = 0; i < trainable_size; i++)
	{
		ccv_array_t* const trainable_indexes = trainable_pos[((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i))->d];
		const ccv_nnc_cmd_t cmd = minimizer_setter(model, (ccv_cnnp_trainable_index_t*)ccv_array_get(trainable_indexes, 0), trainable_indexes->rnum, context);
		_ccv_cnnp_model_graph_exec_symbol_set(model, update_nodes[i], cmd);
		ccv_array_free(trainable_indexes);
	}
}

static void _ccv_cnnp_model_gradient_init(ccv_cnnp_model_t* const model, const int gradient_mode, ccv_nnc_tensor_t* const* const fits, const int fit_size)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_NONE);
	assert(gradient_mode != CCV_CNNP_COMPILED_DATA_GRADIENT_NONE);
	const int evaluate_to_size = compiled_data->evaluate.to_size = ccv_nnc_symbolic_graph_destination_size(model->graph);
	assert(evaluate_to_size > 0);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	compiled_data->evaluate.tos = ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * evaluate_to_size * parallel_count + sizeof(ccv_nnc_graph_exec_t) * evaluate_to_size * parallel_count);
	compiled_data->evaluate.to_ops = (ccv_nnc_graph_exec_t*)(compiled_data->evaluate.tos + evaluate_to_size * parallel_count);
	memcpy(compiled_data->evaluate.tos, ccv_nnc_symbolic_graph_destinations(model->graph), sizeof(ccv_nnc_graph_exec_symbol_t) * evaluate_to_size);
	if (!compiled_data->rewindables)
		compiled_data->rewindables = ccv_array_new(sizeof(ccv_cnnp_rewind_symbol_t), 0, 0);
	ccv_nnc_tensor_symbol_new_hook(model->graph, _ccv_cnnp_model_tensor_symbol_new_hook, compiled_data->rewindables);
	ccv_nnc_tensor_symbol_alias_new_hook(model->graph, _ccv_cnnp_model_tensor_symbol_alias_new_hook, compiled_data->rewindables);
	ccv_nnc_graph_exec_symbol_new_hook(model->graph, _ccv_cnnp_model_graph_exec_symbol_new_hook, compiled_data->rewindables);
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
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimize.minimizer);
	const int trainable_size = compiled_data->trainables->rnum;
	compiled_data->saved_aux = (ccv_nnc_tensor_symbol_map_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_map_t) * saved_aux_size * trainable_size + sizeof(ccv_nnc_tensor_symbol_t) * trainable_size + sizeof(ccv_nnc_graph_exec_symbol_t) * trainable_size);
	compiled_data->updated_trainables = (ccv_nnc_tensor_symbol_t*)(compiled_data->saved_aux + saved_aux_size * trainable_size);
	compiled_data->update_nodes = (ccv_nnc_graph_exec_symbol_t*)(compiled_data->updated_trainables + trainable_size);
	const int trainable_size_maybe_more = gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES ? trainable_size : trainable_size + model->input_size;
	compiled_data->gradients = (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * trainable_size_maybe_more + sizeof(ccv_nnc_graph_exec_symbol_t) * trainable_size_maybe_more * parallel_count);
	compiled_data->backward.tos = (ccv_nnc_graph_exec_symbol_t*)(compiled_data->gradients + trainable_size_maybe_more);
	compiled_data->backward.to_size = trainable_size_maybe_more;
	if (gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES)
		ccv_nnc_symbolic_graph_minimize(model->graph, compiled_data->minimize.minimizer, f, output_size, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), trainable_size, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), compiled_data->gradients, compiled_data->updated_trainables, compiled_data->saved_aux, compiled_data->update_nodes);
	else // Compute minimize with gradients including inputs.
		ccv_nnc_symbolic_graph_minimize(model->graph, compiled_data->minimize.minimizer, f, output_size, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), trainable_size, model->inputs, model->input_size, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), compiled_data->gradients, compiled_data->updated_trainables, compiled_data->saved_aux, compiled_data->update_nodes);
	if (compiled_data->minimize.setter)
		_ccv_cnnp_model_set_minimizer_setter(model, compiled_data->minimize.setter, compiled_data->minimize.context);
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_symbol_t df = ccv_nnc_tensor_symbol_for_backward(model->graph, f[i]);
		// Init this to 1 so we can backprop.
		ccv_nnc_tensor_symbol_set_flags(model->graph, df, CCV_NNC_TENSOR_SYMBOL_INIT_ONES);
	}
	for (i = 0; i < trainable_size_maybe_more; i++)
		compiled_data->backward.tos[i] = ccv_nnc_graph_exec_symbol_for_backward(model->graph, compiled_data->gradients[i]);
	ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS);
	ccv_nnc_symbolic_graph_set_destinations(model->graph, compiled_data->update_nodes, trainable_size);
	if (parallel_count > 1)
	{
		ccv_nnc_symbolic_graph_data_parallel(model->graph, compiled_data->parallel_count,
			0, 0,
			compiled_data->gradients, trainable_size_maybe_more,
			0, 0,
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
		for (i = 0; i < trainable_size_maybe_more; i++)
			for (j = 1; j < parallel_count; j++)
			{
				const ccv_nnc_graph_exec_symbol_t copy = ccv_nnc_graph_exec_symbol_copy(model->graph, compiled_data->backward.tos[i], j);
				if (copy.d != CCV_NNC_NO_GRAPH_EXEC_SYMBOL)
					compiled_data->backward.tos[compiled_data->backward.to_size++] = copy;
			}
	}
	// Only use memory compression if we are in gradient trainable mode.
	if (gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES && compiled_data->memory_compression)
		ccv_nnc_symbolic_graph_memory_compression(model->graph, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
	compiled_data->backward.to_size = _ccv_nnc_array_dedup_graph_exec_symbols(compiled_data->backward.tos, compiled_data->backward.to_size);
	compiled_data->gradient_mode = gradient_mode;
}

void ccv_cnnp_model_tensors_init(const ccv_nnc_symbolic_graph_t* const graph, ccv_cnnp_compiled_data_t* const compiled_data)
{
	assert(!compiled_data->tensors.trainables);
	const int trainable_size = compiled_data->trainables->rnum;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	const int retainable_size = compiled_data->retainables->rnum;
	compiled_data->tensors_init.size = ccv_nnc_tensor_symbol_count(graph);
	compiled_data->tensors_init.v = cccalloc(((compiled_data->tensors_init.size + 31) >> 5), sizeof(uint32_t));
	compiled_data->tensors.trainables = (ccv_nnc_tensor_t**)ccmalloc((sizeof(ccv_nnc_tensor_t*) * trainable_size + sizeof(ccv_nnc_tensor_t*) * retainable_size) * parallel_count);
	compiled_data->tensors.retainables = compiled_data->tensors.trainables + trainable_size * parallel_count;
	int i, j;
	for (i = 0; i < trainable_size; i++)
	{
		const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(trainable.graph, trainable);
		CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
		compiled_data->tensors.trainables[i] = ccv_nnc_tensor_new(0, info, 0);
		for (j = 1; j < parallel_count; j++)
		{
			CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			compiled_data->tensors.trainables[i + j * trainable_size] = ccv_nnc_tensor_new(0, info, 0);
		}
	}
	for (i = 0; i < retainable_size; i++)
	{
		const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(retained.graph, retained);
		CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
		compiled_data->tensors.retainables[i] = ccv_nnc_tensor_new(0, info, 0);
		for (j = 1; j < parallel_count; j++)
		{
			CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			compiled_data->tensors.retainables[i + j * retainable_size] = ccv_nnc_tensor_new(0, info, 0);
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

static void _ccv_cnnp_compiled_data_graph_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	if (compiled_data->graph)
		ccv_nnc_graph_free(compiled_data->graph);
	compiled_data->graph = 0;
	if (compiled_data->tensor_arena)
		ccv_nnc_tensor_arena_free(compiled_data->tensor_arena);
	compiled_data->tensor_arena = 0;
	if (compiled_data->graph_exec_arena)
		ccv_nnc_graph_exec_arena_free(compiled_data->graph_exec_arena);
	compiled_data->graph_exec_arena = 0;
}

static void _ccv_cnnp_compiled_data_gradient_free(ccv_cnnp_compiled_data_t* const compiled_data)
{
	if (compiled_data->gradients)
		ccfree(compiled_data->gradients);
	compiled_data->gradients = 0;
	if (compiled_data->saved_aux)
		ccfree(compiled_data->saved_aux);
	compiled_data->saved_aux = 0;
	if (compiled_data->evaluate.tos)
		ccfree(compiled_data->evaluate.tos);
	compiled_data->evaluate.tos = 0;
	if (compiled_data->backward.from_ops)
		ccfree(compiled_data->backward.from_ops);
	compiled_data->backward.from_ops = 0;
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
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(!fits || output_size == fit_size);
	assert(output_size > 0);
	if (compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_NONE)
		_ccv_cnnp_model_gradient_init(model, CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES, fits, fit_size);
	else if (compiled_data->gradient_mode != CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES) {
		_ccv_cnnp_model_rewind_graph(model);
		_ccv_cnnp_compiled_data_gradient_free(compiled_data);
		compiled_data->gradient_mode = CCV_CNNP_COMPILED_DATA_GRADIENT_NONE;
		_ccv_cnnp_model_gradient_init(model, CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES, fits, fit_size);
	}
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
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
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->tensors.trainables, trainable_size, parallel_count, tensor_binds);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->updated_trainables, compiled_data->tensors.trainables, trainable_size, parallel_count, tensor_binds);
	const int retainable_size = compiled_data->retainables->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, 0), compiled_data->tensors.retainables, retainable_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, 0), compiled_data->tensors.retainables, retainable_size, parallel_count, tensor_binds);
	ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	ccv_array_free(tensor_binds);
	if (tensors_init && parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->tensors_init.v, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->tensors.trainables, compiled_data->trainables->rnum, parallel_count);
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
	const int evaluate_to_size = compiled_data->evaluate.to_size;
	compiled_data->evaluate.to_op_size = 0;
	for (i = 0; i < evaluate_to_size; i++)
	{
		ccv_nnc_graph_exec_t const to = ccv_nnc_graph_exec_from_symbol(compiled_data->graph_exec_arena, compiled_data->evaluate.tos[i]);
		if (to.graph)
			compiled_data->evaluate.to_ops[compiled_data->evaluate.to_op_size++] = to;
	}
	ccv_nnc_graph_static_schedule(compiled_data->graph, compiled_data->stream_type);
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

static void _ccv_cnnp_bind_tensors_to_arena(ccv_nnc_tensor_arena_t* const tensor_arena, const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const tensor_symbols, ccv_nnc_tensor_t* const* const tensors, const int tensor_size, const int parallel_count)
{
	int i, j;
	for (i = 0; i < tensor_size; i++)
	{
		ccv_nnc_tensor_bind_symbol(tensor_arena, tensor_symbols[i], tensors[i]);
		for (j = 1; j < parallel_count; j++)
		{
			const ccv_nnc_tensor_symbol_t copy = ccv_nnc_tensor_symbol_copy(graph, tensor_symbols[i], j);
			if (copy.d != CCV_NNC_NO_TENSOR_SYMBOL)
				ccv_nnc_tensor_bind_symbol(tensor_arena, copy, tensors[i + tensor_size * j]);
		}
	}
}

void ccv_cnnp_model_fit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const fits, const int fit_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
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
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init(model->graph, compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	assert((input_size % parallel_count) == 0);
	assert((output_size % parallel_count) == 0);
	const int input_size_per_p = input_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->inputs, inputs, input_size_per_p, parallel_count, tensor_binds);
	const int output_size_per_p = output_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->outputs, outputs, output_size_per_p, parallel_count, tensor_binds);
	const int trainable_size = compiled_data->trainables->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->tensors.trainables, trainable_size, parallel_count, tensor_binds);
	const int retainable_size = compiled_data->retainables->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, 0), compiled_data->tensors.retainables, retainable_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, 0), compiled_data->tensors.retainables, retainable_size, parallel_count, tensor_binds);
	// If we generated gradient for the graph, only compile part of the graph because the rest is irrelevant for evaluation.
	if (compiled_data->gradient_mode != CCV_CNNP_COMPILED_DATA_GRADIENT_NONE)
		ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), compiled_data->evaluate.tos, compiled_data->evaluate.to_size, &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	else {
		assert(compiled_data->parallel_count <= 1); // I don't know how to handle parallel_count larger than 1.
		ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	}
	ccv_array_free(tensor_binds);
	// If tensor is not init'ed, we need to init states first.
	if (tensors_init && parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->tensors_init.v, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->tensors.trainables, compiled_data->trainables->rnum, parallel_count);
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
	ccv_nnc_graph_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_nnc_graph_autotune(compiled_data->graph, compiled_data->workspace_size, 0, TRAVERSE_FULL);
}

static void _ccv_cnnp_model_gradient_tensors_init(const ccv_nnc_symbolic_graph_t* const graph, ccv_cnnp_compiled_data_t* const compiled_data)
{
	assert(!compiled_data->tensors.gradients);
	const int trainable_size = compiled_data->trainables->rnum;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	compiled_data->tensors.gradients = (ccv_nnc_tensor_t**)ccmalloc(sizeof(ccv_nnc_tensor_t*) * trainable_size * 2 * parallel_count);
	compiled_data->tensors.accum_gradients = compiled_data->tensors.gradients + trainable_size * parallel_count;
	int i, j;
	for (i = 0; i < trainable_size; i++)
	{
		const ccv_nnc_tensor_symbol_t trainable = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, i);
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(trainable.graph, trainable);
		CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
		compiled_data->tensors.gradients[i] = ccv_nnc_tensor_new(0, info, 0);
		compiled_data->tensors.accum_gradients[i] = 0; // delay the accumulated gradient allocation until when we need it.
		for (j = 1; j < parallel_count; j++)
		{
			CCV_TENSOR_SET_DEVICE_ID(info.type, j);
			compiled_data->tensors.gradients[i + j * trainable_size] = ccv_nnc_tensor_new(0, info, 0);
			compiled_data->tensors.accum_gradients[i + j * trainable_size] = 0;
		}
	}
}

// Compile the graph to run ccv_cnnp_model_evaluate with requires_grad = true (MULTISTAGE_MODE).
// Particularly, this method compiles the evaluation and backprop graph (the main graph).
static void _ccv_cnnp_model_multistage_jit_0(ccv_cnnp_model_t* const model, const int disable_outgrad, const int is_test, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i, j;
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	const int target_gradient_mode = disable_outgrad ? CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES : CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS;
	assert(!compiled_data->graph || compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE || compiled_data->gradient_mode != target_gradient_mode);
	compiled_data->graph_mode = CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(output_size > 0);
	// There shouldn't be a loss function if we evaluate with multistage jit.
	assert(compiled_data->loss.cmd == CCV_NNC_NOOP);
	if (compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_NONE)
		_ccv_cnnp_model_gradient_init(model, target_gradient_mode, 0, 0); // The type of outputs and fits should be the same. We only use type here.
	else if (compiled_data->gradient_mode != target_gradient_mode) {
		_ccv_cnnp_model_rewind_graph(model);
		_ccv_cnnp_compiled_data_gradient_free(compiled_data);
		compiled_data->gradient_mode = CCV_CNNP_COMPILED_DATA_GRADIENT_NONE;
		_ccv_cnnp_model_gradient_init(model, target_gradient_mode, 0, 0); // The type of outputs and fits should be the same. We only use type here.
	}
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init(model->graph, compiled_data);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	assert((input_size % parallel_count) == 0);
	assert((output_size % parallel_count) == 0);
	const int input_size_per_p = input_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->inputs, inputs, input_size_per_p, parallel_count, tensor_binds);
	const int output_size_per_p = output_size / parallel_count;
	_ccv_cnnp_model_bind_tensors(model->graph, model->outputs, outputs, output_size_per_p, parallel_count, tensor_binds);
	const int trainable_size = compiled_data->trainables->rnum;
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->tensors.trainables, trainable_size, parallel_count, tensor_binds);
	const int retainable_size = compiled_data->retainables->rnum;
	_ccv_cnnp_model_remove_nocopies(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, 0), compiled_data->tensors.retainables, retainable_size, parallel_count);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->retainables, 0), compiled_data->tensors.retainables, retainable_size, parallel_count, tensor_binds);
	if (!compiled_data->tensors.gradients)
		_ccv_cnnp_model_gradient_tensors_init(model->graph, compiled_data);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->gradients, compiled_data->tensors.gradients, trainable_size, parallel_count, tensor_binds);
	ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(model->graph), compiled_data->backward.tos, compiled_data->backward.to_size, &compiled_data->graph, &compiled_data->tensor_arena, &compiled_data->graph_exec_arena);
	ccv_array_free(tensor_binds);
	if (tensors_init && parallel_count > 1)
		_ccv_cnnp_model_copy_tensors(compiled_data->tensors_init.v, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->tensors.trainables, compiled_data->trainables->rnum, parallel_count);
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
	ccv_cnnp_model_set_is_test(model, is_test, _ccv_cnnp_cmd_update_for_execs, compiled_data->graph_exec_arena);
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
	ccv_nnc_graph_static_schedule(compiled_data->graph, compiled_data->stream_type);
	ccv_nnc_graph_autotune(compiled_data->graph, compiled_data->workspace_size, 0, TRAVERSE_FULL);
}

void ccv_cnnp_model_evaluate(ccv_cnnp_model_t* const model, const ccv_cnnp_evaluate_param_t params, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(output_size == model->output_size * parallel_count);
	assert(input_size == model->input_size * parallel_count);
	assert(model->graph);
	const int target_gradient_mode = params.disable_outgrad ? CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES : CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS;
	if (!compiled_data->graph ||
		(params.requires_grad && (compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE || compiled_data->gradient_mode != target_gradient_mode)) ||
		// If a stream context is provided, we need to recompile because we cannot run them efficiently in FIT_MODE.
		(stream_context && !params.requires_grad && compiled_data->graph_mode != CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE_NO_GRAD))
	{
		_ccv_cnnp_compiled_data_graph_free(compiled_data);
		_ccv_cnnp_compiled_data_backward_free(compiled_data);
		_ccv_cnnp_compiled_data_apply_gradients_free(compiled_data);
		if (params.requires_grad)
			_ccv_cnnp_model_multistage_jit_0(model, params.disable_outgrad, params.is_test, inputs, input_size, outputs, output_size);
		else
			_ccv_cnnp_model_multistage_no_grad_jit(model, inputs, input_size, outputs, output_size);
	} else {
		assert((input_size % parallel_count) == 0);
		assert((output_size % parallel_count) == 0);
		const int input_size_per_p = input_size / parallel_count;
		_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, model->inputs, inputs, input_size_per_p, parallel_count);
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
		ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, TRAVERSE_FULL);
	else
		ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, 0, 0,
			compiled_data->evaluate.to_ops, compiled_data->evaluate.to_op_size);
}

// Compile the graph to run ccv_cnnp_model_backward after ccv_cnnp_model_evaluate with requires_grad = true (MULTISTAGE_MODE).
// Particularly, this method compiles the accumulator graph.
static void _ccv_cnnp_model_multistage_jit_1(ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	ccv_nnc_symbolic_graph_t* accum = ccv_nnc_symbolic_graph_new();
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	const int trainable_size = compiled_data->trainables->rnum;
	int i, j;
	compiled_data->backward.gradients = (ccv_nnc_tensor_symbol_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * trainable_size * parallel_count * 3);
	compiled_data->backward.accum_gradients = compiled_data->backward.gradients + trainable_size * parallel_count;
	compiled_data->backward.updated_accum_gradients = compiled_data->backward.accum_gradients + trainable_size * parallel_count;
	for (i = 0; i < trainable_size; i++)
		for (j = 0; j < parallel_count; j++)
		{
			const ccv_nnc_tensor_param_t info = compiled_data->tensors.gradients[i + j * trainable_size]->info;
			// Now, the old gradient is the accumulated gradient, getting new gradient tensor setup so we can collect them.
			compiled_data->tensors.accum_gradients[i + j * trainable_size] = compiled_data->tensors.gradients[i + j * trainable_size];
			compiled_data->tensors.gradients[i + j * trainable_size] = ccv_nnc_tensor_new(0, info, 0);
			ccv_nnc_tensor_symbol_t inputs[2];
			inputs[0] = compiled_data->backward.accum_gradients[i + j * trainable_size] = ccv_nnc_tensor_symbol_new(accum, info, 0);
			inputs[1] = compiled_data->backward.gradients[i + j * trainable_size] = ccv_nnc_tensor_symbol_new(accum, info, 0);
			ccv_nnc_tensor_symbol_t output = compiled_data->backward.updated_accum_gradients[i + j * trainable_size] = ccv_nnc_tensor_symbol_new(accum, info, 0);
			ccv_nnc_graph_exec_symbol_new(accum, CMD_EWSUM_FORWARD(), inputs, 2, &output, 1, 0);
		}
	ccv_nnc_graph_exec_symbol_autogen(accum, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	_ccv_cnnp_model_bind_tensors(accum, compiled_data->backward.accum_gradients, compiled_data->tensors.accum_gradients, trainable_size * parallel_count, 1, tensor_binds);
	_ccv_cnnp_model_bind_tensors(accum, compiled_data->backward.gradients, compiled_data->tensors.gradients, trainable_size * parallel_count, 1, tensor_binds);
	_ccv_cnnp_model_bind_tensors(accum, compiled_data->backward.updated_accum_gradients, compiled_data->tensors.accum_gradients, trainable_size * parallel_count, 1, tensor_binds);
	ccv_nnc_symbolic_graph_compile(accum, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, SYMBOLIC_GRAPH_SOURCES(accum), SYMBOLIC_GRAPH_DESTINATIONS(accum), &compiled_data->backward.accum, &compiled_data->backward.tensor_arena, &compiled_data->backward.graph_exec_arena);
	ccv_nnc_symbolic_graph_free(accum);
	ccv_nnc_graph_static_schedule(compiled_data->backward.accum, compiled_data->stream_type);
	ccv_array_free(tensor_binds);
}

void ccv_cnnp_model_backward(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const ingrads, const int ingrad_size, ccv_nnc_tensor_t* const* const outgrads, const int outgrad_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(ingrad_size == model->output_size * parallel_count);
	if (outgrad_size > 0)
		{ assert(outgrad_size == model->input_size * parallel_count); }
	assert(model->graph);
	assert(compiled_data->graph);
	const int trainable_size = compiled_data->trainables->rnum;
	// If we need to accumulate the gradients now, do jit on accumulator.
	if (compiled_data->backward.count > 0)
	{
		if (!compiled_data->backward.accum)
			_ccv_cnnp_model_multistage_jit_1(model);
		else {
			// Otherwise, we need to switch accumulated gradients with gradients (so we can do accumulation properly).
			int i;
			for (i = 0; i < trainable_size * parallel_count; i++)
			{
				ccv_nnc_tensor_t* tensor;
				CCV_SWAP(compiled_data->tensors.accum_gradients[i], compiled_data->tensors.gradients[i], tensor);
			}
			// Do rebind in case we messed up the binding (we switch accum_gradients and gradients).
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->backward.tensor_arena, 0, compiled_data->backward.gradients, compiled_data->tensors.gradients, trainable_size * parallel_count, 1);
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->backward.tensor_arena, 0, compiled_data->backward.accum_gradients, compiled_data->tensors.accum_gradients, trainable_size * parallel_count, 1);
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->backward.tensor_arena, 0, compiled_data->backward.updated_accum_gradients, compiled_data->tensors.accum_gradients, trainable_size * parallel_count, 1);
		}
	}
	const int ingrad_size_per_p = ingrad_size / parallel_count;
	const int outgrad_size_per_p = outgrad_size / parallel_count;
	int i, j;
	for (i = 0; i < ingrad_size_per_p; i++)
	{
		const ccv_nnc_tensor_symbol_t ingrad = ccv_nnc_tensor_symbol_for_backward(model->graph, model->outputs[i]);
		ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ingrad, ingrads[i]);
		for (j = 1; j < parallel_count; j++)
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, ingrad, j), ingrads[i + ingrad_size_per_p * j]);
	}
	if (outgrad_size > 0)
	{
		assert(compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS && "shouldn't pass disable_outgrad to ccv_cnnp_model_evaluate before if you plan to compute outgrad");
		for (i = 0; i < outgrad_size_per_p; i++)
		{
			const ccv_nnc_tensor_symbol_t outgrad = ccv_nnc_tensor_symbol_for_backward(model->graph, model->inputs[i]);
			ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, outgrad, outgrads[i]);
			for (j = 1; j < parallel_count; j++)
				ccv_nnc_tensor_bind_symbol(compiled_data->tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, outgrad, j), outgrads[i + outgrad_size_per_p * j]);
		}
	} else {
		assert(compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES ||
			compiled_data->gradient_mode == CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS);
	}
	// Bind to the gradients (if we start to accumulate at 2 (i.e. accum_gradients and gradients binding no longer changes), no need to do the binding.
	if (compiled_data->backward.count <= 1)
		_ccv_cnnp_bind_tensors_to_arena(compiled_data->tensor_arena, model->graph, compiled_data->gradients, compiled_data->tensors.gradients, trainable_size, parallel_count);
	// Run the backward pass.
	ccv_nnc_graph_run(compiled_data->graph, 0, stream_context, 0, compiled_data->backward.from_ops, compiled_data->backward.from_op_size, 0, 0);
	// If we need to run accumulation round, do that now.
	if (compiled_data->backward.count > 0)
		ccv_nnc_graph_run(compiled_data->backward.accum, 0, stream_context, 0, TRAVERSE_FULL);
	// Update the count, this determines whether we need to accumulate or not.
	++compiled_data->backward.count;
}

// Compile the graph to run ccv_cnnp_model_apply_gradients after ccv_cnnp_model_backward (MULTISTAGE_MODE).
// Particularly, this method compiles the trainable update graph.
static void _ccv_cnnp_model_multistage_jit_2(ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	const int trainable_size = compiled_data->trainables->rnum;
	ccv_array_t* const tensor_binds = ccv_array_new(sizeof(ccv_nnc_tensor_bind_t), 0, 0);
	_ccv_cnnp_model_bind_tensors(model->graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->trainables, 0), compiled_data->tensors.trainables, trainable_size, parallel_count, tensor_binds);
	_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->updated_trainables, compiled_data->tensors.trainables, trainable_size, parallel_count, tensor_binds);
	// Bind accumulated gradients.
	if (compiled_data->backward.count > 1)
		_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->gradients, compiled_data->tensors.accum_gradients, trainable_size, parallel_count, tensor_binds);
	else
		_ccv_cnnp_model_bind_tensors(model->graph, compiled_data->gradients, compiled_data->tensors.gradients, trainable_size, parallel_count, tensor_binds);
	// TODO: Need to find the start point for this.
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
	ccv_nnc_symbolic_graph_compile(model->graph, (ccv_nnc_tensor_bind_t*)ccv_array_get(tensor_binds, 0), tensor_binds->rnum, 0, 0, froms, from_size, SYMBOLIC_GRAPH_DESTINATIONS(model->graph), &compiled_data->apply_gradients.graph, &compiled_data->apply_gradients.tensor_arena, &compiled_data->apply_gradients.graph_exec_arena);
	ccv_array_free(tensor_binds);
	ccfree(froms);
	const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(compiled_data->minimize.minimizer);
	for (i = 0; i < saved_aux_size * trainable_size; i++)
	{
		ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_symbol(compiled_data->apply_gradients.tensor_arena, compiled_data->saved_aux[i].source);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &tensor, 1, 0);
		for (j = 1; j < parallel_count; j++)
		{
			ccv_nnc_tensor_t* const copy = ccv_nnc_tensor_from_symbol(compiled_data->apply_gradients.tensor_arena, ccv_nnc_tensor_symbol_copy(model->graph, compiled_data->saved_aux[i].source, j));
			if (copy)
				ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, &copy, 1, 0);
		}
	}
	ccv_nnc_graph_static_schedule(compiled_data->apply_gradients.graph, compiled_data->stream_type);
}

void ccv_cnnp_model_apply_gradients(ccv_cnnp_model_t* const model, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	assert(compiled_data->graph_mode == CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	assert(model->graph);
	assert(compiled_data->graph);
	assert(compiled_data->backward.count > 0);
	if (!compiled_data->apply_gradients.graph)
		_ccv_cnnp_model_multistage_jit_2(model);
	else {
		const int trainable_size = compiled_data->trainables->rnum;
		// Change to bind accum_gradients if we do gradient accumulation (run backward more than once).
		if (compiled_data->backward.count > 1)
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->apply_gradients.tensor_arena, model->graph, compiled_data->gradients, compiled_data->tensors.accum_gradients, trainable_size, parallel_count);
		else
			_ccv_cnnp_bind_tensors_to_arena(compiled_data->apply_gradients.tensor_arena, model->graph, compiled_data->gradients, compiled_data->tensors.gradients, trainable_size, parallel_count);
	}
	ccv_nnc_graph_run(compiled_data->apply_gradients.graph, 0, stream_context, 0, TRAVERSE_FULL);
	// Reset backward count to 0.
	compiled_data->backward.count = 0;
}

void ccv_cnnp_model_set_minimizer(ccv_cnnp_model_t* const model, const ccv_nnc_cmd_t minimizer, const ccv_cnnp_model_minimizer_set_f minimizer_setter, const void* const context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	compiled_data->minimize.minimizer = minimizer;
	compiled_data->minimize.setter = minimizer_setter;
	compiled_data->minimize.context = context;
	if (!compiled_data->update_nodes)
		return;
	int i;
	const int trainable_size = compiled_data->trainables->rnum;
	ccv_nnc_graph_exec_symbol_t* const update_nodes = compiled_data->update_nodes;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = model->graph;
	assert(symbolic_graph);
	for (i = 0; i < trainable_size; i++)
		_ccv_cnnp_model_graph_exec_symbol_set(model, update_nodes[i], minimizer);
	// Use the minimizer to update.
	if (!minimizer_setter)
		return;
	_ccv_cnnp_model_set_minimizer_setter(model, minimizer_setter, context);
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
	const int trainable_size = compiled_data->trainables->rnum;
	ccv_array_free(compiled_data->trainables);
	const int retainable_size = compiled_data->retainables->rnum;
	ccv_array_free(compiled_data->retainables);
	assert(compiled_data->ids.trainables->rnum == trainable_size);
	assert(compiled_data->ids.retainables->rnum == retainable_size);
	for (i = 0; i < trainable_size; i++)
		ccfree(*(char**)ccv_array_get(compiled_data->ids.trainables, i));
	ccv_array_free(compiled_data->ids.trainables);
	for (i = 0; i < retainable_size; i++)
		ccfree(*(char**)ccv_array_get(compiled_data->ids.retainables, i));
	ccv_array_free(compiled_data->ids.retainables);
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	if (compiled_data->tensors.trainables)
	{
		for (i = 0; i < trainable_size * parallel_count; i++)
			ccv_nnc_tensor_free(compiled_data->tensors.trainables[i]);
		for (i = 0; i < retainable_size * parallel_count; i++)
			if (compiled_data->tensors.retainables[i])
				ccv_nnc_tensor_free(compiled_data->tensors.retainables[i]);
		ccfree(compiled_data->tensors.trainables);
	}
	if (compiled_data->tensors.gradients)
	{
		for (i = 0; i < trainable_size * parallel_count; i++)
		{
			ccv_nnc_tensor_free(compiled_data->tensors.gradients[i]);
			if (compiled_data->tensors.accum_gradients[i])
				ccv_nnc_tensor_free(compiled_data->tensors.accum_gradients[i]);
		}
		ccfree(compiled_data->tensors.gradients);
	}
	if (compiled_data->rewindables)
		ccv_array_free(compiled_data->rewindables);
	if (compiled_data->tensors_init.v)
		ccfree(compiled_data->tensors_init.v);
	_ccv_cnnp_compiled_data_graph_free(compiled_data);
	_ccv_cnnp_compiled_data_gradient_free(compiled_data);
	_ccv_cnnp_compiled_data_backward_free(compiled_data);
	_ccv_cnnp_compiled_data_apply_gradients_free(compiled_data);
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
	if (model->name)
		ccfree(model->name);
	ccfree(model);
}
