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

static void _ccv_cnnp_sequential_model_add_to_trainable(ccv_cnnp_model_t* const super, ccv_array_t* const trainables)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_trainable(self->sequence[i], trainables);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_sequential_model_isa = {
	.deinit = _ccv_cnnp_sequential_model_deinit,
	.build = _ccv_cnnp_sequential_model_build,
	.add_to_trainable = _ccv_cnnp_sequential_model_add_to_trainable,
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

static void _ccv_cnnp_functional_model_add_to_trainable(ccv_cnnp_model_t* const super, ccv_array_t* const trainables)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_trainable(self->sequence[i]->model, trainables);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_functional_model_isa = {
	.deinit = _ccv_cnnp_functional_model_deinit,
	.build = _ccv_cnnp_functional_model_build,
	.add_to_trainable = _ccv_cnnp_functional_model_add_to_trainable,
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
	ccv_cnnp_functional_model_t* const functional_model = (ccv_cnnp_functional_model_t*)cccalloc(1, sizeof(ccv_cnnp_functional_model_t) + sizeof(ccv_cnnp_model_t*) * (sequence_size - 1) + sizeof(ccv_nnc_tensor_symbol_t) * output_size);
	functional_model->super.isa = &ccv_cnnp_functional_model_isa;
	functional_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(functional_model->sequence + sequence_size);
	functional_model->super.output_size = output_size;
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
		ccv_nnc_graph_exec_symbol_autogen(model->graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_symbolic_graph_simplify(model->graph,
			SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
				CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
				CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
			model->outputs, model->output_size,
			SYMBOLIC_GRAPH_SOURCES(model->graph), SYMBOLIC_GRAPH_DESTINATIONS(model->graph));
	}
}

void ccv_cnnp_model_fit(ccv_cnnp_model_t* const model, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
}

void ccv_cnnp_model_dot(const ccv_cnnp_model_t* const model, const int flags, FILE* out)
{
	if (model->graph)
		ccv_nnc_symbolic_graph_dot(model->graph, flags, out);
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
	ccfree(model);
}
