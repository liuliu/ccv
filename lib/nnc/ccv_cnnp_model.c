#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

typedef struct {
	void (*init)(ccv_cnnp_model_t* const self); // It can be nil.
	void (*deinit)(ccv_cnnp_model_t* const self); // It can be nil.
	void (*build)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size); // Call this graph to build computation. No need to specify input size or output size, as it is defined along in the model already.
	void (*add_to_trainable)(ccv_cnnp_model_t* const self, ccv_array_t* const trainables); // This is called to add ccv_nnc_tensor_symbol_t to as list of trainables.
} ccv_cnnp_model_vtab_t;

static const ccv_cnnp_model_vtab_t ccv_cnnp_sequential_model_isa;
static const ccv_cnnp_model_vtab_t ccv_cnnp_functional_model_isa;
static const ccv_cnnp_model_vtab_t ccv_cnnp_model_input_isa;

struct ccv_cnnp_model_s {
	const ccv_cnnp_model_vtab_t* isa;
	int input_dim[CCV_NNC_MAX_DIM_ALLOC]; // The input_dim of the model (it may not be applicable if it is a functional model).
	ccv_cnnp_model_io_t io; // The opaque io that can be nil.
	ccv_nnc_symbolic_graph_t* graph;
	int output_size;
	ccv_nnc_tensor_symbol_t* outputs;
};

struct ccv_cnnp_model_io_s {
	uint8_t tbits; // Temporary bits stored in the ccv_cnnp_model_io_t object, whoever uses it should clean it up.
	ccv_nnc_tensor_param_t info;
	ccv_cnnp_model_t* model; // Reference back to the model who holds it. This is required because the model is the one whole holds the io.
	ccv_array_t* incomings; // Array of ccv_cnnp_model_io_t. The order is important because it impacts the order of symbols.
	ccv_array_t* outgoings; // Array of ccv_cnnp_model_io_t.
};

static inline void _ccv_cnnp_model_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	if (outputs && output_size)
		self->isa->build(self, graph, inputs, input_size, outputs, output_size);
	else {
		self->isa->build(self, graph, inputs, input_size, self->outputs, self->output_size);
	}
}

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
		_ccv_cnnp_model_build(sub_model, graph, &input, 1, &output, 1);
		input = output;
	}
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_sequential_model_isa = {
	.deinit = _ccv_cnnp_sequential_model_deinit,
	.build = _ccv_cnnp_sequential_model_build,
};

ccv_cnnp_model_t* ccv_cnnp_sequential_new(ccv_cnnp_model_t* const* const models, const int model_size)
{
	assert(model_size > 0);
	ccv_cnnp_sequential_model_t* const sequential_model = (ccv_cnnp_sequential_model_t*)ccmalloc(sizeof(ccv_cnnp_sequential_model_t) + sizeof(ccv_cnnp_model_t*) * (model_size - 1) + sizeof(ccv_nnc_tensor_symbol_t));
	sequential_model->super.isa = &ccv_cnnp_sequential_model_isa;
	memcpy(sequential_model->super.input_dim, models[0]->input_dim, sizeof(sequential_model->super.input_dim));
	sequential_model->super.io = 0;
	sequential_model->super.graph = 0;
	sequential_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(sequential_model->sequence + model_size);
	sequential_model->super.output_size = 1;
	sequential_model->sequence_size = model_size;
	memcpy(sequential_model->sequence, models, sizeof(ccv_cnnp_model_t*) * model_size);
	return (ccv_cnnp_model_t*)sequential_model;
}

typedef struct {
	ccv_cnnp_model_t super;
	int input_size;
	int output_offset; // The offset records the models that ends. It will be relevant when compute loss.
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
	for (i = 0; i < self->input_size; i++)
		self->sequence[i]->model->outputs[0] = inputs[i]; // Assigning the output symbol of input layer to be the input symbol.
	ccv_array_t* input_symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 1, 0);
	for (i = self->input_size; i < self->sequence_size; i++)
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
		_ccv_cnnp_model_build(sub_model, graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(input_symbols, 0), input_symbols->rnum, 0, 0);
	}
	ccv_array_free(input_symbols);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_functional_model_isa = {
	.deinit = _ccv_cnnp_functional_model_deinit,
	.build = _ccv_cnnp_functional_model_build,
};

#define CCV_CNNP_IS_MODEL_INPUT(x) ((x)->isa == &ccv_cnnp_model_input_isa)

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
				if (!CCV_CNNP_IS_MODEL_INPUT(input->model))
					ccv_array_push(reverse_top, &input);
				else
					++input_count;
			}
	}
	assert(input_count == input_size); // Assuming they all match.
	const int sequence_size = reverse_top->rnum + input_size;
	ccv_cnnp_functional_model_t* const functional_model = (ccv_cnnp_functional_model_t*)ccmalloc(sizeof(ccv_cnnp_functional_model_t) + sizeof(ccv_cnnp_model_t*) * (sequence_size - 1));
	functional_model->super.isa = &ccv_cnnp_functional_model_isa;
	memset(functional_model->super.input_dim, 0, sizeof(functional_model->super.input_dim));
	functional_model->super.graph = 0;
	functional_model->super.io = 0;
	functional_model->super.outputs = 0;
	functional_model->input_size = input_size;
	functional_model->sequence_size = sequence_size;
	functional_model->output_offset = functional_model->sequence_size - output_size;
	memcpy(functional_model->sequence, inputs, sizeof(ccv_cnnp_model_io_t) * input_size);
	for (i = 0; i < reverse_top->rnum; i++)
		functional_model->sequence[input_size + i] = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, reverse_top->rnum - 1 - i);
	ccv_array_free(reverse_top);
	return (ccv_cnnp_model_t*)functional_model;
}

ccv_cnnp_model_io_t ccv_cnnp_model_apply(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t* const inputs, const int input_size)
{
	if (!model->io)
	{
		model->io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
		model->io->incomings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
		model->io->outgoings = 0;
	}
	model->io->info = ccv_nnc_tensor_auto;
	model->io->model = model;
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

static const ccv_cnnp_model_vtab_t ccv_cnnp_model_input_isa = {};

ccv_cnnp_model_io_t ccv_cnnp_model_input(const ccv_nnc_tensor_param_t params)
{
	ccv_cnnp_model_t* const input = (ccv_cnnp_model_t*)ccmalloc(sizeof(ccv_cnnp_model_t) + sizeof(ccv_nnc_tensor_symbol_t));
	input->isa = &ccv_cnnp_model_input_isa;
	memcpy(input->input_dim, params.dim, sizeof(input->input_dim));
	input->graph = 0;
	input->io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
	input->io->outgoings = 0;
	input->io->model = input;
	input->io->info = params;
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
	ccfree(model);
}
