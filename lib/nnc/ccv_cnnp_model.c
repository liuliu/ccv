#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

typedef struct {
	void (*init)(ccv_cnnp_model_t* const self); // It can be nil.
	void (*deinit)(ccv_cnnp_model_t* const self); // It can be nil.
	void (*build)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, ccv_nnc_tensor_symbol_t* const output); // Call this graph to build computation. No need to specify input size or output size, as it is defined along in the model already.
} ccv_cnnp_model_vtab_t;

static const ccv_cnnp_model_vtab_t ccv_cnnp_sequential_model_isa;
static const ccv_cnnp_model_vtab_t ccv_cnnp_input_isa;

struct ccv_cnnp_model_s {
	const ccv_cnnp_model_vtab_t* isa;
	int input_size;
	ccv_cnnp_model_io_t output; // The opaque output io that can be nil.
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_tensor_symbol_t output_symbol;
};

struct ccv_cnnp_model_io_s {
	ccv_nnc_tensor_param_t info;
	ccv_cnnp_model_t* model; // Reference back to the model who holds it. This is required because the model is the one whole holds the io.
	ccv_array_t* outgoings; // Array of ccv_cnnp_model_io_t. The order is important because it impacts the order of symbols.
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

static void _ccv_cnnp_sequential_model_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, ccv_nnc_tensor_symbol_t* const output)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	ccv_nnc_tensor_symbol_t input = inputs[0];
	for (i = 0; i < self->sequence_size; i++)
	{
		ccv_nnc_tensor_symbol_t output;
		ccv_cnnp_model_t* const sub_model = self->sequence[i];
		// Go through each sub model to build the graph.
		sub_model->isa->build(sub_model, graph, &input, &output);
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
	ccv_cnnp_sequential_model_t* const sequential_model = (ccv_cnnp_sequential_model_t*)ccmalloc(sizeof(ccv_cnnp_sequential_model_t) + sizeof(ccv_cnnp_model_t*) * (model_size - 1));
	sequential_model->super.isa = &ccv_cnnp_sequential_model_isa;
	sequential_model->super.input_size = 1;
	sequential_model->super.output = 0;
	sequential_model->super.graph = 0;
	sequential_model->sequence_size = model_size;
	memcpy(sequential_model->sequence, models, sizeof(ccv_cnnp_model_t*) * model_size);
	return (ccv_cnnp_model_t*)sequential_model;
}

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_param_t* inputs; // The inputs turned from ccv_cnnp_model_io_t to ccv_nnc_tensor_param_t.
	int input_size;
	int output_offset; // The offset records the models that ends. It will be relevant when compute loss.
	// The name is similar to sequential model, but it is just topological sorted models.
	int sequence_size;
	ccv_cnnp_model_t* sequence[1];
} ccv_cnnp_functional_model_t;

ccv_cnnp_model_t* ccv_cnnp_model_new(const ccv_cnnp_model_io_t* const inputs, const int input_size, ccv_cnnp_model_io_t* const outputs, const int output_size)
{
	int sequence_size = 1;
	ccv_cnnp_functional_model_t* const functional_model = (ccv_cnnp_functional_model_t*)ccmalloc(sizeof(ccv_cnnp_functional_model_t) + sizeof(ccv_cnnp_model_t*) * (sequence_size - 1));
	return (ccv_cnnp_model_t*)functional_model;
}

ccv_cnnp_model_io_t ccv_cnnp_model_apply(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t* const inputs, const int input_size)
{
	if (!model->output)
	{
		model->output = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
		model->output->outgoings = 0;
	}
	model->output->info = ccv_nnc_tensor_auto;
	model->output->model = model;
	if (model->output->outgoings)
		ccv_array_clear(model->output->outgoings); // New outputs.
	int i;
	for (i = 0; i < input_size; i++)
	{
		if (!inputs[i]->outgoings)
			inputs[i]->outgoings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
		ccv_array_push(inputs[i]->outgoings, &model->output);
	}
	return model->output;
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_input_isa = {};

ccv_cnnp_model_io_t ccv_cnnp_model_input(const ccv_nnc_tensor_param_t params)
{
	ccv_cnnp_model_t* const input = (ccv_cnnp_model_t*)ccmalloc(sizeof(ccv_cnnp_model_t));
	input->isa = &ccv_cnnp_input_isa;
	input->input_size = 0;
	input->graph = 0;
	input->output = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
	input->output->outgoings = 0;
	input->output->model = input;
	input->output->info = params;
	return input->output;
}

void ccv_cnnp_model_free(ccv_cnnp_model_t* const model)
{
	if (model->isa->deinit)
		model->isa->deinit(model);
	if (model->output)
	{
		if (model->output->outgoings)
			ccv_array_free(model->output->outgoings);
		ccfree(model->output);
	}
	ccfree(model);
}
