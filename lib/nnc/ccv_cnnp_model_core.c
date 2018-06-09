#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"

static void _ccv_cnnp_model_add_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_symbol_params(graph, inputs[0]), 0);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_EWSUM_FORWARD(), inputs, input_size, outputs, output_size, 0);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_model_add_isa = {
	.build = _ccv_cnnp_model_add_build,
};

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_add_t;

ccv_cnnp_model_t* ccv_cnnp_model_add(void)
{
	ccv_cnnp_model_add_t* const model_add = (ccv_cnnp_model_add_t*)ccmalloc(sizeof(ccv_cnnp_model_add_t));
	model_add->super.isa = &ccv_cnnp_model_add_isa;
	memset(model_add->super.input_dim, 0, sizeof(model_add->super.input_dim));
	model_add->super.io = 0;
	model_add->super.graph = 0;
	model_add->super.outputs = &model_add->output;
	model_add->super.output_size = 1;
	return (ccv_cnnp_model_t*)model_add;
}

static void _ccv_cnnp_model_concat_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_model_concat_isa = {
	.build = _ccv_cnnp_model_concat_build,
};

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_concat_t;

ccv_cnnp_model_t* ccv_cnnp_model_concat(void)
{
	ccv_cnnp_model_concat_t* const model_concat = (ccv_cnnp_model_concat_t*)ccmalloc(sizeof(ccv_cnnp_model_add_t));
	model_concat->super.isa = &ccv_cnnp_model_concat_isa;
	memset(model_concat->super.input_dim, 0, sizeof(model_concat->super.input_dim));
	model_concat->super.io = 0;
	model_concat->super.graph = 0;
	model_concat->super.outputs = &model_concat->output;
	model_concat->super.output_size = 1;
	return (ccv_cnnp_model_t*)model_concat;
}
