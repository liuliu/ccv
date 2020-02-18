#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"

#pragma mark - Core Layers

static void _ccv_cnnp_add_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_symbol_params(graph, inputs[0]), 0);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_EWSUM_FORWARD(), inputs, input_size, outputs, output_size, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_add_copy(const ccv_cnnp_model_t* const self);

static const ccv_cnnp_model_vtab_t ccv_cnnp_add_isa = {
	.build = _ccv_cnnp_add_build,
	.copy = _ccv_cnnp_add_copy,
};

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_add_t;

ccv_cnnp_model_t* ccv_cnnp_add(const char* const name)
{
	ccv_cnnp_model_add_t* const model_add = (ccv_cnnp_model_add_t*)cccalloc(1, sizeof(ccv_cnnp_model_add_t));
	model_add->super.isa = &ccv_cnnp_add_isa;
	model_add->super.input_size = 1;
	model_add->super.outputs = &model_add->output;
	model_add->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_add->super, name);
	return (ccv_cnnp_model_t*)model_add;
}

static ccv_cnnp_model_t* _ccv_cnnp_add_copy(const ccv_cnnp_model_t* const self)
{
	return ccv_cnnp_add(self->name);
}

static void _ccv_cnnp_concat_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	// TODO: Concatenate is not done yet.
}

static ccv_cnnp_model_t* _ccv_cnnp_concat_copy(const ccv_cnnp_model_t* const self);

static const ccv_cnnp_model_vtab_t ccv_cnnp_concat_isa = {
	.build = _ccv_cnnp_concat_build,
	.copy = _ccv_cnnp_concat_copy,
};

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_concat_t;

ccv_cnnp_model_t* ccv_cnnp_concat(const char* const name)
{
	ccv_cnnp_model_concat_t* const model_concat = (ccv_cnnp_model_concat_t*)cccalloc(1, sizeof(ccv_cnnp_model_concat_t));
	model_concat->super.isa = &ccv_cnnp_concat_isa;
	model_concat->super.input_size = 1;
	model_concat->super.outputs = &model_concat->output;
	model_concat->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_concat->super, name);
	return (ccv_cnnp_model_t*)model_concat;
}

static ccv_cnnp_model_t* _ccv_cnnp_concat_copy(const ccv_cnnp_model_t* const self)
{
	return ccv_cnnp_concat(self->name);
}

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
} ccv_cnnp_model_reshape_t;

static void _ccv_cnnp_reshape_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_reshape_t* const self = (ccv_cnnp_model_reshape_t*)super;
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	assert(ccv_nnc_dimension_count(self->dim) <= ccv_nnc_tensor_count(params));
	memcpy(params.dim, self->dim, sizeof(params.dim));
	outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, inputs[0], self->ofs, self->inc, params, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_reshape_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_reshape_isa = {
	.build = _ccv_cnnp_reshape_build,
	.copy = _ccv_cnnp_reshape_copy,
};

ccv_cnnp_model_t* ccv_cnnp_reshape(const int dim[CCV_NNC_MAX_DIM_ALLOC], const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const char* const name)
{
	ccv_cnnp_model_reshape_t* const model_reshape = (ccv_cnnp_model_reshape_t*)cccalloc(1, sizeof(ccv_cnnp_model_reshape_t));
	model_reshape->super.isa = &ccv_cnnp_reshape_isa;
	model_reshape->super.input_size = 1;
	model_reshape->super.outputs = &model_reshape->output;
	model_reshape->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_reshape->super, name);
	memcpy(model_reshape->dim, dim, sizeof(model_reshape->dim));
	memcpy(model_reshape->ofs, ofs, sizeof(model_reshape->ofs));
	int i, flag = 0;
	for (i = 0; !flag && i < CCV_NNC_MAX_DIM_ALLOC; i++)
		flag = (inc[i] != 0);
	memcpy(model_reshape->inc, flag ? inc : dim, sizeof(model_reshape->inc));
	return (ccv_cnnp_model_t*)model_reshape;
}

static ccv_cnnp_model_t* _ccv_cnnp_reshape_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_reshape_t* const self = (const ccv_cnnp_model_reshape_t*)super;
	return ccv_cnnp_reshape(self->dim, self->ofs, self->inc, self->super.name);
}

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_flatten_t;

static void _ccv_cnnp_flatten_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params = params;
	memset(output_params.dim, 0, sizeof(output_params.dim));
	output_params.dim[0] = ccv_nnc_tensor_get_n(params);
	assert(output_params.dim[0] > 0);
	output_params.dim[1] = ccv_nnc_tensor_count(params) / output_params.dim[0];
	outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, inputs[0], DIM_ALLOC(), output_params.dim, output_params, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_flatten_copy(const ccv_cnnp_model_t* const self);

static const ccv_cnnp_model_vtab_t ccv_cnnp_flatten_isa = {
	.build = _ccv_cnnp_flatten_build,
	.copy = _ccv_cnnp_flatten_copy,
};

ccv_cnnp_model_t* ccv_cnnp_flatten(const char* const name)
{
	ccv_cnnp_model_flatten_t* const model_flatten = (ccv_cnnp_model_flatten_t*)cccalloc(1, sizeof(ccv_cnnp_model_flatten_t));
	model_flatten->super.isa = &ccv_cnnp_flatten_isa;
	model_flatten->super.input_size = 1;
	model_flatten->super.outputs = &model_flatten->output;
	model_flatten->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_flatten->super, name);
	return (ccv_cnnp_model_t*)model_flatten;
}

static ccv_cnnp_model_t* _ccv_cnnp_flatten_copy(const ccv_cnnp_model_t* const self)
{
	return ccv_cnnp_flatten(self->name);
}

#pragma mark - Batch Norm Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t bias;
	ccv_nnc_tensor_symbol_t scale;
	ccv_nnc_graph_exec_symbol_t batch_norm;
	ccv_nnc_cmd_param_t params;
	ccv_array_t* zero_inits;
	ccv_array_t* retainables;
} ccv_cnnp_model_batch_norm_t;

static void _ccv_cnnp_batch_norm_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t bias_params = params;
	memset(bias_params.dim, 0, sizeof(bias_params.dim));
	// If the accuracy is not enough, bump it to 32-bit floating point.
	if (bias_params.datatype != CCV_32F || bias_params.datatype != CCV_64F)
		bias_params.datatype = CCV_32F;
	bias_params.dim[0] = ccv_nnc_tensor_get_c(params);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, params, 0);
	// Both scale and bias are shared between if this model is reused.
	if (!self->scale.graph)
		self->scale = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	if (!self->bias.graph)
		self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	const ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	const ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	// Otherwise, notice mean, var, saved_mean, saved_inv_std are not reused.
	if (!self->zero_inits)
		self->zero_inits = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_push(self->zero_inits, &mean);
	ccv_array_push(self->zero_inits, &var);
	const ccv_nnc_tensor_symbol_t out_mean = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	const ccv_nnc_tensor_symbol_t out_var = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	if (!self->retainables)
		self->retainables = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_push(self->retainables, &out_mean);
	ccv_array_push(self->retainables, &out_var);
	const ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	const ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	const int hw = ccv_nnc_tensor_hw(params, ccv_nnc_tensor_nd(params.dim));
	ccv_nnc_cmd_param_t batch_norm = self->params;
	batch_norm.bnorm.count = hw >= 0 ? CCV_NNC_MAX_DIM + 1 : 1;
	int i;
	batch_norm.bnorm.axis[0] = (params.format == CCV_TENSOR_FORMAT_CHWN) ? 3 : 0;
	if (hw >= 0)
		for (i = 0; i < CCV_NNC_MAX_DIM; i++)
			batch_norm.bnorm.axis[i + 1] = i + hw;
	self->params = batch_norm;
	self->batch_norm = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_BATCH_NORM_FORWARD, 0, batch_norm, 0), TENSOR_SYMBOL_LIST(inputs[0], self->scale, self->bias, mean, var), TENSOR_SYMBOL_LIST(output, out_mean, out_var, saved_mean, saved_inv_std), 0);
	outputs[0] = output;
}

static void _ccv_cnnp_batch_norm_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	if (self->bias.graph)
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
	if (self->scale.graph)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(0, 1), ccv_nnc_no_hint, 0, 0, self->scale);
	int i;
	if (self->zero_inits)
		for (i = 0; i < self->zero_inits->rnum; i++)
			initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, *(ccv_nnc_tensor_symbol_t*)ccv_array_get(self->zero_inits, i));
}

static void _ccv_cnnp_batch_norm_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	if (self->bias.graph)
		add_to_array(parameters, self->bias);
	if (self->scale.graph)
		add_to_array(parameters, self->scale);
}

static void _ccv_cnnp_batch_norm_add_to_output(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	int i;
	if (self->retainables)
		for (i = 0; i < self->retainables->rnum; i++)
		{
			const ccv_nnc_tensor_symbol_t symbol = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(self->retainables, i);
			add_to_array(outputs, symbol);
		}
}

static void _ccv_cnnp_batch_norm_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	if (self->batch_norm.graph)
	{
		self->params.bnorm.is_test = is_test;
		updater(context, self->batch_norm, ccv_nnc_cmd(CCV_NNC_BATCH_NORM_FORWARD, 0, self->params, 0), ccv_nnc_no_hint);
	}
}

static void _ccv_cnnp_batch_norm_deinit(ccv_cnnp_model_t* const super)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	if (self->zero_inits)
		ccv_array_free(self->zero_inits);
	if (self->retainables)
		ccv_array_free(self->retainables);
}

static ccv_cnnp_model_t* _ccv_cnnp_batch_norm_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_batch_norm_isa = {
	.build = _ccv_cnnp_batch_norm_build,
	.init_states = _ccv_cnnp_batch_norm_init_states,
	.add_to_parameter = _ccv_cnnp_batch_norm_add_to_parameter,
	.add_to_output = _ccv_cnnp_batch_norm_add_to_output,
	.copy = _ccv_cnnp_batch_norm_copy,
	.set_is_test = _ccv_cnnp_batch_norm_set_is_test,
	.deinit = _ccv_cnnp_batch_norm_deinit,
};

ccv_cnnp_model_t* ccv_cnnp_batch_norm(const float momentum, const float epsilon, const char* const name)
{
	ccv_cnnp_model_batch_norm_t* const model_batch_norm = (ccv_cnnp_model_batch_norm_t*)cccalloc(1, sizeof(ccv_cnnp_model_batch_norm_t));
	model_batch_norm->super.isa = &ccv_cnnp_batch_norm_isa;
	model_batch_norm->super.input_size = 1;
	model_batch_norm->super.outputs = &model_batch_norm->output;
	model_batch_norm->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_batch_norm->super, name);
	model_batch_norm->scale.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_batch_norm->scale.graph = 0;
	model_batch_norm->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_batch_norm->bias.graph = 0;
	model_batch_norm->params.bnorm.momentum = momentum;
	model_batch_norm->params.bnorm.epsilon = epsilon;
	return (ccv_cnnp_model_t*)model_batch_norm;
}

static ccv_cnnp_model_t* _ccv_cnnp_batch_norm_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_batch_norm_t* const self = (const ccv_cnnp_model_batch_norm_t*)super;
	return ccv_cnnp_batch_norm(self->params.bnorm.momentum, self->params.bnorm.epsilon, self->super.name);
}

#pragma mark - Convolution Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t weights;
	ccv_nnc_tensor_symbol_t bias;
	ccv_nnc_tensor_symbol_t scale;
	int groups;
	int filters;
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_cnnp_param_t params;
} ccv_cnnp_model_convolution_t;

static void _ccv_cnnp_convolution_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_convolution_t* const self = (ccv_cnnp_model_convolution_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	int i;
	const int nd = CCV_NNC_MAX_DIM + 2;
	ccv_nnc_tensor_param_t weights_params = params;
	ccv_nnc_tensor_set_n(&weights_params, self->filters);
	assert(ccv_nnc_tensor_get_c(params) % self->groups == 0);
	ccv_nnc_tensor_set_c(&weights_params, nd, ccv_nnc_tensor_get_c(params) / self->groups);
	const int hw = ccv_nnc_tensor_hw(weights_params, nd);
	assert(hw >= 0);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		weights_params.dim[i + hw] = self->kdim[i];
	if (!self->weights.graph)
		self->weights = ccv_nnc_tensor_symbol_new(graph, weights_params, 0);
	assert(self->weights.graph == graph);
	ccv_nnc_tensor_param_t bias_params = params;
	memset(bias_params.dim, 0, sizeof(bias_params.dim));
	bias_params.dim[0] = self->filters;
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(self->groups, self->filters);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		cmd.info.size.dim[i] = self->kdim[i];
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_hint_tensor_auto(cmd, (ccv_nnc_tensor_param_t []){
			params,
			weights_params,
			bias_params,
		}, 3, self->params.hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_t convolution;
	if (self->params.no_bias)
		convolution = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights), TENSOR_SYMBOL_LIST(output), 0);
	else {
		if (!self->bias.graph)
			self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
		convolution = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights, self->bias), TENSOR_SYMBOL_LIST(output), 0);
	}
	ccv_nnc_graph_exec_symbol_set_hint(graph, convolution, self->params.hint);
	outputs[0] = output;
}

static void _ccv_cnnp_convolution_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_convolution_t* const self = (ccv_cnnp_model_convolution_t*)super;
	const ccv_nnc_tensor_param_t weight_params = ccv_nnc_tensor_symbol_params(graph, self->weights);
	const int n = ccv_max(ccv_nnc_tensor_get_n(weight_params), 1);
	const int count = ccv_nnc_tensor_count(weight_params);
	const float std = sqrtf(2) / sqrtf(count / n);
	const float bound = sqrtf(3) * std;
	initializer(context, CMD_RANDOM_UNIFORM_FORWARD(-bound, bound), ccv_nnc_no_hint, 0, 0, self->weights);
	if (self->bias.graph)
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
	if (self->scale.graph)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(0, 1), ccv_nnc_no_hint, 0, 0, self->scale);
}

static void _ccv_cnnp_convolution_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters)
{
	ccv_cnnp_model_convolution_t* const self = (ccv_cnnp_model_convolution_t*)super;
	add_to_array(parameters, self->weights);
	if (self->bias.graph)
		add_to_array(parameters, self->bias);
	if (self->scale.graph)
		add_to_array(parameters, self->scale);
}

static ccv_cnnp_model_t* _ccv_cnnp_convolution_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_convolution_isa = {
	.build = _ccv_cnnp_convolution_build,
	.init_states = _ccv_cnnp_convolution_init_states,
	.add_to_parameter = _ccv_cnnp_convolution_add_to_parameter,
	.copy = _ccv_cnnp_convolution_copy,
};

ccv_cnnp_model_t* ccv_cnnp_convolution(const int groups, const int filters, const int kdim[CCV_NNC_MAX_DIM_ALLOC], const ccv_cnnp_param_t params, const char* const name)
{
	ccv_cnnp_model_convolution_t* const model_convolution = (ccv_cnnp_model_convolution_t*)cccalloc(1, sizeof(ccv_cnnp_model_convolution_t));
	model_convolution->super.isa = &ccv_cnnp_convolution_isa;
	model_convolution->super.input_size = 1;
	model_convolution->super.outputs = &model_convolution->output;
	model_convolution->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_convolution->super, name);
	model_convolution->weights.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_convolution->weights.graph = 0;
	model_convolution->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_convolution->bias.graph = 0;
	model_convolution->groups = groups;
	model_convolution->filters = filters;
	memcpy(model_convolution->kdim, kdim, sizeof(model_convolution->kdim));
	model_convolution->params = params;
	return (ccv_cnnp_model_t*)model_convolution;
}

static ccv_cnnp_model_t* _ccv_cnnp_convolution_copy(const ccv_cnnp_model_t* const super)
{
	ccv_cnnp_model_convolution_t* const self = (ccv_cnnp_model_convolution_t*)super;
	return ccv_cnnp_convolution(self->groups, self->filters, self->kdim, self->params, self->super.name);
}

#pragma mark - Dense Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t weights;
	ccv_nnc_tensor_symbol_t bias;
	ccv_nnc_tensor_symbol_t scale;
	int count;
	ccv_cnnp_param_t params;
} ccv_cnnp_model_dense_t;

static void _ccv_cnnp_dense_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_dense_t* const self = (ccv_cnnp_model_dense_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t weights_params = params;
	memset(weights_params.dim, 0, sizeof(weights_params.dim));
	weights_params.dim[0] = self->count;
	weights_params.dim[1] = params.dim[ccv_nnc_tensor_nd(params.dim) - 1];
	if (!self->weights.graph)
		self->weights = ccv_nnc_tensor_symbol_new(graph, weights_params, 0);
	assert(self->weights.graph == graph);
	ccv_nnc_tensor_param_t bias_params = params;
	memset(bias_params.dim, 0, sizeof(bias_params.dim));
	bias_params.dim[0] = self->count;
	const ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_hint_tensor_auto(cmd, (ccv_nnc_tensor_param_t []){
			params,
			weights_params,
			bias_params,
		}, 3, self->params.hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	if (self->params.no_bias)
		ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights), TENSOR_SYMBOL_LIST(output), 0);
	else {
		if (!self->bias.graph)
			self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
		ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights, self->bias), TENSOR_SYMBOL_LIST(output), 0);
	}
	outputs[0] = output;
}

static void _ccv_cnnp_dense_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_dense_t* const self = (ccv_cnnp_model_dense_t*)super;
	const ccv_nnc_tensor_param_t weight_params = ccv_nnc_tensor_symbol_params(graph, self->weights);
	const int c = weight_params.dim[1];
	const float std = sqrtf(2) / sqrtf(c);
	const float bound = sqrtf(3) * std;
	initializer(context, CMD_RANDOM_UNIFORM_FORWARD(-bound, bound), ccv_nnc_no_hint, 0, 0, self->weights);
	if (self->bias.graph)
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
	if (self->scale.graph)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(0, 1), ccv_nnc_no_hint, 0, 0, self->scale);
}

static void _ccv_cnnp_dense_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters)
{
	ccv_cnnp_model_dense_t* const self = (ccv_cnnp_model_dense_t*)super;
	add_to_array(parameters, self->weights);
	if (self->bias.graph)
		add_to_array(parameters, self->bias);
	if (self->scale.graph)
		add_to_array(parameters, self->scale);
}

static ccv_cnnp_model_t* _ccv_cnnp_dense_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_dense_isa = {
	.build = _ccv_cnnp_dense_build,
	.init_states = _ccv_cnnp_dense_init_states,
	.add_to_parameter = _ccv_cnnp_dense_add_to_parameter,
	.copy = _ccv_cnnp_dense_copy,
};

ccv_cnnp_model_t* ccv_cnnp_dense(const int count, const ccv_cnnp_param_t params, const char* const name)
{
	ccv_cnnp_model_dense_t* const model_dense = (ccv_cnnp_model_dense_t*)cccalloc(1, sizeof(ccv_cnnp_model_dense_t));
	model_dense->super.isa = &ccv_cnnp_dense_isa;
	model_dense->super.input_size = 1;
	model_dense->super.outputs = &model_dense->output;
	model_dense->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_dense->super, name);
	model_dense->weights.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_dense->weights.graph = 0;
	model_dense->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_dense->bias.graph = 0;
	model_dense->count = count;
	model_dense->params = params;
	return (ccv_cnnp_model_t*)model_dense;
}

static ccv_cnnp_model_t* _ccv_cnnp_dense_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_dense_t* const self = (const ccv_cnnp_model_dense_t*)super;
	return ccv_cnnp_dense(self->count, self->params, self->super.name);
}

#pragma mark - Pool Layers

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_cnnp_param_t params;
} ccv_cnnp_model_pool_t;

static void _ccv_cnnp_max_pool_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_pool_t* const self = (ccv_cnnp_model_pool_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	const int hw = ccv_nnc_tensor_hw(params, ccv_nnc_tensor_nd(params.dim));
	ccv_nnc_cmd_t cmd;
	if (hw >= 0 && self->kdim[0] == 0 && self->kdim[1] == 0)
		cmd = CMD_MAX_POOL_FORWARD(params.dim[hw], params.dim[hw + 1]);
	else
		cmd = CMD_MAX_POOL_FORWARD(self->kdim[0], self->kdim[1]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_hint_tensor_auto(cmd, &params, 1, self->params.hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t pool_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	const ccv_nnc_graph_exec_symbol_t exec = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(pool_output), 0);
	ccv_nnc_graph_exec_symbol_set_hint(graph, exec, self->params.hint);
	outputs[0] = pool_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_max_pool_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_max_pool_isa = {
	.build = _ccv_cnnp_max_pool_build,
	.copy = _ccv_cnnp_max_pool_copy,
};

ccv_cnnp_model_t* ccv_cnnp_max_pool(const int kdim[CCV_NNC_MAX_DIM_ALLOC], const ccv_cnnp_param_t params, const char* const name)
{
	ccv_cnnp_model_pool_t* const model_pool = (ccv_cnnp_model_pool_t*)cccalloc(1, sizeof(ccv_cnnp_model_pool_t));
	model_pool->super.isa = &ccv_cnnp_max_pool_isa;
	model_pool->super.input_size = 1;
	model_pool->super.outputs = &model_pool->output;
	model_pool->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_pool->super, name);
	memcpy(model_pool->kdim, kdim, sizeof(model_pool->kdim));
	model_pool->params = params;
	return (ccv_cnnp_model_t*)model_pool;
}

static ccv_cnnp_model_t* _ccv_cnnp_max_pool_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_pool_t* const self = (const ccv_cnnp_model_pool_t*)super;
	return ccv_cnnp_max_pool(self->kdim, self->params, self->super.name);
}

static void _ccv_cnnp_average_pool_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_pool_t* const self = (ccv_cnnp_model_pool_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	const int hw = ccv_nnc_tensor_hw(params, ccv_nnc_tensor_nd(params.dim));
	ccv_nnc_cmd_t cmd;
	if (hw >= 0 && self->kdim[0] == 0 && self->kdim[1] == 0)
		cmd = CMD_AVERAGE_POOL_FORWARD(params.dim[hw], params.dim[hw + 1]);
	else
		cmd = CMD_AVERAGE_POOL_FORWARD(self->kdim[0], self->kdim[1]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_hint_tensor_auto(cmd, &params, 1, self->params.hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t pool_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	const ccv_nnc_graph_exec_symbol_t exec = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(pool_output), 0);
	ccv_nnc_graph_exec_symbol_set_hint(graph, exec, self->params.hint);
	outputs[0] = pool_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_average_pool_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_average_pool_isa = {
	.build = _ccv_cnnp_average_pool_build,
	.copy = _ccv_cnnp_average_pool_copy,
};

ccv_cnnp_model_t* ccv_cnnp_average_pool(const int kdim[CCV_NNC_MAX_DIM_ALLOC], const ccv_cnnp_param_t params, const char* const name)
{
	ccv_cnnp_model_pool_t* const model_pool = (ccv_cnnp_model_pool_t*)cccalloc(1, sizeof(ccv_cnnp_model_pool_t));
	model_pool->super.isa = &ccv_cnnp_average_pool_isa;
	model_pool->super.input_size = 1;
	model_pool->super.outputs = &model_pool->output;
	model_pool->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_pool->super, name);
	memcpy(model_pool->kdim, kdim, sizeof(model_pool->kdim));
	model_pool->params = params;
	return (ccv_cnnp_model_t*)model_pool;
}

static ccv_cnnp_model_t* _ccv_cnnp_average_pool_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_pool_t* const self = (const ccv_cnnp_model_pool_t*)super;
	return ccv_cnnp_average_pool(self->kdim, self->params, self->super.name);
}

#pragma mark - RELU Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_relu_t;

static void _ccv_cnnp_relu_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t relu = CMD_RELU_FORWARD();
	ccv_nnc_hint_tensor_auto(relu, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t relu_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, relu, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(relu_output), 0);
	outputs[0] = relu_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_relu_copy(const ccv_cnnp_model_t* const self);

static const ccv_cnnp_model_vtab_t ccv_cnnp_relu_isa = {
	.build = _ccv_cnnp_relu_build,
	.copy = _ccv_cnnp_relu_copy,
};

ccv_cnnp_model_t* ccv_cnnp_relu(const char* const name)
{
	ccv_cnnp_model_relu_t* const model_relu = (ccv_cnnp_model_relu_t*)cccalloc(1, sizeof(ccv_cnnp_model_relu_t));
	model_relu->super.isa = &ccv_cnnp_relu_isa;
	model_relu->super.input_size = 1;
	model_relu->super.outputs = &model_relu->output;
	model_relu->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_relu->super, name);
	return (ccv_cnnp_model_t*)model_relu;
}

static ccv_cnnp_model_t* _ccv_cnnp_relu_copy(const ccv_cnnp_model_t* const self)
{
	return ccv_cnnp_relu(self->name);
}

#pragma mark - Softmax Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_softmax_t;

static void _ccv_cnnp_softmax_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t softmax = CMD_SOFTMAX_FORWARD();
	ccv_nnc_hint_tensor_auto(softmax, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t softmax_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, softmax, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(softmax_output), 0);
	outputs[0] = softmax_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_softmax_copy(const ccv_cnnp_model_t* const self);

static const ccv_cnnp_model_vtab_t ccv_cnnp_softmax_isa = {
	.build = _ccv_cnnp_softmax_build,
	.copy = _ccv_cnnp_softmax_copy,
};

ccv_cnnp_model_t* ccv_cnnp_softmax(const char* const name)
{
	ccv_cnnp_model_softmax_t* const model_softmax = (ccv_cnnp_model_softmax_t*)cccalloc(1, sizeof(ccv_cnnp_model_softmax_t));
	model_softmax->super.isa = &ccv_cnnp_softmax_isa;
	model_softmax->super.input_size = 1;
	model_softmax->super.outputs = &model_softmax->output;
	model_softmax->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_softmax->super, name);
	return (ccv_cnnp_model_t*)model_softmax;
}

static ccv_cnnp_model_t* _ccv_cnnp_softmax_copy(const ccv_cnnp_model_t* const self)
{
	return ccv_cnnp_softmax(self->name);
}

#pragma mark - Scalar Mul Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	float a;
} ccv_cnnp_model_scalar_mul_t;

static void _ccv_cnnp_scalar_mul_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_cnnp_model_scalar_mul_t* const self = (ccv_cnnp_model_scalar_mul_t*)super;
	const ccv_nnc_cmd_t scalar_mul = CMD_SCALAR_MUL_FORWARD(self->a);
	ccv_nnc_hint_tensor_auto(scalar_mul, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t scalar_mul_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, scalar_mul, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(scalar_mul_output), 0);
	outputs[0] = scalar_mul_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_scalar_mul_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_scalar_mul_isa = {
	.build = _ccv_cnnp_scalar_mul_build,
	.copy = _ccv_cnnp_scalar_mul_copy,
};

ccv_cnnp_model_t* ccv_cnnp_scalar_mul(const float a, const char* const name)
{
	ccv_cnnp_model_scalar_mul_t* const model_scalar_mul = (ccv_cnnp_model_scalar_mul_t*)cccalloc(1, sizeof(ccv_cnnp_model_scalar_mul_t));
	model_scalar_mul->super.isa = &ccv_cnnp_scalar_mul_isa;
	model_scalar_mul->super.input_size = 1;
	model_scalar_mul->super.outputs = &model_scalar_mul->output;
	model_scalar_mul->super.output_size = 1;
	model_scalar_mul->a = a;
	ccv_cnnp_model_copy_name(&model_scalar_mul->super, name);
	return (ccv_cnnp_model_t*)model_scalar_mul;
}

static ccv_cnnp_model_t* _ccv_cnnp_scalar_mul_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_scalar_mul_t* const self = (const ccv_cnnp_model_scalar_mul_t*)super;
	return ccv_cnnp_scalar_mul(self->a, self->super.name);
}

#pragma mark - Transpose Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int transpose[2];
} ccv_cnnp_model_transpose_t;

static void _ccv_cnnp_transpose_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_transpose_t* const self = (ccv_cnnp_model_transpose_t*)super;
	if (self->transpose[0] == self->transpose[1])
	{
		outputs[0] = inputs[0];
		return;
	}
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t transpose = CMD_TRANSPOSE_FORWARD(self->transpose[0], self->transpose[1]);
	ccv_nnc_hint_tensor_auto(transpose, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t transpose_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, transpose, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(transpose_output), 0);
	outputs[0] = transpose_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_transpose_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_transpose_isa = {
	.build = _ccv_cnnp_transpose_build,
	.copy = _ccv_cnnp_transpose_copy,
};

ccv_cnnp_model_t* ccv_cnnp_transpose(const int axis_a, const int axis_b, const char* const name)
{
	ccv_cnnp_model_transpose_t* const model_transpose = (ccv_cnnp_model_transpose_t*)cccalloc(1, sizeof(ccv_cnnp_model_transpose_t));
	model_transpose->super.isa = &ccv_cnnp_transpose_isa;
	model_transpose->super.input_size = 1;
	model_transpose->super.outputs = &model_transpose->output;
	model_transpose->super.output_size = 1;
	model_transpose->transpose[0] = axis_a;
	model_transpose->transpose[1] = axis_b;
	ccv_cnnp_model_copy_name(&model_transpose->super, name);
	return (ccv_cnnp_model_t*)model_transpose;
}

static ccv_cnnp_model_t* _ccv_cnnp_transpose_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_transpose_t* const self = (const ccv_cnnp_model_transpose_t*)super;
	return ccv_cnnp_transpose(self->transpose[0], self->transpose[1], self->super.name);
}

#pragma mark - Layer Norm Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t bias;
	ccv_nnc_tensor_symbol_t scale;
	ccv_nnc_cmd_param_t params;
} ccv_cnnp_model_layer_norm_t;

static void _ccv_cnnp_layer_norm_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_layer_norm_t* const self = (ccv_cnnp_model_layer_norm_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t bias_params = params;
	// If the accuracy is not enough, bump it to 32-bit floating point.
	if (bias_params.datatype != CCV_32F || bias_params.datatype != CCV_64F)
		bias_params.datatype = CCV_32F;
	const int nd = ccv_nnc_tensor_nd(params.dim);
	int i;
	for (i = 0; i < nd; i++)
		bias_params.dim[i] = 1;
	for (i = 0; i < self->params.lnorm.count; i++)
		bias_params.dim[self->params.lnorm.axis[i]] = params.dim[self->params.lnorm.axis[i]];
	// Both scale and bias are shared between if this model is reused.
	if (!self->scale.graph)
		self->scale = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	if (!self->bias.graph)
		self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, 0);
	const ccv_nnc_cmd_t layer_norm = ccv_nnc_cmd(CCV_NNC_LAYER_NORM_FORWARD, 0, self->params, 0);
	ccv_nnc_tensor_param_t output_params[3];
	ccv_nnc_hint_tensor_auto(layer_norm, (ccv_nnc_tensor_param_t []){
			params,
			bias_params,
			bias_params,
		}, 3, ccv_nnc_no_hint, output_params, 3);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
	const ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(graph, output_params[1], 0);
	const ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(graph, output_params[2], 0);
	ccv_nnc_graph_exec_symbol_new(graph, layer_norm, TENSOR_SYMBOL_LIST(inputs[0], self->scale, self->bias), TENSOR_SYMBOL_LIST(output, saved_mean, saved_inv_std), 0);
	outputs[0] = output;
}

static void _ccv_cnnp_layer_norm_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_layer_norm_t* const self = (ccv_cnnp_model_layer_norm_t*)super;
	if (self->bias.graph)
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
	if (self->scale.graph)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(0, 1), ccv_nnc_no_hint, 0, 0, self->scale);
}

static void _ccv_cnnp_layer_norm_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters)
{
	ccv_cnnp_model_layer_norm_t* const self = (ccv_cnnp_model_layer_norm_t*)super;
	if (self->bias.graph)
		add_to_array(parameters, self->bias);
	if (self->scale.graph)
		add_to_array(parameters, self->scale);
}

static ccv_cnnp_model_t* _ccv_cnnp_layer_norm_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_layer_norm_isa = {
	.build = _ccv_cnnp_layer_norm_build,
	.init_states = _ccv_cnnp_layer_norm_init_states,
	.add_to_parameter = _ccv_cnnp_layer_norm_add_to_parameter,
	.copy = _ccv_cnnp_layer_norm_copy,
};

ccv_cnnp_model_t* ccv_cnnp_layer_norm(const float epsilon, const int axis[CCV_NNC_MAX_DIM_ALLOC], const int axis_count, const char* const name)
{
	ccv_cnnp_model_layer_norm_t* const model_layer_norm = (ccv_cnnp_model_layer_norm_t*)cccalloc(1, sizeof(ccv_cnnp_model_layer_norm_t));
	model_layer_norm->super.isa = &ccv_cnnp_layer_norm_isa;
	model_layer_norm->super.input_size = 1;
	model_layer_norm->super.outputs = &model_layer_norm->output;
	model_layer_norm->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_layer_norm->super, name);
	model_layer_norm->scale.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_layer_norm->scale.graph = 0;
	model_layer_norm->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_layer_norm->bias.graph = 0;
	model_layer_norm->params.lnorm.epsilon = epsilon;
	model_layer_norm->params.lnorm.count = axis_count;
	memcpy(model_layer_norm->params.lnorm.axis, axis, sizeof(model_layer_norm->params.lnorm.axis));
	return (ccv_cnnp_model_t*)model_layer_norm;
}

static ccv_cnnp_model_t* _ccv_cnnp_layer_norm_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_layer_norm_t* const self = (const ccv_cnnp_model_layer_norm_t*)super;
	return ccv_cnnp_layer_norm(self->params.lnorm.epsilon, self->params.lnorm.axis, self->params.lnorm.count, self->super.name);
}

#pragma mark - Batched Matrix Mul Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int transpose_a[2];
	int transpose_b[2];
} ccv_cnnp_model_matmul_t;

static void _ccv_cnnp_matmul_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size == 1);
	ccv_cnnp_model_matmul_t* const self = (ccv_cnnp_model_matmul_t*)super;
	ccv_nnc_tensor_param_t a_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t b_params = ccv_nnc_tensor_symbol_params(graph, inputs[1]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t matmul = CMD_GEMM_FORWARD(self->transpose_a, self->transpose_b);
	ccv_nnc_hint_tensor_auto(matmul, (ccv_nnc_tensor_param_t []){
			a_params,
			b_params,
		}, 2, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t matmul_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, matmul, inputs, input_size, TENSOR_SYMBOL_LIST(matmul_output), 0);
	outputs[0] = matmul_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_matmul_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_matmul_isa = {
	.build = _ccv_cnnp_matmul_build,
	.copy = _ccv_cnnp_matmul_copy,
};

ccv_cnnp_model_t* ccv_cnnp_matmul(const int transpose_a[2], const int transpose_b[2], const char* const name)
{
	ccv_cnnp_model_matmul_t* const model_matmul = (ccv_cnnp_model_matmul_t*)cccalloc(1, sizeof(ccv_cnnp_model_matmul_t));
	model_matmul->super.isa = &ccv_cnnp_matmul_isa;
	model_matmul->super.input_size = 2;
	model_matmul->super.outputs = &model_matmul->output;
	model_matmul->super.output_size = 1;
	model_matmul->transpose_a[0] = transpose_a[0];
	model_matmul->transpose_a[1] = transpose_a[1];
	model_matmul->transpose_b[0] = transpose_b[0];
	model_matmul->transpose_b[1] = transpose_b[1];
	ccv_cnnp_model_copy_name(&model_matmul->super, name);
	return (ccv_cnnp_model_t*)model_matmul;
}

static ccv_cnnp_model_t* _ccv_cnnp_matmul_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_matmul_t* const self = (const ccv_cnnp_model_matmul_t*)super;
	return ccv_cnnp_matmul(self->transpose_a, self->transpose_b, self->super.name);
}

#pragma mark - Dropout Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_graph_exec_symbol_t dropout;
	float p;
} ccv_cnnp_model_dropout_t;

static void _ccv_cnnp_dropout_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params[2];
	ccv_cnnp_model_dropout_t* const self = (ccv_cnnp_model_dropout_t*)super;
	const ccv_nnc_cmd_t dropout = CMD_DROPOUT_FORWARD(self->p);
	ccv_nnc_hint_tensor_auto(dropout, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, output_params, 2);
	const ccv_nnc_tensor_symbol_t dropout_output = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
	const ccv_nnc_tensor_symbol_t mask = ccv_nnc_tensor_symbol_new(graph, output_params[1], 0);
	self->dropout = ccv_nnc_graph_exec_symbol_new(graph, dropout, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(dropout_output, mask), 0);
	outputs[0] = dropout_output;
}

static void _ccv_cnnp_dropout_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_model_dropout_t* const self = (ccv_cnnp_model_dropout_t*)super;
	if (self->dropout.graph)
	{
		if (is_test)
			// During test, the dropout is not applied. Data transfer is perfect because if these are the same tensor, it will skip.
			updater(context, self->dropout, CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint);
		else
			updater(context, self->dropout, CMD_DROPOUT_FORWARD(self->p), ccv_nnc_no_hint);
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_dropout_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_dropout_isa = {
	.build = _ccv_cnnp_dropout_build,
	.set_is_test = _ccv_cnnp_dropout_is_test,
	.copy = _ccv_cnnp_dropout_copy,
};

ccv_cnnp_model_t* ccv_cnnp_dropout(const float p, const char* const name)
{
	ccv_cnnp_model_dropout_t* const model_dropout = (ccv_cnnp_model_dropout_t*)cccalloc(1, sizeof(ccv_cnnp_model_dropout_t));
	model_dropout->super.isa = &ccv_cnnp_dropout_isa;
	model_dropout->super.input_size = 1;
	model_dropout->super.outputs = &model_dropout->output;
	model_dropout->super.output_size = 1;
	model_dropout->p = p;
	ccv_cnnp_model_copy_name(&model_dropout->super, name);
	return (ccv_cnnp_model_t*)model_dropout;
}

static ccv_cnnp_model_t* _ccv_cnnp_dropout_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_dropout_t* const self = (const ccv_cnnp_model_dropout_t*)super;
	return ccv_cnnp_dropout(self->p, self->super.name);
}

#pragma mark - Masked Fill Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	float eq;
	float fill;
} ccv_cnnp_model_masked_fill_t;

static void _ccv_cnnp_masked_fill_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_cnnp_model_masked_fill_t* const self = (ccv_cnnp_model_masked_fill_t*)super;
	const ccv_nnc_tensor_symbol_t masked_fill_output = ccv_nnc_tensor_symbol_new(graph, params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_MASKED_FILL_FORWARD(self->eq, self->fill), TENSOR_SYMBOL_LIST(inputs[0], inputs[1]), TENSOR_SYMBOL_LIST(masked_fill_output), 0);
	outputs[0] = masked_fill_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_masked_fill_copy(const ccv_cnnp_model_t* const super);

static const ccv_cnnp_model_vtab_t ccv_cnnp_masked_fill_isa = {
	.build = _ccv_cnnp_masked_fill_build,
	.copy = _ccv_cnnp_masked_fill_copy,
};

ccv_cnnp_model_t* ccv_cnnp_masked_fill(const float eq, const float fill, const char* const name)
{
	ccv_cnnp_model_masked_fill_t* const model_masked_fill = (ccv_cnnp_model_masked_fill_t*)cccalloc(1, sizeof(ccv_cnnp_model_masked_fill_t));
	model_masked_fill->super.isa = &ccv_cnnp_masked_fill_isa;
	model_masked_fill->super.input_size = 2;
	model_masked_fill->super.outputs = &model_masked_fill->output;
	model_masked_fill->super.output_size = 1;
	model_masked_fill->eq = eq;
	model_masked_fill->fill = fill;
	ccv_cnnp_model_copy_name(&model_masked_fill->super, name);
	return (ccv_cnnp_model_t*)model_masked_fill;
}

static ccv_cnnp_model_t* _ccv_cnnp_masked_fill_copy(const ccv_cnnp_model_t* const super)
{
	const ccv_cnnp_model_masked_fill_t* const self = (const ccv_cnnp_model_masked_fill_t*)super;
	return ccv_cnnp_masked_fill(self->eq, self->fill, self->super.name);
}
