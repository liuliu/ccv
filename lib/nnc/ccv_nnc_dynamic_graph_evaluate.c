#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"
#include "_ccv_cnnp_model.h"

#pragma mark - Level-5.5 API

static int _ccv_cnnp_model_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stateful_exec_t* const stateful_exec = (ccv_nnc_stateful_exec_t*)cmd.data;
	ccv_cnnp_model_t* const model = (ccv_cnnp_model_t*)stateful_exec->data;
	if (cmd.cmd == CCV_NNC_CUSTOM_FORWARD)
	{
		ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
			.disable_outgrad = 0,
			.is_test = 0,
		}, inputs, input_size, outputs, output_size, 0, stream_context);
	} else {
		ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
		const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
		const int ingrad_size = model->output_size * parallel_count;
		assert(ingrad_size <= input_size);
		ccv_cnnp_model_backward(model, inputs, ingrad_size, outputs, output_size, 0, stream_context);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_cnnp_model_tensor_auto(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	ccv_nnc_stateful_exec_t* const stateful_exec = (ccv_nnc_stateful_exec_t*)cmd.data;
	ccv_cnnp_model_t* const model = (ccv_cnnp_model_t*)stateful_exec->data;
	ccv_cnnp_model_tensor_auto(model, outputs, output_size);
}

static void _ccv_cnnp_model_apply_gradients(const ccv_nnc_cmd_t cmd, const ccv_nnc_cmd_t minimizer, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stateful_exec_t* const stateful_exec = (ccv_nnc_stateful_exec_t*)cmd.data;
	ccv_cnnp_model_t* const model = (ccv_cnnp_model_t*)stateful_exec->data;
	ccv_cnnp_model_set_minimizer(model, minimizer, 0, 0);
	ccv_cnnp_model_apply_gradients(model, stream_context);
}

static ccv_nnc_stateful_cmd_vtab_t ccv_cnnp_model_exec_isa = {
	.super = {
		.exec = _ccv_cnnp_model_exec,
		.tensor_auto = _ccv_cnnp_model_tensor_auto,
	},
	.apply_gradients = _ccv_cnnp_model_apply_gradients,
};

void ccv_nnc_dynamic_graph_evaluate(ccv_nnc_dynamic_graph_t* const dynamic_graph, ccv_cnnp_model_t* const model, const ccv_nnc_tensor_variable_t* const inputs, const int input_size, ccv_nnc_tensor_variable_t* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_CUSTOM_FORWARD, (ccv_nnc_cmd_vtab_t*)&ccv_cnnp_model_exec_isa, (ccv_nnc_cmd_param_t){}, 0);
	assert(input_size > 0);
	if (!model->graph)
	{
		ccv_nnc_tensor_param_t input_params[input_size];
		int i;
		for (i = 0; i < input_size; i++)
			input_params[i] = inputs[i]->info;
		ccv_cnnp_model_compile(model, input_params, input_size, CMD_NOOP(), CMD_NOOP());
	}
	int i;
	for (i = 0; i < input_size; i++)
	{
		// Cannot have the parameter be a partial tensor view for model evaluation.
		ccv_nnc_tensor_t* const tensor = inputs[i] ? (ccv_nnc_tensor_t*)CCV_NNC_TENSOR_VIEW(inputs[i]->tensor_view) : 0;
		if (tensor)
			{ assert(!CCV_IS_TENSOR_VIEW(tensor)); }
	}
	if (dynamic_graph->no_grad)
	{
		ccv_nnc_stateful_exec_t stateful_exec = {
			.tensor_tape = tensor_tape,
			.data = model
		};
		cmd.data = &stateful_exec;
		// Parallel parameter doesn't make sense here, the parallel is defined inside the model.
		ccv_nnc_dynamic_graph_exec_ret(dynamic_graph, cmd, ccv_nnc_no_hint, 0, inputs, input_size, outputs, output_size, 0, stream_context, 0);
	} else {
		ccv_nnc_stateful_exec_t* const stateful_exec = (ccv_nnc_stateful_exec_t*)ccmalloc(sizeof(ccv_nnc_stateful_exec_t));
		stateful_exec->tensor_tape = tensor_tape;
		stateful_exec->data = model;
		cmd.data = stateful_exec;
		ccv_nnc_graph_exec_symbol_t symbol;
		ccv_nnc_dynamic_graph_exec_ret(dynamic_graph, cmd, ccv_nnc_no_hint, 0, inputs, input_size, outputs, output_size, 0, stream_context, &symbol);
		assert(symbol.graph);
		int ret;
		khiter_t k = kh_put(stateful_exec, dynamic_graph->stateful_execs, symbol.d, &ret);
		kh_val(dynamic_graph->stateful_execs, k) = stateful_exec;
	}
}

