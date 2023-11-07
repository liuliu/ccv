#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"

// MARK - Add-on Functions

static int _ccv_cnnp_model_clip_grad_norm_reduce_norm2(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[0]->info.type);
	ccv_nnc_tensor_t* const old_norm2 = outputs[1 + device_id * 2];
	ccv_nnc_tensor_t* const norm2 = outputs[1 + device_id * 2 + 1];
	const int tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
	if (tensor_count == 1)
		ccv_nnc_cmd_exec(CMD_MUL_FORWARD(1), hint, flags, TENSOR_LIST(inputs[0], inputs[0]), TENSOR_LIST(norm2), stream_context);
	else {
		ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(), hint, flags, TENSOR_LIST(inputs[0]), TENSOR_LIST(norm2), stream_context);
		ccv_nnc_cmd_exec(CMD_MUL_FORWARD(1), hint, flags, TENSOR_LIST(norm2, norm2), TENSOR_LIST(norm2), stream_context);
	}
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), hint, flags, TENSOR_LIST(old_norm2, norm2), TENSOR_LIST(old_norm2), stream_context);
	return CCV_NNC_EXEC_SUCCESS;
}

static ccv_nnc_cmd_vtab_t clip_grad_norm_reduce_norm2_vtab = {
	.exec = _ccv_cnnp_model_clip_grad_norm_reduce_norm2
};

static int _ccv_cnnp_model_clip_grad_norm_scatter_norm2(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[0]->info.type);
	ccv_nnc_tensor_t* const norm2 = inputs[1 + device_id * 2];
	ccv_nnc_cmd_exec(CMD_MUL_FORWARD(1), hint, flags, TENSOR_LIST(inputs[0], norm2), TENSOR_LIST(outputs[0]), stream_context);
	return CCV_NNC_EXEC_SUCCESS;
}

static ccv_nnc_cmd_vtab_t clip_grad_norm_scatter_norm2_vtab = {
	.exec = _ccv_cnnp_model_clip_grad_norm_scatter_norm2
};

void ccv_cnnp_model_parameters_clip_grad_norm(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, int norm_type, float max_norm, ccv_nnc_stream_context_t* const stream_context)
{
	assert(norm_type == 2);
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	ccv_nnc_tensor_t* norm2[parallel_count * 2];
	ccv_nnc_tensor_t* max_normt[parallel_count];
	const int stream_type = model->compiled_data->stream_type;
	int i;
	if (stream_type == CCV_STREAM_CONTEXT_GPU)
	{
		for (i = 0; i < parallel_count; i++)
		{
			ccv_nnc_tensor_param_t info = {
				.type = CCV_TENSOR_GPU_MEMORY,
				.format = CCV_TENSOR_FORMAT_NHWC,
				.datatype = CCV_32F,
				.dim = {1},
			};
			CCV_TENSOR_SET_DEVICE_ID(info.type, i);
			norm2[i * 2] = ccv_nnc_tensor_new(ccv_nnc_xpu_alloc(&compiled_data->xpu_alloc, i, stream_context, ccv_nnc_tensor_data_size(info)), info, 0);
			norm2[i * 2 + 1] = ccv_nnc_tensor_new(ccv_nnc_xpu_alloc(&compiled_data->xpu_alloc, i, stream_context, ccv_nnc_tensor_data_size(info)), info, 0);
			max_normt[i] = ccv_nnc_tensor_new(ccv_nnc_xpu_alloc(&compiled_data->xpu_alloc, i, stream_context, ccv_nnc_tensor_data_size(info)), info, 0);
		}
	} else {
		for (i = 0; i < parallel_count; i++)
		{
			ccv_nnc_tensor_param_t info = {
				.type = CCV_TENSOR_CPU_MEMORY,
				.format = CCV_TENSOR_FORMAT_NHWC,
				.datatype = CCV_32F,
				.dim = {1},
			};
			norm2[i * 2] = ccv_nnc_tensor_new(0, info, 0);
			norm2[i * 2 + 1] = ccv_nnc_tensor_new(0, info, 0);
			max_normt[i] = ccv_nnc_tensor_new(0, info, 0);
		}
	}
	// zero out old norm2.
	if (parallel_count > 1)
	{
		ccv_nnc_stream_context_t* streams[parallel_count];
		ccv_nnc_stream_signal_t* signal;
		if (stream_context)
			signal = ccv_nnc_stream_context_emit_signal_new(stream_context);
		for (i = 0; i < parallel_count; i++)
		{
			const int stream_type = CCV_TENSOR_GET_MEMORY(norm2[i * 2]->info.type) == CCV_TENSOR_GPU_MEMORY ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
			const int device_id = CCV_TENSOR_GET_DEVICE_ID(norm2[i * 2]->info.type);
			int type = stream_type;
			CCV_STREAM_SET_DEVICE_ID(type, device_id);
			ccv_nnc_stream_context_t* const stream_0 = ccv_cnnp_compiled_data_get_stream(compiled_data, type);
			// Wait signal to finish.
			if (stream_context)
				ccv_nnc_stream_context_wait_signal(stream_0, signal);
			ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(norm2[i * 2]), stream_0);
			if (stream_context)
			{
				ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_emit_signal_new(stream_0);
				ccv_nnc_stream_context_wait_signal(stream_context, signal);
			}
			streams[i] = stream_0;
		}
		// If this should be blocking, blocking it.
		if (!stream_context)
			for (i = 0; i < parallel_count; i++)
				if (streams[i])
					ccv_nnc_stream_context_wait(streams[i]);
	} else {
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(norm2[0]), stream_context);
	}
	// Gather norm2.
	ccv_nnc_cmd_t reduce_cmd = {
		.cmd = CCV_NNC_CUSTOM_FORWARD,
		.isa = &clip_grad_norm_reduce_norm2_vtab,
	};
	ccv_cnnp_model_parameter_gradients_map(model, parameters, reduce_cmd, ccv_nnc_no_hint, 0, 0, 0, norm2, parallel_count * 2, stream_context);
	// Now compute max(max_norm / norm2, 1.0).
	if (parallel_count > 1)
	{
		ccv_nnc_stream_context_t* streams[parallel_count];
		ccv_nnc_stream_signal_t* signal;
		if (stream_context)
			signal = ccv_nnc_stream_context_emit_signal_new(stream_context);
		for (i = 0; i < parallel_count; i++)
		{
			const int stream_type = CCV_TENSOR_GET_MEMORY(norm2[i * 2]->info.type) == CCV_TENSOR_GPU_MEMORY ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
			const int device_id = CCV_TENSOR_GET_DEVICE_ID(norm2[i * 2]->info.type);
			int type = stream_type;
			CCV_STREAM_SET_DEVICE_ID(type, device_id);
			ccv_nnc_stream_context_t* const stream_0 = ccv_cnnp_compiled_data_get_stream(compiled_data, type);
			// Wait signal to finish.
			if (stream_context)
				ccv_nnc_stream_context_wait_signal(stream_0, signal);
			ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(norm2[i * 2]), TENSOR_LIST(norm2[i * 2]), stream_0);
			ccv_nnc_cmd_exec(CMD_SET_FORWARD(max_norm), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(max_normt[i]), stream_0);
			ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(max_normt[i], norm2[i * 2]), TENSOR_LIST(norm2[i * 2]), stream_0);
			ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(norm2[i * 2]), TENSOR_LIST(norm2[i * 2]), stream_0);
			if (stream_context)
			{
				ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_emit_signal_new(stream_0);
				ccv_nnc_stream_context_wait_signal(stream_context, signal);
			}
			streams[i] = stream_0;
		}
		// If this should be blocking, blocking it.
		if (!stream_context)
			for (i = 0; i < parallel_count; i++)
				if (streams[i])
					ccv_nnc_stream_context_wait(streams[i]);
	} else {
		ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(norm2[0]), TENSOR_LIST(norm2[0]), stream_context);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(max_norm), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(max_normt[0]), stream_context);
		ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(max_normt[0], norm2[0]), TENSOR_LIST(norm2[0]), stream_context);
		ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(norm2[0]), TENSOR_LIST(norm2[0]), stream_context);
	}
	ccv_nnc_cmd_t scatter_cmd = {
		.cmd = CCV_NNC_CUSTOM_FORWARD,
		.isa = &clip_grad_norm_scatter_norm2_vtab,
	};
	ccv_cnnp_model_parameter_gradients_map(model, parameters, scatter_cmd, ccv_nnc_no_hint, 0, norm2, parallel_count * 2, 0, 0, stream_context);
	if (stream_type == CCV_STREAM_CONTEXT_GPU)
		for (i = 0; i < parallel_count; i++)
		{
			ccv_nnc_xpu_free(&compiled_data->xpu_alloc, norm2[i * 2]->data.u8);
			ccv_nnc_xpu_free(&compiled_data->xpu_alloc, norm2[i * 2 + 1]->data.u8);
			ccv_nnc_xpu_free(&compiled_data->xpu_alloc, max_normt[i]->data.u8);
		}
	for (i = 0; i < parallel_count; i++)
	{
		ccv_nnc_tensor_free(norm2[i * 2]);
		ccv_nnc_tensor_free(norm2[i * 2 + 1]);
		ccv_nnc_tensor_free(max_normt[i]);
	}
}

// MARK - Add-on Functions

static int _ccv_cnnp_model_isnan(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[0]->info.type);
	ccv_nnc_tensor_t* const old_isnanr = outputs[1 + device_id * 2];
	ccv_nnc_tensor_t* const isnanr = outputs[1 + device_id * 2 + 1];
	ccv_nnc_cmd_t reduce_cmd = CMD_REDUCE_ISNAN_FORWARD();
	reduce_cmd.info.reduce.count = ccv_nnc_tensor_nd(inputs[0]->info.dim);
	int i;
	for (i = 0; i < cmd.info.reduce.count; i++)
		reduce_cmd.info.reduce.axis[i] = i;
	ccv_nnc_cmd_exec(reduce_cmd, hint, flags, TENSOR_LIST(inputs[0]), TENSOR_LIST(isnanr), stream_context);
	ccv_nnc_cmd_exec(CMD_EWSUM_FORWARD(), hint, flags, TENSOR_LIST(old_isnanr, isnanr), TENSOR_LIST(old_isnanr), stream_context);
	return CCV_NNC_EXEC_SUCCESS;
}

static ccv_nnc_cmd_vtab_t reduce_isnan_vtab = {
	.exec = _ccv_cnnp_model_isnan
};

int ccv_cnnp_model_parameter_gradients_isnan(ccv_cnnp_model_t* const model, const ccv_cnnp_model_io_t parameters, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data);
	const int parallel_count = ccv_max(model->parallel_count, 1);
	ccv_nnc_tensor_t* isnanr[parallel_count * 2];
	const int stream_type = model->compiled_data->stream_type;
	int i;
	if (stream_type == CCV_STREAM_CONTEXT_GPU)
	{
		for (i = 0; i < parallel_count; i++)
		{
			ccv_nnc_tensor_param_t info = {
				.type = CCV_TENSOR_GPU_MEMORY,
				.format = CCV_TENSOR_FORMAT_NHWC,
				.datatype = CCV_32S,
				.dim = {1},
			};
			CCV_TENSOR_SET_DEVICE_ID(info.type, i);
			isnanr[i * 2] = ccv_nnc_tensor_new(ccv_nnc_xpu_alloc(&compiled_data->xpu_alloc, i, stream_context, ccv_nnc_tensor_data_size(info)), info, 0);
			isnanr[i * 2 + 1] = ccv_nnc_tensor_new(ccv_nnc_xpu_alloc(&compiled_data->xpu_alloc, i, stream_context, ccv_nnc_tensor_data_size(info)), info, 0);
		}
	} else {
		for (i = 0; i < parallel_count; i++)
		{
			ccv_nnc_tensor_param_t info = {
				.type = CCV_TENSOR_CPU_MEMORY,
				.format = CCV_TENSOR_FORMAT_NHWC,
				.datatype = CCV_32S,
				.dim = {1},
			};
			isnanr[i * 2] = ccv_nnc_tensor_new(0, info, 0);
			isnanr[i * 2 + 1] = ccv_nnc_tensor_new(0, info, 0);
		}
	}
	// zero out old isnanr.
	if (parallel_count > 1)
	{
		ccv_nnc_stream_context_t* streams[parallel_count];
		ccv_nnc_stream_signal_t* signal;
		if (stream_context)
			signal = ccv_nnc_stream_context_emit_signal_new(stream_context);
		for (i = 0; i < parallel_count; i++)
		{
			const int stream_type = CCV_TENSOR_GET_MEMORY(isnanr[i * 2]->info.type) == CCV_TENSOR_GPU_MEMORY ? CCV_STREAM_CONTEXT_GPU : CCV_STREAM_CONTEXT_CPU;
			const int device_id = CCV_TENSOR_GET_DEVICE_ID(isnanr[i * 2]->info.type);
			int type = stream_type;
			CCV_STREAM_SET_DEVICE_ID(type, device_id);
			ccv_nnc_stream_context_t* const stream_0 = ccv_cnnp_compiled_data_get_stream(compiled_data, type);
			// Wait signal to finish.
			if (stream_context)
				ccv_nnc_stream_context_wait_signal(stream_0, signal);
			ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(isnanr[i * 2]), stream_0);
			if (stream_context)
			{
				ccv_nnc_stream_signal_t* const signal = ccv_nnc_stream_context_emit_signal_new(stream_0);
				ccv_nnc_stream_context_wait_signal(stream_context, signal);
			}
			streams[i] = stream_0;
		}
		// If this should be blocking, blocking it.
		if (!stream_context)
			for (i = 0; i < parallel_count; i++)
				if (streams[i])
					ccv_nnc_stream_context_wait(streams[i]);
	} else
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(isnanr[0]), stream_context);
	// Gather isnanr.
	ccv_nnc_cmd_t reduce_cmd = {
		.cmd = CCV_NNC_CUSTOM_FORWARD,
		.isa = &reduce_isnan_vtab,
	};
	ccv_cnnp_model_parameter_gradients_map(model, parameters, reduce_cmd, ccv_nnc_no_hint, 0, 0, 0, isnanr, parallel_count * 2, stream_context);
	for (i = 0; i < parallel_count; i++)
		ccv_nnc_tensor_free(isnanr[i * 2 + 1]);
	int retval = 0;
	if (stream_type == CCV_TENSOR_GPU_MEMORY)
	{
		ccv_nnc_tensor_param_t info = {
			.type = CCV_TENSOR_CPU_MEMORY,
			.format = CCV_TENSOR_FORMAT_NHWC,
			.datatype = CCV_32S,
			.dim = {1},
		};
		ccv_nnc_tensor_t* checknan = ccv_nnc_tensor_new(0, info, 0);
		for (i = 0; i < parallel_count; i++)
		{
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(isnanr[i * 2]), TENSOR_LIST(checknan), 0);
			if (checknan->data.i32[0] > 0)
			{
				retval = 1;
				break;
			}
		}
		ccv_nnc_tensor_free(checknan);
	} else {
		for (i = 0; i < parallel_count; i++)
			if (isnanr[i * 2]->data.i32[0] > 0)
			{
				retval = 1;
				break;
			}
	}
	for (i = 0; i < parallel_count; i++)
		ccv_nnc_tensor_free(isnanr[i * 2]);
	return retval;
}

// MARK - Core Layers

static void _ccv_cnnp_sum_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, ccv_nnc_tensor_symbol_params(graph, inputs[0]), 0);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_EWSUM_FORWARD(), inputs, input_size, outputs, output_size, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_sum_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_sum_isa = {
	.build = _ccv_cnnp_sum_build,
	.copy = _ccv_cnnp_sum_copy,
};

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_sum_t;

ccv_cnnp_model_t* ccv_cnnp_sum(const char* const name)
{
	ccv_cnnp_model_sum_t* const model_sum = (ccv_cnnp_model_sum_t*)cccalloc(1, sizeof(ccv_cnnp_model_sum_t));
	model_sum->super.isa = &ccv_cnnp_sum_isa;
	model_sum->super.input_size = 0;
	model_sum->super.outputs = &model_sum->output;
	model_sum->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_sum->super, name);
	return (ccv_cnnp_model_t*)model_sum;
}

static ccv_cnnp_model_t* _ccv_cnnp_sum_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	return ccv_cnnp_sum(self->name);
}

typedef struct {
	ccv_cnnp_model_t super;
	int axis;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_concat_t;

static void _ccv_cnnp_concat_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_concat_t* const self = (const ccv_cnnp_model_concat_t*)super;
	assert(output_size == 1);
	ccv_nnc_tensor_param_t output_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	int i, j;
	if (output_params.dim[0] == 0)
		for (i = 1; i < input_size; i++)
		{
			output_params = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
			if (output_params.dim[0] != 0)
				break;
		}
	const int nd = ccv_nnc_tensor_nd(output_params.dim);
	const int axis = self->axis;
	assert(axis < nd);
	output_params.dim[axis] = 0;
	int input_is_contiguous = 1;
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
		const int input_nd = ccv_nnc_tensor_nd(input_params.dim);
		if (input_nd == 0)
		{
			input_is_contiguous = 0;
			continue;
		}
		assert(input_nd == nd);
		for (j = 0; j < nd; j++)
			if (j != axis)
				{ assert(input_params.dim[j] == output_params.dim[j]); }
		output_params.dim[axis] += input_params.dim[axis];
	}
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {};
	int stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	ccv_nnc_tensor_get_stride(output_params.dim, stride);
	if (input_is_contiguous)
	{
		ccv_nnc_tensor_symbol_t aliases[input_size];
		for (i = 0; i < input_size; i++)
		{
			const ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
			aliases[i] = ccv_nnc_tensor_symbol_alias_new(graph, outputs[0], ofs, stride, input_params, 0);
			ofs[axis] += input_params.dim[axis];
		}
		// Format transform is more flexible.
		ccv_nnc_graph_exec_symbol_new(graph, CMD_FORMAT_TRANSFORM_FORWARD(), inputs, input_size, aliases, input_size, "concat");
	} else {
		int contiguous_input_size = 0;
		ccv_nnc_tensor_symbol_t contiguous_inputs[input_size];
		ccv_nnc_tensor_symbol_t aliases[input_size];
		for (i = 0; i < input_size; i++)
		{
			const ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
			if (input_params.dim[0] == 0)
				continue;
			contiguous_inputs[contiguous_input_size] = inputs[i];
			aliases[contiguous_input_size] = ccv_nnc_tensor_symbol_alias_new(graph, outputs[0], ofs, stride, input_params, 0);
			ofs[axis] += input_params.dim[axis];
			++contiguous_input_size;
		}
		// Format transform is more flexible.
		ccv_nnc_graph_exec_symbol_new(graph, CMD_FORMAT_TRANSFORM_FORWARD(), contiguous_inputs, contiguous_input_size, aliases, contiguous_input_size, "concat");
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_concat_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_concat_isa = {
	.build = _ccv_cnnp_concat_build,
	.copy = _ccv_cnnp_concat_copy,
};

ccv_cnnp_model_t* ccv_cnnp_concat(const int axis, const char* const name)
{
	ccv_cnnp_model_concat_t* const model_concat = (ccv_cnnp_model_concat_t*)cccalloc(1, sizeof(ccv_cnnp_model_concat_t));
	model_concat->super.isa = &ccv_cnnp_concat_isa;
	model_concat->super.input_size = 0;
	model_concat->super.outputs = &model_concat->output;
	model_concat->super.output_size = 1;
	model_concat->axis = axis;
	ccv_cnnp_model_copy_name(&model_concat->super, name);
	return (ccv_cnnp_model_t*)model_concat;
}

static ccv_cnnp_model_t* _ccv_cnnp_concat_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_concat_t* const self = (const ccv_cnnp_model_concat_t*)super;
	return ccv_cnnp_concat(self->axis, self->super.name);
}

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int format;
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int stride[CCV_NNC_MAX_DIM_ALLOC];
} ccv_cnnp_model_reshape_t;

static void _ccv_cnnp_reshape_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_reshape_t* const self = (ccv_cnnp_model_reshape_t*)super;
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	if (self->format > 0)
		params.format = self->format;
	assert(ccv_nnc_dimension_count(self->dim) <= ccv_nnc_tensor_count(params));
	ccv_nnc_tensor_symbol_t to = ccv_nnc_tensor_symbol_alias_to(graph, inputs[0]);
	int stride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
	if (to.d == CCV_NNC_NO_TENSOR_SYMBOL) // If we are not reshape an alias, it is straightforward.
	{
		memcpy(params.dim, self->dim, sizeof(params.dim));
		int* stride;
		if (self->stride[0] == 0)
		{
			ccv_nnc_tensor_get_stride(self->dim, stride_from_dim);
			stride = stride_from_dim;
		} else
			stride = self->stride;
		outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, inputs[0], self->ofs, stride, params, 0);
	} else {
		// Otherwise, we need to check if it is permute. For permute, we cannot do alias directly.
		// We need to first materialize the permute and then run reshape on top of it, otherwise it will be wrong.
		int old_stride[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_symbol_alias_params(graph, inputs[0], 0, old_stride);
		// We identify permute by checking if the stride is not in descending order.
		// This also covered "permute" through reshape, rather than using ccv_cnnp_permute directly.
		const int nd = ccv_nnc_tensor_nd(params.dim);
		const int new_nd = ccv_nnc_tensor_nd(self->dim);
		int i, no_permute = 1;
		// If the new dim has different nd, or we actually have a stride, we need to check if it is no permute or not.
		if (new_nd != nd || (self->stride[0] != 0 && memcmp(self->stride, old_stride, sizeof(self->stride)) != 0))
			for (i = 1; no_permute && i < nd; i++)
				if (old_stride[i - 1] < old_stride[i])
					no_permute = 0;
		if (no_permute)
		{ // Just straightforward reshape if there is no no permute.
			memcpy(params.dim, self->dim, sizeof(params.dim));
			int* stride;
			if (self->stride[0] == 0)
			{
				if (new_nd != nd) // Cannot use old stride.
				{
					ccv_nnc_tensor_get_stride(self->dim, stride_from_dim);
					stride = stride_from_dim;
				} else
					stride = old_stride;
			} else
				stride = self->stride;
			outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, inputs[0], self->ofs, stride, params, 0);
		} else {
			// Otherwise, we first do format transform to plain tensor and then do reshape.
			ccv_nnc_tensor_symbol_t permuted = ccv_nnc_tensor_symbol_new(graph, params, 0);
			ccv_nnc_graph_exec_symbol_new(graph, CMD_FORMAT_TRANSFORM_FORWARD(), TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(permuted), "reshape");
			memcpy(params.dim, self->dim, sizeof(params.dim));
			int* stride;
			if (self->stride[0] == 0)
			{
				ccv_nnc_tensor_get_stride(self->dim, stride_from_dim);
				stride = stride_from_dim;
			} else
				stride = self->stride;
			// And then we create alias against the permuted one.
			outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, permuted, self->ofs, stride, params, 0);
		}
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_reshape_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_reshape_isa = {
	.build = _ccv_cnnp_reshape_build,
	.copy = _ccv_cnnp_reshape_copy,
};

ccv_cnnp_model_t* ccv_cnnp_reshape(const int format, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC], const char* const name)
{
	ccv_cnnp_model_reshape_t* const model_reshape = (ccv_cnnp_model_reshape_t*)cccalloc(1, sizeof(ccv_cnnp_model_reshape_t));
	model_reshape->super.isa = &ccv_cnnp_reshape_isa;
	model_reshape->super.input_size = 1;
	model_reshape->super.outputs = &model_reshape->output;
	model_reshape->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_reshape->super, name);
	model_reshape->format = format;
	memcpy(model_reshape->dim, dim, sizeof(model_reshape->dim));
	memcpy(model_reshape->ofs, ofs, sizeof(model_reshape->ofs));
	if (stride[0] != 0)
		memcpy(model_reshape->stride, stride, sizeof(model_reshape->stride));
	return (ccv_cnnp_model_t*)model_reshape;
}

static ccv_cnnp_model_t* _ccv_cnnp_reshape_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_reshape_t* const self = (const ccv_cnnp_model_reshape_t*)super;
	return ccv_cnnp_reshape(self->format, self->dim, self->ofs, self->stride, self->super.name);
}

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_identity_t;

static void _ccv_cnnp_identity_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	outputs[0] = inputs[0];
}

static ccv_cnnp_model_t* _ccv_cnnp_identity_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_identity_isa = {
	.build = _ccv_cnnp_identity_build,
	.copy = _ccv_cnnp_identity_copy,
};

ccv_cnnp_model_t* ccv_cnnp_identity(const char* const name)
{
	ccv_cnnp_model_identity_t* const model_identity = (ccv_cnnp_model_identity_t*)cccalloc(1, sizeof(ccv_cnnp_model_identity_t));
	model_identity->super.isa = &ccv_cnnp_identity_isa;
	model_identity->super.input_size = 1;
	model_identity->super.outputs = &model_identity->output;
	model_identity->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_identity->super, name);
	return (ccv_cnnp_model_t*)model_identity;
}

static ccv_cnnp_model_t* _ccv_cnnp_identity_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_identity_t* const self = (const ccv_cnnp_model_identity_t*)super;
	return ccv_cnnp_identity(self->super.name);
}

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int index[CCV_NNC_MAX_DIM_ALLOC];
} ccv_cnnp_model_permute_t;

static void _ccv_cnnp_permute_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_permute_t* const self = (ccv_cnnp_model_permute_t*)super;
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_symbol_t to = ccv_nnc_tensor_symbol_alias_to(graph, inputs[0]);
	const int nd = ccv_nnc_tensor_nd(params.dim);
	int input_dim[CCV_NNC_MAX_DIM_ALLOC];
	memcpy(input_dim, params.dim, sizeof(params.dim));
	int input_stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	int output_stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	if (to.d == CCV_NNC_NO_TENSOR_SYMBOL) // If it is not an alias. Find stride and permute.
	{
		ccv_nnc_tensor_get_stride(input_dim, input_stride);
		int i;
		for (i = 0; i < nd; i++)
		{
			const int idx = self->index[i];
			assert(idx >= 0 && idx < nd);
			params.dim[i] = input_dim[idx];
			output_stride[i] = input_stride[idx];
		}
		outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, inputs[0], ccv_nnc_no_ofs, output_stride, params, 0);
	} else {
		// if it is an alias, we can get the stride from it and use that.
		int input_ofs[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_symbol_alias_params(graph, inputs[0], input_ofs, input_stride);
		assert(input_stride[0] != 0);
		int output_ofs[CCV_NNC_MAX_DIM_ALLOC] = {};
		int i;
		for (i = 0; i < nd; i++)
		{
			const int idx = self->index[i];
			assert(idx >= 0 && idx < nd);
			params.dim[i] = input_dim[idx];
			output_stride[i] = input_stride[idx];
			output_ofs[i] = input_ofs[idx];
		}
		outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, inputs[0], output_ofs, output_stride, params, 0);
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_permute_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_permute_isa = {
	.build = _ccv_cnnp_permute_build,
	.copy = _ccv_cnnp_permute_copy,
};

ccv_cnnp_model_t* ccv_cnnp_permute(const int index[CCV_NNC_MAX_DIM_ALLOC], const char* const name)
{
	ccv_cnnp_model_permute_t* const model_permute = (ccv_cnnp_model_permute_t*)cccalloc(1, sizeof(ccv_cnnp_model_permute_t));
	model_permute->super.isa = &ccv_cnnp_permute_isa;
	model_permute->super.input_size = 1;
	model_permute->super.outputs = &model_permute->output;
	model_permute->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_permute->super, name);
	memcpy(model_permute->index, index, sizeof(model_permute->index));
	return (ccv_cnnp_model_t*)model_permute;
}

static ccv_cnnp_model_t* _ccv_cnnp_permute_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_permute_t* const self = (const ccv_cnnp_model_permute_t*)super;
	return ccv_cnnp_permute(self->index, self->super.name);
}

typedef struct {
	ccv_cnnp_model_t super;
	int index;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_extract_t;

static void _ccv_cnnp_extract_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	ccv_cnnp_model_extract_t* const self = (ccv_cnnp_model_extract_t*)super;
	outputs[0] = inputs[self->index];
}

static ccv_cnnp_model_t* _ccv_cnnp_extract_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_extract_isa = {
	.build = _ccv_cnnp_extract_build,
	.copy = _ccv_cnnp_extract_copy,
};

ccv_cnnp_model_t* ccv_cnnp_extract(const int index, const char* const name)
{
	ccv_cnnp_model_extract_t* const model_extract = (ccv_cnnp_model_extract_t*)cccalloc(1, sizeof(ccv_cnnp_model_extract_t));
	model_extract->index = index;
	model_extract->super.isa = &ccv_cnnp_extract_isa;
	model_extract->super.input_size = 0;
	model_extract->super.outputs = &model_extract->output;
	model_extract->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_extract->super, name);
	return (ccv_cnnp_model_t*)model_extract;
}

static ccv_cnnp_model_t* _ccv_cnnp_extract_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_extract_t* const self = (ccv_cnnp_model_extract_t*)super;
	return ccv_cnnp_extract(self->index, self->super.name);
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
	int stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	ccv_nnc_tensor_get_stride(output_params.dim, stride);
	outputs[0] = ccv_nnc_tensor_symbol_alias_new(graph, inputs[0], DIM_ALLOC(), stride, output_params, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_flatten_copy(const ccv_cnnp_model_t* const self, void* const context);

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

static ccv_cnnp_model_t* _ccv_cnnp_flatten_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	return ccv_cnnp_flatten(self->name);
}

// MARK - Batch Norm Layer

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
	const int nd = ccv_nnc_tensor_nd(params.dim);
	ccv_nnc_tensor_param_t bias_params = params;
	memset(bias_params.dim, 0, sizeof(bias_params.dim));
	// If the accuracy is not enough, bump it to 32-bit floating point.
	if (bias_params.datatype != CCV_32F && bias_params.datatype != CCV_64F)
		bias_params.datatype = CCV_32F;
	bias_params.dim[0] = nd > 1 ? ccv_nnc_tensor_get_c(params) : params.dim[0];
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, params, 0);
	// Both scale and bias are shared between if this model is reused.
	if (!self->scale.graph)
		self->scale = ccv_nnc_tensor_symbol_new(graph, bias_params, "scale");
	if (!self->bias.graph)
		self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, "bias");
	const ccv_nnc_tensor_symbol_t mean = ccv_nnc_tensor_symbol_new(graph, bias_params, "mean");
	const ccv_nnc_tensor_symbol_t var = ccv_nnc_tensor_symbol_new(graph, bias_params, "var");
	// Otherwise, notice mean, var, saved_mean, saved_inv_std are not reused.
	if (!self->zero_inits)
		self->zero_inits = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_push(self->zero_inits, &mean);
	ccv_array_push(self->zero_inits, &var);
	const ccv_nnc_tensor_symbol_t out_mean = ccv_nnc_tensor_symbol_new(graph, bias_params, "out_mean");
	const ccv_nnc_tensor_symbol_t out_var = ccv_nnc_tensor_symbol_new(graph, bias_params, "out_var");
	if (!self->retainables)
		self->retainables = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_push(self->retainables, &out_mean);
	ccv_array_push(self->retainables, &out_var);
	const ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(graph, bias_params, "saved_mean");
	const ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(graph, bias_params, "saved_inv_std");
	const int hw = ccv_nnc_tensor_hw(params, ccv_nnc_tensor_nd(params.dim));
	ccv_nnc_cmd_param_t batch_norm = self->params;
	batch_norm.bnorm.count = hw >= 0 ? CCV_NNC_MAX_DIM + 1 : 1;
	int i;
	batch_norm.bnorm.axis[0] = (params.format == CCV_TENSOR_FORMAT_CHWN) ? 3 : 0;
	if (hw >= 0)
		for (i = 0; i < CCV_NNC_MAX_DIM; i++)
			batch_norm.bnorm.axis[i + 1] = i + hw;
	self->params = batch_norm;
	self->batch_norm = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_BATCH_NORM_FORWARD, 0, batch_norm, 0), TENSOR_SYMBOL_LIST(inputs[0], self->scale, self->bias, mean, var), TENSOR_SYMBOL_LIST(output, out_mean, out_var, saved_mean, saved_inv_std), "batch_norm");
	outputs[0] = output;
}

static void _ccv_cnnp_batch_norm_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	if (self->scale.graph)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(0, 1), ccv_nnc_no_hint, 0, 0, self->scale);
	if (self->bias.graph)
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
	int i;
	if (self->zero_inits)
		for (i = 0; i < self->zero_inits->rnum; i++)
			initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, *(ccv_nnc_tensor_symbol_t*)ccv_array_get(self->zero_inits, i));
}

static void _ccv_cnnp_batch_norm_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	if (self->scale.graph)
		add_to_array(parameters, self->scale, is_trainable);
	if (self->bias.graph)
		add_to_array(parameters, self->bias, is_trainable);
}

static void _ccv_cnnp_batch_norm_add_to_output(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	ccv_cnnp_model_batch_norm_t* const self = (ccv_cnnp_model_batch_norm_t*)super;
	int i;
	if (self->retainables)
		for (i = 0; i < self->retainables->rnum; i++)
		{
			const ccv_nnc_tensor_symbol_t symbol = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(self->retainables, i);
			add_to_array(outputs, symbol, 0);
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

static ccv_cnnp_model_t* _ccv_cnnp_batch_norm_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_batch_norm_isa = {
	.build = _ccv_cnnp_batch_norm_build,
	.init_states = _ccv_cnnp_batch_norm_init_states,
	.add_to_parameter = _ccv_cnnp_batch_norm_add_to_parameter,
	.add_to_output = _ccv_cnnp_batch_norm_add_to_output,
	.copy = _ccv_cnnp_batch_norm_copy,
	.set_is_test = _ccv_cnnp_batch_norm_set_is_test,
	.deinit = _ccv_cnnp_batch_norm_deinit,
};

ccv_cnnp_model_t* ccv_cnnp_batch_norm(const float momentum, const float epsilon, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_batch_norm_t* const model_batch_norm = (ccv_cnnp_model_batch_norm_t*)cccalloc(1, sizeof(ccv_cnnp_model_batch_norm_t));
	model_batch_norm->super.isa = &ccv_cnnp_batch_norm_isa;
	model_batch_norm->super.input_size = 1;
	model_batch_norm->super.outputs = &model_batch_norm->output;
	model_batch_norm->super.output_size = 1;
	model_batch_norm->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_batch_norm->super, name);
	model_batch_norm->scale.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_batch_norm->scale.graph = 0;
	model_batch_norm->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_batch_norm->bias.graph = 0;
	model_batch_norm->params.bnorm.momentum = momentum;
	model_batch_norm->params.bnorm.epsilon = epsilon;
	return (ccv_cnnp_model_t*)model_batch_norm;
}

static ccv_cnnp_model_t* _ccv_cnnp_batch_norm_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_batch_norm_t* const self = (const ccv_cnnp_model_batch_norm_t*)super;
	return ccv_cnnp_batch_norm(self->params.bnorm.momentum, self->params.bnorm.epsilon, self->super.is_trainable, self->super.name);
}

// MARK - Convolution Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t weights;
	ccv_nnc_tensor_symbol_t bias;
	int groups;
	int filters;
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	int no_bias;
	int format;
	ccv_nnc_hint_t hint;
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
	if (self->format)
		weights_params.format = self->format;
	ccv_nnc_tensor_set_n(&weights_params, self->filters);
	assert(ccv_nnc_tensor_get_c(params) % self->groups == 0);
	ccv_nnc_tensor_set_c(&weights_params, nd, ccv_nnc_tensor_get_c(params) / self->groups);
	const int hw = ccv_nnc_tensor_hw(weights_params, nd);
	assert(hw >= 0);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		weights_params.dim[i + hw] = self->kdim[i];
	if (!self->weights.graph)
		self->weights = ccv_nnc_tensor_symbol_new(graph, weights_params, "weights");
	assert(self->weights.graph == graph);
	ccv_nnc_tensor_param_t bias_params = params;
	if (self->format)
		bias_params.format = self->format;
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
		}, 3, self->hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_t convolution;
	if (self->no_bias)
		convolution = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights), TENSOR_SYMBOL_LIST(output), "convolution");
	else {
		if (!self->bias.graph)
			self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, "bias");
		convolution = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights, self->bias), TENSOR_SYMBOL_LIST(output), "convolution");
	}
	ccv_nnc_graph_exec_symbol_set_hint(graph, convolution, self->hint);
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
}

static void _ccv_cnnp_convolution_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_convolution_t* const self = (ccv_cnnp_model_convolution_t*)super;
	add_to_array(parameters, self->weights, is_trainable);
	if (self->bias.graph)
		add_to_array(parameters, self->bias, is_trainable);
}

static ccv_cnnp_model_t* _ccv_cnnp_convolution_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_convolution_isa = {
	.build = _ccv_cnnp_convolution_build,
	.init_states = _ccv_cnnp_convolution_init_states,
	.add_to_parameter = _ccv_cnnp_convolution_add_to_parameter,
	.copy = _ccv_cnnp_convolution_copy,
};

ccv_cnnp_model_t* ccv_cnnp_convolution(const int groups, const int filters, const int kdim[CCV_NNC_MAX_DIM_ALLOC], const int no_bias, ccv_nnc_hint_t hint, const int format, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_convolution_t* const model_convolution = (ccv_cnnp_model_convolution_t*)cccalloc(1, sizeof(ccv_cnnp_model_convolution_t));
	model_convolution->super.isa = &ccv_cnnp_convolution_isa;
	model_convolution->super.input_size = 1;
	model_convolution->super.outputs = &model_convolution->output;
	model_convolution->super.output_size = 1;
	model_convolution->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_convolution->super, name);
	model_convolution->weights.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_convolution->weights.graph = 0;
	model_convolution->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_convolution->bias.graph = 0;
	model_convolution->groups = groups;
	model_convolution->filters = filters;
	memcpy(model_convolution->kdim, kdim, sizeof(model_convolution->kdim));
	model_convolution->no_bias = no_bias;
	model_convolution->hint = hint;
	model_convolution->format = format;
	return (ccv_cnnp_model_t*)model_convolution;
}

static ccv_cnnp_model_t* _ccv_cnnp_convolution_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_convolution_t* const self = (ccv_cnnp_model_convolution_t*)super;
	return ccv_cnnp_convolution(self->groups, self->filters, self->kdim, self->no_bias, self->hint, self->format, self->super.is_trainable, self->super.name);
}

// MARK - Dense Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t weights;
	ccv_nnc_tensor_symbol_t bias;
	int count;
	int no_bias;
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
		self->weights = ccv_nnc_tensor_symbol_new(graph, weights_params, "weights");
	assert(self->weights.graph == graph);
	ccv_nnc_tensor_param_t bias_params = params;
	memset(bias_params.dim, 0, sizeof(bias_params.dim));
	bias_params.dim[0] = self->count;
	ccv_nnc_cmd_t cmd = {0};
	cmd.cmd = CCV_NNC_GEMM_FORWARD;
	cmd.info.blas.a[0] = 1;
	cmd.info.blas.a[1] = 1;
	cmd.info.blas.transpose_b[0] = 0;
	cmd.info.blas.transpose_b[1] = 1;
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_hint_tensor_auto(cmd, (ccv_nnc_tensor_param_t []){
			params,
			weights_params,
			bias_params,
		}, 3, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	if (self->no_bias)
		ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights), TENSOR_SYMBOL_LIST(output), "dense");
	else {
		if (!self->bias.graph)
			self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, "bias");
		ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], self->weights, self->bias), TENSOR_SYMBOL_LIST(output), "dense");
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
}

static void _ccv_cnnp_dense_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_dense_t* const self = (ccv_cnnp_model_dense_t*)super;
	add_to_array(parameters, self->weights, is_trainable);
	if (self->bias.graph)
		add_to_array(parameters, self->bias, is_trainable);
}

static ccv_cnnp_model_t* _ccv_cnnp_dense_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_dense_isa = {
	.build = _ccv_cnnp_dense_build,
	.init_states = _ccv_cnnp_dense_init_states,
	.add_to_parameter = _ccv_cnnp_dense_add_to_parameter,
	.copy = _ccv_cnnp_dense_copy,
};

ccv_cnnp_model_t* ccv_cnnp_dense(const int count, const int no_bias, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_dense_t* const model_dense = (ccv_cnnp_model_dense_t*)cccalloc(1, sizeof(ccv_cnnp_model_dense_t));
	model_dense->super.isa = &ccv_cnnp_dense_isa;
	model_dense->super.input_size = 1;
	model_dense->super.outputs = &model_dense->output;
	model_dense->super.output_size = 1;
	model_dense->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_dense->super, name);
	model_dense->weights.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_dense->weights.graph = 0;
	model_dense->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_dense->bias.graph = 0;
	model_dense->count = count;
	model_dense->no_bias = no_bias;
	return (ccv_cnnp_model_t*)model_dense;
}

static ccv_cnnp_model_t* _ccv_cnnp_dense_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_dense_t* const self = (const ccv_cnnp_model_dense_t*)super;
	return ccv_cnnp_dense(self->count, self->no_bias, self->super.is_trainable, self->super.name);
}

// MARK - Pool Layers

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_hint_t hint;
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
	ccv_nnc_hint_tensor_auto(cmd, &params, 1, self->hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t pool_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	const ccv_nnc_graph_exec_symbol_t exec = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(pool_output), "max_pool");
	ccv_nnc_graph_exec_symbol_set_hint(graph, exec, self->hint);
	outputs[0] = pool_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_max_pool_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_max_pool_isa = {
	.build = _ccv_cnnp_max_pool_build,
	.copy = _ccv_cnnp_max_pool_copy,
};

ccv_cnnp_model_t* ccv_cnnp_max_pool(const int kdim[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_hint_t hint, const char* const name)
{
	ccv_cnnp_model_pool_t* const model_pool = (ccv_cnnp_model_pool_t*)cccalloc(1, sizeof(ccv_cnnp_model_pool_t));
	model_pool->super.isa = &ccv_cnnp_max_pool_isa;
	model_pool->super.input_size = 1;
	model_pool->super.outputs = &model_pool->output;
	model_pool->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_pool->super, name);
	memcpy(model_pool->kdim, kdim, sizeof(model_pool->kdim));
	model_pool->hint = hint;
	return (ccv_cnnp_model_t*)model_pool;
}

static ccv_cnnp_model_t* _ccv_cnnp_max_pool_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_pool_t* const self = (const ccv_cnnp_model_pool_t*)super;
	return ccv_cnnp_max_pool(self->kdim, self->hint, self->super.name);
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
	ccv_nnc_hint_tensor_auto(cmd, &params, 1, self->hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t pool_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	const ccv_nnc_graph_exec_symbol_t exec = ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(pool_output), "average_pool");
	ccv_nnc_graph_exec_symbol_set_hint(graph, exec, self->hint);
	outputs[0] = pool_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_average_pool_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_average_pool_isa = {
	.build = _ccv_cnnp_average_pool_build,
	.copy = _ccv_cnnp_average_pool_copy,
};

ccv_cnnp_model_t* ccv_cnnp_average_pool(const int kdim[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_hint_t hint, const char* const name)
{
	ccv_cnnp_model_pool_t* const model_pool = (ccv_cnnp_model_pool_t*)cccalloc(1, sizeof(ccv_cnnp_model_pool_t));
	model_pool->super.isa = &ccv_cnnp_average_pool_isa;
	model_pool->super.input_size = 1;
	model_pool->super.outputs = &model_pool->output;
	model_pool->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_pool->super, name);
	memcpy(model_pool->kdim, kdim, sizeof(model_pool->kdim));
	model_pool->hint = hint;
	return (ccv_cnnp_model_t*)model_pool;
}

static ccv_cnnp_model_t* _ccv_cnnp_average_pool_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_pool_t* const self = (const ccv_cnnp_model_pool_t*)super;
	return ccv_cnnp_average_pool(self->kdim, self->hint, self->super.name);
}

// MARK - RELU Layer

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
	ccv_nnc_graph_exec_symbol_new(graph, relu, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(relu_output), "relu");
	outputs[0] = relu_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_relu_copy(const ccv_cnnp_model_t* const self, void* const context);

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

static ccv_cnnp_model_t* _ccv_cnnp_relu_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	return ccv_cnnp_relu(self->name);
}

// MARK - Sigmoid Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_sigmoid_t;

static void _ccv_cnnp_sigmoid_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t sigmoid = CMD_SIGMOID_FORWARD();
	ccv_nnc_hint_tensor_auto(sigmoid, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t sigmoid_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, sigmoid, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(sigmoid_output), "sigmoid");
	outputs[0] = sigmoid_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_sigmoid_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_sigmoid_isa = {
	.build = _ccv_cnnp_sigmoid_build,
	.copy = _ccv_cnnp_sigmoid_copy,
};

ccv_cnnp_model_t* ccv_cnnp_sigmoid(const char* const name)
{
	ccv_cnnp_model_sigmoid_t* const model_sigmoid = (ccv_cnnp_model_sigmoid_t*)cccalloc(1, sizeof(ccv_cnnp_model_sigmoid_t));
	model_sigmoid->super.isa = &ccv_cnnp_sigmoid_isa;
	model_sigmoid->super.input_size = 1;
	model_sigmoid->super.outputs = &model_sigmoid->output;
	model_sigmoid->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_sigmoid->super, name);
	return (ccv_cnnp_model_t*)model_sigmoid;
}

static ccv_cnnp_model_t* _ccv_cnnp_sigmoid_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	return ccv_cnnp_sigmoid(self->name);
}

// MARK - Tanh Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_tanh_t;

static void _ccv_cnnp_tanh_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t tanh = CMD_TANH_FORWARD();
	ccv_nnc_hint_tensor_auto(tanh, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t tanh_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, tanh, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(tanh_output), "tanh");
	outputs[0] = tanh_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_tanh_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_tanh_isa = {
	.build = _ccv_cnnp_tanh_build,
	.copy = _ccv_cnnp_tanh_copy,
};

ccv_cnnp_model_t* ccv_cnnp_tanh(const char* const name)
{
	ccv_cnnp_model_tanh_t* const model_tanh = (ccv_cnnp_model_tanh_t*)cccalloc(1, sizeof(ccv_cnnp_model_tanh_t));
	model_tanh->super.isa = &ccv_cnnp_tanh_isa;
	model_tanh->super.input_size = 1;
	model_tanh->super.outputs = &model_tanh->output;
	model_tanh->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_tanh->super, name);
	return (ccv_cnnp_model_t*)model_tanh;
}

static ccv_cnnp_model_t* _ccv_cnnp_tanh_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	return ccv_cnnp_tanh(self->name);
}

// MARK - Swish Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_swish_t;

static void _ccv_cnnp_swish_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t swish = CMD_SWISH_FORWARD();
	ccv_nnc_hint_tensor_auto(swish, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t swish_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, swish, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(swish_output), "swish");
	outputs[0] = swish_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_swish_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_swish_isa = {
	.build = _ccv_cnnp_swish_build,
	.copy = _ccv_cnnp_swish_copy,
};

ccv_cnnp_model_t* ccv_cnnp_swish(const char* const name)
{
	ccv_cnnp_model_swish_t* const model_swish = (ccv_cnnp_model_swish_t*)cccalloc(1, sizeof(ccv_cnnp_model_swish_t));
	model_swish->super.isa = &ccv_cnnp_swish_isa;
	model_swish->super.input_size = 1;
	model_swish->super.outputs = &model_swish->output;
	model_swish->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_swish->super, name);
	return (ccv_cnnp_model_t*)model_swish;
}

static ccv_cnnp_model_t* _ccv_cnnp_swish_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	return ccv_cnnp_swish(self->name);
}

// MARK - GELU Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int tanh;
} ccv_cnnp_model_gelu_t;

static void _ccv_cnnp_gelu_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_gelu_t* const self = (ccv_cnnp_model_gelu_t*)super;
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t gelu = CMD_GELU_FORWARD(self->tanh);
	ccv_nnc_hint_tensor_auto(gelu, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t gelu_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, gelu, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(gelu_output), "gelu");
	outputs[0] = gelu_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_gelu_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_gelu_isa = {
	.build = _ccv_cnnp_gelu_build,
	.copy = _ccv_cnnp_gelu_copy,
};

ccv_cnnp_model_t* ccv_cnnp_gelu(const int tanh, const char* const name)
{
	ccv_cnnp_model_gelu_t* const model_gelu = (ccv_cnnp_model_gelu_t*)cccalloc(1, sizeof(ccv_cnnp_model_gelu_t));
	model_gelu->super.isa = &ccv_cnnp_gelu_isa;
	model_gelu->super.input_size = 1;
	model_gelu->super.outputs = &model_gelu->output;
	model_gelu->super.output_size = 1;
	model_gelu->tanh = tanh;
	ccv_cnnp_model_copy_name(&model_gelu->super, name);
	return (ccv_cnnp_model_t*)model_gelu;
}

static ccv_cnnp_model_t* _ccv_cnnp_gelu_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_gelu_t* const self = (ccv_cnnp_model_gelu_t*)super;
	return ccv_cnnp_gelu(self->tanh, self->super.name);
}

// MARK - Leaky ReLU Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	float negative_slope;
} ccv_cnnp_model_leaky_relu_t;

static void _ccv_cnnp_leaky_relu_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_leaky_relu_t* const self = (ccv_cnnp_model_leaky_relu_t*)super;
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t leaky_relu = CMD_LEAKY_RELU_FORWARD(self->negative_slope);
	ccv_nnc_hint_tensor_auto(leaky_relu, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t leaky_relu_output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, leaky_relu, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(leaky_relu_output), "leaky_relu");
	outputs[0] = leaky_relu_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_leaky_relu_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_leaky_relu_isa = {
	.build = _ccv_cnnp_leaky_relu_build,
	.copy = _ccv_cnnp_leaky_relu_copy,
};

ccv_cnnp_model_t* ccv_cnnp_leaky_relu(const float negative_slope, const char* const name)
{
	ccv_cnnp_model_leaky_relu_t* const model_leaky_relu = (ccv_cnnp_model_leaky_relu_t*)cccalloc(1, sizeof(ccv_cnnp_model_leaky_relu_t));
	model_leaky_relu->super.isa = &ccv_cnnp_leaky_relu_isa;
	model_leaky_relu->super.input_size = 1;
	model_leaky_relu->super.outputs = &model_leaky_relu->output;
	model_leaky_relu->super.output_size = 1;
	model_leaky_relu->negative_slope = negative_slope;
	ccv_cnnp_model_copy_name(&model_leaky_relu->super, name);
	return (ccv_cnnp_model_t*)model_leaky_relu;
}

static ccv_cnnp_model_t* _ccv_cnnp_leaky_relu_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_leaky_relu_t* const self = (ccv_cnnp_model_leaky_relu_t*)super;
	return ccv_cnnp_leaky_relu(self->negative_slope, self->super.name);
}

// MARK - Softmax Layer

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
	ccv_nnc_graph_exec_symbol_new(graph, softmax, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(softmax_output), "softmax");
	outputs[0] = softmax_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_softmax_copy(const ccv_cnnp_model_t* const self, void* const context);

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

static ccv_cnnp_model_t* _ccv_cnnp_softmax_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	return ccv_cnnp_softmax(self->name);
}

// MARK - Add Layer

typedef struct {
	ccv_cnnp_model_t super;
	float p;
	float q;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_add_t;

static void _ccv_cnnp_add_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_add_t* const self = (const ccv_cnnp_model_add_t*)super;
	assert(input_size == 2);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params[2];
	int i;
	for (i = 0; i < 2; i++)
		input_params[i] = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t add = CMD_ADD_FORWARD(self->p, self->q);
	ccv_nnc_hint_tensor_auto(add, input_params, 2, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, add, inputs, input_size, outputs, output_size, "add");
}

static ccv_cnnp_model_t* _ccv_cnnp_add_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_add_isa = {
	.build = _ccv_cnnp_add_build,
	.copy = _ccv_cnnp_add_copy,
};

ccv_cnnp_model_t* ccv_cnnp_add(const float p, const float q, const char* const name)
{
	ccv_cnnp_model_add_t* const model_add = (ccv_cnnp_model_add_t*)cccalloc(1, sizeof(ccv_cnnp_model_add_t));
	model_add->super.isa = &ccv_cnnp_add_isa;
	model_add->super.input_size = 2;
	model_add->super.outputs = &model_add->output;
	model_add->super.output_size = 1;
	model_add->p = p;
	model_add->q = q;
	ccv_cnnp_model_copy_name(&model_add->super, name);
	return (ccv_cnnp_model_t*)model_add;
}

static ccv_cnnp_model_t* _ccv_cnnp_add_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_add_t* const self = (const ccv_cnnp_model_add_t*)super;
	return ccv_cnnp_add(self->p, self->q, self->super.name);
}

// MARK - Mul Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	float p;
} ccv_cnnp_model_mul_t;

static void _ccv_cnnp_mul_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_mul_t* const self = (const ccv_cnnp_model_mul_t*)super;
	assert(input_size == 2);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params[2];
	int i;
	for (i = 0; i < 2; i++)
		input_params[i] = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t mul = CMD_MUL_FORWARD(self->p);
	ccv_nnc_hint_tensor_auto(mul, input_params, 2, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, mul, inputs, input_size, outputs, output_size, "mul");
}

static ccv_cnnp_model_t* _ccv_cnnp_mul_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_mul_isa = {
	.build = _ccv_cnnp_mul_build,
	.copy = _ccv_cnnp_mul_copy,
};

ccv_cnnp_model_t* ccv_cnnp_mul(const float p, const char* const name)
{
	ccv_cnnp_model_mul_t* const model_mul = (ccv_cnnp_model_mul_t*)cccalloc(1, sizeof(ccv_cnnp_model_mul_t));
	model_mul->super.isa = &ccv_cnnp_mul_isa;
	model_mul->super.input_size = 2;
	model_mul->super.outputs = &model_mul->output;
	model_mul->super.output_size = 1;
	model_mul->p = p;
	ccv_cnnp_model_copy_name(&model_mul->super, name);
	return (ccv_cnnp_model_t*)model_mul;
}

static ccv_cnnp_model_t* _ccv_cnnp_mul_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_mul_t* const self = (const ccv_cnnp_model_mul_t*)super;
	return ccv_cnnp_mul(self->p, self->super.name);
}

// MARK - Scalar Mul Layer

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
	ccv_nnc_graph_exec_symbol_new(graph, scalar_mul, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(scalar_mul_output), "scalar_mul");
	outputs[0] = scalar_mul_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_scalar_mul_copy(const ccv_cnnp_model_t* const super, void* const context);

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

static ccv_cnnp_model_t* _ccv_cnnp_scalar_mul_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_scalar_mul_t* const self = (const ccv_cnnp_model_scalar_mul_t*)super;
	return ccv_cnnp_scalar_mul(self->a, self->super.name);
}

// MARK - Transpose Layer

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
	ccv_nnc_graph_exec_symbol_new(graph, transpose, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(transpose_output), "transpose");
	outputs[0] = transpose_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_transpose_copy(const ccv_cnnp_model_t* const super, void* const context);

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

static ccv_cnnp_model_t* _ccv_cnnp_transpose_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_transpose_t* const self = (const ccv_cnnp_model_transpose_t*)super;
	return ccv_cnnp_transpose(self->transpose[0], self->transpose[1], self->super.name);
}

// MARK - Layer Norm Layer

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
	const int nd = ccv_nnc_tensor_nd(params.dim);
	int i;
	for (i = 0; i < nd; i++)
		bias_params.dim[i] = 1;
	for (i = 0; i < self->params.lnorm.count; i++)
		bias_params.dim[self->params.lnorm.axis[i]] = params.dim[self->params.lnorm.axis[i]];
	// Both scale and bias are shared between if this model is reused.
	if (!self->scale.graph)
		self->scale = ccv_nnc_tensor_symbol_new(graph, bias_params, "scale");
	if (!self->bias.graph)
		self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, "bias");
	const ccv_nnc_cmd_t layer_norm = ccv_nnc_cmd(CCV_NNC_LAYER_NORM_FORWARD, 0, self->params, 0);
	ccv_nnc_tensor_param_t output_params[3];
	ccv_nnc_hint_tensor_auto(layer_norm, (ccv_nnc_tensor_param_t []){
			params,
			bias_params,
			bias_params,
		}, 3, ccv_nnc_no_hint, output_params, 3);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
	const ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(graph, output_params[1], "saved_mean");
	const ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(graph, output_params[2], "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(graph, layer_norm, TENSOR_SYMBOL_LIST(inputs[0], self->scale, self->bias), TENSOR_SYMBOL_LIST(output, saved_mean, saved_inv_std), "layer_norm");
	outputs[0] = output;
}

static void _ccv_cnnp_layer_norm_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_layer_norm_t* const self = (ccv_cnnp_model_layer_norm_t*)super;
	if (self->scale.graph)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(0, 1), ccv_nnc_no_hint, 0, 0, self->scale);
	if (self->bias.graph)
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
}

static void _ccv_cnnp_layer_norm_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_layer_norm_t* const self = (ccv_cnnp_model_layer_norm_t*)super;
	if (self->scale.graph)
		add_to_array(parameters, self->scale, is_trainable);
	if (self->bias.graph)
		add_to_array(parameters, self->bias, is_trainable);
}

static ccv_cnnp_model_t* _ccv_cnnp_layer_norm_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_layer_norm_isa = {
	.build = _ccv_cnnp_layer_norm_build,
	.init_states = _ccv_cnnp_layer_norm_init_states,
	.add_to_parameter = _ccv_cnnp_layer_norm_add_to_parameter,
	.copy = _ccv_cnnp_layer_norm_copy,
};

ccv_cnnp_model_t* ccv_cnnp_layer_norm(const float epsilon, const int axis[CCV_NNC_MAX_DIM_ALLOC], const int axis_count, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_layer_norm_t* const model_layer_norm = (ccv_cnnp_model_layer_norm_t*)cccalloc(1, sizeof(ccv_cnnp_model_layer_norm_t));
	model_layer_norm->super.isa = &ccv_cnnp_layer_norm_isa;
	model_layer_norm->super.input_size = 1;
	model_layer_norm->super.outputs = &model_layer_norm->output;
	model_layer_norm->super.output_size = 1;
	model_layer_norm->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_layer_norm->super, name);
	model_layer_norm->scale.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_layer_norm->scale.graph = 0;
	model_layer_norm->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_layer_norm->bias.graph = 0;
	model_layer_norm->params.lnorm.epsilon = epsilon;
	model_layer_norm->params.lnorm.count = axis_count;
	memcpy(model_layer_norm->params.lnorm.axis, axis, sizeof(int) * axis_count);
	return (ccv_cnnp_model_t*)model_layer_norm;
}

static ccv_cnnp_model_t* _ccv_cnnp_layer_norm_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_layer_norm_t* const self = (const ccv_cnnp_model_layer_norm_t*)super;
	return ccv_cnnp_layer_norm(self->params.lnorm.epsilon, self->params.lnorm.axis, self->params.lnorm.count, self->super.is_trainable, self->super.name);
}

// MARK - Group Norm Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t bias;
	ccv_nnc_tensor_symbol_t scale;
	ccv_nnc_cmd_param_t params;
} ccv_cnnp_model_group_norm_t;

static void _ccv_cnnp_group_norm_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_group_norm_t* const self = (ccv_cnnp_model_group_norm_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t bias_params = params;
	const int nd = ccv_nnc_tensor_nd(params.dim);
	int i;
	for (i = 0; i < nd; i++)
		bias_params.dim[i] = 1;
	bias_params.dim[self->params.gnorm.group_axis] = params.dim[self->params.gnorm.group_axis];
	// Both scale and bias are shared between if this model is reused.
	if (!self->scale.graph)
		self->scale = ccv_nnc_tensor_symbol_new(graph, bias_params, "scale");
	if (!self->bias.graph)
		self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, "bias");
	const ccv_nnc_cmd_t group_norm = ccv_nnc_cmd(CCV_NNC_GROUP_NORM_FORWARD, 0, self->params, 0);
	ccv_nnc_tensor_param_t output_params[3];
	ccv_nnc_hint_tensor_auto(group_norm, (ccv_nnc_tensor_param_t []){
			params,
			bias_params,
			bias_params,
		}, 3, ccv_nnc_no_hint, output_params, 3);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
	const ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(graph, output_params[1], "saved_mean");
	const ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(graph, output_params[2], "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(graph, group_norm, TENSOR_SYMBOL_LIST(inputs[0], self->scale, self->bias), TENSOR_SYMBOL_LIST(output, saved_mean, saved_inv_std), "group_norm");
	outputs[0] = output;
}

static void _ccv_cnnp_group_norm_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_group_norm_t* const self = (ccv_cnnp_model_group_norm_t*)super;
	if (self->scale.graph)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(0, 1), ccv_nnc_no_hint, 0, 0, self->scale);
	if (self->bias.graph)
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
}

static void _ccv_cnnp_group_norm_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_group_norm_t* const self = (ccv_cnnp_model_group_norm_t*)super;
	if (self->scale.graph)
		add_to_array(parameters, self->scale, is_trainable);
	if (self->bias.graph)
		add_to_array(parameters, self->bias, is_trainable);
}

static ccv_cnnp_model_t* _ccv_cnnp_group_norm_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_group_norm_isa = {
	.build = _ccv_cnnp_group_norm_build,
	.init_states = _ccv_cnnp_group_norm_init_states,
	.add_to_parameter = _ccv_cnnp_group_norm_add_to_parameter,
	.copy = _ccv_cnnp_group_norm_copy,
};

ccv_cnnp_model_t* ccv_cnnp_group_norm(const int group_axis, const int groups, const float epsilon, const int reduce_axis[CCV_NNC_MAX_DIM_ALLOC], const int axis_count, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_group_norm_t* const model_group_norm = (ccv_cnnp_model_group_norm_t*)cccalloc(1, sizeof(ccv_cnnp_model_group_norm_t));
	model_group_norm->super.isa = &ccv_cnnp_group_norm_isa;
	model_group_norm->super.input_size = 1;
	model_group_norm->super.outputs = &model_group_norm->output;
	model_group_norm->super.output_size = 1;
	model_group_norm->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_group_norm->super, name);
	model_group_norm->scale.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_group_norm->scale.graph = 0;
	model_group_norm->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_group_norm->bias.graph = 0;
	model_group_norm->params.gnorm.group_axis = group_axis;
	model_group_norm->params.gnorm.groups = groups;
	model_group_norm->params.gnorm.epsilon = epsilon;
	model_group_norm->params.gnorm.reduce_count = axis_count;
	memcpy(model_group_norm->params.gnorm.reduce_axis, reduce_axis, sizeof(int) * axis_count);
	return (ccv_cnnp_model_t*)model_group_norm;
}

static ccv_cnnp_model_t* _ccv_cnnp_group_norm_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_group_norm_t* const self = (const ccv_cnnp_model_group_norm_t*)super;
	return ccv_cnnp_group_norm(self->params.gnorm.group_axis, self->params.gnorm.groups, self->params.gnorm.epsilon, self->params.gnorm.reduce_axis, self->params.gnorm.reduce_count, self->super.is_trainable, self->super.name);
}

// MARK - Batched Matrix Mul Layer

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
	ccv_nnc_graph_exec_symbol_new(graph, matmul, inputs, input_size, TENSOR_SYMBOL_LIST(matmul_output), "matmul");
	outputs[0] = matmul_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_matmul_copy(const ccv_cnnp_model_t* const super, void* const context);

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

static ccv_cnnp_model_t* _ccv_cnnp_matmul_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_matmul_t* const self = (const ccv_cnnp_model_matmul_t*)super;
	return ccv_cnnp_matmul(self->transpose_a, self->transpose_b, self->super.name);
}

// MARK - Dropout Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_graph_exec_symbol_t dropout;
	float p;
	int entirety;
} ccv_cnnp_model_dropout_t;

static void _ccv_cnnp_dropout_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params[2];
	ccv_cnnp_model_dropout_t* const self = (ccv_cnnp_model_dropout_t*)super;
	const ccv_nnc_cmd_t dropout = CMD_DROPOUT_FORWARD(self->p, self->entirety);
	ccv_nnc_hint_tensor_auto(dropout, (ccv_nnc_tensor_param_t []){
			params,
		}, 1, ccv_nnc_no_hint, output_params, 2);
	const ccv_nnc_tensor_symbol_t dropout_output = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
	const ccv_nnc_tensor_symbol_t mask = ccv_nnc_tensor_symbol_new(graph, output_params[1], "mask");
	self->dropout = ccv_nnc_graph_exec_symbol_new(graph, dropout, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(dropout_output, mask), "dropout");
	outputs[0] = dropout_output;
}

static void _ccv_cnnp_dropout_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_model_dropout_t* const self = (ccv_cnnp_model_dropout_t*)super;
	if (self->dropout.graph)
	{
		if (is_test)
			// During test, the dropout is not applied. Data transfer is perfect because if these are the same tensor, it will skip.
			updater(context, self->dropout, CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint);
		else
			updater(context, self->dropout, CMD_DROPOUT_FORWARD(self->p, self->entirety), ccv_nnc_no_hint);
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_dropout_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_dropout_isa = {
	.build = _ccv_cnnp_dropout_build,
	.set_is_test = _ccv_cnnp_dropout_set_is_test,
	.copy = _ccv_cnnp_dropout_copy,
};

ccv_cnnp_model_t* ccv_cnnp_dropout(const float p, const int entirety, const char* const name)
{
	ccv_cnnp_model_dropout_t* const model_dropout = (ccv_cnnp_model_dropout_t*)cccalloc(1, sizeof(ccv_cnnp_model_dropout_t));
	model_dropout->super.isa = &ccv_cnnp_dropout_isa;
	model_dropout->super.input_size = 1;
	model_dropout->super.outputs = &model_dropout->output;
	model_dropout->super.output_size = 1;
	model_dropout->p = p;
	model_dropout->entirety = entirety;
	ccv_cnnp_model_copy_name(&model_dropout->super, name);
	return (ccv_cnnp_model_t*)model_dropout;
}

static ccv_cnnp_model_t* _ccv_cnnp_dropout_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_dropout_t* const self = (const ccv_cnnp_model_dropout_t*)super;
	return ccv_cnnp_dropout(self->p, self->entirety, self->super.name);
}

// MARK - Masked Fill Layer

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
	ccv_nnc_graph_exec_symbol_new(graph, CMD_MASKED_FILL_FORWARD(self->eq, self->fill), TENSOR_SYMBOL_LIST(inputs[0], inputs[1]), TENSOR_SYMBOL_LIST(masked_fill_output), "masked_fill");
	outputs[0] = masked_fill_output;
}

static ccv_cnnp_model_t* _ccv_cnnp_masked_fill_copy(const ccv_cnnp_model_t* const super, void* const context);

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

static ccv_cnnp_model_t* _ccv_cnnp_masked_fill_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_masked_fill_t* const self = (const ccv_cnnp_model_masked_fill_t*)super;
	return ccv_cnnp_masked_fill(self->eq, self->fill, self->super.name);
}

// MARK - Index Select Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_index_select_t;

static void _ccv_cnnp_index_select_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_param_t vocab_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	const ccv_nnc_tensor_param_t index_params = ccv_nnc_tensor_symbol_params(graph, inputs[1]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t index_select = CMD_INDEX_SELECT_FORWARD();
	ccv_nnc_hint_tensor_auto(index_select, (ccv_nnc_tensor_param_t []){
			vocab_params,
			index_params,
		}, 2, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, index_select, TENSOR_SYMBOL_LIST(inputs[0], inputs[1]), TENSOR_SYMBOL_LIST(output), "index_select");
	outputs[0] = output;
}

static ccv_cnnp_model_t* _ccv_cnnp_index_select_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_index_select_isa = {
	.build = _ccv_cnnp_index_select_build,
	.copy = _ccv_cnnp_index_select_copy,
};

ccv_cnnp_model_t* ccv_cnnp_index_select(const char* const name)
{
	ccv_cnnp_model_index_select_t* const model_index_select = (ccv_cnnp_model_index_select_t*)cccalloc(1, sizeof(ccv_cnnp_model_index_select_t));
	model_index_select->super.isa = &ccv_cnnp_index_select_isa;
	model_index_select->super.input_size = 2;
	model_index_select->super.outputs = &model_index_select->output;
	model_index_select->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_index_select->super, name);
	return (ccv_cnnp_model_t*)model_index_select;
}

static ccv_cnnp_model_t* _ccv_cnnp_index_select_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_index_select_t* const self = (ccv_cnnp_model_index_select_t*)super;
	return ccv_cnnp_index_select(self->super.name);
}

// MARK - Embedding Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t vocab;
	int datatype;
	int vocab_size;
	int embed_size;
} ccv_cnnp_model_embedding_t;

static void _ccv_cnnp_embedding_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_embedding_t* const self = (ccv_cnnp_model_embedding_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t vocab_params = params;
	memset(vocab_params.dim, 0, sizeof(vocab_params.dim));
	vocab_params.datatype = self->datatype;
	vocab_params.dim[0] = self->vocab_size;
	vocab_params.dim[1] = self->embed_size;
	if (!self->vocab.graph)
		self->vocab = ccv_nnc_tensor_symbol_new(graph, vocab_params, "vocab");
	assert(self->vocab.graph == graph);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t embedding = CMD_INDEX_SELECT_FORWARD();
	ccv_nnc_hint_tensor_auto(embedding, (ccv_nnc_tensor_param_t []){
			vocab_params,
			params,
		}, 2, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, embedding, TENSOR_SYMBOL_LIST(self->vocab, inputs[0]), TENSOR_SYMBOL_LIST(output), "embedding");
	outputs[0] = output;
}

static void _ccv_cnnp_embedding_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_embedding_t* const self = (ccv_cnnp_model_embedding_t*)super;
	const float std = sqrtf(2) / sqrtf(self->vocab_size + self->embed_size);
	const float bound = sqrtf(3) * std;
	initializer(context, CMD_RANDOM_UNIFORM_FORWARD(-bound, bound), ccv_nnc_no_hint, 0, 0, self->vocab);
}

static void _ccv_cnnp_embedding_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_embedding_t* const self = (ccv_cnnp_model_embedding_t*)super;
	add_to_array(parameters, self->vocab, is_trainable);
}

static ccv_cnnp_model_t* _ccv_cnnp_embedding_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_embedding_isa = {
	.build = _ccv_cnnp_embedding_build,
	.init_states = _ccv_cnnp_embedding_init_states,
	.add_to_parameter = _ccv_cnnp_embedding_add_to_parameter,
	.copy = _ccv_cnnp_embedding_copy,
};

ccv_cnnp_model_t* ccv_cnnp_embedding(const int datatype, const int vocab_size, const int embed_size, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_embedding_t* const model_embedding = (ccv_cnnp_model_embedding_t*)cccalloc(1, sizeof(ccv_cnnp_model_embedding_t));
	model_embedding->super.isa = &ccv_cnnp_embedding_isa;
	model_embedding->super.input_size = 1;
	model_embedding->super.outputs = &model_embedding->output;
	model_embedding->super.output_size = 1;
	model_embedding->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_embedding->super, name);
	model_embedding->vocab.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_embedding->vocab.graph = 0;
	assert(datatype == CCV_32F || datatype == CCV_16F);
	model_embedding->datatype = datatype;
	assert(vocab_size > 0);
	model_embedding->vocab_size = vocab_size;
	assert(embed_size > 0);
	model_embedding->embed_size = embed_size;
	return (ccv_cnnp_model_t*)model_embedding;
}

static ccv_cnnp_model_t* _ccv_cnnp_embedding_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_embedding_t* const self = (ccv_cnnp_model_embedding_t*)super;
	return ccv_cnnp_embedding(self->datatype, self->vocab_size, self->embed_size, self->super.is_trainable, self->super.name);
}

// MARK - Pool Layers

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int type;
	float width_scale;
	float height_scale;
} ccv_cnnp_model_upsample_t;

static void _ccv_cnnp_upsample_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_cnnp_model_upsample_t* const self = (ccv_cnnp_model_upsample_t*)super;
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_cmd_t cmd = CMD_UPSAMPLE_FORWARD(self->type, self->width_scale, self->height_scale);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_hint_tensor_auto(cmd, &params, 1, ccv_nnc_no_hint, &output_params, 1);
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0]), TENSOR_SYMBOL_LIST(output), "upsample");
	outputs[0] = output;
}

static ccv_cnnp_model_t* _ccv_cnnp_upsample_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_upsample_isa = {
	.build = _ccv_cnnp_upsample_build,
	.copy = _ccv_cnnp_upsample_copy,
};

ccv_cnnp_model_t* ccv_cnnp_upsample(const int type, const float width_scale, const float height_scale, const char* const name)
{
	ccv_cnnp_model_upsample_t* const model_upsample = (ccv_cnnp_model_upsample_t*)cccalloc(1, sizeof(ccv_cnnp_model_upsample_t));
	model_upsample->super.isa = &ccv_cnnp_upsample_isa;
	model_upsample->super.input_size = 1;
	model_upsample->super.outputs = &model_upsample->output;
	model_upsample->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_upsample->super, name);
	assert(type == CCV_NNC_UPSAMPLE_NEAREST || type == CCV_NNC_UPSAMPLE_BILINEAR);
	model_upsample->type = type;
	model_upsample->width_scale = width_scale;
	model_upsample->height_scale = height_scale;
	return (ccv_cnnp_model_t*)model_upsample;
}

static ccv_cnnp_model_t* _ccv_cnnp_upsample_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_upsample_t* const self = (const ccv_cnnp_model_upsample_t*)super;
	return ccv_cnnp_upsample(self->type, self->width_scale, self->height_scale, self->super.name);
}

// MARK - Reduce Sum Layer

typedef struct {
	ccv_cnnp_model_t super;
	int axis[CCV_NNC_MAX_DIM_ALLOC];
	int count;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_reduce_sum_t;

static void _ccv_cnnp_reduce_sum_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_reduce_sum_t* const self = (const ccv_cnnp_model_reduce_sum_t*)super;
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_cmd_t reduce_sum = CMD_REDUCE_SUM_FORWARD();
	int i;
	for (i = 0; i < self->count; i++)
		reduce_sum.info.reduce.axis[i] = self->axis[i];
	reduce_sum.info.reduce.count = self->count;
	ccv_nnc_hint_tensor_auto(reduce_sum, &input_params, 1, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, reduce_sum, inputs, input_size, outputs, output_size, "reduce_sum");
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_sum_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_reduce_sum_isa = {
	.build = _ccv_cnnp_reduce_sum_build,
	.copy = _ccv_cnnp_reduce_sum_copy,
};

ccv_cnnp_model_t* ccv_cnnp_reduce_sum(const int* const axis, const int axis_count, const char* const name)
{
	ccv_cnnp_model_reduce_sum_t* const model_reduce_sum = (ccv_cnnp_model_reduce_sum_t*)cccalloc(1, sizeof(ccv_cnnp_model_reduce_sum_t));
	model_reduce_sum->super.isa = &ccv_cnnp_reduce_sum_isa;
	model_reduce_sum->super.input_size = 1;
	model_reduce_sum->super.outputs = &model_reduce_sum->output;
	model_reduce_sum->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_reduce_sum->super, name);
	assert(axis_count <= CCV_NNC_MAX_DIM_ALLOC);
	int i;
	for (i = 0; i < axis_count; i++)
		model_reduce_sum->axis[i] = axis[i];
	model_reduce_sum->count = axis_count;
	return (ccv_cnnp_model_t*)model_reduce_sum;
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_sum_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_reduce_sum_t* const self = (const ccv_cnnp_model_reduce_sum_t*)super;
	return ccv_cnnp_reduce_sum(self->axis, self->count, self->super.name);
}

// MARK - Reduce Mean Layer

typedef struct {
	ccv_cnnp_model_t super;
	int axis[CCV_NNC_MAX_DIM_ALLOC];
	int count;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_reduce_mean_t;

static void _ccv_cnnp_reduce_mean_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_reduce_mean_t* const self = (const ccv_cnnp_model_reduce_mean_t*)super;
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_cmd_t reduce_mean = CMD_REDUCE_MEAN_FORWARD();
	int i;
	for (i = 0; i < self->count; i++)
		reduce_mean.info.reduce.axis[i] = self->axis[i];
	reduce_mean.info.reduce.count = self->count;
	ccv_nnc_hint_tensor_auto(reduce_mean, &input_params, 1, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, reduce_mean, inputs, input_size, outputs, output_size, "reduce_mean");
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_mean_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_reduce_mean_isa = {
	.build = _ccv_cnnp_reduce_mean_build,
	.copy = _ccv_cnnp_reduce_mean_copy,
};

ccv_cnnp_model_t* ccv_cnnp_reduce_mean(const int* const axis, const int axis_count, const char* const name)
{
	ccv_cnnp_model_reduce_mean_t* const model_reduce_mean = (ccv_cnnp_model_reduce_mean_t*)cccalloc(1, sizeof(ccv_cnnp_model_reduce_mean_t));
	model_reduce_mean->super.isa = &ccv_cnnp_reduce_mean_isa;
	model_reduce_mean->super.input_size = 1;
	model_reduce_mean->super.outputs = &model_reduce_mean->output;
	model_reduce_mean->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_reduce_mean->super, name);
	assert(axis_count <= CCV_NNC_MAX_DIM_ALLOC);
	int i;
	for (i = 0; i < axis_count; i++)
		model_reduce_mean->axis[i] = axis[i];
	model_reduce_mean->count = axis_count;
	return (ccv_cnnp_model_t*)model_reduce_mean;
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_mean_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_reduce_mean_t* const self = (const ccv_cnnp_model_reduce_mean_t*)super;
	return ccv_cnnp_reduce_mean(self->axis, self->count, self->super.name);
}

// MARK - Reduce Max Layer

typedef struct {
	ccv_cnnp_model_t super;
	int axis[CCV_NNC_MAX_DIM_ALLOC];
	int count;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_reduce_max_t;

static void _ccv_cnnp_reduce_max_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_reduce_max_t* const self = (const ccv_cnnp_model_reduce_max_t*)super;
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_cmd_t reduce_max = CMD_REDUCE_MAX_FORWARD();
	int i;
	for (i = 0; i < self->count; i++)
		reduce_max.info.reduce.axis[i] = self->axis[i];
	reduce_max.info.reduce.count = self->count;
	ccv_nnc_hint_tensor_auto(reduce_max, &input_params, 1, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, reduce_max, inputs, input_size, outputs, output_size, "reduce_max");
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_max_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_reduce_max_isa = {
	.build = _ccv_cnnp_reduce_max_build,
	.copy = _ccv_cnnp_reduce_max_copy,
};

ccv_cnnp_model_t* ccv_cnnp_reduce_max(const int* const axis, const int axis_count, const char* const name)
{
	ccv_cnnp_model_reduce_max_t* const model_reduce_max = (ccv_cnnp_model_reduce_max_t*)cccalloc(1, sizeof(ccv_cnnp_model_reduce_max_t));
	model_reduce_max->super.isa = &ccv_cnnp_reduce_max_isa;
	model_reduce_max->super.input_size = 1;
	model_reduce_max->super.outputs = &model_reduce_max->output;
	model_reduce_max->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_reduce_max->super, name);
	assert(axis_count <= CCV_NNC_MAX_DIM_ALLOC);
	int i;
	for (i = 0; i < axis_count; i++)
		model_reduce_max->axis[i] = axis[i];
	model_reduce_max->count = axis_count;
	return (ccv_cnnp_model_t*)model_reduce_max;
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_max_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_reduce_max_t* const self = (const ccv_cnnp_model_reduce_max_t*)super;
	return ccv_cnnp_reduce_max(self->axis, self->count, self->super.name);
}

// MARK - Reduce Min Layer

typedef struct {
	ccv_cnnp_model_t super;
	int axis[CCV_NNC_MAX_DIM_ALLOC];
	int count;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_reduce_min_t;

static void _ccv_cnnp_reduce_min_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_reduce_min_t* const self = (const ccv_cnnp_model_reduce_min_t*)super;
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_cmd_t reduce_min = CMD_REDUCE_MIN_FORWARD();
	int i;
	for (i = 0; i < self->count; i++)
		reduce_min.info.reduce.axis[i] = self->axis[i];
	reduce_min.info.reduce.count = self->count;
	ccv_nnc_hint_tensor_auto(reduce_min, &input_params, 1, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, reduce_min, inputs, input_size, outputs, output_size, "reduce_min");
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_min_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_reduce_min_isa = {
	.build = _ccv_cnnp_reduce_min_build,
	.copy = _ccv_cnnp_reduce_min_copy,
};

ccv_cnnp_model_t* ccv_cnnp_reduce_min(const int* const axis, const int axis_count, const char* const name)
{
	ccv_cnnp_model_reduce_min_t* const model_reduce_min = (ccv_cnnp_model_reduce_min_t*)cccalloc(1, sizeof(ccv_cnnp_model_reduce_min_t));
	model_reduce_min->super.isa = &ccv_cnnp_reduce_min_isa;
	model_reduce_min->super.input_size = 1;
	model_reduce_min->super.outputs = &model_reduce_min->output;
	model_reduce_min->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_reduce_min->super, name);
	assert(axis_count <= CCV_NNC_MAX_DIM_ALLOC);
	int i;
	for (i = 0; i < axis_count; i++)
		model_reduce_min->axis[i] = axis[i];
	model_reduce_min->count = axis_count;
	return (ccv_cnnp_model_t*)model_reduce_min;
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_min_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_reduce_min_t* const self = (const ccv_cnnp_model_reduce_min_t*)super;
	return ccv_cnnp_reduce_min(self->axis, self->count, self->super.name);
}

// MARK - Reduce Norm2 Layer

typedef struct {
	ccv_cnnp_model_t super;
	int axis[CCV_NNC_MAX_DIM_ALLOC];
	int count;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_reduce_norm2_t;

static void _ccv_cnnp_reduce_norm2_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_reduce_norm2_t* const self = (const ccv_cnnp_model_reduce_norm2_t*)super;
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_cmd_t reduce_norm2 = CMD_REDUCE_NORM2_FORWARD();
	int i;
	for (i = 0; i < self->count; i++)
		reduce_norm2.info.reduce.axis[i] = self->axis[i];
	reduce_norm2.info.reduce.count = self->count;
	ccv_nnc_hint_tensor_auto(reduce_norm2, &input_params, 1, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, reduce_norm2, inputs, input_size, outputs, output_size, "reduce_norm2");
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_norm2_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_reduce_norm2_isa = {
	.build = _ccv_cnnp_reduce_norm2_build,
	.copy = _ccv_cnnp_reduce_norm2_copy,
};

ccv_cnnp_model_t* ccv_cnnp_reduce_norm2(const int* const axis, const int axis_count, const char* const name)
{
	ccv_cnnp_model_reduce_norm2_t* const model_reduce_norm2 = (ccv_cnnp_model_reduce_norm2_t*)cccalloc(1, sizeof(ccv_cnnp_model_reduce_norm2_t));
	model_reduce_norm2->super.isa = &ccv_cnnp_reduce_norm2_isa;
	model_reduce_norm2->super.input_size = 1;
	model_reduce_norm2->super.outputs = &model_reduce_norm2->output;
	model_reduce_norm2->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_reduce_norm2->super, name);
	assert(axis_count <= CCV_NNC_MAX_DIM_ALLOC);
	int i;
	for (i = 0; i < axis_count; i++)
		model_reduce_norm2->axis[i] = axis[i];
	model_reduce_norm2->count = axis_count;
	return (ccv_cnnp_model_t*)model_reduce_norm2;
}

static ccv_cnnp_model_t* _ccv_cnnp_reduce_norm2_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_reduce_norm2_t* const self = (const ccv_cnnp_model_reduce_norm2_t*)super;
	return ccv_cnnp_reduce_norm2(self->axis, self->count, self->super.name);
}

// MARK - Argmax Layer

typedef struct {
	ccv_cnnp_model_t super;
	int axis;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_argmax_t;

static void _ccv_cnnp_argmax_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_argmax_t* const self = (const ccv_cnnp_model_argmax_t*)super;
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_cmd_t argmax = CMD_ARGMAX_FORWARD();
	argmax.info.reduce.axis[0] = self->axis;
	argmax.info.reduce.count = 1;
	ccv_nnc_hint_tensor_auto(argmax, &input_params, 1, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, argmax, inputs, input_size, outputs, output_size, "argmax");
}

static ccv_cnnp_model_t* _ccv_cnnp_argmax_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_argmax_isa = {
	.build = _ccv_cnnp_argmax_build,
	.copy = _ccv_cnnp_argmax_copy,
};

ccv_cnnp_model_t* ccv_cnnp_argmax(const int axis, const char* const name)
{
	ccv_cnnp_model_argmax_t* const model_argmax = (ccv_cnnp_model_argmax_t*)cccalloc(1, sizeof(ccv_cnnp_model_argmax_t));
	model_argmax->super.isa = &ccv_cnnp_argmax_isa;
	model_argmax->super.input_size = 1;
	model_argmax->super.outputs = &model_argmax->output;
	model_argmax->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_argmax->super, name);
	model_argmax->axis = axis;
	return (ccv_cnnp_model_t*)model_argmax;
}

static ccv_cnnp_model_t* _ccv_cnnp_argmax_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_argmax_t* const self = (const ccv_cnnp_model_argmax_t*)super;
	return ccv_cnnp_argmax(self->axis, self->super.name);
}

// MARK - Argmin Layer

typedef struct {
	ccv_cnnp_model_t super;
	int axis;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_argmin_t;

static void _ccv_cnnp_argmin_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	const ccv_cnnp_model_argmin_t* const self = (const ccv_cnnp_model_argmin_t*)super;
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	ccv_nnc_tensor_param_t output_params;
	ccv_nnc_cmd_t argmin = CMD_ARGMIN_FORWARD();
	argmin.info.reduce.axis[0] = self->axis;
	argmin.info.reduce.count = 1;
	ccv_nnc_hint_tensor_auto(argmin, &input_params, 1, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, argmin, inputs, input_size, outputs, output_size, "argmin");
}

static ccv_cnnp_model_t* _ccv_cnnp_argmin_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_argmin_isa = {
	.build = _ccv_cnnp_argmin_build,
	.copy = _ccv_cnnp_argmin_copy,
};

ccv_cnnp_model_t* ccv_cnnp_argmin(const int axis, const char* const name)
{
	ccv_cnnp_model_argmin_t* const model_argmin = (ccv_cnnp_model_argmin_t*)cccalloc(1, sizeof(ccv_cnnp_model_argmin_t));
	model_argmin->super.isa = &ccv_cnnp_argmin_isa;
	model_argmin->super.input_size = 1;
	model_argmin->super.outputs = &model_argmin->output;
	model_argmin->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_argmin->super, name);
	model_argmin->axis = axis;
	return (ccv_cnnp_model_t*)model_argmin;
}

static ccv_cnnp_model_t* _ccv_cnnp_argmin_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_argmin_t* const self = (const ccv_cnnp_model_argmin_t*)super;
	return ccv_cnnp_argmin(self->axis, self->super.name);
}

// MARK - Min Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_min_t;

static void _ccv_cnnp_min_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params[2];
	int i;
	for (i = 0; i < 2; i++)
		input_params[i] = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t min = CMD_MIN_FORWARD();
	ccv_nnc_hint_tensor_auto(min, input_params, 2, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, min, inputs, input_size, outputs, output_size, "min");
}

static ccv_cnnp_model_t* _ccv_cnnp_min_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_min_isa = {
	.build = _ccv_cnnp_min_build,
	.copy = _ccv_cnnp_min_copy,
};

ccv_cnnp_model_t* ccv_cnnp_min(const char* const name)
{
	ccv_cnnp_model_min_t* const model_min = (ccv_cnnp_model_min_t*)cccalloc(1, sizeof(ccv_cnnp_model_min_t));
	model_min->super.isa = &ccv_cnnp_min_isa;
	model_min->super.input_size = 2;
	model_min->super.outputs = &model_min->output;
	model_min->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_min->super, name);
	return (ccv_cnnp_model_t*)model_min;
}

static ccv_cnnp_model_t* _ccv_cnnp_min_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_min_t* const self = (const ccv_cnnp_model_min_t*)super;
	return ccv_cnnp_min(self->super.name);
}

// MARK - Max Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_max_t;

static void _ccv_cnnp_max_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t input_params[2];
	int i;
	for (i = 0; i < 2; i++)
		input_params[i] = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
	ccv_nnc_tensor_param_t output_params;
	const ccv_nnc_cmd_t max = CMD_MAX_FORWARD();
	ccv_nnc_hint_tensor_auto(max, input_params, 2, ccv_nnc_no_hint, &output_params, 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, max, inputs, input_size, outputs, output_size, "max");
}

static ccv_cnnp_model_t* _ccv_cnnp_max_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_max_isa = {
	.build = _ccv_cnnp_max_build,
	.copy = _ccv_cnnp_max_copy,
};

ccv_cnnp_model_t* ccv_cnnp_max(const char* const name)
{
	ccv_cnnp_model_max_t* const model_max = (ccv_cnnp_model_max_t*)cccalloc(1, sizeof(ccv_cnnp_model_max_t));
	model_max->super.isa = &ccv_cnnp_max_isa;
	model_max->super.input_size = 2;
	model_max->super.outputs = &model_max->output;
	model_max->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_max->super, name);
	return (ccv_cnnp_model_t*)model_max;
}

static ccv_cnnp_model_t* _ccv_cnnp_max_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_max_t* const self = (const ccv_cnnp_model_max_t*)super;
	return ccv_cnnp_max(self->super.name);
}

// MARK - LSTM Layer

typedef struct {
	ccv_cnnp_model_t super;
	int masked;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t weights;
	ccv_nnc_tensor_symbol_t reserves;
	ccv_nnc_cmd_param_t params;
	ccv_nnc_graph_exec_symbol_t lstm;
} ccv_cnnp_model_lstm_t;

static int _ccv_cnnp_lstm_weight_dim(int bidirectional, int num_layers, int input_size, int hidden_size, int proj_size, int bias)
{
	const int D = !!bidirectional + 1;
	if (hidden_size == proj_size)
		return (num_layers * (bias ? 8 : 0) + (num_layers - 1) * (hidden_size * 4 * D + hidden_size * 4) + input_size * 4 + hidden_size * 4) * D;
	else
		return (num_layers * (bias ? 8 : 0) + (num_layers - 1) * (proj_size * 4 * D + proj_size * 4) + (proj_size * 4 + input_size * 4) + num_layers * proj_size) * D;
}

static void _ccv_cnnp_lstm_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_model_lstm_t* const self = (ccv_cnnp_model_lstm_t*)super;
	assert(input_size == self->super.input_size);
	assert(output_size == 1);
	const int proj_size = self->params.rnn.proj_size == 0 ? self->params.rnn.hidden_size : self->params.rnn.proj_size;
	ccv_nnc_tensor_param_t input_params[5];
	input_params[0]= ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	if (input_size == 2)
		input_params[1] = ccv_nnc_tensor_symbol_params(graph, inputs[1]);
	input_params[4] = input_params[0];
	memset(input_params[4].dim, 0, sizeof(input_params[4].dim));
	const int x_nd = ccv_nnc_tensor_nd(input_params[0].dim);
	const int feature_count = input_params[0].dim[x_nd - 1];
	input_params[4].dim[0] = _ccv_cnnp_lstm_weight_dim(self->params.rnn.bidirectional, self->params.rnn.num_layers, feature_count, self->params.rnn.hidden_size, proj_size, self->params.rnn.bias);
	input_params[4].dim[1] = self->params.rnn.hidden_size;
	const ccv_nnc_cmd_t lstm = ccv_nnc_cmd(CCV_NNC_LSTM_FORWARD, 0, self->params, 0);
	ccv_nnc_tensor_param_t output_params[4];
	ccv_nnc_hint_tensor_auto(lstm, input_params, 5, ccv_nnc_no_hint, output_params, 4);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
	if (!self->weights.graph)
		self->weights = ccv_nnc_tensor_symbol_new(graph, input_params[4], "weights");
	if (!self->reserves.graph)
		self->reserves = ccv_nnc_tensor_symbol_new(graph, output_params[3], "reserves");
	const ccv_nnc_tensor_symbol_t mask = input_size == 2 ? inputs[1] : NO_TENSOR_SYMBOL;
	self->lstm = ccv_nnc_graph_exec_symbol_new(graph, lstm, TENSOR_SYMBOL_LIST(inputs[0], mask, NO_TENSOR_SYMBOL, NO_TENSOR_SYMBOL, self->weights), TENSOR_SYMBOL_LIST(outputs[0], NO_TENSOR_SYMBOL, NO_TENSOR_SYMBOL, self->reserves), "lstm");
}

static void _ccv_cnnp_lstm_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_lstm_t* const self = (ccv_cnnp_model_lstm_t*)super;
	if (self->weights.graph)
	{
		const float stdv = 1.0 / sqrt(self->params.rnn.hidden_size);
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(-stdv, stdv), ccv_nnc_no_hint, 0, 0, self->weights);
	}
}

static void _ccv_cnnp_lstm_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_lstm_t* const self = (ccv_cnnp_model_lstm_t*)super;
	if (self->weights.graph)
		add_to_array(parameters, self->weights, is_trainable);
}

static void _ccv_cnnp_lstm_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_model_lstm_t* const self = (ccv_cnnp_model_lstm_t*)super;
	if (self->lstm.graph)
	{
		self->params.rnn.is_test = is_test;
		updater(context, self->lstm, ccv_nnc_cmd(CCV_NNC_LSTM_FORWARD, 0, self->params, 0), ccv_nnc_no_hint);
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_lstm_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_lstm_isa = {
	.build = _ccv_cnnp_lstm_build,
	.init_states = _ccv_cnnp_lstm_init_states,
	.add_to_parameter = _ccv_cnnp_lstm_add_to_parameter,
	.copy = _ccv_cnnp_lstm_copy,
	.set_is_test = _ccv_cnnp_lstm_set_is_test,
};

ccv_cnnp_model_t* ccv_cnnp_lstm(const int masked, const int hidden_size, const int proj_size, const int num_layers, const int bias, const int batch_first, const int bidirectional, const float dropout, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_lstm_t* const model_lstm = (ccv_cnnp_model_lstm_t*)cccalloc(1, sizeof(ccv_cnnp_model_lstm_t));
	model_lstm->super.isa = &ccv_cnnp_lstm_isa;
	model_lstm->super.input_size = masked ? 2 : 1;
	model_lstm->super.outputs = &model_lstm->output;
	model_lstm->super.output_size = 1;
	model_lstm->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_lstm->super, name);
	model_lstm->masked = masked;
	model_lstm->weights.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_lstm->weights.graph = 0;
	model_lstm->params.rnn.hidden_size = hidden_size;
	model_lstm->params.rnn.proj_size = proj_size;
	model_lstm->params.rnn.num_layers = num_layers;
	model_lstm->params.rnn.bias = bias;
	model_lstm->params.rnn.batch_first = batch_first;
	model_lstm->params.rnn.bidirectional = bidirectional;
	model_lstm->params.rnn.dropout = dropout;
	return (ccv_cnnp_model_t*)model_lstm;
}

static ccv_cnnp_model_t* _ccv_cnnp_lstm_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_lstm_t* const self = (const ccv_cnnp_model_lstm_t*)super;
	return ccv_cnnp_lstm(self->masked, self->params.rnn.hidden_size, self->params.rnn.proj_size, self->params.rnn.num_layers, self->params.rnn.bias, self->params.rnn.batch_first, self->params.rnn.bidirectional, self->params.rnn.dropout, self->super.is_trainable, self->super.name);
}

/// MARK - Datatype conversion layer.

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	int datatype;
	int ref_to_last;
} ccv_cnnp_model_datatype_conversion_t;

static void _ccv_cnnp_datatype_conversion_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_model_datatype_conversion_t* const self = (ccv_cnnp_model_datatype_conversion_t*)super;
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	if (self->ref_to_last)
	{
		assert(input_size > 1);
		const ccv_nnc_tensor_param_t last_params = ccv_nnc_tensor_symbol_params(graph, inputs[input_size - 1]);
		params.datatype = last_params.datatype;
	} else
		params.datatype = self->datatype;
	assert(output_size == 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_DATATYPE_CONVERSION_FORWARD(), inputs, output_size, outputs, output_size, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_datatype_conversion_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_datatype_conversion_isa = {
	.build = _ccv_cnnp_datatype_conversion_build,
	.copy = _ccv_cnnp_datatype_conversion_copy,
};

ccv_cnnp_model_t* ccv_cnnp_datatype_conversion(const int datatype, const int ref_to_last, const char* const name)
{
	ccv_cnnp_model_datatype_conversion_t* const model_datatype_conversion = (ccv_cnnp_model_datatype_conversion_t*)cccalloc(1, sizeof(ccv_cnnp_model_datatype_conversion_t));
	model_datatype_conversion->super.isa = &ccv_cnnp_datatype_conversion_isa;
	model_datatype_conversion->super.input_size = 0;
	model_datatype_conversion->super.outputs = &model_datatype_conversion->output;
	model_datatype_conversion->super.output_size = 1;
	model_datatype_conversion->datatype = datatype;
	model_datatype_conversion->ref_to_last = ref_to_last;
	ccv_cnnp_model_copy_name(&model_datatype_conversion->super, name);
	return (ccv_cnnp_model_t*)model_datatype_conversion;
}

static ccv_cnnp_model_t* _ccv_cnnp_datatype_conversion_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_datatype_conversion_t* const self = (ccv_cnnp_model_datatype_conversion_t*)super;
	return ccv_cnnp_datatype_conversion(self->datatype, self->ref_to_last, self->super.name);
}

/// MARK - Clamp layer.

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	float min;
	float max;
} ccv_cnnp_model_clamp_t;

static void _ccv_cnnp_clamp_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_model_clamp_t* const self = (ccv_cnnp_model_clamp_t*)super;
	ccv_nnc_tensor_param_t params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	assert(output_size == 1);
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_CLAMP_FORWARD(self->min, self->max), inputs, output_size, outputs, output_size, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_clamp_copy(const ccv_cnnp_model_t* const self, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_clamp_isa = {
	.build = _ccv_cnnp_clamp_build,
	.copy = _ccv_cnnp_clamp_copy,
};

ccv_cnnp_model_t* ccv_cnnp_clamp(const float min, const float max, const char* const name)
{
	ccv_cnnp_model_clamp_t* const model_clamp = (ccv_cnnp_model_clamp_t*)cccalloc(1, sizeof(ccv_cnnp_model_clamp_t));
	model_clamp->super.isa = &ccv_cnnp_clamp_isa;
	model_clamp->super.input_size = 0;
	model_clamp->super.outputs = &model_clamp->output;
	model_clamp->super.output_size = 1;
	model_clamp->min = min;
	model_clamp->max = max;
	ccv_cnnp_model_copy_name(&model_clamp->super, name);
	return (ccv_cnnp_model_t*)model_clamp;
}

static ccv_cnnp_model_t* _ccv_cnnp_clamp_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	ccv_cnnp_model_clamp_t* const self = (ccv_cnnp_model_clamp_t*)super;
	return ccv_cnnp_clamp(self->min, self->max, self->super.name);
}

// MARK - Parameter Layer

typedef struct {
	ccv_cnnp_model_t super;
	float init_bound;
	ccv_nnc_tensor_symbol_t weights;
	ccv_nnc_tensor_param_t weights_params;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_parameter_t;

static void _ccv_cnnp_parameter_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	ccv_cnnp_model_parameter_t* const self = (ccv_cnnp_model_parameter_t*)super;
	if (!self->weights.graph)
		self->weights = ccv_nnc_tensor_symbol_new(graph, self->weights_params, "weights");
	assert(self->weights.graph == graph);
	outputs[0] = self->weights;
}

static void _ccv_cnnp_parameter_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_parameter_t* const self = (ccv_cnnp_model_parameter_t*)super;
	if (self->init_bound > 0)
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(-self->init_bound, self->init_bound), ccv_nnc_no_hint, 0, 0, self->weights);
	else
		initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->weights);
}

static void _ccv_cnnp_parameter_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_parameter_t* const self = (ccv_cnnp_model_parameter_t*)super;
	add_to_array(parameters, self->weights, is_trainable);
}

static ccv_cnnp_model_t* _ccv_cnnp_parameter_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_parameter_isa = {
	.build = _ccv_cnnp_parameter_build,
	.init_states = _ccv_cnnp_parameter_init_states,
	.add_to_parameter = _ccv_cnnp_parameter_add_to_parameter,
	.copy = _ccv_cnnp_parameter_copy,
};

ccv_cnnp_model_t* ccv_cnnp_parameter(const ccv_nnc_tensor_param_t params, const float init_bound, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_parameter_t* const model_parameter = (ccv_cnnp_model_parameter_t*)cccalloc(1, sizeof(ccv_cnnp_model_parameter_t));
	model_parameter->super.isa = &ccv_cnnp_parameter_isa;
	model_parameter->super.input_size = 0;
	model_parameter->super.outputs = &model_parameter->output;
	model_parameter->super.output_size = 1;
	model_parameter->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_parameter->super, name);
	model_parameter->weights.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_parameter->weights.graph = 0;
	model_parameter->weights_params = params;
	return (ccv_cnnp_model_t*)model_parameter;
}

static ccv_cnnp_model_t* _ccv_cnnp_parameter_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_parameter_t* const self = (const ccv_cnnp_model_parameter_t*)super;
	return ccv_cnnp_parameter(self->weights_params, self->init_bound, self->super.is_trainable, self->super.name);
}

// MARK - Scalar Layer

typedef struct {
	ccv_cnnp_model_t super;
	int type;
	int format;
	int datatype;
	float value;
	ccv_nnc_tensor_symbol_t output;
} ccv_cnnp_model_scalar_t;

static void _ccv_cnnp_scalar_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	ccv_cnnp_model_scalar_t* const self = (ccv_cnnp_model_scalar_t*)super;
	ccv_nnc_tensor_param_t params = {
		.type = self->type,
		.format = self->format,
		.datatype = self->datatype,
		.dim = {
			1
		}
	};
	if (input_size > 0)
	{
		ccv_nnc_tensor_param_t input_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
		params.type = input_params.type;
		params.format = input_params.format;
		params.datatype = input_params.datatype;
	}
	outputs[0] = ccv_nnc_tensor_symbol_new(graph, params, 0);
	ccv_nnc_graph_exec_symbol_new(graph, CMD_SET_FORWARD(self->value), 0, 0, outputs, 1, 0);
}

static ccv_cnnp_model_t* _ccv_cnnp_scalar_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_scalar_isa = {
	.build = _ccv_cnnp_scalar_build,
	.copy = _ccv_cnnp_scalar_copy,
};

ccv_cnnp_model_t* ccv_cnnp_scalar(const int type, const int format, const int datatype, const float value, const char* const name)
{
	ccv_cnnp_model_scalar_t* const model_scalar = (ccv_cnnp_model_scalar_t*)cccalloc(1, sizeof(ccv_cnnp_model_scalar_t));
	model_scalar->super.isa = &ccv_cnnp_scalar_isa;
	model_scalar->super.input_size = 0;
	model_scalar->super.outputs = &model_scalar->output;
	model_scalar->super.output_size = 1;
	ccv_cnnp_model_copy_name(&model_scalar->super, name);
	model_scalar->type = type;
	model_scalar->format = format;
	model_scalar->datatype = datatype;
	model_scalar->value = value;
	return (ccv_cnnp_model_t*)model_scalar;
}

static ccv_cnnp_model_t* _ccv_cnnp_scalar_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_scalar_t* const self = (const ccv_cnnp_model_scalar_t*)super;
	return ccv_cnnp_scalar(self->type, self->format, self->datatype, self->value, self->super.name);
}

// MARK - Scaled-Dot Product Attention Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t weights;
	ccv_nnc_tensor_symbol_t bias;
	float scale;
	int is_causal;
	int has_attn_mask;
	int fused_unify_head_weights;
	int no_bias;
} ccv_cnnp_model_scaled_dot_product_attention_t;

static void _ccv_cnnp_scaled_dot_product_attention_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(input_size == 3 || input_size == 4);
	assert(output_size == 1);
	ccv_cnnp_model_scaled_dot_product_attention_t* const self = (ccv_cnnp_model_scaled_dot_product_attention_t*)super;
	const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_symbol_params(graph, inputs[0]);
	const ccv_nnc_tensor_param_t k_params = ccv_nnc_tensor_symbol_params(graph, inputs[1]);
	const ccv_nnc_tensor_param_t v_params = ccv_nnc_tensor_symbol_params(graph, inputs[2]);
	const int v_nd = ccv_nnc_tensor_nd(v_params.dim);
	assert(v_nd == 3 || v_nd == 4);
	const int hEv = (v_nd == 3 ? 1 : v_params.dim[2]) * v_params.dim[v_nd - 1];
	ccv_nnc_tensor_param_t weights_params = q_params;
	memset(weights_params.dim, 0, sizeof(weights_params.dim));
	weights_params.dim[0] = hEv;
	weights_params.dim[1] = hEv;
	ccv_nnc_tensor_param_t bias_params = q_params;
	memset(bias_params.dim, 0, sizeof(bias_params.dim));
	bias_params.dim[0] = hEv;
	ccv_nnc_cmd_t cmd = {0};
	cmd.cmd = CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD;
	cmd.info.scaled_dot_product_attention.scale = self->scale;
	cmd.info.scaled_dot_product_attention.is_causal = self->is_causal;
	ccv_nnc_tensor_param_t output_params[3];
	ccv_nnc_tensor_symbol_t output;
	ccv_nnc_tensor_symbol_t saved_softmax;
	ccv_nnc_tensor_symbol_t saved_v_proj = NO_TENSOR_SYMBOL;
	ccv_nnc_tensor_symbol_t attn_mask = NO_TENSOR_SYMBOL;
	ccv_nnc_tensor_symbol_t weights = NO_TENSOR_SYMBOL;
	ccv_nnc_tensor_symbol_t bias = NO_TENSOR_SYMBOL;
	if (self->has_attn_mask)
		attn_mask = inputs[3];
	if (self->fused_unify_head_weights)
	{
		if (!self->weights.graph)
			self->weights = ccv_nnc_tensor_symbol_new(graph, weights_params, "weights");
		weights = self->weights;
		if (!self->no_bias)
		{
			if (!self->bias.graph)
				self->bias = ccv_nnc_tensor_symbol_new(graph, bias_params, "bias");
			bias = self->bias;
		}
		ccv_nnc_hint_tensor_auto(cmd, (ccv_nnc_tensor_param_t []){
				q_params,
				k_params,
				v_params,
				(ccv_nnc_tensor_param_t){},
				weights_params,
				bias_params,
			}, 6, ccv_nnc_no_hint, output_params, 3);
		output = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
		saved_softmax = ccv_nnc_tensor_symbol_new(graph, output_params[1], 0);
		saved_v_proj = ccv_nnc_tensor_symbol_new(graph, output_params[2], 0);
	} else {
		ccv_nnc_hint_tensor_auto(cmd, (ccv_nnc_tensor_param_t []){
				q_params,
				k_params,
				v_params,
			}, 3, ccv_nnc_no_hint, output_params, 2);
		output = ccv_nnc_tensor_symbol_new(graph, output_params[0], 0);
		saved_softmax = ccv_nnc_tensor_symbol_new(graph, output_params[1], 0);
	}
	ccv_nnc_graph_exec_symbol_new(graph, cmd, TENSOR_SYMBOL_LIST(inputs[0], inputs[1], inputs[2], attn_mask, weights, bias), TENSOR_SYMBOL_LIST(output, saved_softmax, saved_v_proj), "scaled_dot_product_attention");
	outputs[0] = output;
}

static void _ccv_cnnp_scaled_dot_product_attention_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_scaled_dot_product_attention_t* const self = (ccv_cnnp_model_scaled_dot_product_attention_t*)super;
	if (self->weights.graph)
	{
		assert(self->fused_unify_head_weights);
		const ccv_nnc_tensor_param_t weight_params = ccv_nnc_tensor_symbol_params(graph, self->weights);
		const int c = weight_params.dim[1];
		const float std = sqrtf(2) / sqrtf(c);
		const float bound = sqrtf(3) * std;
		initializer(context, CMD_RANDOM_UNIFORM_FORWARD(-bound, bound), ccv_nnc_no_hint, 0, 0, self->weights);
		if (self->bias.graph)
			initializer(context, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, self->bias);
	}
}

static void _ccv_cnnp_scaled_dot_product_attention_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_scaled_dot_product_attention_t* const self = (ccv_cnnp_model_scaled_dot_product_attention_t*)super;
	if (self->weights.graph)
	{
		assert(self->fused_unify_head_weights);
		add_to_array(parameters, self->weights, is_trainable);
		if (self->bias.graph)
			add_to_array(parameters, self->bias, is_trainable);
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_scaled_dot_product_attention_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_scaled_dot_product_attention_isa = {
	.build = _ccv_cnnp_scaled_dot_product_attention_build,
	.copy = _ccv_cnnp_scaled_dot_product_attention_copy,
};

static const ccv_cnnp_model_vtab_t ccv_cnnp_scaled_dot_product_attention_fused_isa = {
	.build = _ccv_cnnp_scaled_dot_product_attention_build,
	.init_states = _ccv_cnnp_scaled_dot_product_attention_init_states,
	.add_to_parameter = _ccv_cnnp_scaled_dot_product_attention_add_to_parameter,
	.copy = _ccv_cnnp_scaled_dot_product_attention_copy,
};

ccv_cnnp_model_t* ccv_cnnp_scaled_dot_product_attention(const float scale, const int is_causal, const int has_attn_mask, const int fused_unify_head_weights, const int no_bias, const int is_trainable, const char* const name)
{
	ccv_cnnp_model_scaled_dot_product_attention_t* const model_scaled_dot_product_attention = (ccv_cnnp_model_scaled_dot_product_attention_t*)cccalloc(1, sizeof(ccv_cnnp_model_scaled_dot_product_attention_t));
	model_scaled_dot_product_attention->super.isa = fused_unify_head_weights ? &ccv_cnnp_scaled_dot_product_attention_fused_isa : &ccv_cnnp_scaled_dot_product_attention_isa;
	model_scaled_dot_product_attention->super.input_size = has_attn_mask ? 4 : 3;
	model_scaled_dot_product_attention->super.outputs = &model_scaled_dot_product_attention->output;
	model_scaled_dot_product_attention->super.output_size = 1;
	model_scaled_dot_product_attention->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_scaled_dot_product_attention->super, name);
	model_scaled_dot_product_attention->weights.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_scaled_dot_product_attention->weights.graph = 0;
	model_scaled_dot_product_attention->bias.d = CCV_NNC_NO_TENSOR_SYMBOL;
	model_scaled_dot_product_attention->bias.graph = 0;
	model_scaled_dot_product_attention->scale = scale;
	model_scaled_dot_product_attention->is_causal = is_causal;
	model_scaled_dot_product_attention->has_attn_mask = has_attn_mask;
	model_scaled_dot_product_attention->fused_unify_head_weights = fused_unify_head_weights;
	model_scaled_dot_product_attention->no_bias = no_bias;
	return (ccv_cnnp_model_t*)model_scaled_dot_product_attention;
}

static ccv_cnnp_model_t* _ccv_cnnp_scaled_dot_product_attention_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_scaled_dot_product_attention_t* const self = (const ccv_cnnp_model_scaled_dot_product_attention_t*)super;
	return ccv_cnnp_scaled_dot_product_attention(self->scale, self->is_causal, self->has_attn_mask, self->fused_unify_head_weights, self->no_bias, self->super.is_trainable, self->super.name);
}
