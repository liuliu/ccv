#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_lstm_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: x, [xs], [hx], [cx], w.
	// output: y, [hy], [cy], r
	if (input_size == 5 && (input_bitmasks[0] & 17u) == ((1u << 0) | (0u << 1) | (0u << 2) | (0u << 3) | (1u << 4)) && (output_bitmasks[0] & 0x9u) == ((1u << 0) | (0u << 1) | (0u << 2) | (1u << 3)))
		return 1;
	return 0;
}

static int _ccv_nnc_lstm_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: dy, [dhy], [dcy], [dr]
	// output: dx, [dxs], [dhx], [dcx].
	if ((input_bitmasks[0] & 4865u) == ((1u << 0) | (0u << 1) | (0u << 2) | (0u << 3) | (0u << 4) | (0u << 5) | (0u << 6) | (0u << 7) | (1u << 8) | (1u << 9) | (0u << 10) | (0u << 11) | (1u << 12)) && (output_bitmasks[0] & 13u) == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3) | (0u << 4)))
		return 1;
	// Output dx, [dxs], [dhx], [dcx] and dw.
	if ((input_bitmasks[0] & 4881u) == ((1u << 0) | (0u << 1) | (0u << 2) | (0u << 3) | (1u << 4) | (0u << 5) | (0u << 6) | (0u << 7) | (1u << 8) | (1u << 9) | (0u << 10) | (0u << 11) | (1u << 12)) && (output_bitmasks[0] & 29u) == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3) | 1u << 4))
		return 1;
	// Output dw (this needs to be invoked after dx, dhx, dcx computed, thus, functionally the same as above).
	if ((input_bitmasks[0] & 4881u) == ((1u << 0) | (0u << 1) | (0u << 2) | (0u << 3) | (1u << 4) | (0u << 5) | (0u << 6) | (0u << 7) | (1u << 8) | (1u << 9) | (0u << 10) | (0u << 11) | (1u << 12)) && (output_bitmasks[0] & 16u) == ((0u << 0) | (0u << 1) | (0u << 2) | (0u << 3) | 1u << 4))
		return 1;
	return 0;
}

typedef size_t(*ccv_nnc_lstm_reserve_space_size_f)(const ccv_nnc_cmd_t cmd, const int datatype, const int feature_size, const int batch_count, const int max_seq_count);

static void _ccv_nnc_lstm_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size >= 1 && input_size >= 5);
	const int x_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	assert(x_nd == 3 || x_nd == 2);
	const int batch_count = x_nd == 3 ? (cmd.rnn.batch_first ? inputs[0].dim[0] : inputs[0].dim[1]) : 1;
	assert(batch_count > 0);
	const int max_seq_count = x_nd == 3 ? (cmd.rnn.batch_first ? inputs[0].dim[1] : inputs[0].dim[0]) : inputs[0].dim[0];
	const int feature_count = inputs[0].dim[x_nd - 1];
	const int proj_size = cmd.rnn.proj_size == 0 ? cmd.rnn.hidden_size : cmd.rnn.proj_size;
	const int output_feature_count = proj_size * (!!cmd.rnn.bidirectional + 1);
	memset(outputs[0].dim, 0, sizeof(outputs[0].dim));
	outputs[0].dim[0] = cmd.rnn.batch_first ? batch_count : max_seq_count;
	outputs[0].dim[1] = cmd.rnn.batch_first ? max_seq_count : batch_count;
	outputs[0].dim[2] = output_feature_count;
	outputs[0].type = inputs[0].type;
	outputs[0].format = inputs[0].format;
	outputs[0].datatype = inputs[0].datatype;
	if (output_size >= 4)
	{
		ccv_nnc_cmd_t lstm = ccv_nnc_cmd(CCV_NNC_LSTM_FORWARD, 0, cmd, 0);
		const int backend = ccv_nnc_cmd_find_backend(lstm, CCV_TENSOR_GET_MEMORY(inputs[0].type), inputs[0].format, inputs[0].datatype);
		lstm.backend = backend;
		ccv_nnc_lstm_reserve_space_size_f reserve_space_size = (ccv_nnc_lstm_reserve_space_size_f)ccv_nnc_cmd_aux(lstm);
		size_t total_size = reserve_space_size(lstm, inputs[0].datatype, feature_count, batch_count, max_seq_count);
		memset(outputs[3].dim, 0, sizeof(outputs[3].dim));
		outputs[3].dim[0] = (int)((total_size + cmd.rnn.hidden_size - 1) / cmd.rnn.hidden_size);
		outputs[3].dim[1] = cmd.rnn.hidden_size;
		outputs[3].type = inputs[0].type;
		outputs[3].format = inputs[0].format;
		outputs[3].datatype = inputs[0].datatype;
	}
	int i;
	if (input_size >= 4 && output_size >= 3)
		for (i = 0; i < 2; i++)
			outputs[i + 1] = inputs[i + 2];
}

REGISTER_COMMAND(CCV_NNC_LSTM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_lstm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_lstm_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_lstm_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_LSTM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_lstm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_lstm_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_LSTM_FORWARD)
#define CMD_LSTM_FORWARD(_hidden_size, _proj_size, _num_layers, _bias, _batch_first, _bidirectional, _dropout, _is_test) ccv_nnc_cmd(CCV_NNC_LSTM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.rnn={.hidden_size=_hidden_size,.proj_size=_proj_size,.num_layers=_num_layers,.bias=_bias,.batch_first=_batch_first,.bidirectional=_bidirectional,.dropout=_dropout,.is_test=_is_test}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_LSTM_BACKWARD)
#define CMD_LSTM_BACKWARD(_hidden_size, _proj_size, _num_layers, _bias, _batch_first, _bidirectional, _dropout, _is_test) ccv_nnc_cmd(CCV_NNC_LSTM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.rnn={.hidden_size=_hidden_size,.proj_size=_proj_size,.num_layers=_num_layers,.bias=_bias,.batch_first=_batch_first,.bidirectional=_bidirectional,.dropout=_dropout,.is_test=_is_test}}), 0)
