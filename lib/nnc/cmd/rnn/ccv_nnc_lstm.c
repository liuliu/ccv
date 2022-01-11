#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_lstm_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: x, [xs], hx, cx, w.
	// output: y, hy, cy, [r]
	if (input_size == 5 && (input_bitmasks[0] & 31u) == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5)) && output_bitmasks[0] == ((1u << 0) | (1u << 1) | (1u << 2)))
		return 1;
	if (input_size == 5 && (input_bitmasks[0] & 31u) == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5)) && output_bitmasks[0] == ((1u << 0) | (1u << 1) | (1u << 2) | (1u << 3)))
		return 1;
	return 0;
}

static int _ccv_nnc_lstm_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: dy, dhy, dcy, [dr]
	// output: dx, [dxs], dhx, dcx.
	if ((input_bitmasks[0] & 8191u) == ((1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (0u << 4) | (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12)) && output_bitmasks[0] == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3) | (0u << 4)))
		return 1;
	// Output dx, [dxs], dhx, dcx and dw.
	if ((input_bitmasks[0] & 8191u) == ((1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (0u << 4) | (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12)) && output_bitmasks[0] == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3) | 1u << 4))
		return 1;
	// Output dw.
	if ((input_bitmasks[0] & 8191u) == ((0u << 0) | (0u << 1) | (0u << 2) | (1u << 3) | (0u << 4) | (0u << 5) | (1u << 6) | (1u << 7) | (0u << 8) | (0u << 9) | (1u << 10) | (1u << 11) | (1u << 12)) && output_bitmasks[0] == ((0u << 0) | (0u << 1) | (0u << 2) | (0u << 3) | 1u << 4))
		return 1;
	return 0;
}

static void _ccv_nnc_lstm_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size < input_size);
	int i;
	for (i = 0; i < output_size; i++)
	{
		outputs[i].type = inputs[i].type;
		outputs[i].format = inputs[i].format;
		outputs[i].datatype = inputs[i].datatype;
	}
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
