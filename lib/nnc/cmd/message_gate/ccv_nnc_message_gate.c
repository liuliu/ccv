#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_message_gate_allow_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	// The output is written after the mix has read all of the state and the message, and each of its
	// values reads only theirs at its position.
	return (input_idx == 0 || input_idx == 1) && output_idx == 0;
}

static int _ccv_nnc_message_gate_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Inputs: the state [..., C], the message [..., C], the down projection [R, 2C] and the forget
	// and gain projections [C, R]. Outputs: the gated state [..., C], then scratch for the
	// normalized state and message side by side [..., 2C] and for the mix [..., R], both FP32.
	if (input_size == 5 && output_size == 3 && (input_bitmasks[0] & 31u) == 31u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	return 0;
}

static int _ccv_nnc_message_gate_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_message_gate_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 5);
	assert(output_size == 3);
	outputs[0] = inputs[0];
	// The scratch outputs replace the last axis with 2C and with R, in FP32.
	const int nd = ccv_nnc_tensor_nd(inputs[0].dim);
	outputs[1] = inputs[0];
	outputs[1].datatype = CCV_32F;
	outputs[1].dim[nd - 1] = inputs[0].dim[nd - 1] * 2;
	outputs[2] = outputs[1];
	outputs[2].dim[nd - 1] = inputs[2].dim[0];
}

REGISTER_COMMAND(CCV_NNC_MESSAGE_GATE_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_message_gate_cpu_ref.c, mps/ccv_nnc_message_gate_mps.m)
{
	registry->bitmask = _ccv_nnc_message_gate_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_message_gate_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_message_gate_allow_inplace;
}

REGISTER_COMMAND(CCV_NNC_MESSAGE_GATE_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_message_gate_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_message_gate_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MESSAGE_GATE_FORWARD)
#define CMD_MESSAGE_GATE_FORWARD(_epsilon) ccv_nnc_cmd(CCV_NNC_MESSAGE_GATE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.message_gate={.epsilon=(_epsilon)}}), 0)
