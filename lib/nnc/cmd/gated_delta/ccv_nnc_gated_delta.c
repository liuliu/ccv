#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_gated_delta_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Inputs: q, k, v, log_decay, beta, state_in.
	// Outputs: y, state_out.
	if (input_size == 6 && output_size == 2 && (input_bitmasks[0] & 63u) == 63u && (output_bitmasks[0] & 3u) == 3u)
		return 1;
	return 0;
}

static int _ccv_nnc_gated_delta_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static int _ccv_nnc_gated_delta_allow_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (input_idx == 5 && output_idx == 1)
		return 1;
	return 0;
}

static void _ccv_nnc_gated_delta_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 6);
	assert(output_size == 2);
	const int q_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	const int k_nd = ccv_nnc_tensor_nd(inputs[1].dim);
	const int v_nd = ccv_nnc_tensor_nd(inputs[2].dim);
	const int log_decay_nd = ccv_nnc_tensor_nd(inputs[3].dim);
	const int beta_nd = ccv_nnc_tensor_nd(inputs[4].dim);
	const int state_nd = ccv_nnc_tensor_nd(inputs[5].dim);
	assert(q_nd == 4);
	assert(k_nd == 4);
	assert(v_nd == 4);
	assert(log_decay_nd == 3);
	assert(beta_nd == 3);
	assert(state_nd == 4);
	assert(inputs[0].dim[0] == inputs[1].dim[0]);
	assert(inputs[0].dim[0] == inputs[2].dim[0]);
	assert(inputs[0].dim[1] == inputs[1].dim[1]);
	assert(inputs[0].dim[1] == inputs[2].dim[1]);
	assert(inputs[0].dim[2] == inputs[1].dim[2]);
	assert(inputs[0].dim[3] == inputs[1].dim[3]);
	assert(inputs[2].dim[2] % inputs[0].dim[2] == 0);
	assert(inputs[3].dim[0] == inputs[0].dim[0]);
	assert(inputs[3].dim[1] == inputs[0].dim[1]);
	assert(inputs[3].dim[2] == inputs[2].dim[2]);
	assert(inputs[4].dim[0] == inputs[3].dim[0]);
	assert(inputs[4].dim[1] == inputs[3].dim[1]);
	assert(inputs[4].dim[2] == inputs[3].dim[2]);
	assert(inputs[5].dim[0] == inputs[0].dim[0]);
	assert(inputs[5].dim[1] == inputs[2].dim[2]);
	assert(inputs[5].dim[2] == inputs[2].dim[3]);
	assert(inputs[5].dim[3] == inputs[0].dim[3]);
	outputs[0] = inputs[2];
	outputs[1] = inputs[5];
}

REGISTER_COMMAND(CCV_NNC_GATED_DELTA_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_gated_delta_cpu_ref.c, mps/ccv_nnc_gated_delta_mps.m)
{
	registry->bitmask = _ccv_nnc_gated_delta_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_gated_delta_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_gated_delta_allow_inplace;
}

REGISTER_COMMAND(CCV_NNC_GATED_DELTA_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
{
	registry->bitmask = _ccv_nnc_gated_delta_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GATED_DELTA_FORWARD)
#define CMD_GATED_DELTA_FORWARD() ccv_nnc_cmd(CCV_NNC_GATED_DELTA_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}), 0)
