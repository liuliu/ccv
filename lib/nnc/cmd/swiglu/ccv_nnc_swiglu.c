#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_swiglu_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return input_size == 3 && output_size == 1 && (input_bitmasks[0] & 7u) == 7u && output_bitmasks[0] == 1u;
}

static int _ccv_nnc_swiglu_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_swiglu_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3);
	assert(output_size == 1);
	outputs[0] = inputs[0];
	if (inputs[0].dim[0] == 0 || inputs[1].dim[0] == 0)
	{
		memset(outputs[0].dim, 0, sizeof(outputs[0].dim));
		return;
	}
	const int a_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	const int gate_w_nd = ccv_nnc_tensor_nd(inputs[1].dim);
	assert(a_nd >= 1 && gate_w_nd == 2);
	outputs[0].dim[a_nd - 1] = inputs[1].dim[gate_w_nd - 2];
}

REGISTER_COMMAND(CCV_NNC_SWIGLU_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_swiglu_cpu_ref.c, mps/ccv_nnc_swiglu_mps.m)
{
	registry->bitmask = _ccv_nnc_swiglu_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_swiglu_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_SWIGLU_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_swiglu_cpu_ref.c, mps/ccv_nnc_swiglu_mps.m)
{
	registry->bitmask = _ccv_nnc_swiglu_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SWIGLU_FORWARD)
#define CMD_SWIGLU_FORWARD(_clamp) ccv_nnc_cmd(CCV_NNC_SWIGLU_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.swiglu={.clamp=_clamp}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SWIGLU_BACKWARD)
#define CMD_SWIGLU_BACKWARD(_clamp) ccv_nnc_cmd(CCV_NNC_SWIGLU_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.swiglu={.clamp=_clamp}}), 0)
