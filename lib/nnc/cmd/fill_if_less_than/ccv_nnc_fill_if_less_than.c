#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_fill_if_less_than_allow_first_replace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == 0 && output_idx == 0;
}

static int _ccv_nnc_fill_if_less_than_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return input_size == 3 && output_size == 1 && (input_bitmasks[0] & 7u) == 7u && output_bitmasks[0] == 1u;
}

static int _ccv_nnc_fill_if_less_than_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return (input_bitmasks[0] & 13u) == 13u && output_bitmasks[0] == 1u;
}

REGISTER_COMMAND(CCV_NNC_FILL_IF_LESS_THAN_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_fill_if_less_than_cpu_ref.c, mps/ccv_nnc_fill_if_less_than_mps.m)
{
	registry->bitmask = _ccv_nnc_fill_if_less_than_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_fill_if_less_than_allow_first_replace;
}

REGISTER_COMMAND(CCV_NNC_FILL_IF_LESS_THAN_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_fill_if_less_than_cpu_ref.c, mps/ccv_nnc_fill_if_less_than_mps.m)
{
	registry->bitmask = _ccv_nnc_fill_if_less_than_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_FILL_IF_LESS_THAN_FORWARD)
#define CMD_FILL_IF_LESS_THAN_FORWARD(_fill) ccv_nnc_cmd(CCV_NNC_FILL_IF_LESS_THAN_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.fill_if_less_than={.value=_fill}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_FILL_IF_LESS_THAN_BACKWARD)
#define CMD_FILL_IF_LESS_THAN_BACKWARD(_fill) ccv_nnc_cmd(CCV_NNC_FILL_IF_LESS_THAN_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.fill_if_less_than={.value=_fill}}, 0)
