#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_scatter_add_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_scatter_add_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 5u) == 5u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_scatter_add_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 2);
	assert(output_size == 1);
	outputs[0] = inputs[0];
	if (cmd.scatter_add.bincount > 0)
		outputs[0].dim[0] = cmd.scatter_add.bincount;
	else
		outputs[0].dim[0] = inputs[1].dim[0];
}

REGISTER_COMMAND(CCV_NNC_SCATTER_ADD_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scatter_add_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_scatter_add_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_scatter_add_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_SCATTER_ADD_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scatter_add_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_scatter_add_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient; // This is just best guess.
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCATTER_ADD_FORWARD)
#define CMD_SCATTER_ADD_FORWARD(_bincount) ccv_nnc_cmd(CCV_NNC_SCATTER_ADD_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scatter_add={.bincount=_bincount}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCATTER_ADD_BACKWARD)
#define CMD_SCATTER_ADD_BACKWARD(_bincount) ccv_nnc_cmd(CCV_NNC_SCATTER_ADD_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scatter_add={.bincount=_bincount}}), 0)
