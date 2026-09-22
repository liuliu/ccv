#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_sol_attention_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return input_size == 3 && output_size == 1 && input_bitmasks[0] == 7u && output_bitmasks[0] == 1u;
}

static int _ccv_nnc_sol_attention_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0; // Forward-only approximation.
}

static void _ccv_nnc_sol_attention_tensor_auto(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3 && output_size == 1);
	assert(ccv_nnc_tensor_nd(inputs[0].dim) == 4);
	outputs[0] = inputs[0];
}

REGISTER_COMMAND(CCV_NNC_SOL_ATTENTION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_sol_attention_cpu_ref.c, mps/ccv_nnc_sol_attention_mps.m)
{
	registry->bitmask = _ccv_nnc_sol_attention_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_sol_attention_tensor_auto;
}

REGISTER_COMMAND(CCV_NNC_SOL_ATTENTION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
{
	registry->bitmask = _ccv_nnc_sol_attention_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SOL_ATTENTION_FORWARD)
#define CMD_SOL_ATTENTION_FORWARD(_scale, _tau, _block_size, _start, _end) ccv_nnc_cmd(CCV_NNC_SOL_ATTENTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.sol_attention={.scale=_scale,.tau=_tau,.block_size=_block_size,.approximation_start=_start,.approximation_end=_end,.local_block_radius=1}}), 0)
