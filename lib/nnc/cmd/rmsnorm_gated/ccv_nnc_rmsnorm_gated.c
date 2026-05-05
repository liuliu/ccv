#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_rmsnorm_gated_allow_first_replace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == 0 && output_idx == 0;
}

static int _ccv_nnc_rmsnorm_gated_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (cmd.rmsnorm_gated.elementwise_affine)
	{
		// 3 inputs (x, gate, gamma)
		// 1 output (y)
		if (input_size == 3 && output_size == 1 && input_bitmasks[0] == 7u && output_bitmasks[0] == 1u)
			return 1;
	} else {
		// 2 inputs (x, gate)
		// 1 output (y)
		if (input_size == 2 && output_size == 1 && input_bitmasks[0] == 3u && output_bitmasks[0] == 1u)
			return 1;
	}
	return 0;
}

static int _ccv_nnc_rmsnorm_gated_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

REGISTER_COMMAND(CCV_NNC_RMSNORM_GATED_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_rmsnorm_gated_cpu_ref.c, mps/ccv_nnc_rmsnorm_gated_mps.m)
{
	registry->bitmask = _ccv_nnc_rmsnorm_gated_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_rmsnorm_gated_allow_first_replace;
}

REGISTER_COMMAND(CCV_NNC_RMSNORM_GATED_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
{
	registry->bitmask = _ccv_nnc_rmsnorm_gated_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_RMSNORM_GATED_FORWARD)
#define CMD_RMSNORM_GATED_FORWARD(_epsilon, _elementwise_affine, ...) ccv_nnc_cmd(CCV_NNC_RMSNORM_GATED_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.rmsnorm_gated={.epsilon=_epsilon,.elementwise_affine=_elementwise_affine,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)
