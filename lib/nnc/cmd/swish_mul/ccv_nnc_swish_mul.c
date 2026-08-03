#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_swish_mul_allow_first_replace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == 0 && output_idx == 0;
}

static int _ccv_nnc_swish_mul_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	const uint64_t required = cmd.swish_mul.weighted ? 7u : 3u;
	const int required_inputs = cmd.swish_mul.weighted ? 3 : 2;
	if (input_size == required_inputs && output_size == 1 && (input_bitmasks[0] & required) == required && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_swish_mul_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (cmd.swish_mul.weighted || cmd.swish_mul.clamp > 1.0e-6f)
		return 0;
	// w.r.t. both value and gate.
	if ((input_bitmasks[0] & 7u) == 7u && output_bitmasks[0] == ((1u << 0) | (1u << 1)))
		return 1;
	// w.r.t. value. Only gate is needed from the forward inputs.
	if ((input_bitmasks[0] & 5u) == 5u && output_bitmasks[0] == ((1u << 0) | (0u << 1)))
		return 1;
	// w.r.t. gate.
	if ((input_bitmasks[0] & 7u) == 7u && output_bitmasks[0] == ((0u << 0) | (1u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_SWISH_MUL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_swish_mul_cpu_ref.c, gpu/ccv_nnc_swish_mul_gpu_ref.cu, mps/ccv_nnc_swish_mul_mps.m)
{
	registry->bitmask = _ccv_nnc_swish_mul_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_swish_mul_allow_first_replace;
}

REGISTER_COMMAND(CCV_NNC_SWISH_MUL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_swish_mul_cpu_ref.c, gpu/ccv_nnc_swish_mul_gpu_ref.cu, mps/ccv_nnc_swish_mul_mps.m)
{
	registry->bitmask = _ccv_nnc_swish_mul_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SWISH_MUL_FORWARD)
#define CMD_SWISH_MUL_FORWARD(_beta, _scale) ccv_nnc_cmd(CCV_NNC_SWISH_MUL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.swish_mul={.beta=_beta,.scale=_scale}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SWISH_MUL_FORWARD)
#define CMD_WEIGHTED_SWISH_MUL_FORWARD(_beta, _scale, _clamp) ccv_nnc_cmd(CCV_NNC_SWISH_MUL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.swish_mul={.beta=_beta,.scale=_scale,.clamp=_clamp,.weighted=1}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SWISH_MUL_BACKWARD)
#define CMD_SWISH_MUL_BACKWARD(_beta, _scale) ccv_nnc_cmd(CCV_NNC_SWISH_MUL_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.swish_mul={.beta=_beta,.scale=_scale}}), 0)
