#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_signed_sqrt_allow_first_replace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == 0 && output_idx == 0;
}

static int _ccv_nnc_signed_sqrt_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return (input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u;
}

static int _ccv_nnc_signed_sqrt_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return (input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 1u;
}

REGISTER_COMMAND(CCV_NNC_SIGNED_SQRT_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_signed_sqrt_cpu_ref.c, mps/ccv_nnc_signed_sqrt_mps.m)
{
	registry->bitmask = _ccv_nnc_signed_sqrt_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_signed_sqrt_allow_first_replace;
}

REGISTER_COMMAND(CCV_NNC_SIGNED_SQRT_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_signed_sqrt_cpu_ref.c, mps/ccv_nnc_signed_sqrt_mps.m)
{
	registry->bitmask = _ccv_nnc_signed_sqrt_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	// Keep separately bound model input / output gradients in distinct tensor
	// headers. The backends also accept an explicitly supplied in-place gradient.
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SIGNED_SQRT_FORWARD)
#define CMD_SIGNED_SQRT_FORWARD(_min) ccv_nnc_cmd(CCV_NNC_SIGNED_SQRT_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.signed_sqrt={.minimum_magnitude=(_min)}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SIGNED_SQRT_BACKWARD)
#define CMD_SIGNED_SQRT_BACKWARD(_min) ccv_nnc_cmd(CCV_NNC_SIGNED_SQRT_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.signed_sqrt={.minimum_magnitude=(_min)}}), 0)
