#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_leaky_relu_allow_first_replace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == 0 && output_idx == 0;
}

static int _ccv_nnc_leaky_relu_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_leaky_relu_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 5u) == ((1u << 0) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_LEAKY_RELU_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_leaky_relu_cpu_ref.c, gpu/ccv_nnc_leaky_relu_gpu_ref.cu, mps/ccv_nnc_leaky_relu_mps.m)
{
	registry->bitmask = _ccv_nnc_leaky_relu_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_leaky_relu_allow_first_replace;
}

REGISTER_COMMAND(CCV_NNC_LEAKY_RELU_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_leaky_relu_cpu_ref.c, gpu/ccv_nnc_leaky_relu_gpu_ref.cu, mps/ccv_nnc_leaky_relu_mps.m)
{
	registry->bitmask = _ccv_nnc_leaky_relu_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	registry->allow_inplace = _ccv_nnc_leaky_relu_allow_first_replace;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_LEAKY_RELU_FORWARD)
#define CMD_LEAKY_RELU_FORWARD(_negative_slope) ccv_nnc_cmd(CCV_NNC_LEAKY_RELU_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.leaky_relu={.negative_slope=_negative_slope}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_LEAKY_RELU_BACKWARD)
#define CMD_LEAKY_RELU_BACKWARD(_negative_slope) ccv_nnc_cmd(CCV_NNC_LEAKY_RELU_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.leaky_relu={.negative_slope=_negative_slope}}, 0)
