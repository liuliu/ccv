#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_random_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_RANDOM_UNIFORM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_rand_uniform_cpu_ref.c, gpu/ccv_nnc_rand_uniform_gpu_ref.cu, mps/ccv_nnc_rand_uniform_mps.m)
{
	registry->bitmask = _ccv_nnc_random_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_RANDOM_UNIFORM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_rand_uniform_cpu_ref.c, gpu/ccv_nnc_rand_uniform_gpu_ref.cu, mps/ccv_nnc_rand_uniform_mps.m)
{
	registry->bitmask = _ccv_nnc_random_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_RANDOM_UNIFORM_FORWARD)
#define CMD_RANDOM_UNIFORM_FORWARD(_lb, _ub) ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_lb, _ub}}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_RANDOM_UNIFORM_BACKWARD)
#define CMD_RANDOM_UNIFORM_BACKWARD(_lb, _ub) ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_lb, _ub}}}, 0)

REGISTER_COMMAND(CCV_NNC_RANDOM_NORMAL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_rand_normal_cpu_ref.c, gpu/ccv_nnc_rand_normal_gpu_ref.cu, mps/ccv_nnc_rand_normal_mps.m)
{
	registry->bitmask = _ccv_nnc_random_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_RANDOM_NORMAL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_rand_normal_cpu_ref.c, gpu/ccv_nnc_rand_normal_gpu_ref.cu, mps/ccv_nnc_rand_normal_mps.m)
{
	registry->bitmask = _ccv_nnc_random_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_RANDOM_NORMAL_FORWARD)
#define CMD_RANDOM_NORMAL_FORWARD(_std, _mean) ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_std, _mean}}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_RANDOM_NORMAL_BACKWARD)
#define CMD_RANDOM_NORMAL_BACKWARD(_std, _mean) ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_std, _mean}}}, 0)
