#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static void _ccv_nnc_reduce_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	outputs[0] = inputs[0];
	int i;
	for (i = 0; i < cmd.reduce.count; i++)
		outputs[0].dim[cmd.reduce.axis[i]] = 1; // Reduce the dimension to 1.
}

static int _ccv_nnc_reduce_isnan_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (input_bitmasks[0] == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_REDUCE_ISNAN_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_reduce_isnan_cpu_ref.c, gpu/ccv_nnc_reduce_isnan_gpu_cudnn.cu, mps/ccv_nnc_reduce_isnan_mps.m)
{
	registry->bitmask = _ccv_nnc_reduce_isnan_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_reduce_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_REDUCE_ISNAN_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_reduce_isnan_cpu_ref.c, gpu/ccv_nnc_reduce_isnan_gpu_cudnn.cu, mps/ccv_nnc_reduce_isnan_mps.m)
{
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_REDUCE_ISNAN_FORWARD)
#define CMD_REDUCE_ISNAN_FORWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_ISNAN_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_REDUCE_ISNAN_BACKWARD)
#define CMD_REDUCE_ISNAN_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_ISNAN_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)

