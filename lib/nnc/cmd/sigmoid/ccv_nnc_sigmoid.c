#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_sigmoid_allow_first_replace(const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == 0 && output_idx == 0;
}

static int _ccv_nnc_sigmoid_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_sigmoid_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// gradient, [x], y
	if ((input_bitmasks[0] & 5u) == 5u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_sigmoid_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	outputs[0] = inputs[0];
}

static void _ccv_nnc_sigmoid_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	outputs[0] = inputs[2];
}

REGISTER_COMMAND(CCV_NNC_SIGMOID_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_sigmoid_cpu_ref.c, gpu/ccv_nnc_sigmoid_gpu_cudnn.cu, mps/ccv_nnc_sigmoid_mps.m)
{
	registry->bitmask = _ccv_nnc_sigmoid_forw_bitmask;
	registry->allow_inplace = _ccv_nnc_sigmoid_allow_first_replace;
	registry->tensor_auto = _ccv_nnc_sigmoid_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_SIGMOID_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_sigmoid_cpu_ref.c, gpu/ccv_nnc_sigmoid_gpu_cudnn.cu, mps/ccv_nnc_sigmoid_mps.m)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_sigmoid_back_bitmask;
	registry->allow_inplace = _ccv_nnc_sigmoid_allow_first_replace;
	registry->tensor_auto = _ccv_nnc_sigmoid_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SIGMOID_FORWARD)
#define CMD_SIGMOID_FORWARD() ccv_nnc_cmd(CCV_NNC_SIGMOID_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SIGMOID_BACKWARD)
#define CMD_SIGMOID_BACKWARD() ccv_nnc_cmd(CCV_NNC_SIGMOID_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
