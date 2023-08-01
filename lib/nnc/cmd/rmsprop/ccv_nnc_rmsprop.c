#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_rmsprop_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 4 inputs (gradient, x, momentum, squared gradient average)
	// 3 outputs (y, new momentum, new squared gradient average)
	if (input_bitmasks[0] == 15u && output_bitmasks[0] == 7u)
		return 1;
	return 0;
}

static int _ccv_nnc_rmsprop_allow_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (input_idx == output_idx + 1)
		return 1;
	return 0;
}

static int _ccv_nnc_rmsprop_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Doesn't support.
	return 0;
}

static void _ccv_nnc_rmsprop_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[0];
}

static void _ccv_nnc_rmsprop_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	// Doesn't support.
}

REGISTER_COMMAND(CCV_NNC_RMSPROP_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_rmsprop_cpu_ref.c, gpu/ccv_nnc_rmsprop_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_rmsprop_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_rmsprop_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_rmsprop_allow_inplace;
}

REGISTER_COMMAND(CCV_NNC_RMSPROP_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_rmsprop_cpu_ref.c, gpu/ccv_nnc_rmsprop_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_rmsprop_back_bitmask;
	registry->tensor_auto = _ccv_nnc_rmsprop_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_RMSPROP_FORWARD)
#define CMD_RMSPROP_FORWARD(_rate, _decay, _alpha, _momentum, _epsilon) ccv_nnc_cmd(CCV_NNC_RMSPROP_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.rmsprop={.rate=_rate,.scale=1,.decay=_decay,.alpha=_alpha,.momentum=_momentum,.epsilon=_epsilon}}), 0)
