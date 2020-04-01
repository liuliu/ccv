#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_adam_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 3 inputs (gradient, x, momentum, velocity)
	// 2 outputs (y, new momentum, new velocity)
	if (input_bitmasks[0] == 15u && output_bitmasks[0] == 7u)
		return 1;
	return 0;
}

static int _ccv_nnc_adam_allow_inplace(const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (input_idx == output_idx + 1)
		return 1;
	return 0;
}

static int _ccv_nnc_adam_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Doesn't support.
	return 0;
}

static void _ccv_nnc_adam_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[0];
}

static void _ccv_nnc_adam_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	// Doesn't support.
}

REGISTER_COMMAND(CCV_NNC_ADAM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_adam_cpu_ref.c, gpu/ccv_nnc_adam_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_adam_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_adam_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_adam_allow_inplace;
}

REGISTER_COMMAND(CCV_NNC_ADAM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_adam_cpu_ref.c, gpu/ccv_nnc_adam_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_adam_back_bitmask;
	registry->tensor_auto = _ccv_nnc_adam_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ADAM_FORWARD)
#define CMD_ADAM_FORWARD(_step, _rate, _beta1, _beta2, _decay, _epsilon) ccv_nnc_cmd(CCV_NNC_ADAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.adam={.step=_step,.rate=_rate,.beta1=_beta1,.beta2=_beta2,.decay=_decay,.epsilon=_epsilon}}), 0)
