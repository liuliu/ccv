#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_sgd_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 3 inputs (gradient, x, momentum)
	// 2 outputs (y, new momentum)
	if (input_bitmasks[0] == 7u && output_bitmasks[0] == 3u)
		return 1;
	return 0;
}

static int _ccv_nnc_sgd_allow_inplace(const int input_idx, const int output_idx)
{
	if (input_idx == output_idx + 1)
		return 1;
	return 0;
}

static int _ccv_nnc_sgd_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Doesn't support.
	return 0;
}

static void _ccv_nnc_sgd_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[0];
}

static void _ccv_nnc_sgd_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	// Doesn't support.
}

REGISTER_COMMAND(CCV_NNC_SGD_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_sgd_cpu_ref.c, gpu/ccv_nnc_sgd_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_sgd_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_sgd_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_sgd_allow_inplace;
}

REGISTER_COMMAND(CCV_NNC_SGD_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_sgd_cpu_ref.c, gpu/ccv_nnc_sgd_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_sgd_back_bitmask;
	registry->tensor_auto = _ccv_nnc_sgd_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SGD_FORWARD)
#define CMD_SGD_FORWARD(_rate, _decay, _momentum, _dampening) ccv_nnc_cmd(CCV_NNC_SGD_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.minimize={.rate=_rate,.decay=_decay,.momentum=_momentum,.dampening=_dampening}}), 0)
