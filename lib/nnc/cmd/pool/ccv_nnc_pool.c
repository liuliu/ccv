#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_max_pool_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_max_pool_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_pool_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = inputs[0];
	ccv_nnc_hint_tensor_forward(cmd, inputs[0], hint, outputs);
}

static void _ccv_nnc_pool_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = inputs[0];
	ccv_nnc_hint_tensor_backward(cmd, inputs[0], hint, outputs);
}

REGISTER_COMMAND(CCV_NNC_MAX_POOL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_max_pool_cpu_ref.c, gpu/ccv_nnc_max_pool_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_max_pool_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_pool_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_MAX_POOL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_max_pool_cpu_ref.c, gpu/ccv_nnc_max_pool_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_max_pool_back_bitmask;
	registry->tensor_auto = _ccv_nnc_pool_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MAX_POOL_FORWARD)
#define CMD_MAX_POOL_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_MAX_POOL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MAX_POOL_BACKWARD)
#define CMD_MAX_POOL_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_MAX_POOL_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)

static int _ccv_nnc_avg_pool_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_avg_pool_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_AVERAGE_POOL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_avg_pool_cpu_ref.c, gpu/ccv_nnc_avg_pool_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_avg_pool_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_pool_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_AVERAGE_POOL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_avg_pool_cpu_ref.c, gpu/ccv_nnc_avg_pool_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_avg_pool_back_bitmask;
	registry->tensor_auto = _ccv_nnc_pool_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_AVERAGE_POOL_FORWARD)
#define CMD_AVERAGE_POOL_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_AVERAGE_POOL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_AVERAGE_POOL_BACKWARD)
#define CMD_AVERAGE_POOL_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_AVERAGE_POOL_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
