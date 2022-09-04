#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_batch_norm_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 5 inputs (x, scale, bias, mean, var)
	// 1 outputs (y)
	if (input_bitmasks[0] == 31u && output_bitmasks[0] == 1u)
		return 1;
	// 5 inputs (x, scale, bias, mean, var)
	// 5 outputs (y, mean, var, saved_mean, saved_inv_var)
	// Both mean and var in output is inplace for the input mean, var
	if (input_bitmasks[0] == 31u && output_bitmasks[0] == 31u)
		return 1;
	return 0;
}

static int _ccv_nnc_batch_norm_enforce_inplace(const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (input_idx == 3 && output_idx == 1)
		return 1;
	if (input_idx == 4 && output_idx == 2)
		return 1;
	return 0;
}

static int _ccv_nnc_batch_norm_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 0b110000001100001
	// Inputs (gradient, 0, 0, 0, 0, x, scale, 0, 0, 0, 0, 0, 0, saved_mean, saved_inv_var)
	// Output the propagated error, dscale and dbias
	if ((input_bitmasks[0] & 24673u) == 24673u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	return 0;
}

static void _ccv_nnc_batch_norm_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 5);
	assert(output_size == 1 || output_size == 5);
	outputs[0] = inputs[0];
	if (output_size == 1)
		return;
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		for (j = 0; j < cmd.bnorm.count; j++)
			outputs[i].dim[cmd.bnorm.axis[j]] = 1; // Reduce the dimension to 1.
	}
}

static void _ccv_nnc_batch_norm_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 15);
	assert(output_size == 5);
	outputs[0] = inputs[0];
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		for (j = 0; j < cmd.bnorm.count; j++)
			outputs[i].dim[cmd.bnorm.axis[j]] = 1; // Reduce the dimension to 1.
	}
}

REGISTER_COMMAND(CCV_NNC_BATCH_NORM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_batch_norm_cpu_ref.c, gpu/ccv_nnc_batch_norm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_batch_norm_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_batch_norm_tensor_auto_forw;
	registry->enforce_inplace = _ccv_nnc_batch_norm_enforce_inplace;
}

REGISTER_COMMAND(CCV_NNC_BATCH_NORM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_batch_norm_cpu_ref.c, gpu/ccv_nnc_batch_norm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_batch_norm_back_bitmask;
	registry->tensor_auto = _ccv_nnc_batch_norm_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_BATCH_NORM_FORWARD)
#define CMD_BATCH_NORM_FORWARD(_epsilon, _is_test, _momentum, ...) ccv_nnc_cmd(CCV_NNC_BATCH_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.bnorm={.epsilon=_epsilon,.is_test=_is_test,.momentum=_momentum,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_BATCH_NORM_BACKWARD)
#define CMD_BATCH_NORM_BACKWARD(_epsilon, _is_test, _momentum, ...) ccv_nnc_cmd(CCV_NNC_BATCH_NORM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.bnorm={.epsilon=_epsilon,.is_test=_is_test,.momentum=_momentum,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)

static int _ccv_nnc_layer_norm_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 3 inputs (x, gamma, beta)
	// 3 outputs (y, saved_mean, saved_inv_std)
	if (input_bitmasks[0] == 7u && output_bitmasks[0] == 7u)
		return 1;
	// 3 inputs (x, gamma, beta)
	// 1 output (y)
	if (input_bitmasks[0] == 7u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_layer_norm_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 0b110011001
	// Inputs (gradient, 0, 0, x, gamma, 0, 0, saved_mean, saved_inv_std)
	// Output the propagated error, dgamma and dbeta
	if ((input_bitmasks[0] & 409u) == 409u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	return 0;
}

static void _ccv_nnc_layer_norm_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3);
	assert(output_size == 1 || output_size == 3);
	outputs[0] = inputs[0];
	if (output_size == 1)
		return;
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		for (j = 0; j < cmd.lnorm.count; j++)
			outputs[i].dim[cmd.lnorm.axis[j]] = 1; // Reduce the dimension to 1.
	}
}

static void _ccv_nnc_layer_norm_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 9);
	assert(output_size == 1 || output_size == 3);
	outputs[0] = inputs[0];
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		for (j = 0; j < cmd.lnorm.count; j++)
			outputs[i].dim[cmd.lnorm.axis[j]] = 1; // Reduce the dimension to 1.
	}
}

REGISTER_COMMAND(CCV_NNC_LAYER_NORM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_layer_norm_cpu_ref.c, gpu/ccv_nnc_layer_norm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_layer_norm_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_layer_norm_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_LAYER_NORM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_layer_norm_cpu_ref.c, gpu/ccv_nnc_layer_norm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_layer_norm_back_bitmask;
	registry->tensor_auto = _ccv_nnc_layer_norm_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_LAYER_NORM_FORWARD)
#define CMD_LAYER_NORM_FORWARD(_epsilon, ...) ccv_nnc_cmd(CCV_NNC_LAYER_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.lnorm={.epsilon=_epsilon,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_LAYER_NORM_BACKWARD)
#define CMD_LAYER_NORM_BACKWARD(_epsilon, ...) ccv_nnc_cmd(CCV_NNC_LAYER_NORM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.lnorm={.epsilon=_epsilon,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)

static int _ccv_nnc_group_norm_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 3 inputs (x, gamma, beta)
	// 3 outputs (y, saved_mean, saved_inv_std)
	if (input_bitmasks[0] == 7u && output_bitmasks[0] == 7u)
		return 1;
	// 3 inputs (x, gamma, beta)
	// 1 output (y)
	if (input_bitmasks[0] == 7u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_group_norm_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 0b110011001
	// Inputs (gradient, 0, 0, x, gamma, 0, 0, saved_mean, saved_inv_std)
	// Output the propagated error, dgamma and dbeta
	if ((input_bitmasks[0] & 409u) == 409u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	return 0;
}

static void _ccv_nnc_group_norm_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3);
	assert(output_size == 1 || output_size == 3);
	outputs[0] = inputs[0];
	if (output_size == 1)
		return;
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		outputs[i].dim[cmd.gnorm.group_axis] = cmd.gnorm.groups; // Reduce to num_groups.
		for (j = 0; j < cmd.gnorm.reduce_count; j++)
			outputs[i].dim[cmd.gnorm.reduce_axis[j]] = 1;
	}
}

static void _ccv_nnc_group_norm_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 9);
	assert(output_size == 1 || output_size == 3);
	outputs[0] = inputs[0];
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		outputs[i].dim[cmd.gnorm.group_axis] = cmd.gnorm.groups; // Reduce the dimension to num_groups.
		for (j = 0; j < cmd.gnorm.reduce_count; j++)
			outputs[i].dim[cmd.gnorm.reduce_axis[j]] = 1;
	}
}

REGISTER_COMMAND(CCV_NNC_GROUP_NORM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_group_norm_cpu_ref.c, gpu/ccv_nnc_group_norm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_group_norm_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_group_norm_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_GROUP_NORM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_group_norm_cpu_ref.c, gpu/ccv_nnc_group_norm_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_group_norm_back_bitmask;
	registry->tensor_auto = _ccv_nnc_group_norm_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GROUP_NORM_FORWARD)
#define CMD_GROUP_NORM_FORWARD(_axis, _groups, _epsilon) ccv_nnc_cmd(CCV_NNC_GROUP_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.gnorm={.axis=_axis,.groups=_groups,.epsilon=_epsilon}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GROUP_NORM_BACKWARD)
#define CMD_GROUP_NORM_BACKWARD(_axis, _groups, _epsilon) ccv_nnc_cmd(CCV_NNC_GROUP_NORM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.gnorm={.axis=_axis,.groups=_groups,.epsilon=_epsilon}}), 0)
