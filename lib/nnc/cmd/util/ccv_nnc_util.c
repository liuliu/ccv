#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_set_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	int i, j, flag = 0;
	int output_bitcount = 0;
	for (i = 0; i < output_bitmask_size; i++)
	{
		for (j = 0; j < 64; j++)
			if (output_bitmasks[i] & (uint64_t)1 << j)
			{
				if (flag)
					return 0;
			} else
				break;
		output_bitcount += j;
		// Trailing zero even if it is not the end of input_bitmask_size, mark flag,
		// if we encounter additional 1, return invalid.
		if (j < 64)
			flag = 1;
		// Always like 1111100000, no 1110010101
		for (; j < 64; j++)
			if (output_bitmasks[i] & (uint64_t)1 << j)
				return 0;
	}
	return output_size == output_bitcount;
}

REGISTER_COMMAND(CCV_NNC_SET_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_set_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_SET_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_set_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SET_FORWARD)
#define CMD_SET_FORWARD(_val) ccv_nnc_cmd(CCV_NNC_SET_FORWARD, 0, CMD_BLAS(_val), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SET_BACKWARD)
#define CMD_SET_BACKWARD(_val) ccv_nnc_cmd(CCV_NNC_SET_BACKWARD, 0, CMD_BLAS(_val), 0)

static int _ccv_nnc_masked_fill_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (input_size == 2 && (input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_masked_fill_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 5u) == ((1u << 0) | (0u << 1) | (1u << 2)) && output_bitmasks[0] == ((1u << 0) | (1u << 1)))
		return 1;
	if ((input_bitmasks[0] & 5u) == ((1u << 0) | (0u << 1) | (1u << 2)) && output_bitmasks[0] == (1u << 0))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_MASKED_FILL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_masked_fill_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_MASKED_FILL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_masked_fill_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient_and_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MASKED_FILL_FORWARD)
#define CMD_MASKED_FILL_FORWARD(_eq, _fill) ccv_nnc_cmd(CCV_NNC_MASKED_FILL_FORWARD, 0, CMD_BLAS(_eq, _fill), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MASKED_FILL_BACKWARD)
#define CMD_MASKED_FILL_BACKWARD(_eq, _fill) ccv_nnc_cmd(CCV_NNC_MASKED_FILL_BACKWARD, 0, CMD_BLAS(_eq, _fill), 0)

static int _ccv_nnc_data_transfer_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	int i, j;
	int input_flag = 0;
	int input_bitcount = 0;
	for (i = 0; i < input_bitmask_size; i++)
	{
		for (j = 0; j < 64; j++)
			if (input_bitmasks[i] & (uint64_t)1 << j)
			{
				if (input_flag)
					return 0;
			} else
				break;
		input_bitcount += j;
		if (j < 64)
			input_flag = 1;
		// Always like 1111100000, no 1110010101
		for (; j < 64; j++)
			if (input_bitmasks[i] & (uint64_t)1 << j)
				return 0;
	}
	int output_flag = 0;
	int output_bitcount = 0;
	for (i = 0; i < output_bitmask_size; i++)
	{
		for (j = 0; j < 64; j++)
			if (output_bitmasks[i] & (uint64_t)1 << j)
			{
				if (output_flag)
					return 0;
			} else
				break;
		output_bitcount += j;
		if (j < 64)
			output_flag = 1;
		for (; j < 64; j++)
			if (output_bitmasks[i] & (uint64_t)1 << j)
				return 0;
	}
	return output_bitcount == input_bitcount && input_size == output_size && input_size == input_bitcount;
}

static int _ccv_nnc_data_transfer_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	int i, j;
	int input_flag = 0;
	int input_bitcount = 0;
	for (i = 0; i < input_bitmask_size; i++)
	{
		for (j = 0; j < 64; j++)
			if (input_bitmasks[i] & (uint64_t)1 << j)
			{
				if (input_flag)
					return 0;
			} else
				break;
		input_bitcount += j;
		if (j < 64)
			input_flag = 1;
		// Always like 1111100000, no 1110010101
		for (; j < 64; j++)
			if (input_bitmasks[i] & (uint64_t)1 << j)
				return 0;
	}
	int output_flag = 0;
	int output_bitcount = 0;
	for (i = 0; i < output_bitmask_size; i++)
	{
		for (j = 0; j < 64; j++)
			if (output_bitmasks[i] & (uint64_t)1 << j)
			{
				if (output_flag)
					return 0;
			} else
				break;
		output_bitcount += j;
		if (j < 64)
			output_flag = 1;
		for (; j < 64; j++)
			if (output_bitmasks[i] & (uint64_t)1 << j)
				return 0;
	}
	return output_bitcount <= input_bitcount && output_bitcount == output_size;
}

REGISTER_COMMAND(CCV_NNC_DATA_TRANSFER_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_DATA_TRANSFER_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_DATA_TRANSFER_FORWARD)
#define CMD_DATA_TRANSFER_FORWARD() ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_DATA_TRANSFER_BACKWARD)
#define CMD_DATA_TRANSFER_BACKWARD() ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_BACKWARD, 0, ccv_nnc_cmd_auto, 0)

REGISTER_COMMAND(CCV_NNC_FORMAT_TRANSFORM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_FORMAT_TRANSFORM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_FORMAT_TRANSFORM_FORWARD)
#define CMD_FORMAT_TRANSFORM_FORWARD() ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_FORMAT_TRANSFORM_BACKWARD)
#define CMD_FORMAT_TRANSFORM_BACKWARD() ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_BACKWARD, 0, ccv_nnc_cmd_auto, 0)

static void _ccv_nnc_transpose_tensor_auto(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
	{
		outputs[i] = inputs[i];
		int t;
		CCV_SWAP(outputs[i].dim[cmd.transpose.axis[0]], outputs[i].dim[cmd.transpose.axis[1]], t);
	}
}

REGISTER_COMMAND(CCV_NNC_TRANSPOSE_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_transpose_tensor_auto;
}

REGISTER_COMMAND(CCV_NNC_TRANSPOSE_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_back_bitmask;
	registry->tensor_auto = _ccv_nnc_transpose_tensor_auto;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_TRANSPOSE_FORWARD)
#define CMD_TRANSPOSE_FORWARD(_axis_a, _axis_b) ccv_nnc_cmd(CCV_NNC_TRANSPOSE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.transpose={.axis={_axis_a, _axis_b}}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_TRANSPOSE_BACKWARD)
#define CMD_TRANSPOSE_BACKWARD(_axis_a, _axis_b) ccv_nnc_cmd(CCV_NNC_TRANSPOSE_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.transpose={.axis={_axis_a, _axis_b}}}), 0)

REGISTER_COMMAND(CCV_NNC_DATATYPE_CONVERSION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_DATATYPE_CONVERSION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, gpu/ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_DATATYPE_CONVERSION_FORWARD)
#define CMD_DATATYPE_CONVERSION_FORWARD() ccv_nnc_cmd(CCV_NNC_DATATYPE_CONVERSION_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_DATATYPE_CONVERSION_BACKWARD)
#define CMD_DATATYPE_CONVERSION_BACKWARD() ccv_nnc_cmd(CCV_NNC_DATATYPE_CONVERSION_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
