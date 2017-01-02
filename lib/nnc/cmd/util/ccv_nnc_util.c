#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_set_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_SET_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_set_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_SET_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_set_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

static int _ccv_nnc_data_transfer_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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
	return output_bitcount == input_bitcount;
}

REGISTER_COMMAND(CCV_NNC_DATA_TRANSFER_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_DATA_TRANSFER_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_FORMAT_TRANSFORM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_FORMAT_TRANSFORM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}
