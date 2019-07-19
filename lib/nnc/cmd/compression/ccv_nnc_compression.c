#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_lssc_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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

static int _ccv_nnc_lssc_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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

static void _ccv_nnc_lssc_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	int i, j;
	assert(output_size <= input_size);
	for (i = 0; i < output_size; i++)
	{
		assert(inputs[i].datatype == CCV_16F);
		const int nd = ccv_nnc_tensor_nd(inputs[i].dim);
		const int hw = ccv_nnc_tensor_hw(inputs[i], nd);
		outputs[i] = inputs[i];
		for (j = 0; j < CCV_NNC_MAX_DIM - 1; j++)
			outputs[i].dim[j + hw] = (inputs[i].dim[j + hw] + 3) / 4;
		outputs[i].dim[CCV_NNC_MAX_DIM - 1 + hw] = (inputs[i].dim[CCV_NNC_MAX_DIM - 1 + hw] + 3) / 4 * 4;
	}
}

static void _ccv_nnc_lssc_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	int i, j;
	assert(output_size <= input_size);
	for (i = 0; i < output_size; i++)
	{
		assert(inputs[i].datatype == CCV_16F);
		const int nd = ccv_nnc_tensor_nd(inputs[i].dim);
		const int hw = ccv_nnc_tensor_hw(inputs[i], nd);
		outputs[i] = inputs[i];
		for (j = 0; j < CCV_NNC_MAX_DIM - 1; j++)
			outputs[i].dim[j + hw] = inputs[i].dim[j + hw] * 4;
		assert(outputs[i].dim[CCV_NNC_MAX_DIM - 1 + hw] % 4 == 0);
	}
}

REGISTER_COMMAND(CCV_NNC_COMPRESSION_LSSC_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_lssc_cpu_ref.c, gpu/ccv_nnc_lssc_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_lssc_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_lssc_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_COMPRESSION_LSSC_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_lssc_cpu_ref.c, gpu/ccv_nnc_lssc_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_lssc_back_bitmask;
	registry->tensor_auto = _ccv_nnc_lssc_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMPRESSION_LSSC_FORWARD)
#define CMD_COMPRESSION_LSSC_FORWARD() ccv_nnc_cmd(CCV_NNC_COMPRESSION_LSSC_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMPRESSION_LSSC_BACKWARD)
#define CMD_COMPRESSION_LSSC_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMPRESSION_LSSC_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
