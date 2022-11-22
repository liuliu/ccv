#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_allreduce_allow_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == output_idx;
}

static int _ccv_nnc_allreduce_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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

static int _ccv_nnc_allreduce_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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

REGISTER_COMMAND(CCV_NNC_COMM_ALLREDUCE_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_comm_gpu_nccl.cu)
{
	registry->bitmask = _ccv_nnc_allreduce_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_allreduce_allow_inplace;
}

REGISTER_COMMAND(CCV_NNC_COMM_ALLREDUCE_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_comm_gpu_nccl.cu)
{
	registry->bitmask = _ccv_nnc_allreduce_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	registry->allow_inplace = _ccv_nnc_allreduce_allow_inplace;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMM_ALLREDUCE_FORWARD)
#define CMD_COMM_ALLREDUCE_FORWARD() ccv_nnc_cmd(CCV_NNC_COMM_ALLREDUCE_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMM_ALLREDUCE_BACKWARD)
#define CMD_COMM_ALLREDUCE_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMM_ALLREDUCE_BACKWARD, 0, ccv_nnc_cmd_auto, 0)

static int _ccv_nnc_arbitary_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return 1;
}

static int _ccv_nnc_broadcast_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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
	return output_bitcount >= 1 && output_bitcount == output_size && input_size == 1 && input_size == input_bitcount;
}

static int _ccv_nnc_broadcast_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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
	return input_bitcount >= 1 && input_bitcount == input_size && output_bitcount == output_size && output_size >= 1;
}

REGISTER_COMMAND(CCV_NNC_COMM_BROADCAST_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_comm_gpu_nccl.cu)
{
	registry->bitmask = _ccv_nnc_broadcast_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

REGISTER_COMMAND(CCV_NNC_COMM_BROADCAST_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_comm_gpu_nccl.cu)
{
	registry->bitmask = _ccv_nnc_broadcast_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMM_BROADCAST_FORWARD)
#define CMD_COMM_BROADCAST_FORWARD() ccv_nnc_cmd(CCV_NNC_COMM_BROADCAST_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMM_ALLREDUCE_BACKWARD)
#define CMD_COMM_BROADCAST_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMM_BROADCAST_BACKWARD, 0, ccv_nnc_cmd_auto, 0)

static int _ccv_nnc_first_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == output_idx && input_idx == 0;
}

static int _ccv_nnc_reduce_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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
	return output_size == input_size && output_bitcount == output_size && input_size >= 1 && input_size == input_bitcount;
}

static int _ccv_nnc_reduce_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
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
	return input_bitcount >= 1 && input_bitcount == input_size && output_bitcount == output_size && output_size >= 1;
}

REGISTER_COMMAND(CCV_NNC_COMM_REDUCE_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_comm_gpu_nccl.cu)
{
	registry->bitmask = _ccv_nnc_reduce_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_first_inplace;
}

REGISTER_COMMAND(CCV_NNC_COMM_REDUCE_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(gpu/ccv_nnc_comm_gpu_nccl.cu)
{
	registry->bitmask = _ccv_nnc_reduce_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMM_REDUCE_FORWARD)
#define CMD_COMM_REDUCE_FORWARD() ccv_nnc_cmd(CCV_NNC_COMM_REDUCE_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_COMM_REDUCE_BACKWARD)
#define CMD_COMM_REDUCE_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMM_REDUCE_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
