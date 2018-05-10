#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_dropout_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 1 input (x)
	// 1 output (y, mask)
	if (input_bitmasks[0] == 1u && output_bitmasks[0] == 3u)
		return 1;
	return 0;
}

static int _ccv_nnc_xy_inplace(const int input_idx, const int output_idx)
{
	if (input_idx == 0 && output_idx == 0)
		return 1;
	return 0;
}

static void _ccv_nnc_dropout_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 2);
	outputs[0] = inputs[0];
	if (output_size == 1)
		return;
	outputs[1] = inputs[0];
	int i;
	// Reset to 0.
	memset(outputs[1].dim, 0, sizeof(outputs[1].dim));
	const int inc = (int)CCV_GET_DATA_TYPE_SIZE(inputs[0].datatype);
	// Align to 128-bytes boundary, for each computed result.
	int line = ((inputs[0].dim[0] + 127) >> 7);
	for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && inputs[0].dim[i] > 0; i++)
		line *= inputs[0].dim[i];
	assert((128 % inc) == 0);
	outputs[1].dim[0] = 128 / inc;
	outputs[1].dim[1] = line; // Aligned to 128 bytes, reserved space.
}

static int _ccv_nnc_dropout_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 0b10001
	// Inputs (dy, 0, 0, 0, mask)
	// Output the propagated error
	if ((input_bitmasks[0] & 17u) == 17u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_DROPOUT_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_dropout_cpu_ref.c, gpu/ccv_nnc_dropout_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_dropout_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_dropout_tensor_auto_forw;
	registry->enforce_inplace = _ccv_nnc_xy_inplace;
}

REGISTER_COMMAND(CCV_NNC_DROPOUT_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_dropout_cpu_ref.c, gpu/ccv_nnc_dropout_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_dropout_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_DROPOUT_FORWARD)
#define CMD_DROPOUT_FORWARD(_p) ccv_nnc_cmd(CCV_NNC_DROPOUT_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.dropout={.p=_p}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_DROPOUT_BACKWARD)
#define CMD_DROPOUT_BACKWARD(_p) ccv_nnc_cmd(CCV_NNC_DROPOUT_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.dropout={.p=_p}}), 0)
