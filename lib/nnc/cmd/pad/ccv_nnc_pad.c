#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_pad_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_pad_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// We don't need the original input since roi align does averaging.
	// We do, however, need the coordinate.
	if ((input_bitmasks[0] & 1u) == (1u << 0) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_pad_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = inputs[0];
	const int nd = ccv_nnc_tensor_nd(outputs[0].dim);
	int i;
	for (i = 0; i < nd; i++)
		outputs[0].dim[i] += cmd.size.dim[i] + cmd.pad.end[i];
}

static void _ccv_nnc_pad_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = inputs[1];
}

REGISTER_COMMAND(CCV_NNC_PAD_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_pad_cpu_ref.c, gpu/ccv_nnc_pad_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_pad_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_pad_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_PAD_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_pad_cpu_ref.c, gpu/ccv_nnc_pad_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_pad_back_bitmask;
	registry->tensor_auto = _ccv_nnc_pad_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_PAD_FORWARD)
#define CMD_PAD_FORWARD(_type, _begin, _end) ccv_nnc_cmd(CCV_NNC_PAD_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={ESCAPE_X _begin}},.pad={.type=_type,.end={ESCAPE_X _end}}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_PAD_BACKWARD)
#define CMD_PAD_BACKWARD(_type, _begin, _end) ccv_nnc_cmd(CCV_NNC_PAD_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={ESCAPE_X _begin}},.pad={.type=_type,.end={ESCAPE_X _end}}}), 0)
