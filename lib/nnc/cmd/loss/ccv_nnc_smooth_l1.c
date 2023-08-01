#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_smooth_l1_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: activation, label
	// output: loss
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_smooth_l1_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: [gradient of loss], activation, label, loss
	// output: w.r.t activation, [label]
	if ((input_bitmasks[0] & 14u) == 14u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_smooth_l1_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size >= 1);
	outputs[0] = inputs[0];
	// The output should have the same dimentionality of the label data.
	memcpy(outputs[0].dim, inputs[1].dim, sizeof(outputs[0].dim));
	const int nd = ccv_nnc_tensor_nd(outputs[0].dim);
	// Set channel to 1 if it is not..
	if (nd > 1 && ccv_nnc_tensor_get_c(outputs[0]) > 1)
		ccv_nnc_tensor_set_c(&outputs[0], nd, 1);
}

static void _ccv_nnc_smooth_l1_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	outputs[0] = inputs[1];
	if (output_size > 1)
		outputs[1] = inputs[2];
}

REGISTER_COMMAND(CCV_NNC_SMOOTH_L1_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_smooth_l1_cpu_ref.c, gpu/ccv_nnc_smooth_l1_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_smooth_l1_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_smooth_l1_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_SMOOTH_L1_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_smooth_l1_cpu_ref.c, gpu/ccv_nnc_smooth_l1_gpu_ref.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_smooth_l1_back_bitmask;
	registry->tensor_auto = _ccv_nnc_smooth_l1_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SMOOTH_L1_FORWARD)
#define CMD_SMOOTH_L1_FORWARD(_b) ccv_nnc_cmd(CCV_NNC_SMOOTH_L1_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.smooth_l1={.beta=_b}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SMOOTH_L1_BACKWARD)
#define CMD_SMOOTH_L1_BACKWARD(_b) ccv_nnc_cmd(CCV_NNC_SMOOTH_L1_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.smooth_l1={.beta=_b}}), 0)
