#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_segmented_swiglu_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return input_size == 6 && output_size == 1 && (input_bitmasks[0] & 63u) == 63u && output_bitmasks[0] == 1u;
}

static int _ccv_nnc_segmented_swiglu_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_segmented_swiglu_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 6);
	assert(output_size == 1);
	outputs[0] = inputs[0];
	const int route_weight_nd = ccv_nnc_tensor_nd(inputs[5].dim);
	const int weight_nd = ccv_nnc_tensor_nd(inputs[3].dim);
	assert(route_weight_nd >= 1 && weight_nd >= 2);
	memset(outputs[0].dim, 0, sizeof(outputs[0].dim));
	memcpy(outputs[0].dim, inputs[5].dim, sizeof(int) * route_weight_nd);
	if (route_weight_nd > 1 && inputs[5].dim[route_weight_nd - 1] == 1)
		outputs[0].dim[route_weight_nd - 1] = inputs[3].dim[weight_nd - 2];
	else {
		assert(route_weight_nd < CCV_NNC_MAX_DIM_ALLOC);
		outputs[0].dim[route_weight_nd] = inputs[3].dim[weight_nd - 2];
	}
}

REGISTER_COMMAND(CCV_NNC_SEGMENTED_SWIGLU_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_segmented_swiglu_cpu_ref.c, mps/ccv_nnc_segmented_swiglu_mps.m)
{
	registry->bitmask = _ccv_nnc_segmented_swiglu_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_segmented_swiglu_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_SEGMENTED_SWIGLU_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_segmented_swiglu_cpu_ref.c, mps/ccv_nnc_segmented_swiglu_mps.m)
{
	registry->bitmask = _ccv_nnc_segmented_swiglu_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SEGMENTED_SWIGLU_FORWARD)
#define CMD_SEGMENTED_SWIGLU_FORWARD(_clamp) ccv_nnc_cmd(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.segmented_swiglu={.clamp=_clamp}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SEGMENTED_SWIGLU_BACKWARD)
#define CMD_SEGMENTED_SWIGLU_BACKWARD(_clamp) ccv_nnc_cmd(CCV_NNC_SEGMENTED_SWIGLU_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.segmented_swiglu={.clamp=_clamp}}), 0)
