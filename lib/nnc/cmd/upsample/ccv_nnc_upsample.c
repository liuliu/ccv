#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"

static void _ccv_nnc_upsample_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 1);
	assert(output_size == 1);
	outputs[0] = inputs[0];
	const int nd = ccv_nnc_tensor_nd(inputs[0].dim);
	if (nd == 2)
	{
		outputs[0].dim[0] = (int)(inputs[0].dim[0] * cmd.upsample.height_scale + 0.5);
		outputs[0].dim[1] = (int)(inputs[0].dim[1] * cmd.upsample.width_scale + 0.5);
	} else if (nd == 3) {
		if (inputs[0].format == CCV_TENSOR_FORMAT_NCHW || inputs[0].format == CCV_TENSOR_FORMAT_CHWN)
		{
			outputs[0].dim[nd - 2] = (int)(inputs[0].dim[nd - 2] * cmd.upsample.height_scale + 0.5);
			outputs[0].dim[nd - 1] = (int)(inputs[0].dim[nd - 1] * cmd.upsample.width_scale + 0.5);
		} else {
			outputs[0].dim[0] = (int)(inputs[0].dim[0] * cmd.upsample.height_scale + 0.5);
			outputs[0].dim[1] = (int)(inputs[0].dim[1] * cmd.upsample.width_scale + 0.5);
		}
	} else if (nd == 4) {
		if (inputs[0].format == CCV_TENSOR_FORMAT_NCHW)
		{
			outputs[0].dim[nd - 2] = (int)(inputs[0].dim[nd - 2] * cmd.upsample.height_scale + 0.5);
			outputs[0].dim[nd - 1] = (int)(inputs[0].dim[nd - 1] * cmd.upsample.width_scale + 0.5);
		} else {
			outputs[0].dim[nd - 3] = (int)(inputs[0].dim[nd - 3] * cmd.upsample.height_scale + 0.5);
			outputs[0].dim[nd - 2] = (int)(inputs[0].dim[nd - 2] * cmd.upsample.width_scale + 0.5);
		}
	}
}

static int _ccv_nnc_upsample_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (input_bitmasks[0] == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_upsample_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Output the propagated error.
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_UPSAMPLE_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_upsample_cpu_ref.c, gpu/ccv_nnc_upsample_gpu_ref.cu, mps/ccv_nnc_upsample_mps.m)
{
	registry->bitmask = _ccv_nnc_upsample_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_upsample_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_UPSAMPLE_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_upsample_cpu_ref.c, gpu/ccv_nnc_upsample_gpu_ref.cu, mps/ccv_nnc_upsample_mps.m)
{
	registry->bitmask = _ccv_nnc_upsample_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_UPSAMPLE_FORWARD)
#define CMD_UPSAMPLE_FORWARD(_type, _width_scale, _height_scale, _align_corners) ccv_nnc_cmd(CCV_NNC_UPSAMPLE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.upsample={.type=_type,.width_scale=_width_scale,.height_scale=_height_scale,.align_corners=_align_corners}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_UPSAMPLE_BACKWARD)
#define CMD_UPSAMPLE_BACKWARD(_type, _width_scale, _height_scale, _align_corners) ccv_nnc_cmd(CCV_NNC_UPSAMPLE_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.upsample={.type=_type,.width_scale=_width_scale,.height_scale=_height_scale,.align_corners=_align_corners}}), 0)
