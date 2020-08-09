#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_roi_align_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_roi_align_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// We don't need the original input since roi align does averaging.
	// We do, however, need the coordinate.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (0u << 1) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_roi_align_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = inputs[0];
	const int hw = ccv_nnc_tensor_hw(outputs[0], ccv_nnc_tensor_nd(outputs[0].dim));
	assert(hw >= 0);
	outputs[0].dim[hw] = ccv_max(cmd.size.dim[0], 1);
	outputs[0].dim[hw + 1] = ccv_max(cmd.size.dim[1], 1);
}

static void _ccv_nnc_roi_align_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = inputs[1];
}

REGISTER_COMMAND(CCV_NNC_ROI_ALIGN_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_roi_align_cpu_ref.c, gpu/ccv_nnc_roi_align_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_roi_align_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_roi_align_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_ROI_ALIGN_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_roi_align_cpu_ref.c, gpu/ccv_nnc_roi_align_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_roi_align_back_bitmask;
	registry->tensor_auto = _ccv_nnc_roi_align_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ROI_ALIGN_FORWARD)
#define CMD_ROI_ALIGN_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_ROI_ALIGN_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ROI_ALIGN_BACKWARD)
#define CMD_ROI_ALIGN_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_ROI_ALIGN_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
