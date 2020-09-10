#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_nms_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 3u)
		return 1;
	return 0;
}

static int _ccv_nnc_nms_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// gradient of sorted, gradient of sorting index, input, output of sorted, output of sorting index.
	if ((input_bitmasks[0] & 17u) == ((1u << 0) | (0u << 1) | (0u << 2) | (0u << 3) | (1u << 4)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_nms_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 2);
	outputs[0] = inputs[0];
	const int nd = ccv_nnc_tensor_nd(inputs[0].dim);
	assert(nd >= 1);
	outputs[1] = inputs[0];
	outputs[1].datatype = CCV_32S;
	memset(outputs[1].dim, 0, sizeof(outputs[1].dim));
	outputs[1].dim[0] = inputs[0].dim[0]; // How many to rank (or batch size).
	outputs[1].dim[1] = (nd <= 2) ? 0 : inputs[0].dim[1]; // How many to rank.
}

static void _ccv_nnc_nms_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0] = inputs[2];
}

REGISTER_COMMAND(CCV_NNC_NMS_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_nms_cpu_ref.c, gpu/ccv_nnc_nms_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_nms_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_nms_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_NMS_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_nms_cpu_ref.c, gpu/ccv_nnc_nms_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_nms_back_bitmask;
	registry->tensor_auto = _ccv_nnc_nms_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_NMS_FORWARD)
#define CMD_NMS_FORWARD(_iou_threshold) ccv_nnc_cmd(CCV_NNC_NMS_FORWARD, 0, ((ccv_nnc_cmd_param_t){.nms={.iou_threshold=_iou_threshold}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_NMS_BACKWARD)
#define CMD_NMS_BACKWARD(_iou_threshold) ccv_nnc_cmd(CCV_NNC_NMS_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.nms={.iou_threshold=_iou_threshold}}), 0)
