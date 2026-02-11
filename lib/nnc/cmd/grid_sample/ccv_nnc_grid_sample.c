#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_grid_sample_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_grid_sample_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 5u) == ((1u << 0) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_GRID_SAMPLE_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_grid_sample_cpu_ref.c, gpu/ccv_nnc_grid_sample_gpu_cudnn.cu, mps/ccv_nnc_grid_sample_mps.m)
{
	registry->bitmask = _ccv_nnc_grid_sample_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_GRID_SAMPLE_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_grid_sample_cpu_ref.c, gpu/ccv_nnc_grid_sample_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_grid_sample_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GRID_SAMPLE_FORWARD)
#define CMD_GRID_SAMPLE_FORWARD(_align_corners) ccv_nnc_cmd(CCV_NNC_GRID_SAMPLE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.grid_sample={.align_corners=(_align_corners)}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GRID_SAMPLE_BACKWARD)
#define CMD_GRID_SAMPLE_BACKWARD(_align_corners) ccv_nnc_cmd(CCV_NNC_GRID_SAMPLE_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.grid_sample={.align_corners=(_align_corners)}}), 0)
