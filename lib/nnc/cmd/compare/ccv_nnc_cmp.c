#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"

static int _ccv_nnc_arbitary_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return 1;
}

static int _ccv_nnc_cmp_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_cmp_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// w.r.t. both x and y, either way, need gradient, input x, input y.
	if ((input_bitmasks[0] & 7u) == 7u && output_bitmasks[0] == ((1u << 0) | (1u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_MIN_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_min_cpu_ref.c, gpu/ccv_nnc_min_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_cmp_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

REGISTER_COMMAND(CCV_NNC_MIN_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_min_cpu_ref.c, gpu/ccv_nnc_min_gpu_ref.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_cmp_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MIN_FORWARD)
#define CMD_MIN_FORWARD() ccv_nnc_cmd(CCV_NNC_MIN_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MIN_BACKWARD)
#define CMD_MIN_BACKWARD() ccv_nnc_cmd(CCV_NNC_MIN_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)

REGISTER_COMMAND(CCV_NNC_MAX_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_max_cpu_ref.c, gpu/ccv_nnc_max_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_cmp_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

REGISTER_COMMAND(CCV_NNC_MAX_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_max_cpu_ref.c, gpu/ccv_nnc_max_gpu_ref.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_cmp_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MAX_FORWARD)
#define CMD_MAX_FORWARD() ccv_nnc_cmd(CCV_NNC_MAX_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MAX_BACKWARD)
#define CMD_MAX_BACKWARD() ccv_nnc_cmd(CCV_NNC_MAX_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)
