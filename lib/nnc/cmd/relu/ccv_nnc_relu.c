#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_relu_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_relu_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_RELU_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_relu_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_relu_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_RELU_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_relu_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_relu_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}
