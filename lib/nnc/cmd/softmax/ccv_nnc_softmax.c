#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_softmax_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_softmax_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_SOFTMAX_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_softmax_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_softmax_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_SOFTMAX_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_softmax_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_softmax_back_bitmask;
}
