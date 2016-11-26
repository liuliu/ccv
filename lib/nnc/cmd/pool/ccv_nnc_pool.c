#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_max_pool_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_max_pool_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 7u) == ((1u << 0) | (1u << 1) | (1u << 2)) && output_bitmask == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_MAX_POOL_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_max_pool_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_max_pool_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_MAX_POOL_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_max_pool_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_max_pool_back_bitmask;
}

static int _ccv_nnc_avg_pool_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_avg_pool_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_AVERAGE_POOL_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_avg_pool_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_avg_pool_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_AVERAGE_POOL_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_avg_pool_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_avg_pool_back_bitmask;
}
