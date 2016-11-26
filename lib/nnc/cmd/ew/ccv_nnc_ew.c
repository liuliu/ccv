#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_ewsum_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if (output_bitmask == 1)
	{
		int i;
		for (i = 0; i < 64; i++)
			if (!(input_bitmask & (uint64_t)1 << i))
				break;
		// Always like 1111100000, no 1110010101
		for (; i < 64; i++)
			if (input_bitmask & (uint64_t)1 << i)
				return 0;
		return 1;
	}
	return 0;
}

static int _ccv_nnc_ewsum_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u)
	{
		int i;
		for (i = 0; i < 64; i++)
			if (!(output_bitmask & (uint64_t)1 << i))
				break;
		// Always like 1111100000, no 1110010101
		for (; i < 64; i++)
			if (output_bitmask & (uint64_t)1 << i)
				return 0;
		return 1;
	}
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWSUM_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewsum_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_EWSUM_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_PASSTHROUGH;
	registry->bitmask = _ccv_nnc_ewsum_back_bitmask;
}

static int _ccv_nnc_ewprod_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if (output_bitmask == 1)
	{
		int i;
		for (i = 0; i < 64; i++)
			if (!(input_bitmask & (uint64_t)1 << i))
				break;
		// Always like 1111100000, no 1110010101
		for (; i < 64; i++)
			if (input_bitmask & (uint64_t)1 << i)
				return 0;
		return 1;
	}
	return 0;
}

static int _ccv_nnc_ewprod_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	int i;
	for (i = 1; i < 64; i++)
		if (!(input_bitmask & (uint64_t)1 << i))
			break;
	const int input_bitcount = i;
	// Always like 1111100000, no 1110010101
	for (; i < 64; i++)
		if (input_bitmask & (uint64_t)1 << i)
			return 0;
	for (i = 0; i < 64; i++)
		if (!(output_bitmask & (uint64_t)1 << i))
			break;
	const int output_bitcount = i;
	for (; i < 64; i++)
		if (output_bitmask & (uint64_t)1 << i)
			return 0;
	return output_bitcount + 2 /* Gradient + Original output */ == input_bitcount;
}

REGISTER_COMMAND(CCV_NNC_EWPROD_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewprod_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_EWPROD_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewprod_back_bitmask;
}

static int _ccv_nnc_ewdiv_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 3u) == ((1u << 0) | (1u << 1)) && output_bitmask == 1u)
		return 1;
	// Nominator can be null (meaning 1).
	if ((input_bitmask & 3u) == ((0u << 0) | (1u << 1)) && output_bitmask == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWDIV_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewdiv_forw_bitmask;
}

static int _ccv_nnc_ewdiv_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	// We don't need to know the original output.
	if ((input_bitmask & (15u & ~((uint64_t)1u << 1))) == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3)) && output_bitmask == ((1u << 0) | (1u << 1)))
		return 1;
	if ((input_bitmask & (15u & ~((uint64_t)1u << 1))) == ((1u << 0) | (0u << 1) | (1u << 2) | (0u << 3)) && output_bitmask == ((1u << 0) | (0u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWDIV_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewdiv_back_bitmask;
}

static int _ccv_nnc_ewexp_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_ewexp_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	// We don't care about the original output.
	if ((input_bitmask & (7u & ~((uint64_t)1u << 1))) == ((1u << 0) | (0u << 1) | (1u << 2)) && output_bitmask == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWEXP_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewexp_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_EWEXP_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewexp_back_bitmask;
}

static int _ccv_nnc_ewlog_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_ewlog_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	// We don't care about the original input.
	if ((input_bitmask & 1u) == 1u && output_bitmask == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWLOG_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewlog_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_EWLOG_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewlog_back_bitmask;
}
