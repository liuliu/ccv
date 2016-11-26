#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_gemm_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 7u) == ((1u << 0) | (1u << 1) | (1u << 2)) && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_gemm_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	// Output the propagated error, gradient w.r.t. w and bias.
	if ((input_bitmask & 7u) == ((1u << 0) | (1u << 1) | (1u << 2) | (0 << 3)) && output_bitmask == ((1u << 0) | (1u << 1) | (1u << 2)))
		return 1;
	// Don't propagate error, only gradient w.r.t. w and bias.
	if ((input_bitmask & 7u) == ((1u << 0) | (1u << 1) | (0 << 2) | (0 << 3)) && output_bitmask == ((0 << 0) | (1u << 1) | (1u << 2)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_GEMM_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_gemm_cpu_ref.c, ccv_nnc_gemm_cpu_opt.c)
{
	registry->bitmask = _ccv_nnc_gemm_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_GEMM_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_gemm_cpu_ref.c, ccv_nnc_gemm_cpu_opt.c)
{
	registry->bitmask = _ccv_nnc_gemm_back_bitmask;
}

static int _ccv_nnc_axpy_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 3u) == ((1u << 0) | (1u << 1)) && output_bitmask == 1u)
		return 1;
	// It is OK to not having y
	if ((input_bitmask & 3u) == ((1u << 0) | (0u << 1)) && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_axpy_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	// w.r.t. both x and y
	if ((input_bitmask & 1u) == 1u && output_bitmask == ((1u << 0) | (1u << 1)))
		return 1;
	// w.r.t. x
	if ((input_bitmask & 1u) == 1u && output_bitmask == ((1u << 0) | (0u << 1)))
		return 1;
	// w.r.t. y
	if ((input_bitmask & 1u) == 1u && output_bitmask == ((0u << 0) | (1u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_AXPY_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_axpy_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_axpy_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_AXPY_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_axpy_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_axpy_back_bitmask;
}
