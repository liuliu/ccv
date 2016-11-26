#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_conv_forw_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if ((input_bitmask & 7u) == ((1u << 0) | (1u << 1) | (1u << 2)) && output_bitmask == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_conv_back_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	// Output the propagated error, gradient w.r.t. w and bias.
	if ((input_bitmask & 15u) == ((1u << 0) | (1u << 1) | (1u << 2) | (0 << 3)) && output_bitmask == ((1u << 0) | (1u << 1) | (1u << 2)))
		return 1;
	// Don't propagate error, only gradient w.r.t. w and bias.
	if ((input_bitmask & 15u) == ((1u << 0) | (1u << 1) | (0 << 2) | (0 << 3)) && output_bitmask == ((0 << 0) | (1u << 1) | (1u << 2)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_CONVOLUTION_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_conv_cpu_ref.c, ccv_nnc_conv_cpu_opt.c, ccv_nnc_conv_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_conv_forw_bitmask;
}

REGISTER_COMMAND(CCV_NNC_CONVOLUTION_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_conv_cpu_ref.c, ccv_nnc_conv_cpu_opt.c, ccv_nnc_conv_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_conv_back_bitmask;
}
