#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_gemm_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_gemm_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Output the propagated error, gradient w.r.t. w and bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2) | (0 << 3)) && output_bitmasks[0] == ((1u << 0) | (1u << 1) | (1u << 2)))
		return 1;
	// Don't propagate error, only gradient w.r.t. w and bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (0 << 2) | (0 << 3)) && output_bitmasks[0] == ((0 << 0) | (1u << 1) | (1u << 2)))
		return 1;
	return 0;
}

static void _ccv_nnc_gemm_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0].type = inputs[0].type;
	outputs[0].format = inputs[0].format;
	outputs[0].datatype = inputs[0].datatype;
	outputs[0].dim[1] = inputs[0].dim[1]; // batch size.
	outputs[0].dim[0] = inputs[1].dim[0]; // from the weight matrix.
	assert(inputs[1].dim[0] == cmd.blas.count);
	assert(inputs[1].dim[0] == inputs[2].dim[0]); // from the bias matrix.
}

REGISTER_COMMAND(CCV_NNC_GEMM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_gemm_cpu_ref.c, ccv_nnc_gemm_cpu_opt.c)
{
	registry->bitmask = _ccv_nnc_gemm_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_gemm_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_GEMM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_gemm_cpu_ref.c, ccv_nnc_gemm_cpu_opt.c)
{
	registry->bitmask = _ccv_nnc_gemm_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

static int _ccv_nnc_axpy_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	// It is OK to not having y
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (0u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_axpy_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// w.r.t. both x and y
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == ((1u << 0) | (1u << 1)))
		return 1;
	// w.r.t. x
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == ((1u << 0) | (0u << 1)))
		return 1;
	// w.r.t. y
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == ((0u << 0) | (1u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_AXPY_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_axpy_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_axpy_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_AXPY_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_axpy_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_axpy_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}
