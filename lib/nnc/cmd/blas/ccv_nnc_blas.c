#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_arbitary_inplace(const int input_idx, const int output_idx)
{
	return 1;
}

static int _ccv_nnc_gemm_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (input_size == 3 && (input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	// No bias is OK.
	if (input_size == 2 && (input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_gemm_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Output the propagated error, gradient w.r.t. w and bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2) | (0 << 3)) && output_bitmasks[0] == ((1u << 0) | (1u << 1) | (1u << 2)))
		return 1;
	// No bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2) | (0 << 3)) && output_bitmasks[0] == ((1u << 0) | (1u << 1) | (0u << 2)))
		return 1;
	// Don't propagate error, only gradient w.r.t. w and bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (0 << 2) | (0 << 3)) && output_bitmasks[0] == ((0 << 0) | (1u << 1) | (1u << 2)))
		return 1;
	// No bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (0 << 2) | (0 << 3)) && output_bitmasks[0] == ((0 << 0) | (1u << 1) | (0u << 2)))
		return 1;
	return 0;
}

static void _ccv_nnc_gemm_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0].type = inputs[0].type;
	outputs[0].format = inputs[0].format;
	outputs[0].datatype = inputs[0].datatype;
	outputs[0].dim[0] = inputs[0].dim[0]; // batch size.
	outputs[0].dim[1] = inputs[1].dim[0]; // from the weight matrix.
	assert(inputs[1].dim[0] == cmd.blas.count);
}

REGISTER_COMMAND(CCV_NNC_GEMM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_gemm_cpu_ref.c, ccv_nnc_gemm_cpu_opt.c, gpu/ccv_nnc_gemm_gpu_cublas.cu)
{
	registry->bitmask = _ccv_nnc_gemm_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_gemm_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_GEMM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_gemm_cpu_ref.c, ccv_nnc_gemm_cpu_opt.c, gpu/ccv_nnc_gemm_gpu_cublas.cu)
{
	registry->bitmask = _ccv_nnc_gemm_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GEMM_FORWARD)
#define CMD_GEMM_FORWARD(_count) ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, 0, CMD_GEMM(_count), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GEMM_BACKWARD)
#define CMD_GEMM_BACKWARD(_count) ccv_nnc_cmd(CCV_NNC_GEMM_BACKWARD, 0, CMD_GEMM(_count), 0)

static int _ccv_nnc_add_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_add_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// w.r.t. both x and y
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == ((1u << 0) | (1u << 1)))
		return 1;
	// w.r.t. x
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == ((1u << 0) | (0u << 1)))
		return 1;
	// w.r.t. y
	if ((input_bitmasks[0] & 1u) == 1u &&  output_bitmasks[0] == ((0u << 0) | (1u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_ADD_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_add_cpu_ref.c, gpu/ccv_nnc_add_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_add_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

REGISTER_COMMAND(CCV_NNC_ADD_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_add_cpu_ref.c, gpu/ccv_nnc_add_gpu_cudnn.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_add_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ADD_FORWARD)
#define CMD_ADD_FORWARD(...) ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, 0, CMD_BLAS(__VA_ARGS__), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ADD_BACKWARD)
#define CMD_ADD_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_ADD_BACKWARD, 0, CMD_BLAS(__VA_ARGS__), 0)

static int _ccv_nnc_mul_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_mul_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// w.r.t. both x and y
	if ((input_bitmasks[0] & 7u) == 7u && output_bitmasks[0] == ((1u << 0) | (1u << 1)))
		return 1;
	// w.r.t. x
	if ((input_bitmasks[0] & 5u) == 5u && output_bitmasks[0] == ((1u << 0) | (0u << 1)))
		return 1;
	// w.r.t. y
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == ((0u << 0) | (1u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_MUL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_mul_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

REGISTER_COMMAND(CCV_NNC_MUL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_mul_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MUL_FORWARD)
#define CMD_MUL_FORWARD(...) ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, 0, CMD_BLAS(__VA_ARGS__), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MUL_BACKWARD)
#define CMD_MUL_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_MUL_BACKWARD, 0, CMD_BLAS(__VA_ARGS__), 0)

static int _ccv_nnc_scalar_mul_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_scalar_mul_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// w.r.t. x
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_SCALAR_MUL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_scalar_mul_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_arbitary_inplace;
}

REGISTER_COMMAND(CCV_NNC_SCALAR_MUL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_scalar_mul_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALAR_MUL_FORWARD)
#define CMD_SCALAR_MUL_FORWARD(...) ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, 0, CMD_BLAS(__VA_ARGS__), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALAR_MUL_BACKWARD)
#define CMD_SCALAR_MUL_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_BACKWARD, 0, CMD_BLAS(__VA_ARGS__), 0)
