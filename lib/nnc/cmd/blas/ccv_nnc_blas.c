#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"

static int _ccv_nnc_same_pos_inplace(const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	// For cudnnOpTensor: "If the input tensor B is the same tensor as the destination tensor C, then the input tensor A also must be the same tensor as the destination tensor C."
	return input_idx == output_idx;
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
	int a_batch_size, a_rows, a_cols, a_batch_inc, a_rows_inc, a_cols_inc;
	int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
	const int a_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	ccv_nnc_tensor_get_matrix_params(inputs[0], 0, inputs[0].dim, cmd.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
	ccv_nnc_tensor_get_matrix_params(inputs[1], 0, inputs[1].dim, cmd.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
	outputs[0].type = inputs[0].type;
	outputs[0].format = inputs[0].format;
	outputs[0].datatype = inputs[0].datatype;
	int b_rows = a_rows, b_cols = w_cols;
	if (a_nd == 1) {
		outputs[0].dim[0] = b_cols;
	} else if (a_nd == 2) {
		outputs[0].dim[0] = b_rows;
		outputs[0].dim[1] = b_cols;
	} else {
		assert(a_nd == 3);
		outputs[0].dim[0] = a_batch_size;
		outputs[0].dim[1] = b_rows;
		outputs[0].dim[2] = b_cols;
	}
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
#define CMD_GEMM_FORWARD(...) ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, 0, CMD_GEMM(__VA_ARGS__), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_GEMM_BACKWARD)
#define CMD_GEMM_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_GEMM_BACKWARD, 0, CMD_GEMM(__VA_ARGS__), 0)

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

static void _ccv_nnc_broadcast_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 2);
	assert(output_size == 1);
	const int a_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	const int b_nd = ccv_nnc_tensor_nd(inputs[1].dim);
	outputs[0] = inputs[0];
	const int c_nd = ccv_max(a_nd, b_nd);
	int i;
	for (i = a_nd - 1; i >= 0; i--)
		outputs[0].dim[i + c_nd - a_nd] = inputs[0].dim[i];
	for (i = b_nd - 1; i >= 0; i--)
		outputs[0].dim[i + c_nd - b_nd] = ccv_max(outputs[0].dim[i + c_nd - b_nd], inputs[1].dim[i]);
}

REGISTER_COMMAND(CCV_NNC_ADD_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_add_cpu_ref.c, gpu/ccv_nnc_add_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_add_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_broadcast_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_same_pos_inplace;
}

REGISTER_COMMAND(CCV_NNC_ADD_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_add_cpu_ref.c, gpu/ccv_nnc_add_gpu_cudnn.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_add_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ADD_FORWARD)
#define CMD_ADD_FORWARD(_p, _q) ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p, _q}}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ADD_BACKWARD)
#define CMD_ADD_BACKWARD(_p, _q) ccv_nnc_cmd(CCV_NNC_ADD_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p, _q}}}, 0)

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
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c, gpu/ccv_nnc_mul_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_mul_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_broadcast_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_same_pos_inplace;
}

REGISTER_COMMAND(CCV_NNC_MUL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c, gpu/ccv_nnc_mul_gpu_cudnn.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_mul_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MUL_FORWARD)
#define CMD_MUL_FORWARD(_p) ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p,}}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MUL_BACKWARD)
#define CMD_MUL_BACKWARD(_p) ccv_nnc_cmd(CCV_NNC_MUL_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p,}}}, 0)

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
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c, gpu/ccv_nnc_mul_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_scalar_mul_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_same_pos_inplace;
}

REGISTER_COMMAND(CCV_NNC_SCALAR_MUL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_mul_cpu_ref.c, gpu/ccv_nnc_mul_gpu_cudnn.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_scalar_mul_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALAR_MUL_FORWARD)
#define CMD_SCALAR_MUL_FORWARD(_a) ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_a,}}}, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALAR_MUL_BACKWARD)
#define CMD_SCALAR_MUL_BACKWARD(_a) ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_a,}}}, 0)
