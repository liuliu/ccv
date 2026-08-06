#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_scaled_dot_product_arg_partition_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Inputs: q, k, headW. Output: selected row ids.
	if (input_size == 3 && output_size == 1 && (input_bitmasks[0] & 7u) == 7u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_scaled_dot_product_arg_partition_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_scaled_dot_product_arg_partition_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3);
	assert(output_size == 1);
	assert(cmd.scaled_dot_product_arg_partition.kth > 0);
	const int q_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	assert(q_nd == 3);
	outputs[0] = inputs[0];
	memset(outputs[0].dim, 0, sizeof(outputs[0].dim));
	outputs[0].dim[0] = inputs[0].dim[0];
	outputs[0].dim[1] = cmd.scaled_dot_product_arg_partition.kth;
	outputs[0].datatype = CCV_32S;
}

static void _ccv_nnc_scaled_dot_product_arg_partition_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[1 + i];
}

REGISTER_COMMAND(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scaled_dot_product_arg_partition_cpu_ref.c, mps/ccv_nnc_scaled_dot_product_arg_partition_mps.m)
{
	registry->bitmask = _ccv_nnc_scaled_dot_product_arg_partition_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_scaled_dot_product_arg_partition_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scaled_dot_product_arg_partition_cpu_ref.c, mps/ccv_nnc_scaled_dot_product_arg_partition_mps.m)
{
	registry->bitmask = _ccv_nnc_scaled_dot_product_arg_partition_back_bitmask;
	registry->tensor_auto = _ccv_nnc_scaled_dot_product_arg_partition_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD)
#define CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(_kth, _scale, _is_causal, _compression_ratio, _query_offset) ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scaled_dot_product_arg_partition={.scale=_scale,.kth=_kth,.is_causal=_is_causal,.compression_ratio=_compression_ratio,.query_offset=_query_offset}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD)
#define CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD(_kth, _scale, _is_causal, _compression_ratio, _query_offset) ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scaled_dot_product_arg_partition={.scale=_scale,.kth=_kth,.is_causal=_is_causal,.compression_ratio=_compression_ratio,.query_offset=_query_offset}}), 0)
