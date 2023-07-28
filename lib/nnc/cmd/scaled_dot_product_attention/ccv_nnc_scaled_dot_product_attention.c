#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_scaled_dot_product_attention_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 5 inputs (query, key, value, [unify head weight], [unify head bias], [attn_mask])
	// 3 outputs (y, [softmax], [qkv])
	if ((input_bitmasks[0] & 7u) == 7u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}


static int _ccv_nnc_allow_query_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (input_idx == 0 && output_idx == 0)
		return 1;
	return 0;
}

static int _ccv_nnc_scaled_dot_product_attention_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 0b110000001100001
	// Inputs (gradient, 0, 0, 0, 0, x, scale, 0, 0, 0, 0, 0, 0, saved_mean, saved_inv_var)
	// Output (dquery, dkey, dvalue, dweight, dbias) [cannot diff against attn_mask]
	if ((input_bitmasks[0] & 24673u) == 24673u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	return 0;
}

static void _ccv_nnc_scaled_dot_product_attention_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	outputs[0] = inputs[0];
	if (output_size == 1)
		return;
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		for (j = 0; j < cmd.bnorm.count; j++)
			outputs[i].dim[cmd.bnorm.axis[j]] = 1; // Reduce the dimension to 1.
	}
}

static void _ccv_nnc_scaled_dot_product_attention_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 15);
	assert(output_size == 5);
	outputs[0] = inputs[0];
	int i, j;
	for (i = 1; i < output_size; i++)
	{
		outputs[i] = inputs[0];
		for (j = 0; j < cmd.bnorm.count; j++)
			outputs[i].dim[cmd.bnorm.axis[j]] = 1; // Reduce the dimension to 1.
	}
}

REGISTER_COMMAND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scaled_dot_product_attention_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_scaled_dot_product_attention_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_scaled_dot_product_attention_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_allow_query_inplace;
}

REGISTER_COMMAND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scaled_dot_product_attention_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_scaled_dot_product_attention_back_bitmask;
	registry->tensor_auto = _ccv_nnc_scaled_dot_product_attention_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD)
#define CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(_scale, _is_causal) ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scaled_dot_product_attention={.scale=_scale,.is_causal=_is_causal}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD)
#define CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(_scale, _is_causal) ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scaled_dot_product_attention={.scale=_scale,.is_causal=_is_causal}}), 0)
