#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_scaled_dot_product_attention_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 6 inputs (query, key, value, [attn_mask], [unify head weight], [unify head bias])
	// 3 outputs (y, [softmax], [qkv])
	if (input_size == 6 && (input_bitmasks[0] & 55u) == 55u && (output_bitmasks[0] & 5u) == 5u)
		return 1;
	if (input_size == 5 && (input_bitmasks[0] & 23u) == 23u && (output_bitmasks[0] & 5u) == 5u)
		return 1;
	if ((input_bitmasks[0] & 55u) == 7u && (output_bitmasks[0] & 5u) == 1u)
		return 1;
	return 0;
}


static int _ccv_nnc_allow_query_inplace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (input_idx == 0 && output_idx == 0)
		return 1;
	return 0;
}

static int _ccv_nnc_scaled_dot_product_attention_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// 1, 0, 0, 8, 16, 32, 64?, 128?, 256?, 512, 1024, 2048?
	// 1, 2, 4, 8, 16, 32
	// Inputs (gradient, 0, 0, q, k, v, [attn_mask], [head weight], [bias], y, saved softmax, qkv)
	// Output (dquery, dkey, dvalue, [attn mask], dweight, dbias) [cannot diff against attn_mask]
	if ((input_bitmasks[0] & 4025u) == 4025u && (output_bitmasks[0] & 63u) == 55u)
		return 1;
	if ((input_bitmasks[0] & 3769u) == 3769u && (output_bitmasks[0] & 31u) == 23u)
		return 1;
	if ((input_bitmasks[0] & 1593u) == 1593u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	return 0;
}

static void _ccv_nnc_scaled_dot_product_attention_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const int q_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	assert(q_nd == 3 || q_nd == 4);
	const int k_nd = ccv_nnc_tensor_nd(inputs[1].dim);
	assert(k_nd == 3 || k_nd == 4);
	const int v_nd = ccv_nnc_tensor_nd(inputs[2].dim);
	assert(v_nd == 3 || v_nd == 4);
	assert(q_nd == k_nd && k_nd == v_nd);
	if (input_size > 4)
	{
		assert(output_size >= 3);
		outputs[0] = inputs[0];
		outputs[0].dim[1] = inputs[0].dim[1]; // sequence length matches query, embedding size matches value * num_head.
		outputs[0].dim[2] = inputs[2].dim[v_nd - 1] * (q_nd == 4 ? inputs[0].dim[2] : 1);
		outputs[0].dim[3] = 0;
		outputs[1] = inputs[0];
		if (q_nd == 4)
		{
			// Switch head v.s. sequence length.
			outputs[1].dim[1] = outputs[1].dim[2];
			outputs[1].dim[2] = inputs[0].dim[1];
		}
		outputs[1].dim[q_nd - 1] = inputs[1].dim[k_nd - 3]; // saved softmax should have sequence length of query x key.
		outputs[2] = inputs[0];
		outputs[2].dim[q_nd - 1] = inputs[2].dim[v_nd - 1]; // sequence length matches query, embedding size matches value.
	} else {
		outputs[0] = inputs[0];
		outputs[0].dim[q_nd - 1] = inputs[2].dim[v_nd - 1]; // sequence length matches query, embedding size matches value.
		if (output_size == 1)
			return;
		assert(output_size > 1);
		outputs[1] = inputs[0];
		if (q_nd == 4)
		{
			// Switch head v.s. sequence length.
			outputs[1].dim[1] = outputs[1].dim[2];
			outputs[1].dim[2] = inputs[0].dim[1];
		}
		outputs[1].dim[q_nd - 1] = inputs[1].dim[k_nd - 3]; // saved softmax should have sequence length of query x key.
	}
}

static void _ccv_nnc_scaled_dot_product_attention_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 6);
	assert(output_size >= 3);
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[3 + i];
}

REGISTER_COMMAND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scaled_dot_product_attention_cpu_ref.c, mps/ccv_nnc_scaled_dot_product_attention_mps.m)
{
	registry->bitmask = _ccv_nnc_scaled_dot_product_attention_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_scaled_dot_product_attention_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_allow_query_inplace;
}

REGISTER_COMMAND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_scaled_dot_product_attention_cpu_ref.c, mps/ccv_nnc_scaled_dot_product_attention_mps.m)
{
	registry->bitmask = _ccv_nnc_scaled_dot_product_attention_back_bitmask;
	registry->tensor_auto = _ccv_nnc_scaled_dot_product_attention_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD)
#define CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(_scale, _is_causal) ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scaled_dot_product_attention={.scale=_scale,.is_causal=_is_causal}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD)
#define CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(_scale, _is_causal) ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.scaled_dot_product_attention={.scale=_scale,.is_causal=_is_causal}}), 0)
