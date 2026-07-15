#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_sparse_indexed_attention_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Inputs: q, dense_k, dense_v, sparse_k, sparse_v, indices, optional sinks. Output: heads.
	const uint64_t required = 63u;
	if (output_size == 1 && (output_bitmasks[0] & 1u) == 1u)
	{
		if (!cmd.sparse_indexed_attention.attention_sinks && input_size == 6 && (input_bitmasks[0] & required) == required)
			return 1;
		if (cmd.sparse_indexed_attention.attention_sinks && input_size == 7 && (input_bitmasks[0] & (required | 64u)) == (required | 64u))
			return 1;
	}
	return 0;
}

static int _ccv_nnc_sparse_indexed_attention_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_sparse_indexed_attention_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 6 || input_size == 7);
	assert(output_size == 1);
	outputs[0] = inputs[0];
}

static void _ccv_nnc_sparse_indexed_attention_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[1 + i];
}

REGISTER_COMMAND(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_sparse_indexed_attention_cpu_ref.c, mps/ccv_nnc_sparse_indexed_attention_mps.m)
{
	registry->bitmask = _ccv_nnc_sparse_indexed_attention_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_sparse_indexed_attention_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_sparse_indexed_attention_cpu_ref.c, mps/ccv_nnc_sparse_indexed_attention_mps.m)
{
	registry->bitmask = _ccv_nnc_sparse_indexed_attention_back_bitmask;
	registry->tensor_auto = _ccv_nnc_sparse_indexed_attention_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD)
#define CMD_SPARSE_INDEXED_ATTENTION_FORWARD(_scale, _is_causal, _attention_sinks) ccv_nnc_cmd(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.sparse_indexed_attention={.scale=_scale,.is_causal=_is_causal,.attention_sinks=_attention_sinks,.sliding_window=0}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD)
#define CMD_SPARSE_INDEXED_ATTENTION_BACKWARD(_scale, _is_causal, _attention_sinks) ccv_nnc_cmd(CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.sparse_indexed_attention={.scale=_scale,.is_causal=_is_causal,.attention_sinks=_attention_sinks,.sliding_window=0}}), 0)
