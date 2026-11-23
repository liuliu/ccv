#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_attention_residual_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Inputs: the block states [n, ..., C], the partial state [..., C] and the queries [Q, C].
	// Outputs: one mixture [..., C] for each query.
	if (input_size != 3 || (input_bitmasks[0] & 7u) != 7u || output_size < 1 || output_size > 64)
		return 0;
	const uint64_t output_bitmask = output_size == 64 ? ~(uint64_t)0 : ((uint64_t)1 << output_size) - 1;
	return (output_bitmasks[0] & output_bitmask) == output_bitmask;
}

static int _ccv_nnc_attention_residual_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_attention_residual_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3);
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[1];
}

REGISTER_COMMAND(CCV_NNC_ATTENTION_RESIDUAL_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_attention_residual_cpu_ref.c, mps/ccv_nnc_attention_residual_mps.m)
{
	registry->bitmask = _ccv_nnc_attention_residual_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_attention_residual_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_ATTENTION_RESIDUAL_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_attention_residual_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_attention_residual_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_ATTENTION_RESIDUAL_FORWARD)
#define CMD_ATTENTION_RESIDUAL_FORWARD(_epsilon, _scale) ccv_nnc_cmd(CCV_NNC_ATTENTION_RESIDUAL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.attention_residual={.epsilon=(_epsilon),.scale=(_scale)}}), 0)
