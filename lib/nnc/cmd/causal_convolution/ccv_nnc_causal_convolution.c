#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_causal_convolution_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Inputs: x [B, T, C], the previous rows [B, kernel_size - 1, C] and the depthwise weights
	// [C, kernel_size]. Outputs: y [B, T, C] and the next rows [B, kernel_size - 1 + state_checkpoint_count, C].
	if (input_size == 3 && output_size == 2 && (input_bitmasks[0] & 7u) == 7u && (output_bitmasks[0] & 3u) == 3u)
		return 1;
	return 0;
}

static int _ccv_nnc_causal_convolution_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_causal_convolution_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3);
	assert(output_size == 2);
	const int x_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	const int channels = inputs[0].dim[x_nd - 1];
	const int kernel_size = ccv_nnc_tensor_count(inputs[2]) / channels;
	assert(inputs[1].dim[1] == kernel_size - 1);
	outputs[0] = inputs[0];
	outputs[1] = inputs[1];
	outputs[1].dim[1] = kernel_size - 1 + cmd.causal_convolution.state_checkpoint_count;
}

REGISTER_COMMAND(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_causal_convolution_cpu_ref.c, mps/ccv_nnc_causal_convolution_mps.m)
{
	registry->bitmask = _ccv_nnc_causal_convolution_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_causal_convolution_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_CAUSAL_CONVOLUTION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_causal_convolution_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_causal_convolution_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD)
#define CMD_CAUSAL_CONVOLUTION_FORWARD(_silu, _state_checkpoint_count) ccv_nnc_cmd(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.causal_convolution={.silu=(_silu),.state_checkpoint_count=(_state_checkpoint_count)}}), 0)
