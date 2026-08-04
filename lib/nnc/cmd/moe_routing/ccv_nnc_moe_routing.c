#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_moe_routing_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return input_size == 3 && output_size == 5 && (input_bitmasks[0] & 7u) == 7u && (output_bitmasks[0] & 31u) == 31u;
}

static int _ccv_nnc_moe_routing_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_moe_routing_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 3);
	assert(output_size == 5);
	const int kth = cmd.moe_routing.kth;
	assert(kth > 0);
	assert(cmd.moe_routing.weight_scale > 0);
	assert(cmd.moe_routing.preselected == 0 || cmd.moe_routing.preselected == 1);
	const int logits_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	const int route_nd = ccv_nnc_tensor_nd(inputs[1].dim);
	const int activation_nd = ccv_nnc_tensor_nd(inputs[2].dim);
	assert(logits_nd == 2);
	assert(activation_nd == 2);
	assert(inputs[0].datatype == CCV_32F || inputs[0].datatype == CCV_16F);
	assert(inputs[2].datatype == CCV_32F || inputs[2].datatype == CCV_16F || inputs[2].datatype == CCV_16BF);
	const int token_count = inputs[0].dim[0];
	const int expert_count = inputs[0].dim[1];
	assert(token_count > 0 && expert_count >= kth);
	assert(inputs[2].dim[0] == token_count);
	if (cmd.moe_routing.preselected)
	{
		assert(route_nd == 2);
		assert(inputs[1].datatype == CCV_32S);
		assert(inputs[1].dim[0] == token_count && inputs[1].dim[1] == kth);
	} else {
		assert(route_nd == 1);
		assert(inputs[1].datatype == CCV_32F || inputs[1].datatype == CCV_16F);
		assert(inputs[1].datatype == inputs[0].datatype);
		assert(inputs[1].dim[0] == expert_count);
	}
	const int pair_count = token_count * kth;
	const int single_input_token = token_count == 1 && (cmd.moe_routing.flags & CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN);
	outputs[0] = inputs[2];
	outputs[0].dim[0] = single_input_token ? 1 : pair_count;
	outputs[1] = inputs[0];
	outputs[1].datatype = CCV_32F;
	memset(outputs[1].dim, 0, sizeof(outputs[1].dim));
	outputs[1].dim[0] = pair_count;
	outputs[2] = outputs[1];
	outputs[2].datatype = CCV_32S;
	outputs[3] = outputs[2];
	outputs[3].dim[0] = ccv_min(pair_count, expert_count);
	outputs[4] = outputs[3];
}

REGISTER_COMMAND(CCV_NNC_MOE_ROUTING_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_moe_routing_cpu_ref.c, mps/ccv_nnc_moe_routing_mps.m)
{
	registry->bitmask = _ccv_nnc_moe_routing_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_moe_routing_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_MOE_ROUTING_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
{
	registry->bitmask = _ccv_nnc_moe_routing_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MOE_ROUTING_FORWARD)
#define CMD_MOE_ROUTING_FORWARD(_kth, _weight_scale, _preselected) ccv_nnc_cmd(CCV_NNC_MOE_ROUTING_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.moe_routing={.kth=(_kth),.weight_scale=(_weight_scale),.preselected=(_preselected)}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MOE_ROUTING_FORWARD)
#define CMD_MOE_ROUTING_FORWARD_FLAGS(_kth, _weight_scale, _preselected, _flags) ccv_nnc_cmd(CCV_NNC_MOE_ROUTING_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.moe_routing={.kth=(_kth),.weight_scale=(_weight_scale),.preselected=(_preselected),.flags=(_flags)}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MOE_ROUTING_BACKWARD)
#define CMD_MOE_ROUTING_BACKWARD(_kth, _weight_scale, _preselected) ccv_nnc_cmd(CCV_NNC_MOE_ROUTING_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.moe_routing={.kth=(_kth),.weight_scale=(_weight_scale),.preselected=(_preselected)}}), 0)
