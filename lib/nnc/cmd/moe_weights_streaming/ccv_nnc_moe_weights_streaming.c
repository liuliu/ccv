#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_moe_weights_streaming_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return input_size == 6 && output_size == 6 && input_bitmask_size > 0 && output_bitmask_size > 0 &&
		(input_bitmasks[0] & 63u) == 63u && (output_bitmasks[0] & 63u) == 63u;
}

static int _ccv_nnc_moe_weights_streaming_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_moe_weights_streaming_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 6);
	assert(output_size == 6);
	assert(cmd.moe_weights_streaming.resident_slots > 0);
	assert(cmd.moe_weights_streaming.routing_width > 0);
	outputs[0] = inputs[0];
	outputs[1] = inputs[1];
	outputs[2] = inputs[2];
	int i;
	for (i = 3; i < 6; i++)
	{
		outputs[i] = inputs[i];
		const int base_datatype = CCV_GET_DATA_TYPE(inputs[i].datatype) == CCV_QX ?
			(inputs[i].datatype & 0xff) : ((inputs[i].datatype >> 12) & 0xff);
		outputs[i].datatype = CCV_QX | CCV_NNC_QX_EPHERMAL_STAGING | base_datatype;
	}
}

REGISTER_COMMAND(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(mps/ccv_nnc_moe_weights_streaming_mps.m)
{
	registry->bitmask = _ccv_nnc_moe_weights_streaming_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_moe_weights_streaming_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_MOE_WEIGHTS_STREAMING_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
{
	registry->bitmask = _ccv_nnc_moe_weights_streaming_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD)
#define CMD_MOE_WEIGHTS_STREAMING_FORWARD(_resident_slots, _routing_width) ccv_nnc_cmd(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.moe_weights_streaming={.resident_slots=(_resident_slots),.routing_width=(_routing_width)}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_MOE_WEIGHTS_STREAMING_BACKWARD)
#define CMD_MOE_WEIGHTS_STREAMING_BACKWARD(_resident_slots, _routing_width) ccv_nnc_cmd(CCV_NNC_MOE_WEIGHTS_STREAMING_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.moe_weights_streaming={.resident_slots=(_resident_slots),.routing_width=(_routing_width)}}), 0)
