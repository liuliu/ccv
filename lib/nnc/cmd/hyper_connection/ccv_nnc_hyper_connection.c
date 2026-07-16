#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"
#include "ccv_nnc_hyper_connection_internal.h"

static int _ccv_nnc_hyper_connection_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (input_size == 3 && output_size == 3 && (input_bitmasks[0] & 7u) == 7u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	if (input_size == 4 && output_size == 3 && (input_bitmasks[0] & 15u) == 15u && (output_bitmasks[0] & 7u) == 7u)
		return 1;
	if (input_size == 4 && output_size == 1 && (input_bitmasks[0] & 15u) == 15u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_hyper_connection_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_hyper_connection_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	if (!((input_size == 3 && output_size == 3) || (input_size == 4 && (output_size == 1 || output_size == 3))))
	{
		assert(0 && "invalid hyper connection input / output arity");
		return;
	}
	const int hc = cmd.hyper_connection.count;
	if (hc <= 0 || hc > 16)
	{
		assert(0 && "hyper connection count must be between 1 and 16");
		return;
	}
	if (output_size == 1)
	{
		outputs[0] = inputs[1];
		assert(_ccv_nnc_hyper_connection_expand_shapes_are_valid(hc, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0]));
		return;
	}
	const int mix_nd = ccv_nnc_tensor_nd(inputs[0].dim);
	if (mix_nd <= 0)
	{
		assert(0 && "hyper connection mix tensor must have a control dimension");
		return;
	}
	outputs[0] = outputs[1] = outputs[2] = inputs[0];
	if (input_size == 3)
	{
		outputs[0].dim[mix_nd - 1] = hc;
		outputs[1].dim[mix_nd - 1] = hc;
		outputs[2].dim[mix_nd - 1] = hc * hc;
	} else {
		outputs[0].dim[mix_nd - 1] = hc;
		outputs[1].dim[mix_nd - 1] = hc * hc;
		outputs[2] = inputs[3];
		const int nd = ccv_nnc_tensor_nd(outputs[2].dim);
		if (nd < 2 || outputs[2].dim[nd - 2] != hc)
		{
			assert(0 && "hyper connection residual tensor must end in [count, hidden]");
			return;
		}
		int i;
		for (i = nd - 2; i < CCV_NNC_MAX_DIM_ALLOC - 1; i++)
			outputs[2].dim[i] = outputs[2].dim[i + 1];
		outputs[2].dim[CCV_NNC_MAX_DIM_ALLOC - 1] = 0;
	}
	const ccv_nnc_tensor_param_t* const residual = input_size == 4 ? &inputs[3] : 0;
	const ccv_nnc_tensor_param_t* const pre = input_size == 3 ? &outputs[0] : 0;
	const ccv_nnc_tensor_param_t* const weighted = input_size == 4 ? &outputs[2] : 0;
	assert(_ccv_nnc_hyper_connection_split_shapes_are_valid(hc, inputs[0], inputs[1], inputs[2], residual, pre, outputs[input_size == 3 ? 1 : 0], outputs[input_size == 3 ? 2 : 1], weighted));
}

REGISTER_COMMAND(CCV_NNC_HYPER_CONNECTION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_hyper_connection_cpu_ref.c, mps/ccv_nnc_hyper_connection_mps.m)
{
	registry->bitmask = _ccv_nnc_hyper_connection_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_hyper_connection_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_HYPER_CONNECTION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_hyper_connection_cpu_ref.c, mps/ccv_nnc_hyper_connection_mps.m)
{
	registry->bitmask = _ccv_nnc_hyper_connection_back_bitmask;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_HYPER_CONNECTION_FORWARD)
#define CMD_HYPER_CONNECTION_FORWARD(_count, _sinkhorn_iterations, _epsilon) ccv_nnc_cmd(CCV_NNC_HYPER_CONNECTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.hyper_connection={.count=(_count),.sinkhorn_iterations=(_sinkhorn_iterations),.epsilon=(_epsilon)}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_HYPER_CONNECTION_BACKWARD)
#define CMD_HYPER_CONNECTION_BACKWARD(_count, _sinkhorn_iterations, _epsilon) ccv_nnc_cmd(CCV_NNC_HYPER_CONNECTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.hyper_connection={.count=(_count),.sinkhorn_iterations=(_sinkhorn_iterations),.epsilon=(_epsilon)}}), 0)
