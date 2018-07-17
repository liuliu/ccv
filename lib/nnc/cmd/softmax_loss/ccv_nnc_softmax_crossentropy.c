#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_softmax_crossentropy_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: activation, label
	// output: [loss], softmax
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 3u)
		return 1;
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 2u)
		return 1;
	return 0;
}

static int _ccv_nnc_softmax_crossentropy_allow_inplace_forw(const int input_idx, const int output_idx)
{
	return (input_idx == 0 && output_idx == 1);
}

static int _ccv_nnc_softmax_crossentropy_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input: [gradient of loss], [gradient of softmax], [activation], label, [loss], softmax
	// output: w.r.t activation, [label]
	if ((input_bitmasks[0] & 41u) == 41u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_softmax_crossentropy_allow_inplace_back(const int input_idx, const int output_idx)
{
	if (input_idx == 1 && output_idx == 0)
		return 1;
	if (input_idx == 2 && output_idx == 0)
		return 1;
	else if (input_idx == 5 && output_idx == 0)
		return 1;
	return 0;
}

static void _ccv_nnc_softmax_crossentropy_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size >= 1);
	outputs[0] = inputs[0];
	if (output_size > 1)
	{
		outputs[1] = inputs[0];
		// The output should have the same dimentionality of the label data.
		memcpy(outputs[1].dim, inputs[1].dim, sizeof(outputs[1].dim));
	}
}

static void _ccv_nnc_softmax_crossentropy_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 6);
	assert(output_size >= 1);
	outputs[0] = inputs[5];
	if (output_size > 1)
		outputs[1] = inputs[3];
}

REGISTER_COMMAND(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_softmax_crossentropy_cpu_ref.c, gpu/ccv_nnc_softmax_crossentropy_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_softmax_crossentropy_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_softmax_crossentropy_tensor_auto_forw;
	registry->allow_inplace = _ccv_nnc_softmax_crossentropy_allow_inplace_forw;
}

REGISTER_COMMAND(CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_softmax_crossentropy_cpu_ref.c, gpu/ccv_nnc_softmax_crossentropy_gpu_cudnn.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_softmax_crossentropy_back_bitmask;
	registry->tensor_auto = _ccv_nnc_softmax_crossentropy_tensor_auto_back;
	registry->allow_inplace = _ccv_nnc_softmax_crossentropy_allow_inplace_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD)
#define CMD_SOFTMAX_CROSSENTROPY_FORWARD() ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD, 0, ccv_nnc_cmd_auto, 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD)
#define CMD_SOFTMAX_CROSSENTROPY_BACKWARD() ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
