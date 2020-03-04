#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_categorical_crossentropy_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_categorical_crossentropy_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 7u) == 7u && (output_bitmasks[0] & 1u) == 1u)
		return 1;
	return 0;
}

static void _ccv_nnc_categorical_crossentropy_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size == 2);
	assert(output_size == 1);
	outputs[0] = inputs[0];
	// The output should have the same dimentionality of the label data.
	memcpy(outputs[0].dim, inputs[1].dim, sizeof(outputs[0].dim));
	const int nd = ccv_nnc_tensor_nd(outputs[0].dim);
	// Set channel to 1 if it is not..
	if (nd > 1 && ccv_nnc_tensor_get_c(outputs[0]) > 1)
		ccv_nnc_tensor_set_c(&outputs[0], nd, 1);
}

static void _ccv_nnc_categorical_crossentropy_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	assert(input_size >= 2);
	assert(output_size >= 1);
	outputs[0] = inputs[1];
	if (output_size > 1)
		outputs[1] = inputs[2];
}

REGISTER_COMMAND(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_categorical_crossentropy_cpu_ref.c, gpu/ccv_nnc_categorical_crossentropy_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_categorical_crossentropy_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_categorical_crossentropy_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_categorical_crossentropy_cpu_ref.c, gpu/ccv_nnc_categorical_crossentropy_gpu_ref.cu)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_categorical_crossentropy_back_bitmask;
	registry->tensor_auto = _ccv_nnc_categorical_crossentropy_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD)
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_0() ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=0,.trim1=1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_CATEGORICAL_CROSSENTROPY_FORWARD")
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_2(_trim0, _trim1) ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=_trim0,.trim1=_trim1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD(...) CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_SEL(CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_F, ##__VA_ARGS__, CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_2, CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_F, CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_0)(__VA_ARGS__)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD)
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_0() ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=0,.trim1=1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_CATEGORICAL_CROSSENTROPY_BACKWARD")
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_2(_trim0, _trim1) ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=_trim0,.trim1=_trim1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD(...) CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_SEL(CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_F, ##__VA_ARGS__, CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_2, CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_F, CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_0)(__VA_ARGS__)
