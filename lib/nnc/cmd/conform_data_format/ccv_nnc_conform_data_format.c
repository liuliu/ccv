#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"

static int _ccv_nnc_conform_data_format_allow_first_replace(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	return input_idx == 0 && output_idx == 0;
}

static int _ccv_nnc_conform_data_format_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return input_bitmask_size > 0 && output_bitmask_size > 0 && (input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u;
}

REGISTER_COMMAND(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_conform_data_format_cpu_ref.c, mps/ccv_nnc_conform_data_format_mps.m)
{
	registry->bitmask = _ccv_nnc_conform_data_format_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
	registry->allow_inplace = _ccv_nnc_conform_data_format_allow_first_replace;
}

REGISTER_COMMAND(CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_conform_data_format_cpu_ref.c, mps/ccv_nnc_conform_data_format_mps.m)
{
	registry->bitmask = _ccv_nnc_conform_data_format_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
	registry->allow_inplace = _ccv_nnc_conform_data_format_allow_first_replace;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD)
#define CMD_CONFORM_DATA_FORMAT_FORWARD(_datatype, _preserved_tail) ccv_nnc_cmd(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.conform_data_format={.datatype=(_datatype),.preserved_tail=(_preserved_tail)}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD)
#define CMD_CONFORM_DATA_FORMAT_BACKWARD(_datatype, _preserved_tail) ccv_nnc_cmd(CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.conform_data_format={.datatype=(_datatype),.preserved_tail=(_preserved_tail)}}), 0)
