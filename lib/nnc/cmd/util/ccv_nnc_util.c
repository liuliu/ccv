#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_set_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	if (output_bitmask == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_SET_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_set_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_SET_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_set_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

static int _ccv_nnc_data_transfer_bitmask(const uint64_t input_bitmask, const uint64_t output_bitmask)
{
	int i;
	for (i = 1; i < 64; i++)
		if (!(input_bitmask & (uint64_t)1 << i))
			break;
	const int input_bitcount = i;
	// Always like 1111100000, no 1110010101
	for (; i < 64; i++)
		if (input_bitmask & (uint64_t)1 << i)
			return 0;
	for (i = 0; i < 64; i++)
		if (!(output_bitmask & (uint64_t)1 << i))
			break;
	const int output_bitcount = i;
	for (; i < 64; i++)
		if (output_bitmask & (uint64_t)1 << i)
			return 0;
	return output_bitcount == input_bitcount;
}

REGISTER_COMMAND(CCV_NNC_DATA_TRANSFER_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_DATA_TRANSFER_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_ref.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_FORMAT_TRANSFORM_FORWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_FORMAT_TRANSFORM_BACKWARD)(ccv_nnc_cmd_registry_t* registry)
	FIND_BACKEND(ccv_nnc_util_cpu_ref.c, ccv_nnc_util_gpu_cudnn.cu)
{
	registry->bitmask = _ccv_nnc_data_transfer_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}
