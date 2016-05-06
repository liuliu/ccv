/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_easy_h
#define GUARD_ccv_nnc_easy_h

#include <ccv.h>
#include <nnc/ccv_nnc.h>

/**
 * Convenience API
 *
 * This header provides convenience APIs for nnc usage. Being convenience API,
 * it is optimized for shorthand coding, and may collide the naming space with
 * others.
 */
// c99 only, make sure your compiler supports that.
#define TENSOR_LIST_X(...) (ccv_nnc_tensor_t* []){__VA_ARGS__}
#define TENSOR_LIST_COUNT_N(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) (_63)
#define TENSOR_LIST_COUNT(...) TENSOR_LIST_COUNT_N(__VA_ARGS__,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
#define TENSOR_LIST(...) TENSOR_LIST_X(__VA_ARGS__), TENSOR_LIST_COUNT(__VA_ARGS__)
#define ONE_CPU_TENSOR(...) ((ccv_nnc_tensor_param_t){.type=CCV_TENSOR_CPU_MEMORY,.format=CCV_TENSOR_FORMAT_NHWC,.dim={__VA_ARGS__}})
// This way, we can do error check on the device type :)
#define ONE_GPU_TENSOR(device_id, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_GPU_MEMORY,.format=CCV_TENSOR_FORMAT_NHWC,.dim={__VA_ARGS__}})

static inline size_t ccv_nnc_tensor_count(const ccv_nnc_tensor_param_t params)
{
	if (params.dim[0] == 0)
		return 0;
	int i;
	size_t count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && params.dim[i] > 0; i++)
		count *= params.dim[i];
	return count;
}

static inline int ccv_nnc_tensor_nd(const ccv_nnc_tensor_param_t params)
{
	int nd = 0;
	while (nd < CCV_NNC_MAX_DIM_ALLOC && params.dim[nd] > 0)
		++nd;
	return nd;
}

#define CMD_CONVOLUTIONAL(_count, ...) ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolutional={.count=_count}})
#define CMD_GENERIC(...) ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}}})
#define CMD_FULL_CONNECT(_count) ((ccv_nnc_cmd_param_t){.full_connect={.count=_count}})

extern const ccv_nnc_hint_t ccv_nnc_default_hint;
extern const ccv_nnc_cmd_param_t ccv_nnc_default_cmd_params;

#endif
