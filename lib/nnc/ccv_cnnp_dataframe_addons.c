#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

static void _ccv_cnnp_array_enum(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	ccv_array_t* const array = (ccv_array_t*)context;
	for (i = 0; i < row_size; i++)
		data[i] = ccv_array_get(array, row_idxs[i]);
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_from_array_new(ccv_array_t* const array)
{
	const ccv_cnnp_column_data_t array_column_data = {
		.data_enum = _ccv_cnnp_array_enum,
		.context = array
	};
	return ccv_cnnp_dataframe_new(&array_column_data, 1, array->rnum);
}

typedef struct {
	int tensor_size;
	int device_id;
} ccv_cnnp_copy_to_gpu_context_t;

static void _ccv_cnnp_tensor_list_deinit(void* const data, void* const context)
{
	ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu = (ccv_cnnp_copy_to_gpu_context_t*)context;
	ccv_nnc_tensor_t** const tensor_list = (ccv_nnc_tensor_t**)data;
	int i;
	for (i = 0; i < copy_to_gpu->tensor_size; i++)
		if (tensor_list[i])
			ccv_nnc_tensor_free(tensor_list[i]);
	ccfree(tensor_list);
}

static void _ccv_cnnp_copy_to_gpu(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu_context = (ccv_cnnp_copy_to_gpu_context_t*)context;
	int i, j;
	for (i = 0; i < batch_size; i++)
	{
		ccv_nnc_tensor_t** inputs = (ccv_nnc_tensor_t**)column_data[0][i];
		ccv_nnc_tensor_t** outputs = (ccv_nnc_tensor_t**)data[i];
		if (!outputs)
		{
			outputs = (ccv_nnc_tensor_t**)(data[i] = ccmalloc(sizeof(ccv_nnc_tensor_t*) * copy_to_gpu_context->tensor_size));
			for (j = 0; j < copy_to_gpu_context->tensor_size; j++)
			{
				ccv_nnc_tensor_param_t params = inputs[j]->info;
				params.type &= ~CCV_TENSOR_CPU_MEMORY;
				params.type |= CCV_TENSOR_GPU_MEMORY; // Change to GPU memory.
				CCV_TENSOR_SET_DEVICE_ID(params.type, copy_to_gpu_context->device_id);
				outputs[j] = ccv_nnc_tensor_new(0, params, 0);
			}
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, copy_to_gpu_context->tensor_size, outputs, copy_to_gpu_context->tensor_size, stream_context);
	}
}

int ccv_cnnp_dataframe_copy_to_gpu(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const int tensor_size, int device_id)
{
	assert(tensor_size > 0);
	int stream_type = CCV_STREAM_CONTEXT_GPU;
	CCV_STREAM_SET_DEVICE_ID(stream_type, device_id);
	ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu_context = (ccv_cnnp_copy_to_gpu_context_t*)ccmalloc(sizeof(ccv_cnnp_copy_to_gpu_context_t));
	copy_to_gpu_context->tensor_size = tensor_size;
	copy_to_gpu_context->device_id = device_id;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_copy_to_gpu, stream_type, _ccv_cnnp_tensor_list_deinit, COLUMN_ID_LIST(column_idx), copy_to_gpu_context, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}

static void _ccv_cnnp_tensor_deinit(void* const data, void* const context)
{
	ccv_nnc_tensor_free((ccv_nnc_tensor_t*)data);
}

static void _ccv_cnnp_tensor_new(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_tensor_param_t params = *(ccv_nnc_tensor_param_t*)context;
	int i;
	for (i = 0; i < row_size; i++)
		if (!data[i])
			data[i] = ccv_nnc_tensor_new(0, params, 0);
}

int ccv_cnnp_dataframe_add_aux_tensors(ccv_cnnp_dataframe_t* const dataframe, const ccv_nnc_tensor_param_t params)
{
	int stream_type = CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY ? CCV_STREAM_CONTEXT_CPU : CCV_STREAM_CONTEXT_GPU;
	CCV_STREAM_SET_DEVICE_ID(stream_type, CCV_TENSOR_GET_DEVICE_ID(params.type));
	ccv_nnc_tensor_param_t* const context = (ccv_nnc_tensor_param_t*)ccmalloc(sizeof(ccv_nnc_tensor_param_t));
	context[0] = params;
	return ccv_cnnp_dataframe_add(dataframe, _ccv_cnnp_tensor_new, stream_type, _ccv_cnnp_tensor_deinit, context, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}
