#include "ccv_nnc.h"

ccv_nnc_tensor_t* ccv_nnc_tensor_new(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	ccv_nnc_tensor_t* tensor;
	// this specific form can be toll-free bridging to ccv_dense_matrix_t
	int tfb = (params.dim[0] > 0 && params.dim[0] <= CCV_MAX_CHANNEL && params.dim[1] > 0 && params.dim[2] > 0 && params.dim[3] == 0);
	if (ptr)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		tensor->sig = 0;
		tensor->refcount = 1;
		tensor->info = params;
		if (tfb)
		{
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_32F | params.dim[0];
			// This corresponding to mat->step
			tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_32F | params.dim[0]));
		} else // This won't be recognized by ccv_dense_matrix_t
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_32F;
		tensor->data.u8 = (uint8_t*)ptr;
		return tensor;
	}
	assert((flags & CCV_TENSOR_CPU_MEMORY) || (flags == 0));
	size_t size = CCV_GET_DATA_TYPE_SIZE(CCV_32F); // Assuming 32-bit float point layout
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (!params.dim[i])
			break;
		size *= params.dim[i];
	}
	tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t) + size);
	tensor->sig = 0;
	tensor->refcount = 1;
	tensor->info = params;
	if (tfb)
	{
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F | params.dim[0];
		// This corresponding to mat->step
		tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_32F | params.dim[0]));
	} else
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F;
	tensor->data.u8 = (uint8_t*)(tensor + 1);
	return tensor;
}

ccv_nnc_tensor_t ccv_nnc_tensor(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	// this specific form can be toll-free bridging to ccv_dense_matrix_t
	int tfb = (params.dim[0] > 0 && params.dim[0] <= CCV_MAX_CHANNEL && params.dim[1] > 0 && params.dim[2] > 0 && params.dim[3] == 0);
	assert(ptr);
	ccv_nnc_tensor_t tensor;
	tensor.sig = 0;
	tensor.refcount = 1;
	tensor.info = params;
	if (tfb)
	{
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F | params.dim[0];
		// This corresponding to mat->step
		tensor.info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_32F | params.dim[0]));
	} else // This won't be recognized by ccv_dense_matrix_t
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F;
	tensor.data.u8 = (uint8_t*)ptr;
	return tensor;
}

void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor)
{
	ccfree(tensor);
}
