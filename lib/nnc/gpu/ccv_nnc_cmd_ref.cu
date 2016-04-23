#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

// nvcc is a C++ compiler, need to specify this is a "C" function to avoid name mangling.
extern "C" void ccv_nnc_gpu_ref_init(ccv_nnc_cmd_api_t cmd_api[]);

// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define set_n_m_dim(x, wd, ad) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1], 0) - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1]); \
		m[x] = wd[x + 1] - n[x] - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1] + wd[x + 1] - ccv_min(ad[x + 1], i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1] + wd[x + 1])); \
	} while (0)

static int _ccv_nnc_data_move(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_unit_t* stream_unit)
{
	assert(input_size == output_size);
	int i;
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_t* a = inputs[i];
		assert(!CCV_IS_TENSOR_VIEW(a));
		ccv_nnc_tensor_t* b = outputs[i];
		assert(!CCV_IS_TENSOR_VIEW(b));
		assert(ccv_nnc_tensor_count(a->info) == ccv_nnc_tensor_count(b->info));
		// Assume it is 32f.
		assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
		assert(CCV_GET_DATA_TYPE(b->type) == CCV_32F);
		size_t size = ccv_nnc_tensor_count(a->info) * sizeof(float);
		if (a->info.type == CCV_TENSOR_CPU_MEMORY &&
			b->info.type == CCV_TENSOR_GPU_MEMORY)
			cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyHostToDevice);
		else if (a->info.type == CCV_TENSOR_GPU_MEMORY &&
				 b->info.type == CCV_TENSOR_CPU_MEMORY)
			cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToHost);
		else if (a->info.type == CCV_TENSOR_CPU_MEMORY &&
				 b->info.type == CCV_TENSOR_CPU_MEMORY)
			cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyHostToHost);
		else if (a->info.type == CCV_TENSOR_GPU_MEMORY &&
				 b->info.type == CCV_TENSOR_GPU_MEMORY)
			cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToDevice);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

//@ccv_nnc_init CCV_NNC_BACKEND_GPU_REF
void ccv_nnc_gpu_ref_init(ccv_nnc_cmd_api_t cmd_api[])
{
	/* Convolutional layer */
	/* Full connect layer */
	/* Max pool layer */
	/* Average pool layer */
	/* Softmax layer */
	/* ReLU activation */
	/* Data transfer */
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER].tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER].exec = _ccv_nnc_data_move;
}
