#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

#include "ccv_nnc_cmd.h"

// nvcc is a C++ compiler, need to specify this is a "C" function to avoid name mangling.
extern "C" void ccv_nnc_gpu_ref_init(ccv_nnc_cmd_api_t cmd_api[]);

static int _ccv_nnc_data_move(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
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
		if (stream_context)
		{
			int device = ccv_nnc_stream_context_get_device(stream_context);
			cudaSetDevice(device);
			cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
			if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY)
				cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyHostToDevice, stream);
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToHost, stream);
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyHostToHost, stream);
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY) {
				int device_a = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
				int device_b = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
				if (device_a == device_b)
					cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToDevice, stream);
				else
					cudaMemcpyPeerAsync(b->data.u8, device_b, a->data.u8, device_a, size, stream);
			}
		} else {
			if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY)
				cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyHostToDevice);
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToHost);
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyHostToHost);
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY) {
				int device_a = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
				int device_b = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
				if (device_a == device_b)
					cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToDevice);
				else
					cudaMemcpyPeer(b->data.u8, device_b, a->data.u8, device_a, size);
			}
		}
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
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].exec = _ccv_nnc_data_move;
}
