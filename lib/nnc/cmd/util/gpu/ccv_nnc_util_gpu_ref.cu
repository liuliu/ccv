extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

static int _ccv_nnc_data_transfer(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	for (i = 0; i < output_size; i++)
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

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}
