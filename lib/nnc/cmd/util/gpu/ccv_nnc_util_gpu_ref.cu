extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

static int _ccv_nnc_data_transfer(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	for (i = 0; i < ccv_min(input_size, output_size); i++)
	{
		const ccv_nnc_tensor_t* a = inputs[i];
		ccv_nnc_tensor_t* b = outputs[i];
		if (a == b)
			continue;
		assert(CCV_IS_TENSOR_CONTIGUOUS(a));
		assert(CCV_IS_TENSOR_CONTIGUOUS(b));
		assert(ccv_nnc_tensor_count(a->info) == ccv_nnc_tensor_count(b->info));
		if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
			{ assert(a->info.datatype == b->info.datatype); }
		else
			{ assert(CCV_GET_DATA_TYPE_SIZE(a->info.datatype) == CCV_GET_DATA_TYPE_SIZE(b->info.datatype)); }
		const size_t size = ccv_nnc_tensor_data_size_without_padding(a->info);
		if (stream_context)
		{
			cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
			if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY)
				CUDA_ENFORCE(cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyHostToDevice, stream));
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				CUDA_ENFORCE(cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToHost, stream));
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				CUDA_ENFORCE(cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyHostToHost, stream));
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY) {
				const int device_a = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
				const int device_b = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
				if (device_a == device_b)
					CUDA_ENFORCE(cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToDevice, stream));
				else
					CUDA_ENFORCE(cudaMemcpyPeerAsync(b->data.u8, device_b, a->data.u8, device_a, size, stream));
			}
		} else {
			if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY)
				CUDA_ENFORCE(cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyHostToDevice));
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				CUDA_ENFORCE(cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToHost));
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
				CUDA_ENFORCE(cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyHostToHost));
			else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY) {
				const int device_a = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
				const int device_b = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
				if (device_a == device_b)
					CUDA_ENFORCE(cudaMemcpy(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToDevice));
				else
					CUDA_ENFORCE(cudaMemcpyPeer(b->data.u8, device_b, a->data.u8, device_a, size));
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_64S | CCV_32S | CCV_8U | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_64S | CCV_32S | CCV_8U | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_data_conversion_kernel(const size_t count, const NUM1* const a, NUM2* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = a[i];
	}
}

static int _ccv_nnc_datatype_conversion(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
		assert(a != b); // Cannot do inplace transform.
		assert(a->info.format == b->info.format);
		assert(CCV_TENSOR_GET_DEVICE_ID(a->info.type) == CCV_TENSOR_GET_DEVICE_ID(b->info.type));
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		const size_t tensor_count = ccv_nnc_tensor_count(a->info);
		if (a->info.datatype == b->info.datatype) {
			// If it is the same, just do a normal data transfer.
			const size_t size = tensor_count * CCV_GET_DATA_TYPE_SIZE(a->type);
			cudaMemcpyAsync(b->data.u8, a->data.u8, size, cudaMemcpyDeviceToDevice, stream);
		} else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_16F) {
			assert(CCV_IS_TENSOR_CONTIGUOUS(a));
			assert(CCV_IS_TENSOR_CONTIGUOUS(b));
			assert(tensor_count == ccv_nnc_tensor_count(b->info));
			_ccv_nnc_data_conversion_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, a->data.f32, (__half*)b->data.f16);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32F) {
			assert(CCV_IS_TENSOR_CONTIGUOUS(a));
			assert(CCV_IS_TENSOR_CONTIGUOUS(b));
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			assert(tensor_count == ccv_nnc_tensor_count(b->info));
			_ccv_nnc_data_conversion_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, (__half*)a->data.f16, b->data.f32);
		} else if (a->info.datatype == CCV_64F && b->info.datatype == CCV_32F) {
			assert(CCV_IS_TENSOR_CONTIGUOUS(a));
			assert(CCV_IS_TENSOR_CONTIGUOUS(b));
			assert(tensor_count == ccv_nnc_tensor_count(b->info));
			_ccv_nnc_data_conversion_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, a->data.f64, b->data.f32);
		} else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_64F) {
			assert(CCV_IS_TENSOR_CONTIGUOUS(a));
			assert(CCV_IS_TENSOR_CONTIGUOUS(b));
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			assert(tensor_count == ccv_nnc_tensor_count(b->info));
			_ccv_nnc_data_conversion_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, a->data.f32, b->data.f64);
		} else if (a->info.datatype == CCV_64F && b->info.datatype == CCV_16F) {
			assert(CCV_IS_TENSOR_CONTIGUOUS(a));
			assert(CCV_IS_TENSOR_CONTIGUOUS(b));
			assert(tensor_count == ccv_nnc_tensor_count(b->info));
			_ccv_nnc_data_conversion_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, a->data.f64, (__half*)b->data.f16);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_64F) {
			assert(CCV_IS_TENSOR_CONTIGUOUS(a));
			assert(CCV_IS_TENSOR_CONTIGUOUS(b));
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			assert(tensor_count == ccv_nnc_tensor_count(b->info));
			_ccv_nnc_data_conversion_kernel<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, (__half*)a->data.f16, b->data.f64);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_datatype_conversion;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATATYPE_CONVERSION_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_datatype_conversion;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_masked_fill_kernel(const size_t a_count, const size_t b_count, const NUM2 p, const NUM1 q, const NUM1* const a, const NUM2* const b, NUM1* const c)
{
	CUDA_1D_KERNEL_LOOP(i, a_count) {
		c[i] = (b[i % b_count] == p) ? q : a[i];
	}
}

static void _ccv_nnc_masked_fill_gpu_ref(const float p, const float q, ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b, ccv_nnc_tensor_view_t* const c, ccv_nnc_stream_context_t* const stream_context)
{
	assert(c->info.datatype == a->info.datatype);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const size_t a_count = ccv_nnc_tensor_count(a->info);
	const size_t b_count = ccv_nnc_tensor_count(b->info);
	assert(a_count >= b_count);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	if (a->info.datatype == CCV_32F && b->info.datatype == CCV_32F)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, p, q, a->data.f32, b->data.f32, c->data.f32);
	else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_32S)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, (int)(p + 0.5), q, a->data.f32, b->data.i32, c->data.f32);
	else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_16F)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, (__half)p, q, a->data.f32, (__half*)b->data.f16, c->data.f32);
	else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32F)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, p, (__half)q, (__half*)a->data.f16, b->data.f32, (__half*)c->data.f16);
	else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32S)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, (int)(p + 0.5), (__half)q, (__half*)a->data.f16, b->data.i32, (__half*)c->data.f16);
	else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_16F)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, (__half)p, (__half)q, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)c->data.f16);
	else if (a->info.datatype == CCV_32S && b->info.datatype == CCV_32F)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, p, (int)(q + 0.5), a->data.i32, b->data.f32, c->data.i32);
	else if (a->info.datatype == CCV_32S && b->info.datatype == CCV_32S)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, (int)(p + 0.5), (int)(q + 0.5), a->data.i32, b->data.i32, c->data.i32);
	else if (a->info.datatype == CCV_32S && b->info.datatype == CCV_16F)
		_ccv_nnc_masked_fill_kernel<<<CUDA_GET_BLOCKS(a_count), CUDA_NUM_THREADS, 0, stream>>>(a_count, b_count, (__half)p, (int)(q + 0.5), a->data.i32, (__half*)b->data.f16, c->data.i32);
}

static int _ccv_nnc_masked_fill_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	assert(inputs[0]);
	assert(inputs[1]);
	assert(outputs[0]);
	_ccv_nnc_masked_fill_gpu_ref(cmd.info.blas.a[0], cmd.info.blas.a[1], (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0], stream_context);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_masked_fill_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	_ccv_nnc_masked_fill_gpu_ref(cmd.info.blas.a[0], 0, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0], stream_context);
	// TODO: doesn't really support taking gradient on mask.
	// if (output_size >= 2 && outputs[1])
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MASKED_FILL_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_masked_fill_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MASKED_FILL_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_masked_fill_back;
}
