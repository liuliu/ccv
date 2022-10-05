#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

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
		assert(CCV_GET_DATA_TYPE_SIZE(a->info.datatype) == CCV_GET_DATA_TYPE_SIZE(b->info.datatype));
		const size_t size = (ssize_t)ccv_nnc_tensor_count(a->info) * CCV_GET_DATA_TYPE_SIZE(a->info.datatype);
		if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY)
		{
			unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)a->data.u8 & -4096);
			const off_t offset_a = (uintptr_t)a->data.u8 - (uintptr_t)aligned_ptr;
			const size_t aligned_size = ((size + offset_a + 4095) & -4096);
			id<MTLBuffer> buffer_a = [ccv_nnc_default_device() newBufferWithBytesNoCopy:aligned_ptr length:aligned_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
			id<MTLBuffer> buffer_b = mpgetbuffer(b->data.u8, b);
			const off_t offset_b = mpgetoffset(b->data.u8);
			@autoreleasepool {
				id<MTLCommandBuffer> command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
				id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
				[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:size];
				[encoder endEncoding];
				[command_buffer commit];
				[command_buffer waitUntilCompleted];
			}
		} else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY) {
			id<MTLBuffer> buffer_a = mpgetbuffer(a->data.u8, a);
			const off_t offset_a = mpgetoffset(a->data.u8);
			unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)b->data.u8 & -4096);
			const off_t offset_b = (uintptr_t)b->data.u8 - (uintptr_t)aligned_ptr;
			const size_t aligned_size = ((size + offset_b + 4095) & -4096);
			id<MTLBuffer> buffer_b = [ccv_nnc_default_device() newBufferWithBytesNoCopy:aligned_ptr length:aligned_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
			@autoreleasepool {
				id<MTLCommandBuffer> command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
				id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
				[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:size];
				[encoder endEncoding];
				[command_buffer commit];
				[command_buffer waitUntilCompleted];
			}
		} else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
			memcpy(b->data.u8, a->data.u8, size);
		else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY) {
			const int device_a = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
			const int device_b = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
			assert(device_a == device_b);
			id<MTLBuffer> buffer_a = mpgetbuffer(a->data.u8, a);
			id<MTLBuffer> buffer_b = mpgetbuffer(b->data.u8, b);
			const off_t offset_a = mpgetoffset(a->data.u8);
			const off_t offset_b = mpgetoffset(b->data.u8);
			@autoreleasepool {
				id<MTLCommandBuffer> command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
				id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
				[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:size];
				[encoder endEncoding];
				[command_buffer commit];
				[command_buffer waitUntilCompleted];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY | CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}

static int _ccv_nnc_transpose(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		for (i = 0; i < output_size; i++)
		{
			const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[i];
			ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[i];
			MPSGraph *graph = [MPSGraph new];
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensor* mps_b = [graph transposeTensor:mps_a dimension:cmd.info.transpose.axis[0] withDimension:cmd.info.transpose.axis[1] name:nil];
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_b, b);
			[graph release];
		}
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_transpose;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_TRANSPOSE_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_transpose;
}

static int _ccv_nnc_set_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i, j;
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		for (i = 0; i < output_size; i++)
		{
			ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[i];
			MPSGraph *graph = [MPSGraph new];
			NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
			const int nd = ccv_nnc_tensor_nd(a->info.dim);
			for (j = 0; j < nd; j++)
				[shape addObject:@(a->info.dim[j])];
			MPSGraphTensor* mps_a = [graph constantWithScalar:cmd.info.blas.a[0] shape:shape dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			[shape release];
			ccv_nnc_mps_graph_result(graph, command_buffer, @{}, mps_a, a);
			[graph release];
		}
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_set_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i, j;
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		for (i = 0; i < output_size; i++)
		{
			ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[i];
			MPSGraph *graph = [MPSGraph new];
			NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
			const int nd = ccv_nnc_tensor_nd(a->info.dim);
			for (j = 0; j < nd; j++)
				[shape addObject:@(a->info.dim[j])];
			MPSGraphTensor* mps_a = [graph constantWithScalar:0 shape:shape dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			[shape release];
			ccv_nnc_mps_graph_result(graph, command_buffer, @{}, mps_a, a);
			[graph release];
		}
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_back;
}

static int _ccv_nnc_format_transform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		for (i = 0; i < output_size; i++)
		{
			const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[i];
			ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[i];
			MPSGraph *graph = [MPSGraph new];
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			if (mps_a != mps_input_a)
				ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, b);
			else
				ccv_nnc_mps_export_data(data_a, command_buffer, b);
			[graph release];
		}
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FORMAT_TRANSFORM_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
}
