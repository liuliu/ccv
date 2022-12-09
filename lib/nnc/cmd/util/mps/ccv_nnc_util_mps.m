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
			unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)a->data.u8 & -PAGE_SIZE);
			const off_t offset_a = (uintptr_t)a->data.u8 - (uintptr_t)aligned_ptr;
			const size_t aligned_size = ((size + offset_a + PAGE_SIZE - 1) & -PAGE_SIZE);
			id<MTLBuffer> buffer_b = mpgetbuffer(b);
			const off_t offset_b = mpgetoffset(b);
			@autoreleasepool {
				id<MTLBuffer> buffer_a = [ccv_nnc_default_device() newBufferWithBytesNoCopy:aligned_ptr length:aligned_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
				id<MTLCommandBuffer> command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
				id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
				[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:size];
				[encoder endEncoding];
				ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
				[buffer_a release];
			}
		} else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY) {
			id<MTLBuffer> buffer_a = mpgetbuffer(a);
			const off_t offset_a = mpgetoffset(a);
			unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)b->data.u8 & -PAGE_SIZE);
			const off_t offset_b = (uintptr_t)b->data.u8 - (uintptr_t)aligned_ptr;
			const size_t aligned_size = ((size + offset_b + PAGE_SIZE - 1) & -PAGE_SIZE);
			@autoreleasepool {
				id<MTLBuffer> buffer_b = [ccv_nnc_default_device() newBufferWithBytesNoCopy:aligned_ptr length:aligned_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
				id<MTLCommandBuffer> command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
				id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
				[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:size];
				[encoder endEncoding];
				ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
				[buffer_b release];
			}
		} else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_CPU_MEMORY)
			memcpy(b->data.u8, a->data.u8, size);
		else if (CCV_TENSOR_GET_MEMORY(a->info.type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(b->info.type) == CCV_TENSOR_GPU_MEMORY) {
			const int device_a = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
			const int device_b = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
			assert(device_a == device_b);
			id<MTLBuffer> buffer_a = mpgetbuffer(a);
			id<MTLBuffer> buffer_b = mpgetbuffer(b);
			const off_t offset_a = mpgetoffset(a);
			const off_t offset_b = mpgetoffset(b);
			@autoreleasepool {
				id<MTLCommandBuffer> command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
				id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
				[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:size];
				[encoder endEncoding];
				ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
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
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs + i, 1, outputs + i, 1);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_b = [graph transposeTensor:mps_a dimension:cmd.info.transpose.axis[0] withDimension:cmd.info.transpose.axis[1] name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1);
		}
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
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
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, 0, 0, outputs + i, 1);
			NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
			const int nd = ccv_nnc_tensor_nd(a->info.dim);
			for (j = 0; j < nd; j++)
				[shape addObject:@(a->info.dim[j])];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, 0, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_a = [graph constantWithScalar:cmd.info.blas.a[0] shape:shape dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				[resultTensors addObject:mps_a];
			});
			[shape release];
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[], &a, (int*[]){ a->info.dim }, (int*[]){ a->stride }, 1);
		}
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
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
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, 0, 0, outputs + i, 1);
			NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
			const int nd = ccv_nnc_tensor_nd(a->info.dim);
			for (j = 0; j < nd; j++)
				[shape addObject:@(a->info.dim[j])];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, 0, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_a = [graph constantWithScalar:0 shape:shape dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				[resultTensors addObject:mps_a];
			});
			[shape release];
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[], &a, (int*[]){ a->info.dim }, (int*[]){ a->stride }, 1);
		}
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
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
			ccv_nnc_tensor_view_t bt = ccv_nnc_get_tensor_view(outputs[i]);
			MPSGraph *graph = [MPSGraph new];
			graph.options = MPSGraphOptionsSynchronizeResults;
			int adim[CCV_NNC_MAX_DIM_ALLOC];
			int astride[CCV_NNC_MAX_DIM_ALLOC];
			int bdim[CCV_NNC_MAX_DIM_ALLOC];
			int bstride[CCV_NNC_MAX_DIM_ALLOC];
			if (a->info.format == bt.info.format)
			{
				memcpy(adim, a->info.dim, sizeof(adim));
				if (CCV_IS_TENSOR_VIEW(a))
					memcpy(astride, a->stride, sizeof(astride));
				memcpy(bdim, bt.info.dim, sizeof(bdim));
				if (CCV_IS_TENSOR_VIEW(&bt))
					memcpy(bstride, bt.stride, sizeof(bstride));
			} else {
				ccv_nnc_tensor_view_get_dim(a, adim);
				ccv_nnc_tensor_view_get_stride(a, astride);
				ccv_nnc_tensor_view_get_dim(&bt, bdim);
				ccv_nnc_tensor_view_get_stride(&bt, bstride);
				if (a->info.format == CCV_TENSOR_FORMAT_NHWC)
				{
					if (bt.info.format == CCV_TENSOR_FORMAT_NCHW)
					{
						int c = bdim[1];
						bdim[1] = bdim[2];
						bdim[2] = bdim[3];
						bdim[3] = c;
						c = bstride[1];
						bstride[1] = bstride[2];
						bstride[2] = bstride[3];
						bstride[3] = c;
					} else {
						assert(bt.info.format == CCV_TENSOR_FORMAT_CHWN);
						int t;
						CCV_SWAP(bdim[0], bdim[3], t);
						CCV_SWAP(bstride[0], bstride[3], t);
					}
				} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW) {
					if (bt.info.format == CCV_TENSOR_FORMAT_NHWC)
					{
						int c = bdim[3];
						bdim[3] = bdim[2];
						bdim[2] = bdim[1];
						bdim[1] = c;
						c = bstride[3];
						bstride[3] = bstride[2];
						bstride[2] = bstride[1];
						bstride[1] = c;
					} else {
						assert(bt.info.format == CCV_TENSOR_FORMAT_CHWN);
						int n = bdim[3];
						bdim[3] = bdim[2];
						bdim[2] = bdim[1];
						bdim[1] = bdim[0];
						bdim[0] = n;
						n = bstride[3];
						bstride[3] = bstride[2];
						bstride[2] = bstride[1];
						bstride[1] = bstride[0];
						bstride[0] = n;
					}
				} else if (a->info.format == CCV_TENSOR_FORMAT_CHWN) {
					if (bt.info.format == CCV_TENSOR_FORMAT_NCHW)
					{
						int n = bdim[0];
						bdim[0] = bdim[1];
						bdim[1] = bdim[2];
						bdim[2] = bdim[3];
						bdim[3] = n;
						n = bstride[0];
						bstride[0] = bstride[1];
						bstride[1] = bstride[2];
						bstride[2] = bstride[3];
						bstride[3] = n;
					} else {
						assert(bt.info.format == CCV_TENSOR_FORMAT_NHWC);
						int t;
						CCV_SWAP(bdim[0], bdim[3], t);
						CCV_SWAP(bstride[0], bstride[3], t);
					}
				}
				// Mark this as tensor view as we changed its stride and dim.
				bt.type |= CCV_TENSOR_VIEW;
				memcpy(bt.info.dim, bdim, sizeof(bdim));
				memcpy(bt.stride, bstride, sizeof(bstride));
			}
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim, astride, &mps_input_a);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			if (mps_a != mps_input_a)
				ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, &bt, bdim, bstride);
			else
				ccv_nnc_mps_export_data(data_a, command_buffer, &bt, bdim, bstride);
			[graph release];
		}
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
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

static int _ccv_nnc_datatype_conversion(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		for (i = 0; i < output_size; i++)
		{
			const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
			ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
			assert(a != b); // Cannot do inplace transform.
			assert(a->info.format == b->info.format);
			assert(CCV_TENSOR_GET_DEVICE_ID(a->info.type) == CCV_TENSOR_GET_DEVICE_ID(b->info.type));
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			if (CCV_IS_TENSOR_VIEW(a)) // Only allocate on-demand MPSGraph if a is a tensor view.
			{
				MPSGraph *graph = [MPSGraph new];
				graph.options = MPSGraphOptionsSynchronizeResults;
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				if (mps_a != mps_input_a)
					ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, b, b->info.dim, b->stride);
				else
					ccv_nnc_mps_export_data(data_a, command_buffer, b, b->info.dim, b->stride);
				[graph release];
			} else
				ccv_nnc_mps_export_data(data_a, command_buffer, b, b->info.dim, b->stride);
		}
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_datatype_conversion;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATATYPE_CONVERSION_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_datatype_conversion;
}
