#include "ccv_nnc_mps.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>

id<MTLDevice> ccv_nnc_default_device(void)
{
	static __thread id<MTLDevice> device;
	if (device == nil)
		device = MTLCreateSystemDefaultDevice();
	return device;
}

static id<MTLCommandQueue> _ccv_nnc_default_queue(void)
{
	static __thread id<MTLCommandQueue> queue;
	if (queue == nil)
		queue = [ccv_nnc_default_device() newCommandQueue];
	return queue;
}

void* mpmalloc(int device, size_t size)
{
	size_t aligned_size = ((size + PAGE_SIZE - 1) & -PAGE_SIZE);
	void* ptr;
	ccmemalign(&ptr, PAGE_SIZE, aligned_size);
	return ptr;
}

void mpfree(int device, void* ptr)
{
	ccfree(ptr);
}

int mpregmp(int device_id, mpmp_f func, void* const context)
{
	return 0;
}

void mpunregmp(const int id)
{
}

id<MTLBuffer> mpgetbuffer(void* ptr, const ccv_nnc_tensor_t* const tensor)
{
	unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)ptr & -PAGE_SIZE);
	const off_t offset_a = (uintptr_t)ptr - (uintptr_t)aligned_ptr;
	size_t size = CCV_GET_DATA_TYPE_SIZE(tensor->info.datatype);
	if (CCV_IS_TENSOR_VIEW(tensor))
	{
		int i, j = 0;
		const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
		for (i = 1; i < nd; i++)
			if (((ccv_nnc_tensor_view_t*)tensor)->stride[i] > ((ccv_nnc_tensor_view_t*)tensor)->stride[j])
				j = i;
		size *= tensor->info.dim[j] * ((ccv_nnc_tensor_view_t*)tensor)->stride[j];
	} else
		size *= ccv_nnc_tensor_count(tensor->info);
	const size_t aligned_size = ((size + offset_a + PAGE_SIZE - 1) & -PAGE_SIZE);
	id<MTLBuffer> buffer = [[ccv_nnc_default_device() newBufferWithBytesNoCopy:aligned_ptr length:aligned_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil] autorelease];
	assert(buffer != nil);
	return buffer;
}

off_t mpgetoffset(void* ptr)
{
	unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)ptr & -4096);
	return (off_t)((uintptr_t)ptr - (uintptr_t)aligned_ptr);
}

// Stream context
ccv_nnc_stream_context_t* ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	return stream_context;
}

void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context)
{
}

void ccv_nnc_stream_compat_add_callback(ccv_nnc_stream_context_t* const stream, const ccv_nnc_callback_f callback, const ccv_nnc_async_callback_f async_callback, void* const callback_context)
{
}

int co_stream_compat_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream)
{
	return 0;
}

void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
}

typedef struct {
	ccv_nnc_stream_context_t super;
	// Left for implementation yet, the CPU support for stream context.
	size_t workspace_size;
	void* workspace;
} ccv_nnc_stream_mps_t;

static __thread ccv_nnc_stream_mps_t ccv_nnc_per_thread_stream_mps = {
	.super = {
		.type = CCV_STREAM_CONTEXT_CPU,
	},
};

void* ccv_nnc_stream_compat_get_workspace(const ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem)
{
	ccv_nnc_stream_mps_t* stream_mps = (ccv_nnc_stream_mps_t*)stream_context;
	if (!stream_mps)
		stream_mps = &ccv_nnc_per_thread_stream_mps;
	assert(mem == CCV_TENSOR_CPU_MEMORY);
	if (stream_mps->workspace_size >= workspace_size)
		return stream_mps->workspace;
	stream_mps->workspace_size = workspace_size;
	if (stream_mps->workspace)
		ccfree(stream_mps->workspace);
	stream_mps->workspace = 0;
	ccmemalign(&stream_mps->workspace, 64, workspace_size);
	return stream_mps->workspace;
}

void ccv_nnc_stream_compat_drain(ccv_nnc_stream_context_t* const stream_context)
{
}

ccv_nnc_stream_signal_t* ccv_nnc_init_stream_signal(ccv_nnc_stream_signal_t* const signal)
{
	return signal;
}

void ccv_nnc_stream_compat_emit_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal)
{
}

void ccv_nnc_stream_compat_wait_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal)
{
}

void ccv_nnc_deinit_stream_signal(ccv_nnc_stream_signal_t* const signal)
{
}

int ccv_nnc_gpu_device_count(void)
{
	return 1;
}

MPSCommandBuffer* ccv_nnc_stream_context_get_command_buffer(ccv_nnc_stream_context_t* const stream_context)
{
	return [MPSCommandBuffer commandBufferFromCommandQueue:_ccv_nnc_default_queue()];
}

MPSDataType ccv_nnc_mps_datatype(const int datatype)
{
	switch (datatype)
	{
		case CCV_8U:
			return MPSDataTypeUInt8;
		case CCV_32S:
			return MPSDataTypeInt32;
		case CCV_64S:
			return MPSDataTypeInt64;
		case CCV_16F:
			return MPSDataTypeFloat16;
		case CCV_32F:
			return MPSDataTypeFloat32;
		case CCV_64F:
			assert(0 && "doesn't support double precision");
	}
	return MPSDataTypeFloat32;
}

MPSGraphTensorNamedDataLayout ccv_nnc_mps_tensor_data_layout(const int format)
{
	switch (format)
	{
		case CCV_TENSOR_FORMAT_NCHW:
			return MPSGraphTensorNamedDataLayoutNCHW;
		case CCV_TENSOR_FORMAT_NHWC:
			return MPSGraphTensorNamedDataLayoutNHWC;
		case CCV_TENSOR_FORMAT_CHWN:
			assert(0 && "doesn't support CHWN");
	}
	return MPSGraphTensorNamedDataLayoutNCHW;
}

MPSGraphTensor* ccv_nnc_mps_graph_tensor_input(MPSGraph* graph, const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC], MPSGraphTensor** input)
{
	const off_t offset = mpgetoffset(tensor_view->data.u8);
	assert(offset % (CCV_GET_DATA_TYPE_SIZE(tensor_view->info.datatype)) == 0);
	const off_t offc = offset / CCV_GET_DATA_TYPE_SIZE(tensor_view->info.datatype);
	const int nd = ccv_nnc_tensor_nd(dim);
	int i;
	NSInteger full_count, partial_count;
	if (CCV_IS_TENSOR_VIEW(tensor_view))
	{
		// Figure out if there are permutations based on strides, if there are, find the permutation and apply to the tensor.
		// Use the found permutation to alter strides and check whether we have the contiguous tensor, if not, we cannot proceed.
		int sorted_dim[CCV_NNC_MAX_DIM_ALLOC];
		int sorted_stride[CCV_NNC_MAX_DIM_ALLOC];
		int sorted_idx[CCV_NNC_MAX_DIM_ALLOC];
		for (i = 0; i < nd; i++)
			sorted_dim[i] = dim[i], sorted_stride[i] = stride[i], sorted_idx[i] = i;
		int j, t;
		for (i = 0; i < nd - 1; i++)
		{
			int idx = i;
			for (j = i + 1; j < nd; j++)
				if (sorted_stride[idx] < sorted_stride[j])
					idx = j;
			if (idx == i)
				continue;
			CCV_SWAP(sorted_stride[i], sorted_stride[idx], t);
			CCV_SWAP(sorted_dim[i], sorted_dim[idx], t);
			CCV_SWAP(sorted_idx[i], sorted_idx[idx], t);
		}
		int full_dim[CCV_NNC_MAX_DIM_ALLOC];
		full_dim[0] = sorted_dim[0];
		int flag = 0;
		for (i = 1; i < nd; i++)
		{
			assert(sorted_stride[i - 1] % sorted_stride[i] == 0);
			full_dim[i] = sorted_stride[i - 1] / sorted_stride[i];
			if (!flag)
				flag = (full_dim[i] != sorted_dim[i]);
		}
		MPSGraphTensor* desc;
		NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
		for (i = 0; i < nd; i++)
			[shape addObject:@(full_dim[i])];
		if (offset)
		{
			partial_count = sorted_stride[0] * sorted_dim[0];
			full_count = partial_count + offc;
			desc = [graph placeholderWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
			*input = desc;
			desc = [graph sliceTensor:desc dimension:0 start:offc length:partial_count name:nil];
			desc = [graph reshapeTensor:desc withShape:shape name:nil];
		} else {
			desc = [graph placeholderWithShape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
			[shape release];
			*input = desc;
		}
		if (flag) // If we sliced this tensor before.
		{
			NSMutableArray<NSNumber*>* starts = [NSMutableArray new];
			NSMutableArray<NSNumber*>* ends = [NSMutableArray new];
			NSMutableArray<NSNumber*>* strides = [NSMutableArray new];
			for (i = 0; i < nd; i++)
			{
				[starts addObject:@(0)];
				[ends addObject:@(sorted_dim[i])];
				[strides addObject:@(1)];
			}
			desc = [graph sliceTensor:desc starts:starts ends:ends strides:strides name:nil];
			[starts release];
			[ends release];
			[strides release];
		}
		/* This requires macOS 13. When that released, use permutation.
		flag = 0;
		for (i = 0; !flag && i < nd; i++)
			flag = (sorted_idx[i] != i);
		if (flag) // If we need to permute this tensor.
		{
			int reverse_idx[CCV_NNC_MAX_DIM_ALLOC]; // This is on the new order, which old axis we are pointing to.
			for (i = 0; i < nd; i++)
				reverse_idx[sorted_idx[i]] = i;
			NSMutableArray<NSNumber*>* permutation = [NSMutableArray new];
			for (i = 0; i < nd; i++)
				[permutation addObject:@(reverse_idx[i])];
			desc = [graph transposeTensor:desc permutation:permutation name:nil];
			[permutation release];
		} */
		for (i = 0; i < nd - 1; i++)
			while (sorted_idx[i] != i)
			{
				desc = [graph transposeTensor:desc dimension:i withDimension:sorted_idx[i] name:nil];
				int t = sorted_idx[i];
				sorted_idx[i] = sorted_idx[t];
				sorted_idx[t] = t;
			}
		return desc;
	} else {
		NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
		for (i = 0; i < nd; i++)
			[shape addObject:@(dim[i])];
		MPSGraphTensor* desc;
		if (offset)
		{
			partial_count = dim[0];
			for (i = 1; i < nd; i++)
				partial_count *= dim[i];
			full_count = offc + partial_count;
			desc = [graph placeholderWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
			*input = desc;
			desc = [graph sliceTensor:desc dimension:0 start:offc length:partial_count name:nil];
			desc = [graph reshapeTensor:desc withShape:shape name:nil];
		} else {
			desc = [graph placeholderWithShape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
			[shape release];
			*input = desc;
		}
		return desc;
	}
}

MPSGraphTensorData* ccv_nnc_mps_graph_tensor_data(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	const off_t offset = mpgetoffset(tensor_view->data.u8);
	assert(offset % (CCV_GET_DATA_TYPE_SIZE(tensor_view->info.datatype)) == 0);
	const off_t offc = offset / CCV_GET_DATA_TYPE_SIZE(tensor_view->info.datatype);
	const int nd = ccv_nnc_tensor_nd(dim);
	int i;
	NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
	NSInteger full_count, partial_count;
	if (CCV_IS_TENSOR_VIEW(tensor_view))
	{
		int sorted_dim[CCV_NNC_MAX_DIM_ALLOC];
		int sorted_stride[CCV_NNC_MAX_DIM_ALLOC];
		for (i = 0; i < nd; i++)
			sorted_dim[i] = dim[i], sorted_stride[i] = stride[i];
		int j, t;
		for (i = 0; i < nd - 1; i++)
		{
			int idx = i;
			for (j = i + 1; j < nd; j++)
				if (sorted_stride[idx] < sorted_stride[j])
					idx = j;
			if (idx == i)
				continue;
			CCV_SWAP(sorted_stride[i], sorted_stride[idx], t);
			CCV_SWAP(sorted_dim[i], sorted_dim[idx], t);
		}
		int full_dim[CCV_NNC_MAX_DIM_ALLOC];
		full_dim[0] = sorted_dim[0];
		for (i = 1; i < nd; i++)
		{
			assert(sorted_stride[i - 1] % sorted_stride[i] == 0);
			full_dim[i] = sorted_stride[i - 1] / sorted_stride[i];
		}
		if (offset)
		{
			partial_count = sorted_stride[0] * sorted_dim[0];
			full_count = partial_count + offc;
			[shape addObject:@(full_count)];
		} else
			for (i = 0; i < nd; i++)
				[shape addObject:@(full_dim[i])];
	} else {
		if (offset)
		{
			partial_count = dim[0];
			for (i = 1; i < nd; i++)
				partial_count *= dim[i];
			full_count = offc + partial_count;
			[shape addObject:@(full_count)];
		} else
			for (i = 0; i < nd; i++)
				[shape addObject:@(dim[i])];
	}
	id<MTLBuffer> buffer = mpgetbuffer(tensor_view->data.u8, (ccv_nnc_tensor_t*)tensor_view);
	MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype)];
	[shape release];
	return [data autorelease];
}

void ccv_nnc_mps_export_data(MPSGraphTensorData* data, MPSCommandBuffer* command_buffer, ccv_nnc_tensor_view_t* const tensor, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	id<MTLBuffer> buffer = mpgetbuffer(tensor->data.u8, (ccv_nnc_tensor_t*)tensor);
	NSInteger rowStrides[CCV_NNC_MAX_DIM_ALLOC];
	int stride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
	const int nd = ccv_nnc_tensor_nd(dim);
	const int* dstride;
	if (!CCV_IS_TENSOR_VIEW(tensor))
	{
		ccv_nnc_tensor_get_stride(dim, stride_from_dim);
		dstride = stride_from_dim;
	} else
		dstride = stride;
	int i;
	for (i = 0; i < nd; i++)
		rowStrides[nd - 1 - i] = CCV_GET_DATA_TYPE_SIZE(tensor->info.datatype) * dstride[i];
	MPSNDArray* ndarray = data.mpsndarray;
	off_t offset = mpgetoffset(tensor->data.u8);
	[ndarray exportDataWithCommandBuffer:command_buffer toBuffer:buffer destinationDataType:ccv_nnc_mps_datatype(tensor->info.datatype) offset:offset rowStrides:rowStrides];
}

void ccv_nnc_mps_graph_result(MPSGraph* graph, MPSCommandBuffer* command_buffer, MPSGraphTensorDataDictionary* feeds, MPSGraphTensor* output, ccv_nnc_tensor_view_t* const data, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	off_t offset = mpgetoffset(data->data.u8);
	if (CCV_IS_TENSOR_CONTIGUOUS(data) && offset == 0)
	{
		MPSGraphTensorData* tensor_data = ccv_nnc_mps_graph_tensor_data(data, dim, stride);
		[graph encodeToCommandBuffer:command_buffer feeds:feeds targetOperations:nil resultsDictionary:@{output: tensor_data} executionDescriptor:nil];
		return;
	}
	MPSGraphTensorDataDictionary* result = [graph encodeToCommandBuffer:command_buffer feeds:feeds targetTensors:@[output] targetOperations:nil executionDescriptor:nil];
	MPSGraphTensorData* tensor_data = result[output];
	ccv_nnc_mps_export_data(tensor_data, command_buffer, data, dim, stride);
}
