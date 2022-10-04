#include "ccv_nnc_mps.h"
#include "ccv_internal.h"
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

id<MTLCommandQueue> ccv_nnc_default_queue(void)
{
	static __thread id<MTLCommandQueue> queue;
	if (queue == nil)
		queue = [ccv_nnc_default_device() newCommandQueue];
	return queue;
}

void* mpmalloc(int device, size_t size)
{
	id<MTLBuffer> buffer = [ccv_nnc_default_device() newBufferWithLength:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
	return (void*)buffer;
}

static const char kMPSGraphTensorDataMTLBufferOffsetKey[0];

void mpsetoffset(void* ptr, off_t off)
{
	id<MTLBuffer> buffer = (id<MTLBuffer>)ptr;
	objc_setAssociatedObject(buffer, &kMPSGraphTensorDataMTLBufferOffsetKey, (id)(intptr_t)off, OBJC_ASSOCIATION_ASSIGN);
}

off_t mpgetoffset(void* ptr)
{
	id<MTLBuffer> buffer = (id<MTLBuffer>)ptr;
	return (off_t)(intptr_t)objc_getAssociatedObject(buffer, &kMPSGraphTensorDataMTLBufferOffsetKey);
}

void mpfree(int device, void* ptr)
{
	id<MTLBuffer> buffer = (id<MTLBuffer>)ptr;
	[buffer release];
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

id<MTLCommandBuffer> ccv_nnc_stream_context_get_command_buffer(ccv_nnc_stream_context_t* const stream_context)
{
	return [[[ccv_nnc_default_queue() commandBuffer] retain] autorelease];
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

MPSGraphTensor* ccv_nnc_mps_graph_tensor_input(MPSGraph* graph, const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC], MPSGraphTensor** input)
{
	off_t offset = mpgetoffset(tensor_view->data.u8);
	assert(offset == 0); // I need to dig more into supporting non-zero offset.
	const int nd = ccv_nnc_tensor_nd(dim);
	int i;
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
		NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
		for (i = 0; i < nd; i++)
			[shape addObject:@(full_dim[i])];
		MPSGraphTensor* desc = [graph placeholderWithShape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
		[shape release];
		*input = desc;
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
			NSMutableArray<NSNumber*>* permutation = [NSMutableArray new];
			for (i = 0; i < nd; i++)
				[permutation addObject:@(sorted_idx[i])];
			desc = [graph transposeTensor:desc permutation:permutation name:nil];
			[permutation release];
		}
		*/
		for (i = 0; i < nd - 1; i++)
			if (sorted_idx[i] != i && sorted_idx[i] > i)
				desc = [graph transposeTensor:desc dimension:i withDimension:sorted_idx[i] name:nil];
		return desc;
	} else {
		NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
		for (i = 0; i < nd; i++)
			[shape addObject:@(dim[i])];
		MPSGraphTensor* desc = [graph placeholderWithShape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
		[shape release];
		*input = desc;
		return desc;
	}
}

MPSGraphTensorData* ccv_nnc_mps_graph_tensor_data(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	off_t offset = mpgetoffset(tensor_view->data.u8);
	assert(offset == 0); // I need to dig more into supporting non-zero offset.
	const int nd = ccv_nnc_tensor_nd(dim);
	int i;
	NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
	if (CCV_IS_TENSOR_VIEW(tensor_view))
	{
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
		for (i = 1; i < nd; i++)
		{
			assert(sorted_stride[i - 1] % sorted_stride[i] == 0);
			full_dim[i] = sorted_stride[i - 1] / sorted_stride[i];
		}
		for (i = 0; i < nd; i++)
			[shape addObject:@(full_dim[i])];
	} else
		for (i = 0; i < nd; i++)
			[shape addObject:@(dim[i])];
	id<MTLBuffer> buffer = (id<MTLBuffer>)tensor_view->data.u8;
	MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype)];
	[shape release];
	return [data autorelease];
}

MPSGraphTensor* ccv_nnc_mps_graph_tensor_result(MPSGraph* graph, MPSGraphTensor* tensor, ccv_nnc_tensor_view_t* tensor_view)
{
	return tensor;
}
