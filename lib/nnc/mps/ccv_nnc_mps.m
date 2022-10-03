#include "ccv_nnc_mps.h"
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

void* ccv_nnc_stream_compat_get_workspace(const ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem)
{
	return 0;
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
