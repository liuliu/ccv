#include "ccv_nnc_mps.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/ccv_nnc_easy.h"
#include "3rdparty/khash/khash.h"
#include <string.h>
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>
#import <os/lock.h>
#import <sys/utsname.h>

int ccv_nnc_is_a13_and_below(void)
{
	static dispatch_once_t once;
	static int is_a13_and_below;
	dispatch_once(&once, ^{
		struct utsname info;
		uname(&info);
		const size_t len = strnlen(info.machine, sizeof(info.machine));
		if (len >= 9) // Compare against iPhone model.
		{
			if (memcmp(info.machine, "iPhone10,", 9) == 0 ||
				memcmp(info.machine, "iPhone11,", 9) == 0 ||
				memcmp(info.machine, "iPhone12,", 9) == 0)
				is_a13_and_below = 1;
		} else if (len >= 6) {
			if (memcmp(info.machine, "iPad7,", 6) == 0 ||
				memcmp(info.machine, "iPad8,", 6) == 0)
				is_a13_and_below = 1;
		} else if (len >= 7) {
			if (memcmp(info.machine, "iPad11,", 7) == 0 ||
				memcmp(info.machine, "iPad12,", 7) == 0)
				is_a13_and_below = 1;
		}
	});
	return is_a13_and_below;
}

id<MTLDevice> ccv_nnc_default_device(void)
{
	static dispatch_once_t once;
	static id<MTLDevice> device;
	dispatch_once(&once, ^{
		device = MTLCreateSystemDefaultDevice();
	});
	return device;
}

MPSGraphDevice* _ccv_nnc_default_mps_device(void)
{
	static dispatch_once_t once;
	static MPSGraphDevice* device;
	dispatch_once(&once, ^{
		device = [[MPSGraphDevice deviceWithMTLDevice:ccv_nnc_default_device()] retain];
	});
	return device;
}

static os_unfair_lock queue_lock; 
#define OLD_MAX_COMMAND_BUFFER_SIZE (32)
#define OLD_LIMITED_COMMAND_BUFFER_SIZE (8)
static id<MTLCommandBuffer> old_last_command_buffers[OLD_MAX_COMMAND_BUFFER_SIZE];
static id<MTLCommandBuffer> last_command_buffer;

static id<MTLCommandQueue> _ccv_nnc_default_queue(void)
{
	static dispatch_once_t once;
	static id<MTLCommandQueue> queue;
	dispatch_once(&once, ^{
		queue = [ccv_nnc_default_device() newCommandQueue];
	});
	return queue;
}

typedef struct {
	int device_id;
	mpmp_f func;
	void* ctx;
} mpmp_t;

static pthread_mutex_t g_mp_mutex = PTHREAD_MUTEX_INITIALIZER;
static ccv_array_t* g_mp_h;
static int g_mp_slot;

int mpregmp(int device_id, mpmp_f func, void* const context)
{
	assert(func);
	pthread_mutex_lock(&g_mp_mutex);
	if (!g_mp_h)
	{
		g_mp_h = ccv_array_new(sizeof(mpmp_t), 1, 0);
		g_mp_slot = -1;
	}
	mpmp_t mp = {
		device_id, func, context,
	};
	int slot = g_mp_slot;
	if (g_mp_slot >= 0)
	{
		assert(g_mp_slot < g_mp_h->rnum);
		*(mpmp_t*)ccv_array_get(g_mp_h, g_mp_slot) = mp;
		int i;
		for (i = g_mp_slot + 1; i < g_mp_h->rnum; i++)
			if (((mpmp_t*)ccv_array_get(g_mp_h, i))->func == 0)
			{
				g_mp_slot = i;
				break;
			}
		if (g_mp_slot == slot)
			g_mp_slot = -1; // Cannot find a slot.
	} else {
		ccv_array_push(g_mp_h, &mp);
		slot = g_mp_h->rnum - 1;
	}
	pthread_mutex_unlock(&g_mp_mutex);
	return slot;
}

void mpunregmp(const int slot)
{
	pthread_mutex_lock(&g_mp_mutex);
	assert(slot < g_mp_h->rnum);
	if (g_mp_slot < 0 || slot < g_mp_slot)
		g_mp_slot = slot;
	*(mpmp_t*)ccv_array_get(g_mp_h, g_mp_slot) = (mpmp_t){};
	pthread_mutex_unlock(&g_mp_mutex);
}

static void mptrigmp(void)
{
	ccv_nnc_synchronize_stream_context(0);
	pthread_mutex_lock(&g_mp_mutex);
	int i;
	for (i = 0; i < g_mp_h->rnum; i++)
	{
		mpmp_t* const mp = (mpmp_t*)ccv_array_get(g_mp_h, i);
		if (mp->device_id == 0 && mp->func)
			mp->func(0, mp->ctx);
	}
	pthread_mutex_unlock(&g_mp_mutex);
	ccv_nnc_mps_clear_graph_executable_cache();
}

void* mpheapalloc(int device, size_t size)
{
	MTLHeapDescriptor* descriptor = [MTLHeapDescriptor new];
	descriptor.size = size;
	descriptor.type = MTLHeapTypePlacement;
	descriptor.cpuCacheMode = MTLCPUCacheModeWriteCombined;
	descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
	id<MTLHeap> heap = [ccv_nnc_default_device() newHeapWithDescriptor:descriptor];
	if (heap == nil)
	{
		mptrigmp();
		heap = [ccv_nnc_default_device() newHeapWithDescriptor:descriptor];
		assert(heap != nil);
	}
	[descriptor release];
	return (void*)heap;
}

void mpheapfree(int device, void* ptr)
{
	id<MTLHeap> heap = (id<MTLHeap>)ptr;
	[heap release];
}

void* mpobjmalloc(int device, size_t size)
{
	id<MTLBuffer> buffer = [ccv_nnc_default_device() newBufferWithLength:size options:MTLResourceStorageModePrivate];
	if (buffer == nil)
	{
		mptrigmp();
		buffer = [ccv_nnc_default_device() newBufferWithLength:size options:MTLResourceStorageModePrivate];
		assert(buffer != nil);
	}
	return (void*)buffer;
}

void mpobjfree(int device, void* ptr)
{
	id<MTLBuffer> buffer = (id<MTLBuffer>)ptr;
	[buffer release];
}

void* mpobjcreate(void* ptr, off_t offset, size_t size)
{
	id<MTLHeap> heap = (id<MTLHeap>)ptr;
	MTLSizeAndAlign sizeAndAlign = [ccv_nnc_default_device() heapBufferSizeAndAlignWithLength:size options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModePrivate];
	assert(offset % sizeAndAlign.align == 0);
	id<MTLBuffer> buffer = [heap newBufferWithLength:sizeAndAlign.size options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModePrivate offset:offset];
	if (buffer == nil)
	{
		mptrigmp();
		buffer = [heap newBufferWithLength:sizeAndAlign.size options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModePrivate offset:offset];
		assert(buffer != nil);
	}
	[buffer makeAliasable];
	return buffer;
}

id<MTLBuffer> mpgetbuffer(const ccv_nnc_tensor_t* const tensor)
{
	return (id<MTLBuffer>)tensor->data.u8;
}

off_t mpgetoffset(const ccv_nnc_tensor_t* const tensor)
{
	return tensor->dataof;
}

void mpmemcpy(void* dest, const off_t dest_off, const int dest_type, const void* src, const off_t src_off, const int src_type, size_t n)
{
	if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_GPU_MEMORY)
	{
		unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)src & -PAGE_SIZE);
		const off_t offset_a = (uintptr_t)src - (uintptr_t)aligned_ptr + src_off;
		const size_t aligned_size = ((n + offset_a + PAGE_SIZE - 1) & -PAGE_SIZE);
		id<MTLBuffer> buffer_b = (id<MTLBuffer>)dest;
		const off_t offset_b = dest_off;
		@autoreleasepool {
			id<MTLBuffer> buffer_a = [ccv_nnc_default_device() newBufferWithBytesNoCopy:aligned_ptr length:aligned_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
			id<MTLCommandBuffer> command_buffer = [MPSCommandBuffer commandBufferFromCommandQueue:_ccv_nnc_default_queue()];
			id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
			[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:n];
			[encoder endEncoding];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
			[buffer_a release];
		}
	} else if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_CPU_MEMORY) {
		id<MTLBuffer> buffer_a = (id<MTLBuffer>)src;
		const off_t offset_a = src_off;
		unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)dest & -PAGE_SIZE);
		const off_t offset_b = (uintptr_t)dest - (uintptr_t)aligned_ptr;
		const size_t aligned_size = ((n + offset_b + PAGE_SIZE - 1) & -PAGE_SIZE);
		@autoreleasepool {
			id<MTLBuffer> buffer_b = [ccv_nnc_default_device() newBufferWithBytesNoCopy:aligned_ptr length:aligned_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
			id<MTLCommandBuffer> command_buffer = [MPSCommandBuffer commandBufferFromCommandQueue:_ccv_nnc_default_queue()];
			id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
			[encoder copyFromBuffer:buffer_a sourceOffset:offset_a toBuffer:buffer_b destinationOffset:offset_b size:n];
			[encoder endEncoding];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
			[buffer_b release];
		}
	} else {
		assert(0 && "can only copy from GPU to CPU or vice versa");
	}
}

// MPSGraphExecutable cache.
static inline uint32_t twang_32from64(uint64_t key)
{
	key = (~key) + (key << 18);
	key = key ^ (key >> 31);
	key = key * 21;
	key = key ^ (key >> 11);
	key = key + (key << 6);
	key = key ^ (key >> 22);
	return (uint32_t)(key);
}

static inline khint32_t _kh_graph_key_executable_hash_func(const ccv_nnc_mps_graph_key_t key)
{
	uint32_t h = key.cmd;
	int i, j;
	uint32_t* data = (uint32_t*)&key.params;
	for (i = 0; i < sizeof(key.params) / sizeof(uint32_t); i++)
		h = twang_32from64(((uint64_t)h << 32) | data[i]);
	data = (uint32_t*)&key.hint;
	for (i = 0; i < sizeof(key.hint) / sizeof(uint32_t); i++)
		h = twang_32from64(((uint64_t)h << 32) | data[i]);
	h = twang_32from64(((uint64_t)h << 32) | key.input_size);
	h = twang_32from64(((uint64_t)h << 32) | key.output_size);
	for (i = 0; i < key.input_size; i++)
	{
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].format);
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].datatype);
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].dataof);
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].nd);
		for (j = 0; j < key.inputs[i].nd; j++)
		{
			h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].dim[j]);
			h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].stride[j]);
		}
	}
	for (i = 0; i < key.output_size; i++)
	{
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].format);
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].datatype);
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].dataof);
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].nd);
		for (j = 0; j < key.outputs[i].nd; j++)
		{
			h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].dim[j]);
			h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].stride[j]);
		}
	}
	return (khint32_t)h;
}

static inline int _kh_graph_key_executable_hash_equal(const ccv_nnc_mps_graph_key_t a, const ccv_nnc_mps_graph_key_t b)
{
	if (a.cmd != b.cmd || a.flags != b.flags || a.input_size != b.input_size || a.output_size != b.output_size)
		return 0;
	if (memcmp(&a.params, &b.params, sizeof(a.params)) != 0)
		return 0;
	if (memcmp(&a.hint, &b.hint, sizeof(a.hint)) != 0)
		return 0;
	int i, j;
	for (i = 0; i < a.input_size; i++)
	{
		if (a.inputs[i].format != b.inputs[i].format || a.inputs[i].datatype != b.inputs[i].datatype || a.inputs[i].nd != b.inputs[i].nd || a.inputs[i].dataof != b.inputs[i].dataof)
			return 0;
		for (j = 0; j < a.inputs[i].nd; j++)
			if (a.inputs[i].dim[j] != b.inputs[i].dim[j] || a.inputs[i].stride[j] != b.inputs[i].stride[j])
				return 0;
	}
	for (i = 0; i < a.output_size; i++)
	{
		if (a.outputs[i].format != b.outputs[i].format || a.outputs[i].datatype != b.outputs[i].datatype || a.outputs[i].nd != b.outputs[i].nd || a.outputs[i].dataof != b.outputs[i].dataof)
			return 0;
		for (j = 0; j < a.outputs[i].nd; j++)
			if (a.outputs[i].dim[j] != b.outputs[i].dim[j] || a.outputs[i].stride[j] != b.outputs[i].stride[j])
				return 0;
	}
	return 1;
}

typedef struct {
	int indice_size;
	int* indices;
	MPSGraphExecutable* exec;
} ccv_nnc_graph_val_t;

KHASH_INIT(graph_executable_cache, ccv_nnc_mps_graph_key_t, ccv_nnc_graph_val_t, 1, _kh_graph_key_executable_hash_func, _kh_graph_key_executable_hash_equal)

static khash_t(graph_executable_cache)* g_graph_executable_cache = 0;

static inline void ccv_nnc_mps_graph_key_free(ccv_nnc_mps_graph_key_t key)
{
	if (key.inputs)
		ccfree(key.inputs);
}

void ccv_nnc_mps_clear_graph_executable_cache(void)
{
	if (!g_graph_executable_cache)
		return;
	khiter_t k;
	for (k = kh_begin(g_graph_executable_cache); k < kh_end(g_graph_executable_cache); k++)
	{
		if (!kh_exist(g_graph_executable_cache, k))
			continue;
		ccv_nnc_mps_graph_key_free(kh_key(g_graph_executable_cache, k));
		if (kh_val(g_graph_executable_cache, k).indices)
			ccfree(kh_val(g_graph_executable_cache, k).indices);
		[kh_val(g_graph_executable_cache, k).exec release];
		kh_del(graph_executable_cache, g_graph_executable_cache, k);
	}
}

MPSGraphExecutable* ccv_nnc_mps_graph_executable_cache(const ccv_nnc_mps_graph_key_t key, int* indices, void(NS_NOESCAPE ^block)(MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors))
{
	if (!g_graph_executable_cache)
		g_graph_executable_cache = kh_init(graph_executable_cache);
	int ret = 0;
	khiter_t k = kh_put(graph_executable_cache, g_graph_executable_cache, key, &ret);
	if (ret != 0)
	{
		MPSGraph* graph = [MPSGraph new];
		graph.options = MPSGraphOptionsSynchronizeResults;
		NSMutableArray<MPSGraphTensor*> *inputTensors = [NSMutableArray new];
		NSMutableArray<MPSGraphShapedType*>* inputShapedTypes = [NSMutableArray new];
		NSMutableArray<MPSGraphTensor*>* targetTensors = [NSMutableArray new];
		block(graph, inputTensors, inputShapedTypes, targetTensors);
		assert(inputTensors.count == inputShapedTypes.count);
		MPSGraphCompilationDescriptor* compilationDescriptor = [MPSGraphCompilationDescriptor new];
		// Need more investigation into what this does.
		// compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel1;
		compilationDescriptor.optimizationProfile = MPSGraphOptimizationProfilePerformance;
		MPSGraphExecutable* executable = [[graph compileWithDevice:_ccv_nnc_default_mps_device() feeds:[NSDictionary dictionaryWithObjects:inputShapedTypes forKeys:inputTensors] targetTensors:targetTensors targetOperations:nil compilationDescriptor:compilationDescriptor] retain];
		executable.options = MPSGraphOptionsSynchronizeResults;
		[compilationDescriptor release];
		[graph release];
		kh_val(g_graph_executable_cache, k).exec = executable;
		kh_val(g_graph_executable_cache, k).indice_size = (int)inputTensors.count;
		kh_val(g_graph_executable_cache, k).indices = inputTensors.count > 0 ? (int*)ccmalloc(sizeof(int) * inputTensors.count) : 0;
		assert(inputTensors.count == executable.feedTensors.count);
		int i;
		for (i = 0; i < executable.feedTensors.count; i++)
			indices[i] = kh_val(g_graph_executable_cache, k).indices[i] = (int)[inputTensors indexOfObject:executable.feedTensors[i]];
		[inputTensors release];
		[inputShapedTypes release];
		[targetTensors release];
	} else {
		ccv_nnc_mps_graph_key_free(key);
		int i;
		for (i = 0; i < kh_val(g_graph_executable_cache, k).indice_size; i++)
			indices[i] = kh_val(g_graph_executable_cache, k).indices[i];
	}
	return kh_val(g_graph_executable_cache, k).exec;
}

ccv_nnc_mps_graph_key_t ccv_nnc_mps_graph_key_new(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	ccv_nnc_mps_graph_key_t key = {
		.cmd = cmd.cmd,
		.hint = hint,
		.params = cmd.info,
		.inputs = 0,
		.input_size = 0,
		.outputs = 0,
		.output_size = 0
	};
	if (input_size == 0 && output_size == 0)
		return key;
	assert(input_size >= 0 && output_size >= 0);
	key.input_size = input_size;
	key.output_size = output_size;
	key.inputs = (ccv_nnc_mps_graph_tensor_shape_t*)ccmalloc(sizeof(ccv_nnc_mps_graph_tensor_shape_t) * (input_size + output_size));
	key.outputs = key.inputs + input_size;
	int i, j;
	for (i = 0; i < input_size; i++)
	{
		memset(key.inputs[i].dim, 0, sizeof(key.inputs[i].dim));
		memset(key.inputs[i].stride, 0, sizeof(key.inputs[i].stride));
		if (!inputs[i])
		{
			key.inputs[i].format = 0;
			key.inputs[i].datatype = 0;
			key.inputs[i].dataof = 0;
			key.inputs[i].nd = 0;
			continue;
		}
		key.inputs[i].format = inputs[i]->info.format;
		key.inputs[i].datatype = inputs[i]->info.datatype;
		key.inputs[i].dataof = inputs[i]->dataof;
		const int nd = key.inputs[i].nd = ccv_nnc_tensor_nd(inputs[i]->info.dim);
		for (j = 0; j < nd; j++)
			key.inputs[i].dim[j] = inputs[i]->info.dim[j];
		if (CCV_IS_TENSOR_VIEW(inputs[i]))
			for (j = 0; j < nd; j++)
				key.inputs[i].stride[j] = ((ccv_nnc_tensor_view_t*)inputs[i])->stride[j];
	}
	for (i = 0; i < output_size; i++)
	{
		memset(key.outputs[i].dim, 0, sizeof(key.outputs[i].dim));
		memset(key.outputs[i].stride, 0, sizeof(key.outputs[i].stride));
		if (!outputs[i])
		{
			key.outputs[i].format = 0;
			key.outputs[i].datatype = 0;
			key.outputs[i].dataof = 0;
			key.outputs[i].nd = 0;
			continue;
		}
		key.outputs[i].format = outputs[i]->info.format;
		key.outputs[i].datatype = outputs[i]->info.datatype;
		key.outputs[i].dataof = outputs[i]->dataof;
		const int nd = key.outputs[i].nd = ccv_nnc_tensor_nd(outputs[i]->info.dim);
		for (j = 0; j < nd; j++)
			key.outputs[i].dim[j] = outputs[i]->info.dim[j];
		if (CCV_IS_TENSOR_VIEW(outputs[i]))
			for (j = 0; j < nd; j++)
				key.outputs[i].stride[j] = ((ccv_nnc_tensor_view_t*)outputs[i])->stride[j];
	}
	return key;
}

// Stream context
ccv_nnc_stream_context_t* ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	return stream_context;
}

static int enable_unbounded_command_buffers = 1;

void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context)
{
	os_unfair_lock_lock(&queue_lock);
	id<MTLCommandBuffer> command_buffer = last_command_buffer;
	last_command_buffer = nil;
	int i;
	const int buffer_size = enable_unbounded_command_buffers ? OLD_MAX_COMMAND_BUFFER_SIZE : OLD_LIMITED_COMMAND_BUFFER_SIZE;
	for (i = 0; i < buffer_size; i++)
	{
		[old_last_command_buffers[i] release];
		old_last_command_buffers[i] = nil;
	}
	os_unfair_lock_unlock(&queue_lock);
	[command_buffer waitUntilCompleted];
	[command_buffer release];
}

void ccv_nnc_stream_compat_add_callback(ccv_nnc_stream_context_t* const stream, const ccv_nnc_callback_f callback, const ccv_nnc_async_callback_f async_callback, void* const callback_context)
{
	os_unfair_lock_lock(&queue_lock);
	id<MTLCommandBuffer> command_buffer = [last_command_buffer retain];
	os_unfair_lock_unlock(&queue_lock);
	if (command_buffer == nil)
	{
		callback(callback_context);
		return;
	}
	[command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
		ccv_nnc_async_callback_t async = {
			.callback_context = callback_context,
			.fn = callback
		};
		async_callback(&async);
	}];
	[command_buffer release];
}

int co_stream_compat_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream)
{
	os_unfair_lock_lock(&queue_lock);
	id<MTLCommandBuffer> command_buffer = [last_command_buffer retain];
	os_unfair_lock_unlock(&queue_lock);
	if (command_buffer == nil)
		return 1;
	co_scheduler_t* const scheduler = self->scheduler;
	pthread_mutex_lock(&scheduler->mutex);
	++scheduler->stream_await_count;
	pthread_mutex_unlock(&scheduler->mutex);
	[command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
		pthread_mutex_lock(&scheduler->mutex);
		_co_prepend_task(scheduler, self);
		--scheduler->stream_await_count;
		pthread_cond_signal(&scheduler->wait);
		pthread_mutex_unlock(&scheduler->mutex);
	}];
	[command_buffer release];
	return 0;
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

void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_mps_t* stream_mps = (ccv_nnc_stream_mps_t*)stream_context;
	if (stream_mps->workspace)
		ccfree(stream_mps->workspace);
}

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
	ccv_nnc_stream_mps_t* stream_mps = (ccv_nnc_stream_mps_t*)stream_context;
	if (!stream_mps)
		stream_mps = &ccv_nnc_per_thread_stream_mps;
	if (stream_mps->workspace)
	{
		ccfree(stream_mps->workspace);
		stream_mps->workspace = 0;
		stream_mps->workspace_size = 0;
	}
}

// We don't need to support signal as of now because we share one queue. When we multiplex on multiple queues, we need to have a signal implementation.
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

void ccv_nnc_mps_unbounded_command_buffers(int state)
{
	enable_unbounded_command_buffers = state;
}

void ccv_nnc_stream_context_commit_command_buffer(ccv_nnc_stream_context_t* const stream_context, MPSCommandBuffer* command_buffer)
{
	int i;
	const int buffer_size = enable_unbounded_command_buffers ? OLD_MAX_COMMAND_BUFFER_SIZE : OLD_LIMITED_COMMAND_BUFFER_SIZE;
	if (!stream_context)
	{
		id<MTLCommandBuffer> committed_command_buffer = [command_buffer.commandBuffer retain];
		[command_buffer commit];
		os_unfair_lock_lock(&queue_lock);
		[last_command_buffer release];
		last_command_buffer = nil;
		for (i = 0; i < buffer_size; i++)
		{
			[old_last_command_buffers[i] release];
			old_last_command_buffers[i] = nil;
		}
		os_unfair_lock_unlock(&queue_lock);
		[committed_command_buffer waitUntilCompleted];
		[committed_command_buffer release];
		return;
	}
	os_unfair_lock_lock(&queue_lock);
	id<MTLCommandBuffer> old_last_command_buffer;
	if (buffer_size > 0)
	{
		old_last_command_buffer = old_last_command_buffers[0];
		for (i = 0; i < buffer_size - 1; i++)
			old_last_command_buffers[i] = old_last_command_buffers[i + 1];
		old_last_command_buffers[buffer_size - 1] = last_command_buffer;
	} else
		old_last_command_buffer = [command_buffer.commandBuffer retain];
	last_command_buffer = [command_buffer.commandBuffer retain];
	os_unfair_lock_unlock(&queue_lock);
	[command_buffer.commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
		os_unfair_lock_lock(&queue_lock);
		if (buffer == last_command_buffer)
		{
			[last_command_buffer release];
			last_command_buffer = nil;
		} else {
			int i;
			for (i = 0; i < buffer_size; i++)
				if (buffer == old_last_command_buffers[i])
				{
					[old_last_command_buffers[i] release];
					old_last_command_buffers[i] = nil;
					break;
				}
		}
		os_unfair_lock_unlock(&queue_lock);
	}];
	[command_buffer commit];
	// Wait if we need to bound how many in-flight command buffers there are. This helps memory usage.
	[old_last_command_buffer waitUntilCompleted];
	[old_last_command_buffer release];
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
		case CCV_8S:
			return MPSDataTypeInt8;
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
	const off_t offset = mpgetoffset((ccv_nnc_tensor_t*)tensor_view);
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
		NSInteger remaining_start = 0;
		if (offset)
		{
			partial_count = ccv_nnc_dimension_upper_bound(dim, stride);
			remaining_start = ccv_min(sorted_dim[0] * sorted_stride[0] - partial_count, offc);
			assert(remaining_start <= offc);
			full_count = offc - remaining_start + sorted_dim[0] * sorted_stride[0];
			desc = [graph placeholderWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
			*input = desc;
			desc = [graph sliceTensor:desc dimension:0 start:offc - remaining_start length:sorted_dim[0] * sorted_stride[0] name:nil];
			desc = [graph reshapeTensor:desc withShape:shape name:nil];
		} else {
			desc = [graph placeholderWithShape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype) name:nil];
			*input = desc;
		}
		[shape release];
		if (flag) // If we sliced this tensor before.
		{
			NSMutableArray<NSNumber*>* starts = [NSMutableArray new];
			NSMutableArray<NSNumber*>* ends = [NSMutableArray new];
			NSMutableArray<NSNumber*>* strides = [NSMutableArray new];
			for (i = 0; i < nd; i++)
			{
				NSInteger start = 0;
				if (full_dim[i] > sorted_dim[i])
				{
					start = ccv_min(remaining_start / sorted_stride[i], full_dim[i] - sorted_dim[i]);
					remaining_start -= start * sorted_stride[i];
				}
				[starts addObject:@(start)];
				[ends addObject:@(sorted_dim[i] + start)];
				[strides addObject:@(1)];
			}
			assert(remaining_start == 0);
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
			*input = desc;
		}
		[shape release];
		return desc;
	}
}

CCV_WARN_UNUSED(MPSGraphShapedType*) ccv_nnc_mps_graph_tensor_input_shape(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	const off_t offset = mpgetoffset((ccv_nnc_tensor_t*)tensor_view);
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
		int flag = 0;
		for (i = 1; i < nd; i++)
		{
			assert(sorted_stride[i - 1] % sorted_stride[i] == 0);
			full_dim[i] = sorted_stride[i - 1] / sorted_stride[i];
			if (!flag)
				flag = (full_dim[i] != sorted_dim[i]);
		}
		MPSGraphShapedType* shapedType;
		NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
		for (i = 0; i < nd; i++)
			[shape addObject:@(full_dim[i])];
		NSInteger remaining_start = 0;
		if (offset)
		{
			partial_count = ccv_nnc_dimension_upper_bound(dim, stride);
			remaining_start = ccv_min(sorted_dim[0] * sorted_stride[0] - partial_count, offc);
			assert(remaining_start <= offc);
			full_count = offc - remaining_start + sorted_dim[0] * sorted_stride[0];
			shapedType = [[MPSGraphShapedType alloc] initWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype)];
		} else {
			shapedType = [[MPSGraphShapedType alloc] initWithShape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype)];
		}
		[shape release];
		return [shapedType autorelease];
	} else {
		NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
		for (i = 0; i < nd; i++)
			[shape addObject:@(dim[i])];
		MPSGraphShapedType* shapedType;
		if (offset)
		{
			partial_count = dim[0];
			for (i = 1; i < nd; i++)
				partial_count *= dim[i];
			full_count = offc + partial_count;
			shapedType = [[MPSGraphShapedType alloc] initWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype)];
		} else {
			shapedType = [[MPSGraphShapedType alloc] initWithShape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype)];
		}
		[shape release];
		return [shapedType autorelease];
	}
}

MPSGraphTensorData* ccv_nnc_mps_graph_tensor_data(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	const off_t offset = mpgetoffset((ccv_nnc_tensor_t*)tensor_view);
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
			partial_count = ccv_nnc_dimension_upper_bound(dim, stride);
			NSInteger remaining_start = ccv_min(sorted_dim[0] * sorted_stride[0] - partial_count, offc);
			assert(remaining_start <= offc);
			full_count = offc - remaining_start + sorted_dim[0] * sorted_stride[0];
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
	id<MTLBuffer> buffer = mpgetbuffer((ccv_nnc_tensor_t*)tensor_view);
	MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:ccv_nnc_mps_datatype(tensor_view->info.datatype)];
	[shape release];
	return [data autorelease];
}

void ccv_nnc_mps_export_data(MPSGraphTensorData* data, MPSCommandBuffer* command_buffer, ccv_nnc_tensor_view_t* const tensor, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	id<MTLBuffer> buffer = mpgetbuffer((ccv_nnc_tensor_t*)tensor);
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
	off_t offset = mpgetoffset((ccv_nnc_tensor_t*)tensor);
	[ndarray exportDataWithCommandBuffer:command_buffer toBuffer:buffer destinationDataType:ccv_nnc_mps_datatype(tensor->info.datatype) offset:offset rowStrides:rowStrides];
}

void ccv_nnc_mps_graph_result(MPSGraph* graph, MPSCommandBuffer* command_buffer, MPSGraphTensorDataDictionary* feeds, MPSGraphTensor* output, ccv_nnc_tensor_view_t* const data, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	off_t offset = mpgetoffset((ccv_nnc_tensor_t*)data);
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

void ccv_nnc_mps_graph_executable_result(MPSGraphExecutable* executable, MPSCommandBuffer* command_buffer, NSArray<MPSGraphTensorData*>* inputsArray, ccv_nnc_tensor_view_t* const* const data, int* dim[CCV_NNC_MAX_DIM_ALLOC], int* stride[CCV_NNC_MAX_DIM_ALLOC], const int size)
{
	int i, flag = 0;
	for (i = 0; !flag && i < size; i++)
	{
		off_t offset = mpgetoffset((ccv_nnc_tensor_t*)data[i]);
		flag = !(CCV_IS_TENSOR_CONTIGUOUS(data[i]) && offset == 0);
	}
	if (!flag)
	{
		NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray new];
		for (i = 0; i < size; i++)
			[results addObject:ccv_nnc_mps_graph_tensor_data(data[i], dim[i], stride[i])];
		[executable encodeToCommandBuffer:command_buffer inputsArray:inputsArray resultsArray:results executionDescriptor:nil];
		[results release];
		return;
	}
	NSArray<MPSGraphTensorData*>* result = [executable encodeToCommandBuffer:command_buffer inputsArray:inputsArray resultsArray:nil executionDescriptor:nil];
	for (i = 0; i < size; i++)
	{
		MPSGraphTensorData* tensor_data = result[i];
		ccv_nnc_mps_export_data(tensor_data, command_buffer, data[i], dim[i], stride[i]);
	}
}
