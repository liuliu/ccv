#include "ccv_nnc_mps.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/_ccv_nnc_stream.h"
#include "nnc/ccv_nnc_easy.h"
#include "3rdparty/khash/khash.h"
#include <string.h>
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>
#import <os/lock.h>
#import <sys/stat.h>
#import <sys/utsname.h>
#import <sys/mman.h>
#import <mach/vm_page_size.h>

id<MTLDevice> ccv_nnc_default_device(void)
{
	static dispatch_once_t once;
	static id<MTLDevice> device;
	dispatch_once(&once, ^{
		device = MTLCreateSystemDefaultDevice();
	});
	return device;
}

@interface MTLFileBackedBuffer: NSObject
@property (nonatomic, copy) NSString* path;
@property (nonatomic, assign) NSUInteger size;
@property (nonatomic, assign) NSUInteger offset;
@end

@interface MTLWholeFileMapping: NSObject
@property (nonatomic, assign) void* base;
@property (nonatomic, assign) NSUInteger size;
@end

ccv_nnc_mfa_context_t* ccv_nnc_default_mfa_context(void)
{
	static dispatch_once_t once;
	static ccv_nnc_mfa_context_t* context;
	dispatch_once(&once, ^{
		context = ccv_nnc_init_mfa_context((__bridge mtl_device_t*)ccv_nnc_default_device());
	});
	return context;
}

MPSGraphDevice* ccv_nnc_default_mps_device(void)
{
	static dispatch_once_t once;
	static MPSGraphDevice* device;
	dispatch_once(&once, ^{
		device = [[MPSGraphDevice deviceWithMTLDevice:ccv_nnc_default_device()] retain];
	});
	return device;
}

#define CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MAGIC (0x43315657ULL << 32)
#define CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MASK (0xffffffffULL << 32)

static inline uint32_t* _ccv_nnc_mps_tensor_fast_fence_token(ccv_nnc_tensor_t* const tensor)
{
	return (uint32_t*)&tensor->sig;
}

static int _ccv_nnc_mps_tensor_fast_fence_pending_value(const ccv_nnc_tensor_t* const tensor, uint32_t* const pending_ref)
{
	const uint64_t sig = __atomic_load_n(&tensor->sig, __ATOMIC_ACQUIRE);
	if ((sig & CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MASK) != CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MAGIC)
		return 0;
	const uint32_t pending = (uint32_t)sig;
	if ((pending & 1) == 0)
		return 0;
	if (pending_ref)
		*pending_ref = pending;
	return 1;
}

int ccv_nnc_mps_tensor_fast_fence_mark_pending(ccv_nnc_tensor_t* const tensor)
{
	if (!ccv_nnc_mfa_prepare_fast_fence(ccv_nnc_default_mfa_context()))
		return 0;
	const uint64_t sig = __atomic_load_n(&tensor->sig, __ATOMIC_ACQUIRE);
	uint32_t pending;
	if ((sig & CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MASK) != CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MAGIC)
	{
		if (sig != 0)
			return 0;
		pending = 1;
		__atomic_store_n(&tensor->sig, CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MAGIC | pending, __ATOMIC_RELEASE);
		return 1;
	}
	uint32_t* const token = _ccv_nnc_mps_tensor_fast_fence_token(tensor);
	const uint32_t current = __atomic_load_n(token, __ATOMIC_ACQUIRE);
	pending = (current >= 0xfffffffdU) ? 1 : current + ((current & 1) ? 2 : 1);
	__atomic_store_n(token, pending, __ATOMIC_RELEASE);
	return 1;
}

int ccv_nnc_mps_tensor_fast_fence_pending(const ccv_nnc_tensor_t* const tensor)
{
	return _ccv_nnc_mps_tensor_fast_fence_pending_value(tensor, 0);
}

void ccv_nnc_mps_tensor_fast_fence_clear(ccv_nnc_tensor_t* const tensor)
{
	uint64_t sig = __atomic_load_n(&tensor->sig, __ATOMIC_ACQUIRE);
	for (;;)
	{
		if ((sig & CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MASK) != CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MAGIC)
			return;
		const uint32_t pending = (uint32_t)sig;
		if ((pending & 1) == 0)
			return;
		const uint64_t complete_sig = CCV_NNC_MPS_TENSOR_FAST_FENCE_SIG_MAGIC | (uint32_t)(pending + 1);
		if (__atomic_compare_exchange_n(&tensor->sig, &sig, complete_sig, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
			return;
	}
}

void ccv_nnc_mps_tensor_fast_fence_wait(ccv_nnc_tensor_t* const tensor)
{
	if (!_ccv_nnc_mps_tensor_fast_fence_pending_value(tensor, 0))
		return;
	uint32_t* const token = _ccv_nnc_mps_tensor_fast_fence_token(tensor);
	while (__atomic_load_n(token, __ATOMIC_ACQUIRE) & 1) {}
}

int ccv_nnc_mps_encode_tensor_fast_fence(MPSCommandBuffer* const command_buffer, ccv_nnc_tensor_t* const tensor, id<MTLBuffer> const buffer, unsigned char* const aligned_ptr, const size_t aligned_size, const off_t offset, const size_t size)
{
	uint32_t pending;
	if (!_ccv_nnc_mps_tensor_fast_fence_pending_value(tensor, &pending))
		return 0;
	const uint64_t word_offset_64 = (uint64_t)offset / sizeof(uint32_t);
	const uint64_t word_count_64 = ((uint64_t)(offset & (sizeof(uint32_t) - 1)) + size + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	if (word_offset_64 > UINT32_MAX || word_count_64 > UINT32_MAX || word_count_64 == 0)
		return 0;
	const uintptr_t aligned_start = (uintptr_t)aligned_ptr;
	const uintptr_t aligned_end = aligned_start + aligned_size;
	uint32_t* const token = _ccv_nnc_mps_tensor_fast_fence_token(tensor);
	const uintptr_t token_ptr = (uintptr_t)token;
	id<MTLBuffer> token_buffer = buffer;
	size_t token_offset = 0;
	int release_token_buffer = 0;
	if (token_ptr >= aligned_start && token_ptr + sizeof(uint32_t) <= aligned_end)
		token_offset = token_ptr - aligned_start;
	else {
		unsigned char* const token_aligned_ptr = (unsigned char*)(token_ptr & -vm_page_size);
		token_offset = token_ptr - (uintptr_t)token_aligned_ptr;
		token_buffer = [ccv_nnc_default_device() newBufferWithBytesNoCopy:token_aligned_ptr length:vm_page_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
		if (!token_buffer)
			return 0;
		release_token_buffer = 1;
	}
	MTLCommandBatch* const command_batch = ccv_nnc_start_command_batch_from_command_buffer((__bridge mtl_command_buffer_t*)command_buffer.commandBuffer, 0);
	ccv_nnc_mfa_fast_fence_params_t params = {
		.word_offset = (uint32_t)word_offset_64,
		.word_count = (uint32_t)word_count_64,
		.pending = pending,
		.complete = pending + 1,
	};
	mtl_buffer_t* tensors[] = {
		(__bridge mtl_buffer_t*)buffer,
		(__bridge mtl_buffer_t*)token_buffer,
		0
	};
	size_t tensor_offsets[] = {
		0,
		token_offset
	};
	const int encoded = ccv_nnc_mfa_encode_fast_fence(ccv_nnc_default_mfa_context(), params, command_batch, tensors, tensor_offsets);
	ccv_nnc_finish_command_batch(command_batch);
	if (release_token_buffer)
		[token_buffer release];
	return encoded;
}

static os_unfair_lock queue_lock;
static os_unfair_lock buffer_lock;
#define CCV_NNC_MPS_MAX_COMMAND_BUFFER_WATERMARK (32)
#define CCV_NNC_MPS_DEFAULT_COMMAND_BUFFER_WATERMARK (8)
static MPSCommandBuffer* current_mps_command_buffer;
static int current_mps_command_buffer_command_count;
static __thread MPSCommandBuffer* checked_out_mps_command_buffer;
static __thread int checked_out_mps_command_buffer_command_count;
static id<MTLCommandBuffer> old_last_command_buffers[CCV_NNC_MPS_MAX_COMMAND_BUFFER_WATERMARK];
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
	*(mpmp_t*)ccv_array_get(g_mp_h, slot) = (mpmp_t){};
	if (g_mp_slot < 0 || slot < g_mp_slot)
		g_mp_slot = slot;
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
	descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
#ifdef __x86_64__
	descriptor.storageMode = MTLStorageModePrivate;
#else
	descriptor.storageMode = MTLStorageModeShared;
#endif
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
#ifdef __x86_64__
	id<MTLBuffer> buffer = [ccv_nnc_default_device() newBufferWithLength:size options:MTLResourceStorageModePrivate];
#else
	id<MTLBuffer> buffer = [ccv_nnc_default_device() newBufferWithLength:size options:MTLResourceStorageModeShared];
#endif
	if (buffer == nil)
	{
		mptrigmp();
		buffer = [ccv_nnc_default_device() newBufferWithLength:size options:MTLResourceStorageModeShared];
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
#ifdef __x86_64__
	MTLSizeAndAlign sizeAndAlign = [ccv_nnc_default_device() heapBufferSizeAndAlignWithLength:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate];
	assert(offset % sizeAndAlign.align == 0);
	id<MTLBuffer> buffer = [heap newBufferWithLength:sizeAndAlign.size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate offset:offset];
#else
	MTLSizeAndAlign sizeAndAlign = [ccv_nnc_default_device() heapBufferSizeAndAlignWithLength:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
	assert(offset % sizeAndAlign.align == 0);
	id<MTLBuffer> buffer = [heap newBufferWithLength:sizeAndAlign.size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared offset:offset];
#endif
	if (buffer == nil)
	{
		mptrigmp();
#ifdef __x86_64__
		buffer = [heap newBufferWithLength:sizeAndAlign.size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate offset:offset];
#else
		buffer = [heap newBufferWithLength:sizeAndAlign.size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared offset:offset];
#endif
		assert(buffer != nil);
	}
	[buffer makeAliasable];
	return buffer;
}

@implementation MTLFileBackedBuffer
@end

@implementation MTLWholeFileMapping

- (void)dealloc
{
	if (_base && _size)
		munmap(_base, _size);
	[super dealloc];
}

@end

static MTLWholeFileMapping* _ccv_nnc_mps_whole_file_mapping(const char* const file, const size_t required_size, const off_t offset)
{
	static dispatch_once_t once;
	static NSMapTable* mappings;
	static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
	dispatch_once(&once, ^{
		mappings = [[NSMapTable alloc] initWithKeyOptions:NSPointerFunctionsStrongMemory valueOptions:NSPointerFunctionsWeakMemory capacity:1];
	});
	int fd = open(file, O_RDONLY, 0);
	if (fd < 0)
		return nil;
	struct stat status;
	if (fstat(fd, &status) != 0 || status.st_size <= 0 || (uint64_t)status.st_size > NSUIntegerMax || offset < 0 ||
		(uint64_t)offset > (uint64_t)status.st_size || required_size > (uint64_t)status.st_size - (uint64_t)offset)
	{
		close(fd);
		return nil;
	}
	NSString* const key = [NSString stringWithFormat:@"%llu:%llu:%llu",
		(unsigned long long)status.st_dev,
		(unsigned long long)status.st_ino,
		(unsigned long long)status.st_size];
	os_unfair_lock_lock(&lock);
	MTLWholeFileMapping* mapping = [[mappings objectForKey:key] retain];
	if (!mapping)
	{
		void* const base = mmap(0, status.st_size, PROT_READ, MAP_SHARED, fd, 0);
		if (base != MAP_FAILED)
		{
			madvise(base, status.st_size, MADV_NORMAL);
			mapping = [MTLWholeFileMapping new];
			mapping.base = base;
			mapping.size = status.st_size;
			[mappings setObject:mapping forKey:key];
		}
	}
	os_unfair_lock_unlock(&lock);
	close(fd);
	return mapping;
}

static char _ccv_nnc_mps_whole_file_mapping_owner_key;

id<MTLBuffer> mpgetbuffer(const ccv_nnc_tensor_t* const tensor)
{
	id obj = (id)tensor->data.u8;
	if ([obj isKindOfClass:[MTLFileBackedBuffer class]])
	{
		MTLFileBackedBuffer* fileBackedBuffer = (MTLFileBackedBuffer*)obj;
		int fd = open(fileBackedBuffer.path.UTF8String, O_RDONLY, 0);
		size_t size = fileBackedBuffer.size;
		off_t offset = (off_t)fileBackedBuffer.offset;
		void* bufptr = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, offset);
		close(fd);
		unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)bufptr & -vm_page_size);
		assert(aligned_ptr == bufptr);
		madvise(bufptr, size, MADV_SEQUENTIAL | MADV_WILLNEED);
		if (ccv_nnc_flags() & CCV_NNC_DISABLE_MMAP_MTL_BUFFER)
		{
#ifdef __x86_64__
			obj = [[ccv_nnc_default_device() newBufferWithBytes:bufptr length:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate] autorelease];
#else
			obj = [[ccv_nnc_default_device() newBufferWithBytes:bufptr length:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared] autorelease];
#endif
			munmap(bufptr, size);
		} else
			obj = [[ccv_nnc_default_device() newBufferWithBytesNoCopy:bufptr length:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:^(void *ptr, NSUInteger len) {
				munmap(ptr, len);
			}] autorelease];
	}
	return (id<MTLBuffer>)obj;
}

off_t mpgetoffset(const ccv_nnc_tensor_t* const tensor)
{
	return tensor->dataof;
}

void* mpmemmap(const char* file, const size_t size, const off_t offset, const int flags)
{
	@autoreleasepool {
		if (flags & CCV_NNC_TENSOR_MEMORY_MAP_ON_DEMAND)
		{
			// ON_DEMAND takes precedence over WHOLE_FILE. Combining the two is not
			// specialized yet and retains the existing per-range lazy behavior.
			MTLFileBackedBuffer* fileBackedBuffer = [MTLFileBackedBuffer new];
			fileBackedBuffer.path = @(file);
			fileBackedBuffer.size = size;
			fileBackedBuffer.offset = offset;
			assert(offset % vm_page_size == 0);
			return (void*)fileBackedBuffer;
		} else {
			if ((flags & CCV_NNC_TENSOR_MEMORY_MAP_WHOLE_FILE) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MMAP_MTL_BUFFER))
			{
				MTLWholeFileMapping* const mapping = _ccv_nnc_mps_whole_file_mapping(file, size, offset);
				if (mapping)
				{
					void* const bufptr = (unsigned char*)mapping.base + offset;
					unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)bufptr & -vm_page_size);
					assert(aligned_ptr == bufptr);
					id obj = [ccv_nnc_default_device() newBufferWithBytesNoCopy:bufptr length:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:nil];
					if (obj)
						objc_setAssociatedObject(obj, &_ccv_nnc_mps_whole_file_mapping_owner_key, mapping, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
					[mapping release];
					if (obj)
						return obj;
				}
			}
			int fd = open(file, O_RDONLY, 0);
			void* bufptr = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, offset);
			close(fd);
			unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)bufptr & -vm_page_size);
			assert(aligned_ptr == bufptr);
			madvise(bufptr, size, MADV_SEQUENTIAL | MADV_WILLNEED);
			id obj;
			if (ccv_nnc_flags() & CCV_NNC_DISABLE_MMAP_MTL_BUFFER)
			{
#ifdef __x86_64__
				obj = [ccv_nnc_default_device() newBufferWithBytes:bufptr length:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate];
#else
				obj = [ccv_nnc_default_device() newBufferWithBytes:bufptr length:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
#endif
				munmap(bufptr, size);
			} else
				obj = [ccv_nnc_default_device() newBufferWithBytesNoCopy:bufptr length:size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared deallocator:^(void *ptr, NSUInteger len) {
					munmap(ptr, len);
				}];
			return obj;
		}
	}
}

void mpmemcpy(void* dest, const off_t dest_off, const int dest_type, const void* src, const off_t src_off, const int src_type, size_t n)
{
	if (n <= 0)
		return;
	if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_GPU_MEMORY)
	{
		unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)src & -vm_page_size);
		const off_t offset_a = (uintptr_t)src - (uintptr_t)aligned_ptr + src_off;
		const size_t aligned_size = ((n + offset_a + vm_page_size - 1) & -vm_page_size);
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
		unsigned char* const aligned_ptr = (unsigned char*)((uintptr_t)dest & -vm_page_size);
		const off_t offset_b = (uintptr_t)dest - (uintptr_t)aligned_ptr;
		const size_t aligned_size = ((n + offset_b + vm_page_size - 1) & -vm_page_size);
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
	h = twang_32from64(((uint64_t)h << 32) | key.index);
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
	if (a.cmd != b.cmd || a.flags != b.flags || a.index != b.index || a.input_size != b.input_size || a.output_size != b.output_size)
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
	ccv_nnc_mfa_clear_pipeline_cache(ccv_nnc_default_mfa_context());
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
		compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel0;
		compilationDescriptor.optimizationProfile = MPSGraphOptimizationProfilePerformance;
		MPSGraphExecutable* executable = [[graph compileWithDevice:ccv_nnc_default_mps_device() feeds:[NSDictionary dictionaryWithObjects:inputShapedTypes forKeys:inputTensors] targetTensors:targetTensors targetOperations:nil compilationDescriptor:compilationDescriptor] retain];
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

ccv_nnc_mps_graph_key_t ccv_nnc_mps_graph_key_new(const ccv_nnc_cmd_t cmd, const int index, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	ccv_nnc_mps_graph_key_t key = {
		.cmd = cmd.cmd,
		.index = index,
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
	// Initialize the MFA context.
	ccv_nnc_default_mfa_context();
	return stream_context;
}

static int command_buffers_watermark = CCV_NNC_MPS_DEFAULT_COMMAND_BUFFER_WATERMARK;

static int _ccv_nnc_mps_encoders_per_command_buffer(void)
{
	return ccv_max(1, (command_buffers_watermark + CCV_NNC_MPS_MAX_COMMAND_BUFFER_WATERMARK - 1) / CCV_NNC_MPS_MAX_COMMAND_BUFFER_WATERMARK);
}

static int _ccv_nnc_mps_tracked_command_buffer_watermark(void)
{
	return ccv_min(command_buffers_watermark, CCV_NNC_MPS_MAX_COMMAND_BUFFER_WATERMARK);
}

static int _ccv_nnc_mps_current_command_buffer_needs_commit(const int command_count)
{
	const int encoders_per_command_buffer = _ccv_nnc_mps_encoders_per_command_buffer();
	return command_count >= encoders_per_command_buffer;
}

static void _ccv_nnc_mps_reset_current_command_buffer_accounting(void)
{
	current_mps_command_buffer_command_count = 0;
}

static void _ccv_nnc_mps_reset_checked_out_command_buffer_accounting(void)
{
	checked_out_mps_command_buffer_command_count = 0;
}

static void _ccv_nnc_mps_release_tracked_command_buffer(id<MTLCommandBuffer> mtl_command_buffer, const int buffer_size)
{
	id<MTLCommandBuffer> found_buffer = nil;
	os_unfair_lock_lock(&queue_lock);
	if (mtl_command_buffer == last_command_buffer)
	{
		found_buffer = last_command_buffer;
		last_command_buffer = nil;
	} else {
		int i;
		for (i = 0; i < buffer_size; i++)
			if (mtl_command_buffer == old_last_command_buffers[i])
			{
				found_buffer = old_last_command_buffers[i];
				old_last_command_buffers[i] = nil;
				break;
			}
	}
	os_unfair_lock_unlock(&queue_lock);
	[found_buffer release];
}

static void _ccv_nnc_mps_track_submitted_command_buffer(id<MTLCommandBuffer> mtl_command_buffer, const int buffer_size)
{
	if (!mtl_command_buffer)
		return;
	int i;
	os_unfair_lock_lock(&queue_lock);
	id<MTLCommandBuffer> old_last_command_buffer;
	if (buffer_size > 0)
	{
		old_last_command_buffer = old_last_command_buffers[0];
		for (i = 0; i < buffer_size - 1; i++)
			old_last_command_buffers[i] = old_last_command_buffers[i + 1];
		old_last_command_buffers[buffer_size - 1] = last_command_buffer;
	} else
		old_last_command_buffer = [mtl_command_buffer retain];
	last_command_buffer = [mtl_command_buffer retain];
	os_unfair_lock_unlock(&queue_lock);

	if (mtl_command_buffer.status >= MTLCommandBufferStatusCompleted)
		_ccv_nnc_mps_release_tracked_command_buffer(mtl_command_buffer, buffer_size);
	[old_last_command_buffer waitUntilCompleted];
	[old_last_command_buffer release];
}

static void _ccv_nnc_mps_commit_command_buffer(MPSCommandBuffer* const command_buffer)
{
	if (!command_buffer)
		return;
	id<MTLCommandBuffer> const mtl_command_buffer = [command_buffer.commandBuffer retain];
	const int buffer_size = _ccv_nnc_mps_tracked_command_buffer_watermark();
	[mtl_command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
		_ccv_nnc_mps_release_tracked_command_buffer(buffer, buffer_size);
	}];
	[command_buffer commit];
	_ccv_nnc_mps_track_submitted_command_buffer(mtl_command_buffer, buffer_size);
	[command_buffer release];
	[mtl_command_buffer release];
}

static void _ccv_nnc_mps_account_current_command_buffer(id<MTLCommandBuffer> const mtl_command_buffer, const int command_count, const int may_commit)
{
	if (!mtl_command_buffer || !checked_out_mps_command_buffer || checked_out_mps_command_buffer.commandBuffer != mtl_command_buffer)
		return;
	if (command_count > 0)
		checked_out_mps_command_buffer_command_count += command_count;
	if (!may_commit)
		return;
	MPSCommandBuffer* command_buffer = nil;
	const int needs_commit = _ccv_nnc_mps_current_command_buffer_needs_commit(checked_out_mps_command_buffer_command_count);
	os_unfair_lock_lock(&buffer_lock);
	if (!current_mps_command_buffer && !needs_commit)
	{
		current_mps_command_buffer = checked_out_mps_command_buffer;
		current_mps_command_buffer_command_count = checked_out_mps_command_buffer_command_count;
		checked_out_mps_command_buffer = nil;
		_ccv_nnc_mps_reset_checked_out_command_buffer_accounting();
	} else {
		command_buffer = checked_out_mps_command_buffer;
		checked_out_mps_command_buffer = nil;
		_ccv_nnc_mps_reset_checked_out_command_buffer_accounting();
	}
	os_unfair_lock_unlock(&buffer_lock);
	_ccv_nnc_mps_commit_command_buffer(command_buffer);
}

void ccv_nnc_stream_compat_commit(ccv_nnc_stream_context_t* const stream_context)
{
	MPSCommandBuffer* command_buffer = nil;
	os_unfair_lock_lock(&buffer_lock);
	command_buffer = current_mps_command_buffer;
	current_mps_command_buffer = nil;
	_ccv_nnc_mps_reset_current_command_buffer_accounting();
	os_unfair_lock_unlock(&buffer_lock);
	_ccv_nnc_mps_commit_command_buffer(command_buffer);
}

void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_compat_commit((ccv_nnc_stream_context_t*)stream_context);
	os_unfair_lock_lock(&queue_lock);
	id<MTLCommandBuffer> command_buffer = last_command_buffer;
	last_command_buffer = nil;
	int i;
	const int buffer_size = _ccv_nnc_mps_tracked_command_buffer_watermark();
	id<MTLCommandBuffer> old_buffers[buffer_size];
	for (i = 0; i < buffer_size; i++)
	{
		old_buffers[i] = old_last_command_buffers[i];
		old_last_command_buffers[i] = nil;
	}
	os_unfair_lock_unlock(&queue_lock);
	[command_buffer waitUntilCompleted];
	[command_buffer release];
	for (i = 0; i < buffer_size; i++)
		if (old_buffers[i])
		{
			[old_buffers[i] waitUntilCompleted];
			[old_buffers[i] release];
		}
}

void ccv_nnc_stream_compat_add_callback(ccv_nnc_stream_context_t* const stream, const ccv_nnc_callback_f callback, const ccv_nnc_async_callback_f async_callback, void* const callback_context)
{
	ccv_nnc_stream_compat_commit(stream);
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
	ccv_nnc_stream_compat_commit(stream);
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
	ccv_nnc_stream_compat_commit(stream_context);
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
	ccv_nnc_stream_compat_commit(stream_context);
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

MTLCommandBatch* ccv_nnc_stream_context_start_command_batch(ccv_nnc_stream_context_t* const stream_context)
{
	if (stream_context)
	{
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		return ccv_nnc_start_command_batch_from_command_buffer((__bridge mtl_command_buffer_t*)command_buffer.commandBuffer, 0);
	}
	ccv_nnc_stream_compat_commit(0);
	return ccv_nnc_start_command_batch((__bridge mtl_command_queue_t*)_ccv_nnc_default_queue());
}

MPSCommandBuffer* ccv_nnc_stream_context_start_mps_command_buffer(ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
	{
		ccv_nnc_stream_compat_commit(0);
		return [MPSCommandBuffer commandBufferFromCommandQueue:_ccv_nnc_default_queue()];
	}
	if (checked_out_mps_command_buffer)
		return [[checked_out_mps_command_buffer retain] autorelease];
	os_unfair_lock_lock(&buffer_lock);
	MPSCommandBuffer* command_buffer = current_mps_command_buffer;
	current_mps_command_buffer = nil;
	checked_out_mps_command_buffer_command_count = current_mps_command_buffer_command_count;
	_ccv_nnc_mps_reset_current_command_buffer_accounting();
	os_unfair_lock_unlock(&buffer_lock);
	if (!command_buffer)
	{
		command_buffer = [[MPSCommandBuffer commandBufferFromCommandQueue:_ccv_nnc_default_queue()] retain];
		_ccv_nnc_mps_reset_checked_out_command_buffer_accounting();
	}
	checked_out_mps_command_buffer = command_buffer;
	return [[command_buffer retain] autorelease];
}

int ccv_nnc_mps_queue_watermark(void)
{
	return command_buffers_watermark;
}

void ccv_nnc_mps_set_queue_watermark(int watermark)
{
	command_buffers_watermark = ccv_max(watermark, 0);
}

void ccv_nnc_mps_set_binary_artifacts(const char** const paths_to_read, const int paths_to_read_size, const char* const path_to_write)
{
	ccv_nnc_mfa_set_binary_archives(ccv_nnc_default_mfa_context(), paths_to_read, paths_to_read_size, path_to_write);
}

void ccv_nnc_stream_context_finish_command_buffer(ccv_nnc_stream_context_t* const stream_context, MPSCommandBuffer* mps_command_buffer, MTLCommandBatch* command_batch)
{
	if (stream_context)
	{
		if (command_batch)
		{
			id<MTLCommandBuffer> const command_buffer = (id<MTLCommandBuffer>)command_batch->command_buffer;
			const int command_count = command_batch->batched_command_count;
			ccv_nnc_finish_command_batch(command_batch);
			_ccv_nnc_mps_account_current_command_buffer(command_buffer, command_count, 1);
		} else
			_ccv_nnc_mps_account_current_command_buffer(mps_command_buffer.commandBuffer, 1, 1);
		return;
	}
	id<MTLCommandBuffer> mtl_command_buffer = mps_command_buffer ? mps_command_buffer.commandBuffer : (id<MTLCommandBuffer>)command_batch->command_buffer;
	id<MTLCommandBuffer> committed_command_buffer = [mtl_command_buffer retain];
	if (mps_command_buffer)
		[mps_command_buffer commit];
	else
		ccv_nnc_finish_command_batch(command_batch);
	[committed_command_buffer waitUntilCompleted];
	[committed_command_buffer release];
}

void ccv_nnc_stream_context_finish_command_batch(ccv_nnc_stream_context_t* const stream_context, MTLCommandBatch* command_batch)
{
	ccv_nnc_stream_context_finish_command_buffer(stream_context, nil, command_batch);
}

void ccv_nnc_stream_context_finish_mps_command_buffer(ccv_nnc_stream_context_t* const stream_context, MPSCommandBuffer* command_buffer)
{
	ccv_nnc_stream_context_finish_command_buffer(stream_context, command_buffer, NULL);
}

MPSCommandBuffer* ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(ccv_nnc_stream_context_t* const stream_context, MTLCommandBatch* command_batch)
{
	id<MTLCommandBuffer> command_buffer = (id<MTLCommandBuffer>)command_batch->command_buffer;
	const int command_count = command_batch->batched_command_count;
	command_batch->command_buffer = 0;
	ccv_nnc_finish_command_batch(command_batch);
	if (stream_context)
		_ccv_nnc_mps_account_current_command_buffer(command_buffer, command_count, 0);
	MPSCommandBuffer* const mps_command_buffer = [MPSCommandBuffer commandBufferWithCommandBuffer:command_buffer];
	if (stream_context)
	{
		if (checked_out_mps_command_buffer && checked_out_mps_command_buffer.commandBuffer == command_buffer)
		{
			[mps_command_buffer retain];
			[checked_out_mps_command_buffer release];
			checked_out_mps_command_buffer = mps_command_buffer;
		}
	}
	return mps_command_buffer;
}

MPSDataType ccv_nnc_mps_datatype(int datatype)
{
	if (CCV_GET_DATA_TYPE(datatype) == CCV_QX)
		datatype = (datatype & 0xff) << 12;
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
		case CCV_16BF:
			return MPSDataTypeBFloat16;
		case CCV_32F:
			return MPSDataTypeFloat32;
		case CCV_QX:
			assert(0 && "doesn't support quantization");
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
	const int datatype = CCV_GET_DATA_TYPE(tensor_view->info.datatype) == CCV_QX ? ((tensor_view->info.datatype & 0xff) << 12) : tensor_view->info.datatype;
	assert(offset % (CCV_GET_DATA_TYPE_SIZE(datatype)) == 0);
	const off_t offc = offset / CCV_GET_DATA_TYPE_SIZE(datatype);
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
				if ((sorted_stride[idx] < sorted_stride[j]) || (sorted_stride[idx] == sorted_stride[j] && sorted_dim[idx] < sorted_dim[j]))
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
			desc = [graph placeholderWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(datatype) name:nil];
			*input = desc;
			desc = [graph sliceTensor:desc dimension:0 start:offc - remaining_start length:sorted_dim[0] * sorted_stride[0] name:nil];
			desc = [graph reshapeTensor:desc withShape:shape name:nil];
		} else {
			desc = [graph placeholderWithShape:shape dataType:ccv_nnc_mps_datatype(datatype) name:nil];
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
			desc = [graph placeholderWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(datatype) name:nil];
			*input = desc;
			desc = [graph sliceTensor:desc dimension:0 start:offc length:partial_count name:nil];
			desc = [graph reshapeTensor:desc withShape:shape name:nil];
		} else {
			desc = [graph placeholderWithShape:shape dataType:ccv_nnc_mps_datatype(datatype) name:nil];
			*input = desc;
		}
		[shape release];
		return desc;
	}
}

CCV_WARN_UNUSED(MPSGraphShapedType*) ccv_nnc_mps_graph_tensor_input_shape(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	const off_t offset = mpgetoffset((ccv_nnc_tensor_t*)tensor_view);
	const int datatype = CCV_GET_DATA_TYPE(tensor_view->info.datatype) == CCV_QX ? ((tensor_view->info.datatype & 0xff) << 12) : tensor_view->info.datatype;
	assert(offset % (CCV_GET_DATA_TYPE_SIZE(datatype)) == 0);
	const off_t offc = offset / CCV_GET_DATA_TYPE_SIZE(datatype);
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
				if ((sorted_stride[idx] < sorted_stride[j]) || (sorted_stride[idx] == sorted_stride[j] && sorted_dim[idx] < sorted_dim[j]))
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
			shapedType = [[MPSGraphShapedType alloc] initWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(datatype)];
		} else {
			shapedType = [[MPSGraphShapedType alloc] initWithShape:shape dataType:ccv_nnc_mps_datatype(datatype)];
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
			shapedType = [[MPSGraphShapedType alloc] initWithShape:@[@(full_count)] dataType:ccv_nnc_mps_datatype(datatype)];
		} else {
			shapedType = [[MPSGraphShapedType alloc] initWithShape:shape dataType:ccv_nnc_mps_datatype(datatype)];
		}
		[shape release];
		return [shapedType autorelease];
	}
}

MPSGraphTensorData* ccv_nnc_mps_graph_tensor_data_with_buffer(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC], void* buffer, const off_t offset)
{
	const int datatype = CCV_GET_DATA_TYPE(tensor_view->info.datatype) == CCV_QX ? ((tensor_view->info.datatype & 0xff) << 12) : tensor_view->info.datatype;
	assert(offset % (CCV_GET_DATA_TYPE_SIZE(datatype)) == 0);
	const off_t offc = offset / CCV_GET_DATA_TYPE_SIZE(datatype);
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
				if ((sorted_stride[idx] < sorted_stride[j]) || (sorted_stride[idx] == sorted_stride[j] && sorted_dim[idx] < sorted_dim[j]))
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
	MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithMTLBuffer:(id<MTLBuffer>)buffer shape:shape dataType:ccv_nnc_mps_datatype(datatype)];
	[shape release];
	return [data autorelease];
}

MPSGraphTensorData* ccv_nnc_mps_graph_tensor_data(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	return ccv_nnc_mps_graph_tensor_data_with_buffer(tensor_view, dim, stride, mpgetbuffer((ccv_nnc_tensor_t*)tensor_view), mpgetoffset((ccv_nnc_tensor_t*)tensor_view));
}

static MPSGraphTensorData* ccv_nnc_mps_graph_output_tensor_data(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	const int nd = ccv_nnc_tensor_nd(dim);
	int i;
	NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
	const int datatype = CCV_GET_DATA_TYPE(tensor_view->info.datatype) == CCV_QX ? ((tensor_view->info.datatype & 0xff) << 12) : tensor_view->info.datatype;
	assert(CCV_IS_TENSOR_CONTIGUOUS(tensor_view));
	assert(mpgetoffset((ccv_nnc_tensor_t*)tensor_view) == 0);
	for (i = 0; i < nd; i++)
		[shape addObject:@(dim[i])];
	void* buffer = mpgetbuffer((ccv_nnc_tensor_t*)tensor_view);
	MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithMTLBuffer:(id<MTLBuffer>)buffer shape:shape dataType:ccv_nnc_mps_datatype(datatype)];
	[shape release];
	return [data autorelease];
}

MPSGraphTensorData* ccv_nnc_mps_graph_constant_data(const float val, const int datatype)
{
	id<MTLBuffer> buffer;
	assert(datatype == CCV_16F || datatype == CCV_32F || datatype == CCV_16BF);
	if (datatype == CCV_16F)
	{
		uint16_t half_bytes;
		ccv_float_to_half_precision(&val, &half_bytes, 1);
#ifdef __x86_64__
		buffer = [ccv_nnc_default_device() newBufferWithLength:sizeof(uint16_t) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked];
		mpmemcpy(buffer, 0, CCV_TENSOR_GPU_MEMORY, &half_bytes, 0, CCV_TENSOR_CPU_MEMORY, sizeof(uint16_t));
#else
		buffer = [ccv_nnc_default_device() newBufferWithBytes:&half_bytes length:sizeof(uint16_t) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
#endif
	} else if (datatype == CCV_16BF) {
		uint16_t half_bytes;
		ccv_float_to_bfloat(&val, &half_bytes, 1);
#ifdef __x86_64__
		buffer = [ccv_nnc_default_device() newBufferWithLength:sizeof(uint16_t) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked];
		mpmemcpy(buffer, 0, CCV_TENSOR_GPU_MEMORY, &half_bytes, 0, CCV_TENSOR_CPU_MEMORY, sizeof(uint16_t));
#else
		buffer = [ccv_nnc_default_device() newBufferWithBytes:&half_bytes length:sizeof(uint16_t) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
#endif
	} else {
#ifdef __x86_64__
		buffer = [ccv_nnc_default_device() newBufferWithLength:sizeof(float) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked];
		mpmemcpy(buffer, 0, CCV_TENSOR_GPU_MEMORY, &val, 0, CCV_TENSOR_CPU_MEMORY, sizeof(float));
#else
		buffer = [ccv_nnc_default_device() newBufferWithBytes:&val length:sizeof(float) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
#endif
	}
	MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:@[@1] dataType:ccv_nnc_mps_datatype(datatype)];
	[buffer release];
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
	MPSGraphCompilationDescriptor* compilationDescriptor = [MPSGraphCompilationDescriptor new];
	// Need more investigation into what this does.
	compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel0;
	compilationDescriptor.optimizationProfile = MPSGraphOptimizationProfilePerformance;
	MPSGraphExecutionDescriptor* executionDescriptor = [MPSGraphExecutionDescriptor new];
	executionDescriptor.compilationDescriptor = compilationDescriptor;
	if (CCV_IS_TENSOR_CONTIGUOUS(data) && offset == 0)
	{
		MPSGraphTensorData* tensor_data = ccv_nnc_mps_graph_tensor_data(data, dim, stride);
		[graph encodeToCommandBuffer:command_buffer feeds:feeds targetOperations:nil resultsDictionary:@{output: tensor_data} executionDescriptor:executionDescriptor];
		[executionDescriptor release];
		[compilationDescriptor release];
		return;
	}
	MPSGraphTensorDataDictionary* result = [graph encodeToCommandBuffer:command_buffer feeds:feeds targetTensors:@[output] targetOperations:nil executionDescriptor:executionDescriptor];
	[executionDescriptor release];
	[compilationDescriptor release];
	MPSGraphTensorData* tensor_data = result[output];
	ccv_nnc_mps_export_data(tensor_data, command_buffer, data, dim, stride);
}

void ccv_nnc_mps_graph_executable_result(MPSGraphExecutable* executable, MPSCommandBuffer* command_buffer, NSArray<MPSGraphTensorData*>* inputsArray, ccv_nnc_tensor_view_t* const* const data, int* dim[CCV_NNC_MAX_DIM_ALLOC], int* stride[CCV_NNC_MAX_DIM_ALLOC], const int size, const int force_export_data)
{
	int i, flag = force_export_data;
	for (i = 0; !flag && i < size; i++)
	{
		off_t offset = mpgetoffset((ccv_nnc_tensor_t*)data[i]);
		flag = !(CCV_IS_TENSOR_CONTIGUOUS(data[i]) && offset == 0);
	}
	if (!flag)
	{
		NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray new];
		for (i = 0; i < size; i++)
			[results addObject:ccv_nnc_mps_graph_output_tensor_data(data[i], dim[i], stride[i])];
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
