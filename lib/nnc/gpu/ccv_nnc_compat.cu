#include "ccv_nnc_compat.h"
extern "C" {
#include <nnc/ccv_nnc_easy.h>
#include <nnc/_ccv_nnc_stream.h>
}

typedef struct {
	cump_f func;
	void* ctx;
} cump_t;

static pthread_mutex_t g_mp_mutex = PTHREAD_MUTEX_INITIALIZER;
static ccv_array_t* g_mp_h;
static int g_mp_slot;

int curegmp(cump_f func, void* const context)
{
	assert(func);
	pthread_mutex_lock(&g_mp_mutex);
	if (!g_mp_h)
	{
		g_mp_h = ccv_array_new(sizeof(cump_t), 1, 0);
		g_mp_slot = -1;
	}
	cump_t mp = {
		func, context,
	};
	int slot = g_mp_slot;
	if (g_mp_slot >= 0)
	{
		assert(g_mp_slot < g_mp_h->rnum);
		*(cump_t*)ccv_array_get(g_mp_h, g_mp_slot) = mp;
		int i;
		for (i = g_mp_slot + 1; i < g_mp_h->rnum; i++)
			if (((cump_t*)ccv_array_get(g_mp_h, i))->func == 0)
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

void cuunregmp(const int slot)
{
	pthread_mutex_lock(&g_mp_mutex);
	assert(slot < g_mp_h->rnum);
	if (g_mp_slot < 0 || slot < g_mp_slot)
		g_mp_slot = slot;
	*(cump_t*)ccv_array_get(g_mp_h, g_mp_slot) = (cump_t){};
	pthread_mutex_unlock(&g_mp_mutex);
}

void cutrigmp(void)
{
	int device_id;
	CUDA_ENFORCE(cudaGetDevice(&device_id));
	pthread_mutex_lock(&g_mp_mutex);
	int i;
	for (i = 0; i < g_mp_h->rnum; i++)
	{
		cump_t* const mp = (cump_t*)ccv_array_get(g_mp_h, i);
		if (mp->func)
			mp->func(mp->ctx);
	}
	pthread_mutex_unlock(&g_mp_mutex);
	// Set back the device id.
	CUDA_ENFORCE(cudaSetDevice(device_id));
}

void* cumalloc(int device, size_t size)
{
	void* ptr = 0;
	CUDA_ENFORCE(cudaSetDevice(device));
	cudaMalloc(&ptr, size);
	if (ptr == 0)
	{
		cutrigmp(); // Trigger memory pressure. And then do it again.
		cudaMalloc(&ptr, size);
	}
	return ptr;
}

void cufree(int device, void* ptr)
{
	CUDA_ENFORCE(cudaSetDevice(device));
	CUDA_ENFORCE(cudaFree(ptr));
}

void cudevice(int device)
{
	if (device >= 0)
		CUDA_ENFORCE(cudaSetDevice(device));
}

void cumemcpy(void* dest, const int dest_type, const void* src, const int src_type, size_t n)
{
	if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_GPU_MEMORY) {
		const int device_b = CCV_TENSOR_GET_DEVICE_ID(dest_type);
		CUDA_ENFORCE(cudaSetDevice(device_b));
		CUDA_ENFORCE(cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice));
	} else if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_CPU_MEMORY) {
		const int device_a = CCV_TENSOR_GET_DEVICE_ID(src_type);
		CUDA_ENFORCE(cudaSetDevice(device_a));
		CUDA_ENFORCE(cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost));
	} else if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_CPU_MEMORY)
		CUDA_ENFORCE(cudaMemcpy(dest, src, n, cudaMemcpyHostToHost));
	else if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_GPU_MEMORY) {
		const int device_a = CCV_TENSOR_GET_DEVICE_ID(src_type);
		const int device_b = CCV_TENSOR_GET_DEVICE_ID(dest_type);
		CUDA_ENFORCE(cudaSetDevice(device_b));
		if (device_a == device_b)
			CUDA_ENFORCE(cudaMemcpy(dest, src, n, cudaMemcpyDeviceToDevice));
		else
			CUDA_ENFORCE(cudaMemcpyPeer(dest, device_b, src, device_a, n));
	}
}

void* cuhostalloc(size_t size)
{
	void* ptr = 0;
	cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
	return ptr;
}

void cuhostfree(void* ptr)
{
	cudaFreeHost(ptr);
}

int curegister(void* ptr, size_t size)
{
	return cudaSuccess == cudaHostRegister(ptr, size, cudaHostRegisterPortable);
}

void cuunregister(void* ptr)
{
	CUDA_ENFORCE(cudaHostUnregister(ptr));
}

typedef struct {
	cudaStream_t stream;
	cublasHandle_t cublas;
	struct {
		int n;
		__half* data;
	} ones_16;
	struct {
		int n;
		float* data;
	} ones_32;
	struct {
		int n;
		double* data;
	} ones_64;
	size_t workspace_size;
	void* workspace;
#ifdef HAVE_CUDNN
	cudnnHandle_t cudnn;
	void* rngs; // user-allocated GPU memory that will hold random number generator states.
#endif
} ccv_nnc_stream_context_device_local_t;

typedef struct {
#ifdef HAVE_NCCL
	ncclComm_t* comms;
	int comm_count;
#endif
} ccv_nnc_stream_resource_container_compat_t;

typedef struct {
	ccv_nnc_stream_context_t super;
	struct {
		size_t workspace_size;
		void* workspace;
	} cpu;
	union {
		ccv_nnc_stream_context_device_local_t _inline_gpu;
		struct {
			ccv_nnc_stream_context_device_local_t* _heap_gpus;
			int _heap_gpu_size;
		};
	};
} ccv_nnc_stream_context_compat_t;

static ccv_nnc_stream_context_device_local_t* _ccv_nnc_stream_compat_device_local(ccv_nnc_stream_context_compat_t* const stream_compat)
{
	int device_id = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
	if (device_id == CCV_STREAM_GET_DEVICE_ID(CCV_COMPUTE_DEVICE_ANY))
	{
		CUDA_ENFORCE(cudaGetDevice(&device_id));
		if (stream_compat->_heap_gpu_size <= device_id)
		{
			if (!stream_compat->_heap_gpus)
				stream_compat->_heap_gpus = (ccv_nnc_stream_context_device_local_t*)cccalloc(device_id + 1, sizeof(ccv_nnc_stream_context_device_local_t));
			else {
				stream_compat->_heap_gpus = (ccv_nnc_stream_context_device_local_t*)ccrealloc(stream_compat->_heap_gpus, sizeof(ccv_nnc_stream_context_device_local_t) * (device_id + 1));
				memset(stream_compat->_heap_gpus + stream_compat->_heap_gpu_size, 0, sizeof(ccv_nnc_stream_context_device_local_t) * (device_id + 1 - stream_compat->_heap_gpu_size));
			}
			stream_compat->_heap_gpu_size = device_id + 1;
		}
		return stream_compat->_heap_gpus + device_id;
	} else {
		CUDA_ENFORCE(cudaSetDevice(device_id));
		return &stream_compat->_inline_gpu;
	}
}

static ccv_nnc_stream_context_compat_t* _ccv_nnc_default_stream_compat()
{
	static __thread ccv_nnc_stream_context_compat_t ccv_nnc_per_thread_gpu_stream_context = {
		.super = {
			.type = CCV_STREAM_CONTEXT_GPU | CCV_COMPUTE_DEVICE_ANY,
		},
	};
	return &ccv_nnc_per_thread_gpu_stream_context;
}

typedef struct {
	ccv_nnc_stream_signal_t super;
	cudaEvent_t event;
} ccv_nnc_stream_compat_signal_t;

ccv_nnc_stream_signal_t* ccv_nnc_init_stream_signal(ccv_nnc_stream_signal_t* const signal)
{
	assert(CCV_STREAM_GET_CONTEXT(((int*)signal)[0]) == CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)ccrealloc(signal, sizeof(ccv_nnc_stream_compat_signal_t));
	const int device = CCV_STREAM_GET_DEVICE_ID(compat_signal->super.type);
	CUDA_ENFORCE(cudaSetDevice(device));
	CUDA_ENFORCE(cudaEventCreateWithFlags(&compat_signal->event, cudaEventDisableTiming));
	return (ccv_nnc_stream_signal_t*)compat_signal;
}

void ccv_nnc_stream_compat_emit_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)signal;
	CUDA_ENFORCE(cudaEventRecord(compat_signal->event, device_local->stream));
}

void ccv_nnc_stream_compat_wait_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)signal;
	CUDA_ENFORCE(cudaStreamWaitEvent(device_local->stream, compat_signal->event, 0));
}

void ccv_nnc_deinit_stream_signal(ccv_nnc_stream_signal_t* const signal)
{
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)signal;
	const int device = CCV_STREAM_GET_DEVICE_ID(compat_signal->super.type);
	CUDA_ENFORCE(cudaSetDevice(device));
	CUDA_ENFORCE(cudaEventDestroy(compat_signal->event));
}

int ccv_nnc_gpu_device_count(void)
{
	int count = 0;
	CUDA_ENFORCE(cudaGetDeviceCount(&count));
	return count;
}

ccv_nnc_stream_context_t* ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	assert(CCV_STREAM_GET_CONTEXT(((int*)stream_context)[0]) == CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)ccrealloc(stream_context, sizeof(ccv_nnc_stream_context_compat_t));
	const ccv_nnc_stream_context_t super = stream_compat->super;
	memset(stream_compat, 0, sizeof(ccv_nnc_stream_context_compat_t));
	stream_compat->super = super;
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	CUDA_ENFORCE(cudaStreamCreate(&device_local->stream));
	return (ccv_nnc_stream_context_t*)stream_compat;
}

void* ccv_nnc_stream_compat_get_workspace(const ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem)
{
	if (workspace_size <= 0)
		return 0;
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	if (mem == CCV_TENSOR_CPU_MEMORY)
	{
		if (stream_compat->cpu.workspace_size >= workspace_size)
			return stream_compat->cpu.workspace;
		stream_compat->cpu.workspace_size = workspace_size;
		if (stream_compat->cpu.workspace)
			ccfree(stream_compat->cpu.workspace);
		stream_compat->cpu.workspace = 0;
		const int success = ccmemalign(&stream_compat->cpu.workspace, 16, workspace_size);
		return success != 0 ? 0 : stream_compat->cpu.workspace;
	} else if (mem == CCV_TENSOR_GPU_MEMORY) {
		ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
		if (device_local->workspace_size >= workspace_size)
			return device_local->workspace;
		int device_id;
		CUDA_ENFORCE(cudaGetDevice(&device_id));
		device_local->workspace_size = workspace_size;
		if (device_local->workspace)
			CUDA_ENFORCE(cudaFree(device_local->workspace));
		device_local->workspace = cumalloc(device_id, workspace_size);
		return device_local->workspace;
	}
	return 0;
}

void ccv_nnc_stream_compat_drain(ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	if (stream_compat->cpu.workspace)
	{
		ccfree(stream_compat->cpu.workspace);
		stream_compat->cpu.workspace = 0;
		stream_compat->cpu.workspace_size = 0;
	}
	const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
	if (device == CCV_STREAM_GET_DEVICE_ID(CCV_COMPUTE_DEVICE_ANY))
	{
		int i;
		for (i = 0; i < stream_compat->_heap_gpu_size; i++)
			if (stream_compat->_heap_gpus[i].workspace)
			{
				CUDA_ENFORCE(cudaSetDevice(i));
				CUDA_ENFORCE(cudaFree(stream_compat->_heap_gpus[i].workspace));
				stream_compat->_heap_gpus[i].workspace = 0;
				stream_compat->_heap_gpus[i].workspace_size = 0;
			}
	} else if (stream_compat->_inline_gpu.workspace) {
		CUDA_ENFORCE(cudaSetDevice(device));
		CUDA_ENFORCE(cudaFree(stream_compat->_inline_gpu.workspace));
		stream_compat->_inline_gpu.workspace = 0;
		stream_compat->_inline_gpu.workspace_size = 0;
	}
}

void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	CUDA_ENFORCE(cudaStreamSynchronize(device_local->stream));
}

typedef struct {
	ccv_nnc_async_callback_t async;
	ccv_nnc_async_callback_f fn;
} ccv_nnc_compat_callback_t;

#if CUDA_VERSION >= 10000
static void _co_stream_compat_callback(void* userdata)
#else
static void _co_stream_compat_callback(cudaStream_t stream, cudaError_t status, void* userdata)
#endif
{
	ccv_nnc_compat_callback_t* const callback = (ccv_nnc_compat_callback_t*)userdata;
	callback->fn(&callback->async);
}

void ccv_nnc_stream_compat_add_callback(ccv_nnc_stream_context_t* const stream, const ccv_nnc_callback_f callback, const ccv_nnc_async_callback_f async_callback, void* const callback_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	// If the stream is completed, no need to wait.
	if (cudaStreamQuery(device_local->stream) == cudaSuccess)
	{
		callback(callback_context);
		return;
	}
	ccv_nnc_compat_callback_t* const ctx = (ccv_nnc_compat_callback_t*)ccmalloc(sizeof(ccv_nnc_compat_callback_t));
	ctx->fn = async_callback;
	ctx->async.fn = callback;
	ctx->async.callback_context = callback_context;
#if CUDA_VERSION >= 10000
	cudaLaunchHostFunc(device_local->stream, _co_stream_compat_callback, ctx);
#else
	cudaStreamAddCallback(device_local->stream, _co_stream_compat_callback, ctx, 0);
#endif
}

#if CUDA_VERSION >= 10000
static void _co_stream_compat_resume(void* userdata)
#else
static void _co_stream_compat_resume(cudaStream_t stream, cudaError_t status, void* userdata)
#endif
{
	co_routine_t* const task = (co_routine_t*)userdata;
	co_scheduler_t* const scheduler = task->scheduler;
	pthread_mutex_lock(&scheduler->mutex);
	_co_prepend_task(scheduler, task);
	--scheduler->stream_await_count;
	pthread_cond_signal(&scheduler->wait);
	pthread_mutex_unlock(&scheduler->mutex);
}

int co_stream_compat_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	// If the stream is completed, no need to wait.
	if (cudaStreamQuery(device_local->stream) == cudaSuccess)
		return 1;
	co_scheduler_t* const scheduler = self->scheduler;
	pthread_mutex_lock(&scheduler->mutex);
	++scheduler->stream_await_count;
#if CUDA_VERSION >= 10000
	cudaLaunchHostFunc(device_local->stream, _co_stream_compat_resume, self);
#else
	cudaStreamAddCallback(device_local->stream, _co_stream_compat_resume, self, 0);
#endif
	pthread_mutex_unlock(&scheduler->mutex);
	return 0;
}

void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (stream_compat->cpu.workspace)
		ccfree(stream_compat->cpu.workspace);
	const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
	if (device == CCV_STREAM_GET_DEVICE_ID(CCV_COMPUTE_DEVICE_ANY))
	{
		int i;
		for (i = 0; i < stream_compat->_heap_gpu_size; i++)
			if (stream_compat->_heap_gpus[i].workspace)
			{
				CUDA_ENFORCE(cudaSetDevice(i));
				if (stream_compat->_heap_gpus[i].workspace)
					CUDA_ENFORCE(cudaFree(stream_compat->_heap_gpus[i].workspace));
				CUDA_ENFORCE(cudaStreamDestroy(stream_compat->_heap_gpus[i].stream));
				if (stream_compat->_heap_gpus[i].cublas)
					CUBLAS_ENFORCE(cublasDestroy(stream_compat->_heap_gpus[i].cublas));
				if (stream_compat->_heap_gpus[i].ones_16.data)
					CUDA_ENFORCE(cudaFree(stream_compat->_heap_gpus[i].ones_16.data));
				if (stream_compat->_heap_gpus[i].ones_32.data)
					CUDA_ENFORCE(cudaFree(stream_compat->_heap_gpus[i].ones_32.data));
				if (stream_compat->_heap_gpus[i].ones_64.data)
					CUDA_ENFORCE(cudaFree(stream_compat->_heap_gpus[i].ones_64.data));
#ifdef HAVE_CUDNN
				if (stream_compat->_heap_gpus[i].cudnn)
					CUDNN_ENFORCE(cudnnDestroy(stream_compat->_heap_gpus[i].cudnn));
				if (stream_compat->_heap_gpus[i].rngs)
					CUDA_ENFORCE(cudaFree(stream_compat->_heap_gpus[i].rngs));
#endif
			}
	} else {
		CUDA_ENFORCE(cudaSetDevice(device));
		if (stream_compat->_inline_gpu.workspace)
		{
			CUDA_ENFORCE(cudaFree(stream_compat->_inline_gpu.workspace));
		}
		CUDA_ENFORCE(cudaStreamDestroy(stream_compat->_inline_gpu.stream));
		if (stream_compat->_inline_gpu.cublas)
			CUBLAS_ENFORCE(cublasDestroy(stream_compat->_inline_gpu.cublas));
		if (stream_compat->_inline_gpu.ones_16.data)
			CUDA_ENFORCE(cudaFree(stream_compat->_inline_gpu.ones_16.data));
		if (stream_compat->_inline_gpu.ones_32.data)
			CUDA_ENFORCE(cudaFree(stream_compat->_inline_gpu.ones_32.data));
		if (stream_compat->_inline_gpu.ones_64.data)
			CUDA_ENFORCE(cudaFree(stream_compat->_inline_gpu.ones_64.data));
#ifdef HAVE_CUDNN
		if (stream_compat->_inline_gpu.cudnn)
			CUDNN_ENFORCE(cudnnDestroy(stream_compat->_inline_gpu.cudnn));
		if (stream_compat->_inline_gpu.rngs)
			CUDA_ENFORCE(cudaFree(stream_compat->_inline_gpu.rngs));
#endif
	}
#ifdef HAVE_NCCL
	if (stream_compat->super._inline_container[0])
	{
		ccv_nnc_stream_resource_container_compat_t* const resource_container_compat = (ccv_nnc_stream_resource_container_compat_t*)stream_compat->super._inline_container[0];
		if (resource_container_compat->comms)
		{
			int i;
			for (i = 0; i < resource_container_compat->comm_count; i++)
				NCCL_ENFORCE(ncclCommDestroy(resource_container_compat->comms[i]));
			ccfree(resource_container_compat->comms);
		}
		ccfree(resource_container_compat);
	}
#endif
}

int ccv_nnc_stream_context_get_device(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
	{
		int device = 0;
		CUDA_ENFORCE(cudaGetDevice(&device));
		return device;
	}
	const ccv_nnc_stream_context_compat_t* stream_compat = (const ccv_nnc_stream_context_compat_t*)stream_context;
	return CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
}

cudaStream_t ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	return device_local->stream;
}

cublasHandle_t ccv_nnc_stream_context_get_cublas(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	if (!device_local->cublas)
	{
		cublasCreate(&device_local->cublas);
		if (!device_local->cublas)
		{
			cutrigmp(); // Trigger memory pressure. And then do it again.
			CUBLAS_ENFORCE(cublasCreate(&device_local->cublas));
		}
		CUBLAS_ENFORCE(cublasSetStream(device_local->cublas, device_local->stream));
	}
	return device_local->cublas;
}

// A simple kernel to set all values to 1.
template<typename NUM>
__global__ static void _ones(NUM* x, int n)
{
	const int thidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thidx < n)
		x[thidx] = 1.;
}

template<typename ONES>
static void* _ccv_nnc_stream_context_get_ones(ONES &device_ones, const int n, cudaStream_t &stream)
{
	if (!device_ones.data || n > device_ones.n)
	{
		if (device_ones.data)
			cudaFree(device_ones.data);
		device_ones.n = n;
		cudaMalloc(&device_ones.data, sizeof(device_ones.data[0]) * n);
		if (!device_ones.data)
		{
			cutrigmp(); // Trigger memory pressure. And then do it again.
			CUDA_ENFORCE(cudaMalloc(&device_ones.data, sizeof(device_ones.data[0]) * n));
		}
		const int block_x = (n + 255) >> 8;
		_ones<<<block_x, 256, 0, stream>>>(device_ones.data, n);
	}
	return device_ones.data;
}

void* ccv_nnc_stream_context_get_ones(const ccv_nnc_stream_context_t* const stream_context, const int n, const int datatype)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	switch (datatype)
	{
		case CCV_16F:
			return _ccv_nnc_stream_context_get_ones(device_local->ones_16, n, device_local->stream);
		case CCV_64F:
			return _ccv_nnc_stream_context_get_ones(device_local->ones_64, n, device_local->stream);
		case CCV_32F:
		default:
			return _ccv_nnc_stream_context_get_ones(device_local->ones_32, n, device_local->stream);
	}
}

cudaDataType_t ccv_nnc_cuda_datatype(const int datatype)
{
	switch (datatype)
	{
		case CCV_8U:
			return CUDA_R_8I;
		case CCV_32S:
			return CUDA_R_32F;
		case CCV_16F:
			return CUDA_R_16F;
		case CCV_32F:
			return CUDA_R_32F;
		case CCV_64F:
			return CUDA_R_64F;
	}
	return CUDA_R_32F;
}

cudaDataType_t ccv_nnc_cuda_compute_datatype(const int datatype)
{
	switch (datatype)
	{
		case CCV_8U:
		case CCV_32S:
		case CCV_16F:
		case CCV_32F:
			return CUDA_R_32F;
		case CCV_64F:
			return CUDA_R_64F;
	}
	return CUDA_R_32F;
}

#ifdef HAVE_CUDNN

cudnnDataType_t ccv_nnc_cudnn_datatype(const int datatype)
{
	switch (datatype)
	{
		case CCV_8U:
			return CUDNN_DATA_INT8;
		case CCV_32S:
			return CUDNN_DATA_INT32;
		case CCV_16F:
			return CUDNN_DATA_HALF;
		case CCV_32F:
			return CUDNN_DATA_FLOAT;
		case CCV_64F:
			return CUDNN_DATA_DOUBLE;
	}
	return CUDNN_DATA_FLOAT;
}

cudnnHandle_t ccv_nnc_stream_context_get_cudnn(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	if (!device_local->cudnn)
	{
		cudnnCreate(&device_local->cudnn);
		if (!device_local->cudnn)
		{
			cutrigmp(); // Trigger memory pressure. And then do it again.
			CUDNN_ENFORCE(cudnnCreate(&device_local->cudnn));
		}
		CUDNN_ENFORCE(cudnnSetStream(device_local->cudnn, device_local->stream));
	}
	return device_local->cudnn;
}

cudnnActivationDescriptor_t ccv_nnc_stream_context_get_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnActivationDescriptor_t desc;
	cudnnCreateActivationDescriptor(&desc);
	return desc;
}

cudnnConvolutionDescriptor_t ccv_nnc_stream_context_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnConvolutionDescriptor_t desc;
	cudnnCreateConvolutionDescriptor(&desc);
	return desc;
}

cudnnDropoutDescriptor_t ccv_nnc_stream_context_get_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, const float p)
{
	cudnnDropoutDescriptor_t desc;
	cudnnCreateDropoutDescriptor(&desc);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat();
	ccv_nnc_stream_context_device_local_t* const device_local = _ccv_nnc_stream_compat_device_local(stream_compat);
	size_t state_size;
	cudnnDropoutGetStatesSize(cudnn, &state_size);
	const uint64_t seed = ccv_nnc_stream_context_genrand_uint32((ccv_nnc_stream_context_t*)stream_compat);
	if (device_local->rngs)
	{
#if CUDNN_VERSION >= 7100
		cudnnRestoreDropoutDescriptor(desc, cudnn, p, device_local->rngs, state_size, seed);
#else
		cudnnSetDropoutDescriptor(desc, cudnn, p, device_local->rngs, state_size, seed);
#endif
	} else {
		CUDA_ENFORCE(cudaMalloc(&device_local->rngs, state_size));
		cudnnSetDropoutDescriptor(desc, cudnn, p, device_local->rngs, state_size, seed);
	}
	return desc;
}

cudnnFilterDescriptor_t ccv_nnc_stream_context_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnFilterDescriptor_t desc;
	cudnnCreateFilterDescriptor(&desc);
	return desc;
}

cudnnOpTensorDescriptor_t ccv_nnc_stream_context_get_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnOpTensorDescriptor_t desc;
	cudnnCreateOpTensorDescriptor(&desc);
	return desc;
}

cudnnPoolingDescriptor_t ccv_nnc_stream_context_get_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnPoolingDescriptor_t desc;
	cudnnCreatePoolingDescriptor(&desc);
	return desc;
}

cudnnReduceTensorDescriptor_t ccv_nnc_stream_context_get_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnReduceTensorDescriptor_t desc;
	cudnnCreateReduceTensorDescriptor(&desc);
	return desc;
}

cudnnRNNDescriptor_t ccv_nnc_stream_context_get_rnn_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnRNNDescriptor_t desc;
	cudnnCreateRNNDescriptor(&desc);
	return desc;
}

cudnnRNNDataDescriptor_t ccv_nnc_stream_context_get_rnn_data_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnRNNDataDescriptor_t desc;
	cudnnCreateRNNDataDescriptor(&desc);
	return desc;
}

cudnnTensorDescriptor_t ccv_nnc_stream_context_get_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnTensorDescriptor_t desc;
	cudnnCreateTensorDescriptor(&desc);
	return desc;
}

void ccv_nnc_stream_context_return_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnActivationDescriptor_t activation_desc)
{
	cudnnDestroyActivationDescriptor(activation_desc);
}

void ccv_nnc_stream_context_return_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnConvolutionDescriptor_t convolution_desc)
{
	cudnnDestroyConvolutionDescriptor(convolution_desc);
}

void ccv_nnc_stream_context_return_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnDropoutDescriptor_t dropout_desc)
{
	cudnnDestroyDropoutDescriptor(dropout_desc);
}

void ccv_nnc_stream_context_return_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnFilterDescriptor_t filter_desc)
{
	cudnnDestroyFilterDescriptor(filter_desc);
}

void ccv_nnc_stream_context_return_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnOpTensorDescriptor_t op_tensor_desc)
{
	cudnnDestroyOpTensorDescriptor(op_tensor_desc);
}

void ccv_nnc_stream_context_return_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnPoolingDescriptor_t pooling_desc)
{
	cudnnDestroyPoolingDescriptor(pooling_desc);
}

void  ccv_nnc_stream_context_return_rnn_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnRNNDescriptor_t rnn_desc)
{
	cudnnDestroyRNNDescriptor(rnn_desc);
}

void  ccv_nnc_stream_context_return_rnn_data_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnRNNDataDescriptor_t rnn_data_desc)
{
	cudnnDestroyRNNDataDescriptor(rnn_data_desc);
}

void ccv_nnc_stream_context_return_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnReduceTensorDescriptor_t reduce_tensor_desc)
{
	cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc);
}

void ccv_nnc_stream_context_return_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnTensorDescriptor_t tensor_desc)
{
	cudnnDestroyTensorDescriptor(tensor_desc);
}

ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* const tensor)
{
	ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc = {
		stream_context,
		ccv_nnc_stream_context_get_tensor_descriptor(stream_context),
		tensor->data,
	};
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {};
	int stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	const int axis_count = ccv_nnc_tensor_nd(tensor->info.dim);
	const int* const inc = CCV_IS_TENSOR_VIEW(tensor) ? tensor->inc : tensor->info.dim;
	int i;
	for (i = axis_count; i < CCV_NNC_MAX_DIM + 2; i++)
		dim[i] = stride[i] = 1;
	dim[axis_count - 1] = tensor->info.dim[axis_count - 1];
	stride[axis_count - 1] = 1;
	for (i = axis_count - 2; i >= 0; i--)
	{
		dim[i] = tensor->info.dim[i];
		stride[i] = stride[i + 1] * inc[i + 1];
	}
	if (axis_count <= 4)
	{
		CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(tensor_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), dim[0], dim[1], dim[2], dim[3], stride[0], stride[1], stride[2], stride[3]));
	} else {
		CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(tensor_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), axis_count, dim, stride));
	}
	return tensor_desc;
}

ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* const tensor)
{
	ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc = {
		stream_context,
		ccv_nnc_stream_context_get_tensor_descriptor(stream_context),
		tensor->data,
	};
	// Fill up dimensions with 1s.
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {};
	int stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	const int axis_count = ccv_nnc_tensor_nd(tensor->info.dim);
	const int* const inc = CCV_IS_TENSOR_VIEW(tensor) ? tensor->inc : tensor->info.dim;
	int i;
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		switch (axis_count)
		{
			case 1:
				dim[0] = dim[2] = dim[3] = 1;
				dim[1] = tensor->info.dim[0];
				stride[0] = inc[0];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1;
				break;
			case 2:
				dim[0] = tensor->info.dim[0];
				dim[1] = tensor->info.dim[1];
				dim[2] = dim[3] = 1;
				stride[0] = inc[1];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1;
				break;
			case CCV_NNC_MAX_DIM + 1:
				dim[0] = 1;
				dim[1] = tensor->info.dim[0];
				stride[CCV_NNC_MAX_DIM + 1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i + 1];
					stride[i + 1] = stride[i + 2] * inc[i + 1];
				}
				stride[0] = stride[1] * inc[0];
				break;
			case CCV_NNC_MAX_DIM + 2:
				stride[CCV_NNC_MAX_DIM + 1] = 1;
				dim[CCV_NNC_MAX_DIM + 1] = tensor->info.dim[CCV_NNC_MAX_DIM + 1];
				for (i = CCV_NNC_MAX_DIM; i >= 0; i--)
				{
					dim[i] = tensor->info.dim[i];
					stride[i] = stride[i + 1] * inc[i + 1];
				}
				break;
			default:
				assert(0);
		}
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		switch (axis_count)
		{
			case 1:
				dim[0] = dim[2] = dim[3] = 1;
				dim[1] = tensor->info.dim[0];
				stride[0] = inc[0];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1; // Even though technically this should be inc[1] (because hw is after c), however, make it 1 doesn't have any differences and more versatile.
				break;
			case 2:
				dim[0] = tensor->info.dim[0];
				dim[1] = tensor->info.dim[1];
				dim[2] = dim[3] = 1;
				stride[0] = inc[1];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1; // Even though technically this should be inc[1] (because hw is after c), however, make it 1 doesn't have any differences and more versatile.
				break;
			case CCV_NNC_MAX_DIM + 1:
				dim[0] = 1;
				dim[1] = tensor->info.dim[CCV_NNC_MAX_DIM];
				stride[1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i];
					stride[i + 2] = (i == CCV_NNC_MAX_DIM - 1) ? inc[i + 1] : stride[i + 3] * inc[i + 1];
				}
				stride[0] = stride[2] * inc[0];
				break;
			case CCV_NNC_MAX_DIM + 2:
				dim[0] = tensor->info.dim[0];
				dim[1] = tensor->info.dim[CCV_NNC_MAX_DIM + 1];
				stride[1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i + 1];
					stride[i + 2] = (i == CCV_NNC_MAX_DIM - 1) ? inc[i + 2] : stride[i + 3] * inc[i + 2];
				}
				stride[0] = stride[2] * inc[1];
				break;
			default:
				assert(0);
		}
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_CHWN) {
		switch (axis_count)
		{
			case 1:
				dim[0] = dim[2] = dim[3] = 1;
				dim[1] = tensor->info.dim[0];
				stride[0] = inc[0];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1;
				break;
			case 2:
				dim[0] = tensor->info.dim[1];
				dim[1] = tensor->info.dim[0];
				dim[2] = dim[3] = 1;
				stride[0] = 1;
				stride[1] = inc[1];
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = inc[1];
				break;
			case CCV_NNC_MAX_DIM + 1:
				dim[0] = 1;
				dim[1] = tensor->info.dim[0];
				stride[CCV_NNC_MAX_DIM + 1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i + 1];
					stride[i + 1] = stride[i + 2] * inc[i + 1];
				}
				stride[0] = stride[1] * inc[0];
				break;
			case CCV_NNC_MAX_DIM + 2:
				dim[0] = tensor->info.dim[CCV_NNC_MAX_DIM + 1];
				stride[0] = 1;
				dim[CCV_NNC_MAX_DIM + 1] = tensor->info.dim[CCV_NNC_MAX_DIM];
				stride[CCV_NNC_MAX_DIM + 1] = inc[CCV_NNC_MAX_DIM + 1];
				for (i = CCV_NNC_MAX_DIM; i > 0; i--)
				{
					dim[i] = tensor->info.dim[i - 1];
					stride[i] = stride[i + 1] * inc[i]; // inc[i] is actually the one before.
				}
				break;
			default:
				assert(0);
		}
	}
	if (CCV_NNC_MAX_DIM == 2)
	{
		CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(tensor_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), dim[0], dim[1], dim[2], dim[3], stride[0], stride[1], stride[2], stride[3]));
	} else {
		CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(tensor_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), CCV_NNC_MAX_DIM + 2, dim, stride));
	}
	return tensor_desc;
}

void ccv_nnc_cudnn_deinit_tensor_view_descriptor(const ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc)
{
	ccv_nnc_stream_context_return_tensor_descriptor(tensor_desc.stream_context, tensor_desc.descriptor);
}

ccv_nnc_cudnn_filter_descriptor_t ccv_nnc_cudnn_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_t* const tensor)
{
	ccv_nnc_cudnn_filter_descriptor_t filter_desc = {
		stream_context,
		ccv_nnc_stream_context_get_filter_descriptor(stream_context),
		tensor->data,
	};
	assert(CCV_IS_TENSOR_CONTIGUOUS(tensor));
	const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
	assert(nd == CCV_NNC_MAX_DIM + 2);
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {};
	int i;
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		for (i = 0; i < nd; i++)
			dim[i] = tensor->info.dim[i];
		if (nd == 4)
		{
			CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(filter_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), CUDNN_TENSOR_NCHW, dim[0], dim[1], dim[2], dim[3]));
		} else {
			CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(filter_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), CUDNN_TENSOR_NCHW, nd, dim));
		}
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		dim[0] = tensor->info.dim[0];
		dim[1] = tensor->info.dim[nd - 1];
		for (i = 2; i < nd; i++)
			dim[i] = tensor->info.dim[i - 1];
		if (nd == 4)
		{
			CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(filter_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), CUDNN_TENSOR_NHWC, dim[0], dim[1], dim[2], dim[3]));
		} else {
			CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(filter_desc.descriptor, ccv_nnc_cudnn_datatype(tensor->info.datatype), CUDNN_TENSOR_NHWC, nd, dim));
		}
	}
	return filter_desc;
}

void ccv_nnc_cudnn_deinit_filter_descriptor(const ccv_nnc_cudnn_filter_descriptor_t filter_desc)
{
	ccv_nnc_stream_context_return_filter_descriptor(filter_desc.stream_context, filter_desc.descriptor);
}

ccv_nnc_cudnn_convolution_descriptor_t ccv_nnc_cudnn_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_hint_t hint, const int datatype)
{
	ccv_nnc_cudnn_convolution_descriptor_t convolution_desc = {
		stream_context,
		ccv_nnc_stream_context_get_convolution_descriptor(stream_context),
	};
	int i;
	int p[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		p[i] = ccv_max(hint.border.begin[i], hint.border.end[i]);
	int v[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		v[i] = hint.stride.dim[i];
	if (CCV_NNC_MAX_DIM == 2)
	{
		CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(convolution_desc.descriptor, p[0], p[1], v[0], v[1], 1, 1, CUDNN_CROSS_CORRELATION, ccv_nnc_cudnn_datatype(datatype)));
	} else {
		int u[CCV_NNC_MAX_DIM];
		for (i = 0; i < CCV_NNC_MAX_DIM; i++)
			u[i] = 1;
		CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(convolution_desc.descriptor, CCV_NNC_MAX_DIM, p, v, u, CUDNN_CROSS_CORRELATION, ccv_nnc_cudnn_datatype(datatype)));
	}
	CUDNN_ENFORCE(cudnnSetConvolutionMathType(convolution_desc.descriptor, CUDNN_TENSOR_OP_MATH));
	return convolution_desc;
}

void ccv_nnc_cudnn_deinit_convolution_descriptor(const ccv_nnc_cudnn_convolution_descriptor_t convolution_desc)
{
	ccv_nnc_stream_context_return_convolution_descriptor(convolution_desc.stream_context, convolution_desc.descriptor);
}
#endif

#ifdef HAVE_NCCL
static void _ccv_nnc_nccl_redo_comms(ncclComm_t* const comms, const int comm_count, const int device_count)
{
	int i;
	for (i = 0; i < comm_count; i++)
		NCCL_ENFORCE(ncclCommDestroy(comms[i]));
	int devs[device_count];
	for (i = 0; i < device_count; i++)
		devs[i] = i;
	NCCL_ENFORCE(ncclCommInitAll(comms, device_count, devs));
}

ncclComm_t ccv_nnc_nccl_get_comm(ccv_nnc_stream_context_t* const stream, const int device_count, const int device_id)
{
	assert(device_count > 0);
	if (stream)
	{
		ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
		if (!stream_compat->super.resource_container)
			stream_compat->super.resource_container = stream_compat->super._inline_container;
		if (!stream_compat->super.resource_container[0])
			stream_compat->super.resource_container[0] = (ccv_nnc_stream_resource_container_t*)cccalloc(1, sizeof(ccv_nnc_stream_resource_container_compat_t));
		ccv_nnc_stream_resource_container_compat_t* const resource_container_compat = (ccv_nnc_stream_resource_container_compat_t*)stream_compat->super.resource_container[0];
		if (resource_container_compat->comms && resource_container_compat->comm_count == device_count)
			return resource_container_compat->comms[device_id];
		if (resource_container_compat->comms)
			resource_container_compat->comms = (ncclComm_t*)ccrealloc(resource_container_compat->comms, sizeof(ncclComm_t) * device_count);
		else
			resource_container_compat->comms = (ncclComm_t*)ccmalloc(sizeof(ncclComm_t) * device_count);
		_ccv_nnc_nccl_redo_comms(resource_container_compat->comms, resource_container_compat->comm_count, device_count);
		resource_container_compat->comm_count = device_count;
		return resource_container_compat->comms[device_id];
	} else {
		static ncclComm_t comms[CCV_TENSOR_GET_DEVICE_ID(CCV_COMPUTE_DEVICE_ANY)];
		static int comm_count = 0;
		if (comm_count != device_count)
		{
			_ccv_nnc_nccl_redo_comms(comms, comm_count, device_count);
			comm_count = device_count;
		}
		return comms[device_id];
	}
}

ncclDataType_t ccv_nnc_nccl_datatype(const int datatype)
{
	switch (datatype)
	{
		case CCV_8U:
			return ncclUint8;
		case CCV_32S:
			return ncclInt;
		case CCV_16F:
			return ncclHalf;
		case CCV_32F:
			return ncclFloat;
		case CCV_64F:
			return ncclDouble;
	}
	return ncclFloat;
}
#endif
